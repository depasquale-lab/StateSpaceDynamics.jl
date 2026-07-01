# Per-component allocation profile for the Gaussian LDS BTD/suf-based fit!.
# Goal: spot any hot-loop function still allocating after the suf-based
# E-step + IW/MN-MAP M-step unification.
#
# Usage: julia --project=benchmark/comparison benchmark/profiling/profile_allocations.jl
#
# Companion to alloc_profile.jl: that script profiles the high-level E-step
# stages (smooth! / aggregate / elbo! / mstep!); this one drills down into the
# inner Newton/BTD kernels and the individual suf-based M-step updates.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "comparison"))

using StateSpaceDynamics
using LinearAlgebra
using Random
using BenchmarkTools

# Build a single-trial Gaussian LDS plus a sampled observation sequence.
function create_test_lds(latent_dim::Int, obs_dim::Int, seq_length::Int)
    A = random_rotation_matrix(latent_dim)
    C = randn(obs_dim, latent_dim)
    Q = Matrix(I(latent_dim) * 0.1)
    R = Matrix(I(obs_dim) * 0.5)
    b = zeros(latent_dim)
    d = zeros(obs_dim)
    x0_mean = zeros(latent_dim)
    x0_cov = Matrix(I(latent_dim) * 0.1)

    state_model = GaussianStateModel(A, Q, b, x0_mean, x0_cov)
    obs_model = GaussianObservationModel(C, R, d)
    lds = LinearDynamicalSystem(state_model, obs_model)

    rng = MersenneTwister(42)
    x, y = rand(rng, lds, seq_length)
    return lds, x, y
end

const T = Float64
const latent_dim = 64
const obs_dim = 10
const seq_length = 100

println("="^60)
println("Allocation Profiling for Gaussian LDS fit! (suf-based path)")
println("="^60)

lds, x, y = create_test_lds(latent_dim, obs_dim, seq_length)
model = LinearDynamicalSystem(lds.state_model, lds.obs_model)

# Warm up
println("\n[Warming up...]")
fit!(deepcopy(model), y; max_iter=1, progress=false)

# Full-fit baseline
println("\n[Full benchmark: fit! with max_iter=10]")
result = @benchmark fit!(deepcopy($model), $y; max_iter=10, progress=false)
display(result)

println("\n\n" * "="^60)
println("Per-component breakdown:")
println("="^60)

tsteps = size(y, 2)
T_max = tsteps

# Multi-trial API takes a Vector{Matrix}; single trial wraps as a 1-element vec.
y_multi = [y]
tsteps_per_trial = [tsteps]

# Workspace pool + FilterSmooth + suff-stats aggregator, set up exactly as
# `_fit_tridiag!` does for the no-input (ux_dim = uy_dim = 0) case.
tfs = StateSpaceDynamics.initialize_FilterSmooth(model, tsteps_per_trial)
sws_pool = [
    StateSpaceDynamics.SmoothWorkspace(T, latent_dim, obs_dim, T_max; ux_dim=0, uy_dim=0)
    for _ in 1:Threads.maxthreadid()
]
sws = sws_pool[1]
suf = StateSpaceDynamics._initialize_td_sufficient_statistics(T, model, tsteps_per_trial)
ux_seq = [zeros(T, 0, tsteps)]
uy_seq = [zeros(T, 0, tsteps)]
StateSpaceDynamics._td_init_const_blocks!(
    sws, model, tsteps_per_trial, y_multi, ux_seq, uy_seq
)

# ---- E-step stages ----------------------------------------------------------

println("\n[smooth! - single trial]")
fs = tfs[1]
result_smooth = @benchmark StateSpaceDynamics.smooth!($model, $fs, $y, $sws)
display(result_smooth)

println("\n[smooth! - multi-trial (1 E-step)]")
result_smooth_multi = @benchmark StateSpaceDynamics.smooth!(
    $model, $tfs, $y_multi, $sws_pool, $ux_seq, $uy_seq
)
display(result_smooth_multi)

println("\n[_aggregate_td_suff_stats! - 1 call]")
StateSpaceDynamics.smooth!(model, tfs, y_multi, sws_pool, ux_seq, uy_seq)
result_agg = @benchmark StateSpaceDynamics._aggregate_td_suff_stats!(
    $suf, $tfs, $model, $ux_seq, $uy_seq, $y_multi, $sws
)
display(result_agg)

# Refresh the aggregated stats so the suf-based benches below read valid data.
StateSpaceDynamics._aggregate_td_suff_stats!(suf, tfs, model, ux_seq, uy_seq, y_multi, sws)

println("\n[elbo! - suf-based]")
total_entropy = sum(fsi.entropy for fsi in tfs.FilterSmooths)
result_elbo = @benchmark StateSpaceDynamics.elbo!($model, $suf, $sws, $total_entropy)
display(result_elbo)

println("\n[mstep! - full]")
result_mstep = @benchmark StateSpaceDynamics.mstep!($model, $suf, $sws)
display(result_mstep)

# ---- ELBO Q-terms (suf-based) ----------------------------------------------

println("\n[Q_state! - suf-based]")
result_qstate = @benchmark StateSpaceDynamics.Q_state!($sws, $model, $suf)
display(result_qstate)

println("\n[Q_obs! - suf-based]")
result_qobs = @benchmark StateSpaceDynamics.Q_obs!($sws, $model, $suf)
display(result_qobs)

println("\n[compute_smooth_constants!]")
result_csc = @benchmark StateSpaceDynamics.compute_smooth_constants!($sws, $model)
display(result_csc)

# ---- Inner Newton / BTD kernels --------------------------------------------
# These operate on the per-timestep latent trajectory; rebuild the smooth
# constants first so the Cholesky-derived buffers in `sws` are populated.
StateSpaceDynamics.compute_smooth_constants!(sws, model)
StateSpaceDynamics.smooth!(model, fs, y, sws)
x_mat = fs.x_smooth  # (D × T) smoothed latent trajectory

println("\n[Gradient!]")
result_grad = @benchmark StateSpaceDynamics.Gradient!($sws, $model, $y, $x_mat)
display(result_grad)

println("\n[Hessian!]")
result_hess = @benchmark StateSpaceDynamics.Hessian!($sws, $model, $y, $x_mat)
display(result_hess)

println("\n[block_tridiagonal_inverse_logdet!]")
btd = sws.btd
StateSpaceDynamics.Hessian!(sws, model, y, x_mat)
StateSpaceDynamics._negate_blocks!(btd)
result_btil = @benchmark StateSpaceDynamics.block_tridiagonal_inverse_logdet!(
    $(fs.p_smooth),
    $(fs.p_smooth_tt1),
    $(btd.neg_sub),
    $(btd.neg_diag),
    $(btd.neg_super),
    $btd,
)
display(result_btil)

println("\n[block_tridiagonal_solve!]")
StateSpaceDynamics.Gradient!(sws, model, y, x_mat)
copyto!(sws.grad_vec, 1, sws.grad_buf, 1, length(sws.grad_vec))
sws.grad_vec .*= -1.0
StateSpaceDynamics.Hessian!(sws, model, y, x_mat)
StateSpaceDynamics._negate_blocks!(btd)
result_solve = @benchmark StateSpaceDynamics.block_tridiagonal_solve!(
    $(sws.X₀), $(btd.neg_sub), $(btd.neg_diag), $(btd.neg_super), $(sws.grad_vec), $btd
)
display(result_solve)

# ---- Individual suf-based M-step updates ------------------------------------

println("\n[update_initial_state_mean!]")
result_x0 = @benchmark StateSpaceDynamics.update_initial_state_mean!($model, $suf)
display(result_x0)

println("\n[update_initial_state_covariance!]")
result_p0 = @benchmark StateSpaceDynamics.update_initial_state_covariance!(
    $model, $suf, $sws
)
display(result_p0)

println("\n[update_A_b!]")
result_ab = @benchmark StateSpaceDynamics.update_A_b!($model, $suf, $sws)
display(result_ab)

println("\n[update_Q!]")
result_q = @benchmark StateSpaceDynamics.update_Q!($model, $suf, $sws)
display(result_q)

println("\n[update_C_d!]")
result_cd = @benchmark StateSpaceDynamics.update_C_d!($model, $suf, $sws)
display(result_cd)

println("\n[update_R!]")
result_r = @benchmark StateSpaceDynamics.update_R!($model, $suf, $sws)
display(result_r)

println("\n" * "="^60)
println("Summary of major allocation sources:")
println("="^60)

println("\nsmooth! (single):     $(BenchmarkTools.prettymemory(result_smooth.memory))")
println("smooth! (multi):      $(BenchmarkTools.prettymemory(result_smooth_multi.memory))")
println("_aggregate_suff:      $(BenchmarkTools.prettymemory(result_agg.memory))")
println("elbo!:                $(BenchmarkTools.prettymemory(result_elbo.memory))")
println("mstep!:               $(BenchmarkTools.prettymemory(result_mstep.memory))")
println("Q_state!:             $(BenchmarkTools.prettymemory(result_qstate.memory))")
println("Q_obs!:               $(BenchmarkTools.prettymemory(result_qobs.memory))")

# Estimate per-iteration footprint (one E-step + ELBO + M-step).
per_iter =
    result_smooth_multi.memory +
    result_agg.memory +
    result_elbo.memory +
    result_mstep.memory
println("\nEstimated per-iteration:    $(BenchmarkTools.prettymemory(per_iter))")
println("Estimated for 10 iterations: $(BenchmarkTools.prettymemory(10 * per_iter))")
