# Quick per-call allocation profile for the BTD path. Goal: spot any
# hot-loop functions still allocating, post-PDMats refactor.
#
# Usage: julia --project=benchmarking benchmarking/alloc_profile.jl

using StateSpaceDynamics
using LinearAlgebra, Random
using BenchmarkTools

const D, p, T, N = 5, 10, 200, 8

function make_lds(seed=42)
    rng = MersenneTwister(seed)
    A = StateSpaceDynamics.random_rotation_matrix(D, rng)
    Q = (M = randn(rng, D, D); M*M' + 1e-3 * I)
    x0 = randn(rng, D)
    P0 = (M = randn(rng, D, D); M*M' + 1e-3 * I)
    C = randn(rng, p, D)
    R = (M = randn(rng, p, p); M*M' + 1e-3 * I)
    b = randn(rng, D)
    d = randn(rng, p)
    sm = GaussianStateModel(; A=Matrix(A), Q=Matrix(Q), b=b, x0=x0, P0=Matrix(P0))
    om = GaussianObservationModel(; C=C, R=Matrix(R), d=d)
    return LinearDynamicalSystem(sm, om)
end

rng = MersenneTwister(123)
lds = make_lds()
_, y = StateSpaceDynamics.rand(rng, lds, fill(T, N))

# Warm precompile
StateSpaceDynamics.fit!(make_lds(), y; max_iter=1, progress=false)

println("BTD path — fit! over $N trials of length $T, latent $D, obs $p")
println("="^70)

# Whole-fit baseline
println("\n[fit! max_iter=20]")
b_fit = @benchmark StateSpaceDynamics.fit!(
    lds_copy, $y; max_iter=20, tol=0.0, progress=false
) setup=(lds_copy = $(make_lds)()) samples=5 seconds=15
display(b_fit)

# Set up workspace + tfs for component-level benches
tsteps_per_trial = fill(T, N)
T_max = maximum(tsteps_per_trial)
tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, tsteps_per_trial)
sws_pool = [
    StateSpaceDynamics.SmoothWorkspace(Float64, D, p, T_max; u_dim=0, d_dim=0)
    for _ in 1:Threads.maxthreadid()
]
suf = StateSpaceDynamics._initialize_td_sufficient_statistics(Float64, lds, tsteps_per_trial)
u_seq = [zeros(0, T) for _ in 1:N]
v_seq = [zeros(0, T) for _ in 1:N]
StateSpaceDynamics._td_init_const_blocks!(sws_pool[1], lds, tsteps_per_trial, y, u_seq, v_seq)

println("\n\n[compute_smooth_constants! (1 call)]")
display(@benchmark StateSpaceDynamics.compute_smooth_constants!($sws_pool[1], $lds))

println("\n[smooth! multi-trial (1 E-step)]")
display(@benchmark StateSpaceDynamics.smooth!($lds, $tfs, $y, $sws_pool, $u_seq, $v_seq))

println("\n[_aggregate_td_suff_stats! (1 call)]")
StateSpaceDynamics.smooth!(lds, tfs, y, sws_pool, u_seq, v_seq)
display(@benchmark StateSpaceDynamics._aggregate_td_suff_stats!(
    $suf, $tfs, $lds, $u_seq, $v_seq, $y, $sws_pool[1]
))

println("\n[calculate_elbo (1 call)]")
total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths)
display(@benchmark StateSpaceDynamics.calculate_elbo($lds, $suf, $sws_pool[1], $total_entropy))

println("\n[mstep! (1 call)]")
display(@benchmark StateSpaceDynamics.mstep!($lds, $suf, $sws_pool[1]))
