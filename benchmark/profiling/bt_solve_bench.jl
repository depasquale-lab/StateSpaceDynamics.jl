using StateSpaceDynamics, LinearAlgebra, Random, SparseArrays, BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 3.0
BenchmarkTools.DEFAULT_PARAMETERS.samples = 5

function make_poisson_lds(D, p; seed=42)
    rng = MersenneTwister(seed)
    A = 0.9 .* StateSpaceDynamics.random_rotation_matrix(D, rng)
    Q = Matrix(0.1 * I(D))
    x0 = zeros(D)
    P0 = Matrix(0.1 * I(D))
    C = 0.3 .* randn(rng, p, D)
    d = log.(0.5 .+ rand(rng, p))
    b = zeros(D)
    sm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
    om = PoissonObservationModel(; C=C, d=d)
    return LinearDynamicalSystem(sm, om)
end

const D, p, T_t = 5, 10, 200
lds = make_poisson_lds(D, p)
rng = MersenneTwister(123)
_, y_multi = StateSpaceDynamics.rand(rng, lds, fill(T_t, 1))
y = y_multi[1]

# Set up the workspace + run smooth! once so the Hessian buffers are populated.
sws = StateSpaceDynamics.SmoothWorkspace(Float64, D, p, T_t; ux_dim=0, uy_dim=0)
StateSpaceDynamics.compute_smooth_constants!(sws, lds)

x0_vec = zeros(D * T_t)
x0_mat = reshape(x0_vec, D, T_t)
g_mat = similar(x0_mat)
p_mat = similar(x0_mat)
btd = sws.btd

# Time the Hessian build
println("=== Hessian build + neg ===")
b = @benchmark begin
    StateSpaceDynamics._fill_hessian_blocks_poisson!($sws, $lds, $x0_mat)
    StateSpaceDynamics._negate_blocks!($btd, $T_t)
end
println(
    "  t=$(round(median(b).time / 1e6; digits=3)) ms allocs=$(b.allocs) mem=$(b.memory) B"
)

# Build Hessian once, then time the BT solve
StateSpaceDynamics._fill_hessian_blocks_poisson!(sws, lds, x0_mat)
StateSpaceDynamics._negate_blocks!(btd, T_t)
neg_sub_v = view(btd.neg_sub, 1:(T_t - 1))
neg_diag_v = view(btd.neg_diag, 1:T_t)
neg_super_v = view(btd.neg_super, 1:(T_t - 1))

# Vec form (matches what newton_smooth! does)
fill!(g_mat, 1.0)
g_vec = vec(g_mat)
p_vec = vec(p_mat)

println("\n=== block_tridiagonal_solve! ===")
b2 = @benchmark StateSpaceDynamics.block_tridiagonal_solve!(
    $p_vec, $neg_sub_v, $neg_diag_v, $neg_super_v, $g_vec, $btd
)
println(
    "  t=$(round(median(b2).time / 1e6; digits=3)) ms allocs=$(b2.allocs) mem=$(b2.memory) B",
)

# Time the workspace joint_loglikelihood! (the per-eval cost in the line search)
println("\n=== sum(joint_loglikelihood!(ws, x, lds, y, lognorm_t)) (one phi-eval) ===")
lognorm_t = StateSpaceDynamics._poisson_lognorm_t(y)
b3 = @benchmark sum(
    StateSpaceDynamics.joint_loglikelihood!($sws, $x0_mat, $lds, $y, $lognorm_t)
)
println(
    "  t=$(round(median(b3).time / 1e6; digits=3)) ms allocs=$(b3.allocs) mem=$(b3.memory) B",
)

# Compare to UMFPACK solve on equivalent sparse Hessian
println("\n=== UMFPACK sparse solve (for comparison) ===")
hess_ws = StateSpaceDynamics.BlockTridiagonalWorkspace(Float64, D, T_t)
_H = StateSpaceDynamics.Hessian!(hess_ws, lds, y, x0_mat)
neg_H = -_H
b4 = @benchmark $neg_H \ $g_vec
println(
    "  t=$(round(median(b4).time / 1e6; digits=3)) ms allocs=$(b4.allocs) mem=$(round(b4.memory/1024; digits=1)) KB",
)
