using StateSpaceDynamics, LinearAlgebra, Random, Optim, LineSearches, SparseArrays
using BenchmarkTools
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

# The package dropped the allocating Gradient/Hessian wrappers; build the
# workspaces once and shim the old allocating API over the `!` versions.
const grad_ws = StateSpaceDynamics.SmoothWorkspace(Float64, D, p, T_t; ux_dim=0, uy_dim=0)
StateSpaceDynamics.compute_smooth_constants!(grad_ws, lds)
const hess_ws = StateSpaceDynamics.BlockTridiagonalWorkspace(Float64, D, T_t)

function nll(vec_x::AbstractVector{Float64})
    x = reshape(vec_x, D, T_t)
    return -sum(StateSpaceDynamics.loglikelihood(x, lds, y))
end

function g!(g::Vector{Float64}, vec_x::Vector{Float64})
    x = reshape(vec_x, D, T_t)
    grad = StateSpaceDynamics.Gradient!(grad_ws, lds, y, x)
    g .= vec(-grad)
    return g
end

function h!(h::SparseMatrixCSC{Float64,Int}, vec_x::Vector{Float64})
    x = reshape(vec_x, D, T_t)
    H = StateSpaceDynamics.Hessian!(hess_ws, lds, y, x)
    # Negate in-place — Hessian returns the joint log-likelihood Hessian
    # (negative-definite at the MAP); Optim minimizes, so we flip sign.
    h.nzval .= -H.nzval
    return nothing
end

X₀ = zeros(D * T_t)

# Pre-build with sparse Hessian storage — mirrors what main's smooth!
# does (`spzeros(T, length(X₀), length(X₀))` + `h!(initial_h, X₀)`).
initial_f = nll(X₀)
initial_g = similar(X₀)
g!(initial_g, X₀)
# Build the sparse pattern by calling Hessian once and copying its
# structure (zeros), then h! overwrites the values.
_H = StateSpaceDynamics.Hessian!(hess_ws, lds, y, reshape(X₀, D, T_t))
initial_h = SparseMatrixCSC{Float64,Int}(
    _H.m, _H.n, _H.colptr, _H.rowval, zeros(length(_H.nzval))
)
h!(initial_h, X₀)

td = TwiceDifferentiable(nll, g!, h!, X₀, initial_f, initial_g, initial_h)
opts = Optim.Options(; g_abstol=1e-8, x_abstol=1e-8, f_abstol=1e-8, iterations=100)

res = optimize(td, copy(X₀), Newton(; linesearch=LineSearches.BackTracking()), opts)
println("Optim Newton (main-style):")
println("  iterations = $(res.iterations)")
println("  f_calls    = $(res.f_calls)")
println("  g_calls    = $(res.g_calls)")
println("  h_calls    = $(res.h_calls)")

b_optim = @benchmark optimize(
    $td, copy($X₀), Newton(; linesearch=LineSearches.BackTracking()), $opts
)
println("  time       = $(round(median(b_optim).time / 1e6; digits=2)) ms")
println(
    "  mem        = $(round(b_optim.memory / 1024; digits=1)) KB / $(b_optim.allocs) allocs"
)

# Hand-rolled benchmark
tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, [T_t])
sws = StateSpaceDynamics.SmoothWorkspace(Float64, D, p, T_t; ux_dim=0, uy_dim=0)
for _ in 1:3
    StateSpaceDynamics.smooth!(lds, tfs[1], y, sws)
end

b_ours = @benchmark StateSpaceDynamics.smooth!($lds, $(tfs[1]), $y, $sws)
println("\nHand-rolled newton_smooth! (current dev_ryan_):")
println("  time       = $(round(median(b_ours).time / 1e6; digits=2)) ms")
println(
    "  mem        = $(round(b_ours.memory / 1024; digits=2)) KB / $(b_ours.allocs) allocs"
)
