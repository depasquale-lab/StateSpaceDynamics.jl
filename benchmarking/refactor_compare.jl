# Compare the refactored BTD/Kalman/Poisson paths against the pre-refactor
# baseline. Designed to run from either branch with the *same* script.
#
# Each block uses only the public API (`fit!`, `LinearDynamicalSystem`,
# `GaussianStateModel`, `GaussianObservationModel`, `PoissonObservationModel`)
# so the same benchmark runs on `main` (pre-refactor) and `dev_ryan_`
# (post-refactor). Output is CSV — diff the two CSVs side by side.
#
# Usage:
#   julia --project=benchmarking benchmarking/refactor_compare.jl > refactor_dev.csv
#   git checkout main
#   julia --project=benchmarking benchmarking/refactor_compare.jl > refactor_main.csv

using StateSpaceDynamics
using BenchmarkTools
using LinearAlgebra
using Random
using Printf

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 2.0
BenchmarkTools.DEFAULT_PARAMETERS.samples = 5

const SCENARIOS = [
    # (label, D_latent, p_obs, T_per_trial, N_trials)
    ("small",          3,  5,  100,  4),
    ("medium",         5, 10,  200,  8),
    ("large",          8, 16,  500, 16),
    ("long_single",    4,  8, 2000,  1),
    ("many_short",     3,  5,   50, 64),
]

const MAX_EM = 20

function make_gaussian_lds(D, p; seed=42, kalman=false)
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
    return LinearDynamicalSystem(sm, om; kalman_filter=kalman)
end

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

function bench_gaussian_td(label, D, p, T, N)
    lds = make_gaussian_lds(D, p)
    rng = MersenneTwister(123)
    _, y = StateSpaceDynamics.rand(rng, lds, fill(T, N))

    # warm up the precompile path
    StateSpaceDynamics.fit!(make_gaussian_lds(D, p), y; max_iter=1, progress=false)

    bench = @benchmark StateSpaceDynamics.fit!(
        lds_copy, $y; max_iter=$MAX_EM, tol=0.0, progress=false
    ) setup=(lds_copy = $(make_gaussian_lds)($D, $p))
    return (
        scenario=label, path="TD", D=D, p=p, T=T, N=N,
        time_s=median(bench).time / 1e9,
        mem_mb=bench.memory / 1024^2,
        allocs=bench.allocs,
    )
end

function bench_gaussian_kalman(label, D, p, T, N)
    lds = make_gaussian_lds(D, p; kalman=true)
    rng = MersenneTwister(123)
    # Kalman path expects 3D y
    _, y_vec = StateSpaceDynamics.rand(rng, lds, fill(T, N))
    y3 = cat(y_vec...; dims=3)

    StateSpaceDynamics.fit!(make_gaussian_lds(D, p; kalman=true), y3; max_iter=1, progress=false)

    bench = @benchmark StateSpaceDynamics.fit!(
        lds_copy, $y3; max_iter=$MAX_EM, tol=0.0, progress=false
    ) setup=(lds_copy = $(make_gaussian_lds)($D, $p; kalman=true))
    return (
        scenario=label, path="Kalman", D=D, p=p, T=T, N=N,
        time_s=median(bench).time / 1e9,
        mem_mb=bench.memory / 1024^2,
        allocs=bench.allocs,
    )
end

function bench_poisson(label, D, p, T, N)
    lds = make_poisson_lds(D, p)
    rng = MersenneTwister(123)
    _, y = StateSpaceDynamics.rand(rng, lds, fill(T, N))

    StateSpaceDynamics.fit!(make_poisson_lds(D, p), y; max_iter=1, progress=false)

    bench = @benchmark StateSpaceDynamics.fit!(
        lds_copy, $y; max_iter=$MAX_EM, tol=0.0, progress=false
    ) setup=(lds_copy = $(make_poisson_lds)($D, $p))
    return (
        scenario=label, path="Poisson", D=D, p=p, T=T, N=N,
        time_s=median(bench).time / 1e9,
        mem_mb=bench.memory / 1024^2,
        allocs=bench.allocs,
    )
end

println("scenario,path,D,p,T,N,time_s,mem_mb,allocs")
for (label, D, p, T, N) in SCENARIOS
    @info "Benchmarking $label (D=$D, p=$p, T=$T, N=$N)"
    for f in (bench_gaussian_td, bench_gaussian_kalman, bench_poisson)
        try
            r = f(label, D, p, T, N)
            @printf("%s,%s,%d,%d,%d,%d,%.6f,%.3f,%d\n",
                    r.scenario, r.path, r.D, r.p, r.T, r.N,
                    r.time_s, r.mem_mb, r.allocs)
            flush(stdout)
        catch e
            @warn "Failed $(f) for $label: $e"
            @printf("%s,%s,%d,%d,%d,%d,NaN,NaN,NaN\n", label, "?", D, p, T, N)
            flush(stdout)
        end
    end
end
