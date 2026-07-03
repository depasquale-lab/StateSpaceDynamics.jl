#=
AirspeedVelocity / PkgBenchmark entry point.

Defines `const SUITE::BenchmarkGroup`, which `benchpkg` runs against multiple
git revisions to track performance across commits and PRs. Keep this file
self-contained and side-effect-free apart from populating `SUITE`.

Run locally with:
    julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
    benchpkg StateSpaceDynamics --rev=dd/HEAD --bench-on=HEAD   # see .github/BENCHMARKING.md
=#

using StateSpaceDynamics
using BenchmarkTools
using LinearAlgebra
using StableRNGs

const SUITE = BenchmarkGroup()

# Gaussian LDS smoothing 
SUITE["GaussianLDS"] = BenchmarkGroup(["LDS", "Gaussian"])
for latent_dim in (2, 4, 8), obs_dim in (5, 10, 20)
    obs_dim < latent_dim && continue
    for T in (100, 500)
        rng = StableRNG(1234)

        A = 0.95 * Matrix(I, latent_dim, latent_dim)
        Q = Matrix(0.1 * I, latent_dim, latent_dim)
        b = zeros(latent_dim)
        x0 = zeros(latent_dim)
        P0 = Matrix(0.1 * I, latent_dim, latent_dim)

        C = randn(rng, obs_dim, latent_dim)
        R = Matrix(0.1 * I, obs_dim, obs_dim)
        d = zeros(obs_dim)

        state_model = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
        obs_model = GaussianObservationModel(; C=C, R=R, d=d)
        model = LinearDynamicalSystem(;
            state_model=state_model,
            obs_model=obs_model,
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            fit_bool=fill(true, 6),
        )

        _, y = rand(rng, model, T)

        SUITE["GaussianLDS"]["smooth", "latent=$latent_dim", "obs=$obs_dim", "T=$T"] = @benchmarkable smooth(
            $model, $y
        ) samples = 10 seconds = 5
    end
end

# Poisson LDS smoothing
SUITE["PoissonLDS"] = BenchmarkGroup(["LDS", "Poisson"])
for latent_dim in (2, 4), obs_dim in (5, 10)
    obs_dim < latent_dim && continue
    for T in (100, 500)
        rng = StableRNG(1234)

        A = 0.95 * Matrix(I, latent_dim, latent_dim)
        Q = Matrix(0.1 * I, latent_dim, latent_dim)
        b = zeros(latent_dim)
        x0 = zeros(latent_dim)
        P0 = Matrix(0.1 * I, latent_dim, latent_dim)

        C = abs.(randn(rng, obs_dim, latent_dim))
        d = log.(fill(0.1, obs_dim))

        state_model = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
        obs_model = PoissonObservationModel(; C=C, d=d)
        model = LinearDynamicalSystem(;
            state_model=state_model,
            obs_model=obs_model,
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            fit_bool=fill(true, 5),
        )

        _, y = rand(rng, model, T)

        SUITE["PoissonLDS"]["smooth", "latent=$latent_dim", "obs=$obs_dim", "T=$T"] = @benchmarkable smooth(
            $model, $y
        ) samples = 10 seconds = 5
    end
end
