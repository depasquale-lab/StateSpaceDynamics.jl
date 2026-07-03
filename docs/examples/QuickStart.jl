# # Quick Start
#
# A minimal end-to-end Poisson Linear Dynamical System example, mirroring the
# code block in the README. If this tutorial errors during `Pkg.test()`, the
# README quick-start is out of sync with the package.

using StateSpaceDynamics
using LinearAlgebra
using StableRNGs

rng = StableRNG(1234);

# ## True model
#
# A 2-D rotational latent system with 3 Poisson observation channels.

x0 = [1.0, -1.0]
P0 = Matrix(Diagonal([0.1, 0.1]))

A = 0.95 * [cos(0.1) -sin(0.1); sin(0.1) cos(0.1)]
Q = Matrix(Diagonal([0.01, 0.01]))
b = zeros(2)

C = [1.2 1.2; 1.2 1.2; 1.2 1.2]
d = log.([0.1, 0.1, 0.1])

state_model = GaussianStateModel(; A=A, Q=Q, b=b, P0=P0, x0=x0)
obs_model = PoissonObservationModel(; C=C, d=d)
plds_true = LinearDynamicalSystem(;
    state_model=state_model,
    obs_model=obs_model,
    latent_dim=2,
    obs_dim=3,
    fit_bool=fill(true, 5),
);

# ## Sample
#
# `rand` with a `Vector{Int}` of per-trial timestep counts returns
# `Vector{Matrix}` for both latents and observations (one entry per trial).

tsteps = 100
trials = 10
latents, observations = rand(rng, plds_true, fill(tsteps, trials));

# ## Fit
#
# Initialise a fresh model and run [`fit!`](@ref). The return value is the ELBO
# trajectory.

A_init = random_rotation_matrix(2, rng)
Q_init = Matrix(0.1 * I(2))
P0_init = Matrix(0.1 * I(2))
x0_init = zeros(2)
b_init = zeros(2)
C_init = rand(rng, 3, 2)
d_init = zeros(3)

plds_naive = LinearDynamicalSystem(;
    state_model=GaussianStateModel(;
        A=A_init, Q=Q_init, b=b_init, P0=P0_init, x0=x0_init
    ),
    obs_model=PoissonObservationModel(; C=C_init, d=d_init),
    latent_dim=2,
    obs_dim=3,
    fit_bool=fill(true, 5),
)

elbos = fit!(plds_naive, observations; max_iter=15, tol=1e-3);

# ## Tests  #src

using SSDTest  #src
using Test  #src

test_em_improves(elbos)  #src
test_lds_dimensions(plds_true; latent_dim=2, obs_dim=3)  #src
test_lds_dimensions(plds_naive; latent_dim=2, obs_dim=3)  #src
@test length(latents) == trials  #src
@test length(observations) == trials  #src
@test all(o -> size(o) == (3, tsteps), observations)  #src
