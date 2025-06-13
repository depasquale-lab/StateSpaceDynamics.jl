# StateSpaceDynamics.jl: A Julia package for probabilistic state space models (SSMs)

[![StateSpaceDynamics-CI](https://github.com/rsenne/ssm_julia/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/rsenne/ssm_julia/actions/workflows/run_tests.yaml)
[![codecov](https://codecov.io/github/depasquale-lab/StateSpaceDynamics.jl/graph/badge.svg?token=EQ6B9RJBQ8)](https://codecov.io/github/depasquale-lab/StateSpaceDynamics.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://depasquale-lab.github.io/StateSpaceDynamics.jl/dev/)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://depasquale-lab.github.io/StateSpaceDynamics.jl/stable)
[![status](https://joss.theoj.org/papers/0bcb7b5a500055bb4f9fc5aec65c177b/status.svg)](https://joss.theoj.org/papers/0bcb7b5a500055bb4f9fc5aec65c177b)

## Description

StateSpaceDynamics.jl is a comprehensive and self-contained Julia package for working with probabilistic state space models (SSMs). It implements a wide range of state-space models, taking inspiration from the [SSM](https://github.com/lindermanlab/ssm) package written in Python by the Linderman Lab. This package is designed to be fast, flexible, and all-encompassing, leveraging Julia's speed and expressiveness to provide researchers and data scientists with a powerful toolkit for state-space modeling.

This package is geared towards applications in neuroscience, so the models incorporate a certain neuroscience flavor (e.g., many of our models are trialized as common in experimental paradigms). However, the models are general enough to be used in other fields such as finance, robotics, and many other domains involving sequential data analysis.

We are continuously working to expand our model offerings. If you have suggestions for additional models or features, please open an issue on our GitHub repository.

## Installation

You can install StateSpaceDynamics.jl using Julia's package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add StateSpaceDynamics
```

## Usage

StateSpaceDynamics.jl is designed to be user friendly with intuitive syntax. Below is an example of how to fit a Poisson Linear Dynamical System (PLDS) to data.

```julia
using StateSpaceDynamics
using LinearAlgebra
using StableRNGs

# Set seed for reproducibility
rng = StableRNG(1234);

# create a toy system
# initial conditions
x0 = [1.0, -1.0] # initial state
P0 = Matrix(Diagonal([0.1, 0.1])) # initial state covariance

# state model parameters
A = [cos(0.1) -sin(0.1); sin(0.1) cos(0.1)] # transition matrix
Q = Matrix(Diagonal([0.01, 0.01])) # process noise

# observation model parameters
C = [1.2 1.2; 1.2 1.2; 1.2 1.2] # observation matrix
log_d = log.([0.1, 0.1, 0.1]) # log of the natural parameters of the Poisson distribution

# generate data
tsteps = 100
trials = 10

gaussian_state_model = GaussianStateModel(;A=A, Q=Q, P0=P0, x0=x0)
poisson_obs_model = PoissonObservationModel(;C=C, log_d=log_d)

plds_true = LinearDynamicalSystem(;state_model=gaussian_state_model, obs_model=poisson_obs_model, latent_dim=2, obs_dim=3, fit_bool=fill(true, 6))
latents, observations = rand(rng, plds_true; tsteps=tsteps, ntrials=trials)

# fit the data to a new naive model
A_init = random_rotation_matrix(2, rng)
Q_init = Matrix(0.1 * I(2))
P0_init = Matrix(0.1 * I(2))
x0_init = zeros(2)

C_init = rand(3, 2)
log_d_init = zeros(3)

plds_true = LinearDynamicalSystem(;state_model=GaussianStateModel(;A=A_init, Q=Q_init, P0=P0_init, x0=x0_init), obs_model=PoissonObservationModel(;C=C_init, log_d=log_d_init), latent_dim=2, obs_dim=3, fit_bool=fill(true, 6))
fit!(plds_true, observations; max_iter=15, tol=1e-3)
```

## Available Models

- [x] Mixture Models
  - [x] Gaussian Mixture Models
  - [x] Poisson Mixture Models
  - [ ] Binomial Mixture Models
  - [ ] Negative Binomial Mixture Models
  - [ ] Student's t Mixture Models
- [x] Hidden Markov Models
  - [x] Gaussian HMMs
  - [ ] Poisson HMMs
  - [ ] Binomial HMMs
  - [ ] Negative Binomial HMMs
  - [ ] Autoregressive HMMs (ARHMM)
- [x] Linear Dynamical Systems
  - [x] Gaussian Linear Dynamical Systems (Kalman Filter)
  - [x] Poisson Linear Dynamical Systems (PLDS)
  - [ ] PFLDS
  - [x] Switching Linear Dynamical Systems (SLDS)
  - [ ] Recurrent Switching Linear Dynamical Systems (rSLDS)
- [x] Generalized Linear Models (Not state space models but needed for HMM-GLMs)
  - [x] Gaussian GLMs
  - [x] Poisson GLMs
  - [x] Binomial GLMs
  - [ ] Negative Binomial GLMs
- [x] HMM-GLM's
  - [x] Gaussian HMM-GLMs
  - [x] Poisson HMM-GLMs
  - [x] Bernoulli HMM-GLMs
  - [ ] Negative Binomial HMM-GLMs
  - [ ] Multinomial HMM-GLMs

## Related Packages

- [HiddenMarkovModels.jl](https://github.com/maxmouchet/HiddenMarkovModels.jl): A Julia package for Hidden Markov Models. We recommend this package if you are   exclusively interested in HMMs. We plan to integrate with this package in the future.

- [StateSpaceLearning.jl](https://github.com/LAMPSPUC/StateSpaceLearning.jl) : A Julia package for time series analysis using state space models.

- [ssm](https://github.com/lindermanlab/ssm) : A python package for state space models.

## References

1. Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.

2. Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT press.

3. Sarrka S., Svensson, L. (2023). Bayesian Filtering and Smoothing. Cambridge University Press.

4. Paninski, L. et al. (2010). A new look at state-space models for neural data. Journal of computational neuroscience, 29(1-2), 107-126.

5. Macke, J. H. et al. (2011). Empirical models of spiking in neural populations. Advances in neural information processing systems, 24, 1350-1358.