# StateSpaceDynamics.jl: A Julia package for probabilistic state space models (SSMs)

[![StateSpaceDynamics-CI](https://github.com/depasquale-lab/StateSpaceDynamics.jl/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/depasquale-lab/StateSpaceDynamics.jl/actions/workflows/run_tests.yaml)
[![Benchmarks](https://github.com/depasquale-lab/StateSpaceDynamics.jl/actions/workflows/airspeed.yml/badge.svg)](https://github.com/depasquale-lab/StateSpaceDynamics.jl/actions/workflows/airspeed.yml)
[![codecov](https://codecov.io/github/depasquale-lab/StateSpaceDynamics.jl/graph/badge.svg?token=EQ6B9RJBQ8)](https://codecov.io/github/depasquale-lab/StateSpaceDynamics.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![Docs: Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://depasquale-lab.github.io/StateSpaceDynamics.jl/dev/)
[![Docs: Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://depasquale-lab.github.io/StateSpaceDynamics.jl/stable)
[![status](https://joss.theoj.org/papers/0bcb7b5a500055bb4f9fc5aec65c177b/status.svg)](https://joss.theoj.org/papers/0bcb7b5a500055bb4f9fc5aec65c177b)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17583787.svg)](https://doi.org/10.5281/zenodo.17583787)
[![All Contributors](https://img.shields.io/github/all-contributors/depasquale-lab/StateSpaceDynamics.jl?color=ee8449&style=flat-square)](#contributors)

## Description

StateSpaceDynamics.jl is a fast, self-contained Julia package for linear dynamical systems (LDS) and related latent state-space models. It covers Gaussian and Poisson LDS, Switching LDS (SLDS), and Probabilistic PCA, taking inspiration from [ssm](https://github.com/lindermanlab/ssm) (Linderman Lab) and [Dynamax](https://probml.github.io/dynamax/). It leverages Julia's speed and expressiveness to give researchers and data scientists a powerful, efficient toolkit for latent state-space modeling.

This package is geared towards applications in neuroscience, so the models incorporate a certain neuroscience flavor (e.g., many of our models are trialized, as is common in experimental paradigms). However, the models are general enough to be used in other fields such as finance, robotics, and many other domains involving sequential data analysis.

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
A = 0.95 * [cos(0.1) -sin(0.1); sin(0.1) cos(0.1)] # transition matrix
Q = Matrix(Diagonal([0.01, 0.01])) # process noise

# observation model parameters
C = [1.2 1.2; 1.2 1.2; 1.2 1.2] # observation matrix
d = log.([0.1, 0.1, 0.1])       # baseline log-rates: λ_i = exp(C_i' x + d_i)

# generate data
tsteps = 100
trials = 10

b = zeros(2)                                       # dynamics bias

gaussian_state_model = GaussianStateModel(;A=A, Q=Q, b=b, P0=P0, x0=x0)
poisson_obs_model = PoissonObservationModel(;C=C, d=d)

plds_true = LinearDynamicalSystem(;state_model=gaussian_state_model,
                                   obs_model=poisson_obs_model,
                                   latent_dim=2, obs_dim=3, fit_bool=fill(true, 5))

# Multi-trial sampling: pass per-trial timestep counts as a Vector. Returns
# `Vector{Matrix}` for both latents and observations (one entry per trial).
latents, observations = rand(rng, plds_true, fill(tsteps, trials))

# fit the data to a new naive model
A_init = random_rotation_matrix(2, rng)
Q_init = Matrix(0.1 * I(2))
P0_init = Matrix(0.1 * I(2))
x0_init = zeros(2)
b_init = zeros(2)

C_init = rand(3, 2)
d_init = zeros(3)

plds_naive = LinearDynamicalSystem(;state_model=GaussianStateModel(;A=A_init, Q=Q_init, b=b_init, P0=P0_init, x0=x0_init), obs_model=PoissonObservationModel(;C=C_init, d=d_init), latent_dim=2, obs_dim=3, fit_bool=fill(true, 5))
elbos = fit!(plds_naive, observations; max_iter=15, tol=1e-3)
```

## Inference

For inference in non-conjugate LDS models (e.g., PoissonLDS), StateSpaceDynamics.jl uses the **Laplace approximation** to estimate the posterior distribution over latent states.

This procedure begins by maximizing the joint log-probability of the latent states and observations (i.e., the complete-data log-likelihood) to obtain a **maximum a posteriori (MAP)** estimate of the latent trajectory. We then approximate the posterior distribution with a Gaussian centered at this MAP estimate, where the covariance is derived from the inverse Hessian of the negative log-joint evaluated at the mode.

This approximation allows us to perform tractable, efficient inference in otherwise intractable models while still capturing uncertainty about the latent states. In the case of Gaussian observations and latents, this is equivalent to the canonical RTS smoothing algorithm.

## Community Guidelines

- Maintain professional, respectful discourse
- Stay focused on `StateSpaceDynamics.jl` development and usage
- Provide reproducible examples for bug reports and feature requests
- Search existing issues before posting

Help us maintain a welcoming environment for researchers and developers.

## Available Models

- [x] Linear Dynamical Systems
  - [x] Gaussian Linear Dynamical Systems (Kalman Filter)
  - [x] Poisson Linear Dynamical Systems (PLDS)
  - [ ] PFLDS
  - [x] Switching Linear Dynamical Systems (SLDS)
  - [ ] Recurrent Switching Linear Dynamical Systems (rSLDS)
- [x] Probabilistic PCA (PPCA)

## Related Packages

- [HiddenMarkovModels.jl](https://github.com/maxmouchet/HiddenMarkovModels.jl): A Julia package for Hidden Markov Models. We recommend this package if you need HMMs; StateSpaceDynamics.jl uses it internally for SLDS forward-backward.

- [StateSpaceLearning.jl](https://github.com/LAMPSPUC/StateSpaceLearning.jl): A Julia package for time series analysis using state space models.

- [ssm](https://github.com/lindermanlab/ssm): A python package for state space models.

- [dynamax](https://github.com/probml/dynamax): A python package built on JAX for state space modelling (supersedes ssm).

- [StateSpaceAnalysis.jl](https://github.com/harrisonritz/StateSpaceAnalysis.jl): A highly performant Julia package for fitting Gaussian LDS models. Currently being absorbed into this package.

## Contributing

If you would like to contribute, report a bug, request a new feature, or simply give feedback, please feel free to [open an issue](https://github.com/depasquale-lab/StateSpaceDynamics.jl/issues) and we will get back to you in a timely manner.

## Citing

If you use our software in your research please cite our JOSS paper using the following bibtex citation:

```bibtex
@article{Senne_StateSpaceDynamics_jl_A_Julia_2025,
  author = {Senne, Ryan and Loschinskey, Zachary and Fourie, James and Loughridge, Carson and DePasquale, Brian D.},
  doi = {10.21105/joss.08077},
  journal = {Journal of Open Source Software},
  month = nov,
  number = {115},
  pages = {8077},
  title = {{StateSpaceDynamics.jl: A Julia package for probabilistic state space models (SSMs)}},
  url = {https://joss.theoj.org/papers/10.21105/joss.08077},
  volume = {10},
  year = {2025}
  }
```

## Contributors

Thanks go to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome!

## References

1. Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.

2. Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT press.

3. Sarrka S., Svensson, L. (2023). Bayesian Filtering and Smoothing. Cambridge University Press.

4. Paninski, L. et al. (2010). A new look at state-space models for neural data. Journal of computational neuroscience, 29(1-2), 107-126.

5. Macke, J. H. et al. (2011). Empirical models of spiking in neural populations. Advances in neural information processing systems, 24, 1350-1358.
