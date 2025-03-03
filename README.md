# StateSpaceDynamics.jl: A Julia package for probabilistic state space models (SSMs)

[![StateSpaceDynamics-CI](https://github.com/rsenne/ssm_julia/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/rsenne/ssm_julia/actions/workflows/run_tests.yaml)
[![codecov](https://codecov.io/github/depasquale-lab/StateSpaceDynamics.jl/graph/badge.svg?token=EQ6B9RJBQ8)](https://codecov.io/github/depasquale-lab/StateSpaceDynamics.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

## Description

StateSpaceDynamics.jl is a comprehensive and self-contained Julia package for working with probabilistic state space models (SSMs). It implements a wide range of state-space models, taking inspiration from the [SSM](https://github.com/lindermanlab/ssm) package written in Python by the Linderman Lab. This package is designed to be fast, flexible, and all-encompassing, leveraging Julia's speed and expressiveness to provide researchers and data scientists with a powerful toolkit for state-space modeling.

This package is geared towards applications in neuroscience, so the models incorparate a certain neuroscience flavor (e.g., many of our models are trialized as common in experiemntal paradigms). However, the models are general enough to be used in other fields such as finance, robotics, and many other domains involving sequential data analysis.

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
tSteps = 100
trials = 10

true_plds = PoissonLDS(;A=A, Q=Q, C=C, log_d=log_d, x0=x0, P0=P0, obs_dim=3, latent_dim=2)
latents, observations = sample(true_plds, tSteps, trials)

# fit the model 
plds = PoissonLDS(;obs_dim=3, latent_dim=2)
fit!(plds, observations)
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
  - [ ] Switching Linear Dynamical Systems (SLDS)
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

- [HiddenMarkovModels.jl](https://github.com/maxmouchet/HiddenMarkovModels.jl): A Julia package for Hidden Markov Models.

- [StateSpaceLearning.jl](https://github.com/LAMPSPUC/StateSpaceLearning.jl) : A Julia package for time series analysis using state space models.

- [ssm](https://github.com/lindermanlab/ssm) : A python package for state space models.

## References

1. Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.

2. Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT press.

3. Sarrka S., Svensson, L. (2023). Bayesian Filtering and Smoothing. Cambridge University Press.

4. Paninski, L. et al. (2010). A new look at state-space models for neural data. Journal of computational neuroscience, 29(1-2), 107-126.

5. Macke, J. H. et al. (2011). Empirical models of spiking in neural populations. Advances in neural information processing systems, 24, 1350-1358.