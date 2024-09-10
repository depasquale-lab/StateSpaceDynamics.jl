# StateSpaceDynamics.jl: A Julia package for probabilistic state space models (SSMs)

[![SSM-CI](https://github.com/rsenne/ssm_julia/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/rsenne/ssm_julia/actions/workflows/run_tests.yaml)
[![codecov](https://codecov.io/github/depasquale-lab/StateSpaceDynamics.jl/graph/badge.svg?token=EQ6B9RJBQ8)](https://codecov.io/github/depasquale-lab/StateSpaceDynamics.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This package implements a number of state-space models in julia. It take inspiration from the [SSM package](https://github.com/lindermanlab/ssm) written in python from the Linderman Lab. Currently, the following models are implemented:

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

We are hoping to add more models in the future. If you have any suggestions, please open an issue.
