# SSM (Statespace Models) _in julia_

[![SSM-CI](https://github.com/rsenne/ssm_julia/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/rsenne/ssm_julia/actions/workflows/run_tests.yaml)
[![codecov](https://codecov.io/gh/rsenne/ssm_julia/graph/badge.svg?token=EQ6B9RJBQ8)](https://codecov.io/gh/rsenne/ssm_julia)

This package implements a number of state-space models in julia. It take inspiration from the [SSM package](https://github.com/lindermanlab/ssm) written in python from the Linderman Lab. Currently, the following models are implemented:

- [x] Mixture Models
    - [x] Gaussian Mixture Models
    - [ ] Poisson Mixture Models
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
    - [ ] Poisson Linear Dynamical Systems (PLDS)
    - [ ] PFLDS
    - [ ] Switching Linear Dynamical Systems (SLDS)
    - [ ] Recurrent Switching Linear Dynamical Systems (rSLDS)
- [ ] HMM-LDS
    - [ ] Gaussian HMM-LDS
    - [ ] Poisson HMM-LDS
- [x] Generalized Linear Models (Not state space models but needed for HMM-GLMs)
    - [x] Gaussian GLMs
    - [x] Poisson GLMs
    - [x] Binomial GLMs
    - [ ] Negative Binomial GLMs
- [x] HMM-GLM's
    - [x] Gaussian HMM-GLMs
    - [ ] Poisson HMM-GLMs
    - [ ] Binomial HMM-GLMs
    - [ ] Negative Binomial HMM-GLMs
    - [ ] Multinomial HMM-GLMs
- [x] Misc.
    - [x] Autoregressive Models
    - [ ] Vector AutoRegressive Models

We are hoping to add more models in the future. If you have any suggestions, please open an issue.
