# Getting Started

StateSpaceDynamics.jl is a comprehensive Julia package for state space modeling, designed specifically with neuroscientific applications in mind. The package provides efficient implementations of various state space models along with tools for parameter estimation, state inference, and model selection.

## Installation

To install
[StateSpaceDynamics.jl](https://github.com/depasquale-lab/StateSpaceDynamics.jl), start up
Julia and type the following code snippet into the REPL.

```julia
using Pkg
Pkg.add("StateSpaceDynamics")
```

or alternatively, you can enter the package manager by typing `]` and then run:

```julia
add StateSpaceDynamics
```

## What are State Space Models?

State space models are a class of probabilistic models that describe the evolution of a system through two main components - a latent and observation process. The latent process is a stochastic process that is not directly observed, but is used to generate the observed data. The observation process is a conditional distribution that describes how the observed data is generated from the latent process.

In their most general form, state space models can be written as:

```math
\begin{align*}
    x_{t+1} &\sim p(x_{t+1} | x_t) \\
    y_t &\sim p(y_t | x_t)
\end{align*}
```

where ``x_t`` is the latent state at time ``t`` and ``y_t`` is the observed data at time ``t``.

### Example: Linear Dynamical Systems

A fundamental example is the Linear Dynamical System (LDS), which combines linear dynamics with Gaussian noise. The LDS can be expressed in two equivalent forms:

1. Equation form:

```math
\begin{align*}
    x_{t+1} &= A x_t + b + \epsilon_t \\
    y_t &= C x_t + d + \delta_t
\end{align*}
```

where:

* ``\mathbf{A}`` is the state transition matrix
* ``\mathbf{C}`` is the observation matrix  
* ``\mathbf{b}`` and ``\mathbf{d}`` are bias terms
* ``\boldsymbol{\epsilon}_t`` and ``\boldsymbol{\delta}_t`` are Gaussian noise terms with covariances ``\mathbf{Q}`` and ``\mathbf{R}`` respectively

2. Distributional form:

```math
\begin{align*}
    x_{t+1} &\sim \mathcal{N}(A x_t + b, Q) \\
    y_t &\sim \mathcal{N}(C x_t + d, R)
\end{align*}
```

where ``\mathbf{Q}`` and ``\mathbf{R}`` are the state and observation noise covariance matrices, respectively.

## Models Implemented

StateSpaceDynamics.jl implements several types of state space models:

1. **Linear Dynamical Systems (LDS)**
   * Gaussian LDS
   * Poisson LDS

2. **Hidden Markov Models (HMM)**
   * Gaussian emissions
   * Regression-based emissions
      * Gaussian regression
      * Bernoulli regression
      * Poisson regression
      * Autoregressive emissions

## Quick Start

Here's a simple example on how to create a Gaussian SSM.

```julia
using StateSpaceDynamics
using LinearAlgebra

# Define model dimensions
latent_dim = 3
obs_dim = 10

# Define state model parameters
A = 0.95 * I(latent_dim)
Q = 0.01 * I(latent_dim)
x0 = zeros(latent_dim)
P0 = 0.1 * I(latent_dim)
state_model = GaussianStateModel(A, Q, x0, P0)

# Define observation model parameters
C = ones(obs_dim, latent_dim)
R = Matrix(0.5 * I(obs_dim))
obs_model = GaussianObservationModel(C, R)

# Construct the LDS
lds = LinearDynamicalSystem(state_model, obs_model, latent_dim, obs_dim, fill(true, 6))
```

## Contributing

If you encounter a bug or would like to contribute to the package, please [open an issue](https://github.com/depasquale-lab/StateSpaceDynamics.jl/issues) on our GitHub repository. Once the suggested change has received positive feedback, feel free to submit a PR adhering to the [BlueStyle](https://github.com/JuliaDiff/BlueStyle) guide.

Please include or update **tests** for any user-facing change. Tests live in the `test/` folder and are run with:

```julia
julia --project -e 'using Pkg; Pkg.test()'
# or from the Pkg REPL by typing "]":
add StateSpaceDynamics
```

## Citing StateSpaceDynamics.jl

If you use this software in your research, please cite our publication in the Journal Of Open Source Software:

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
