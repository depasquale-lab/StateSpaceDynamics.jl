# Getting Started

StateSpaceDynamics.jl is a comprehensive Julia package for state space modeling, designed specifically with neuroscientific applications in mind. The package provides efficient implementations of various state space models along with tools for parameter estimation, state inference, and model selection.

## Installation

To install
[StateSpaceDynamics.jl](https://github.com/depasquale-lab/StateSpaceDynamics.jl), start up
Julia and type the following code-snipped into the REPL. 

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
   - Gaussian LDS
   - Poisson LDS

2. **Hidden Markov Models (HMM)**
   - Gaussian emissions
   - Regression-based emissions
      - Gaussian regression
      - Bernoulli regression
      - Poisson regression
      - Autoregressive emissions

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
P0 = I(latent_dim)
state_model = GaussianStateModel(A, Q, x0, P0)

# Define observation model parameters
C = randn(obs_dim, latent_dim)
R = 0.1 * I(obs_dim)
obs_model = GaussianObservationModel(C, R)

# Construct the LDS
lds = LinearDynamicalSystem(state_model, obs_model, latent_dim, obs_dim, fill(true, 6))
```

## Contributing

If you encounter a bug or would like to contribute to the package, please [open an issue](https://github.com/depasquale-lab/StateSpaceDynamics.jl/issues) on our GitHub repository. Once the suggested change has received positive feedback feel free to submit a PR adhering to the [blue](https://github.com/JuliaDiff/BlueStyle) style guide.

## Citing StateSpaceDynamics.jl

Our work is currently under review in the Journal of Open Source Software. For now, if you use StateSpaceDynamics.jl in your research, please use the following bibtex citation:

```bibtex
@software{Senne_Zenodo_SSD,
  author       = {Ryan Senne and Zachary Loschinskey and James Fourie and Carson Loughridge and Brian DePasquale},
  title        = {StateSpaceDynamics.jl},
  month        = jun,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.15668420},
  url          = {https://doi.org/10.5281/zenodo.15668420}
}
```
