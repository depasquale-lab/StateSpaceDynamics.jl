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

where $x_t$ is the latent state at time $t$ and $y_t$ is the observed data at time $t$.

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

where $Q$ and $R$ are the state and observation noise covariance matrices, respectively.

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

Here's a simple example using a Linear Dynamical System:

```julia
using StateSpaceDynamics

# Create a Gaussian LDS
lds = GaussianLDS(
    latent_dim=3,    # 3D latent state
    obs_dim=10       # 10D observations
)

# Generate synthetic data
x, y = sample(lds, 1000)  # 1000 timepoints

# Fit the model
fit!(lds, y)

# Get smoothed state estimates
x_smoothed = smooth(lds, y)
```

## Contributing

If you encounter a bug or would like to contribute to the package, come find us on Github.

- [rsenne/ssm_julia](https://github.com/depasquale-lab/StateSpaceDynamics.jl)

## Citing StateSpaceDynamics.jl

If you use StateSpaceDynamics.jl in your research, please cite the following:

[Citation information to be added upon publication]
