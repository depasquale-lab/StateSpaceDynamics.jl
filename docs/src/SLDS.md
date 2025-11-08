# What is a Switching Linear Dynamical System?

```@meta
CollapsedDocStrings = true
```

A **Switching Linear Dynamical System (SLDS)** is a powerful probabilistic model that combines the temporal structure of linear dynamical systems with the discrete switching behavior of Hidden Markov Models. SLDS can model complex time series data that exhibits multiple dynamical regimes, where the system can switch between different linear dynamics over time.

An SLDS extends the standard Linear Dynamical System (LDS) by introducing a discrete latent state that determines which linear dynamics are active at each time step. This makes SLDS particularly suitable for modeling systems with:

- **Multiple operational modes** (e.g., different flight phases of an aircraft)
- **Regime changes** (e.g., economic cycles, behavioral states)
- **Non-stationary dynamics** where linear dynamics change over time
- **Hybrid systems** combining discrete and continuous states

```@docs
SLDS
```

## Mathematical Formulation

An SLDS with $K$ discrete states is defined by the following generative model:

```math
\begin{align*}
    z_1 &\sim \text{Cat}(\pi_k) \\
    x_1 &\sim \mathcal{N}(\mu_{0}, P_{0}) \\
    z_t &\mid z_{t-1} \sim \text{Cat}(A_{z_{t-1}, :}) \\
    x_t &\mid x_{t-1}, z_t \sim \mathcal{N}(F_{z_t} x_{t-1} + b_{z_t}, Q_{z_t}) \\
    y_t &\mid x_t, z_t \sim \mathcal{N}(C_{z_t} x_t + d_{z_t}, R_{z_t})
\end{align*}
```

Where:

- ``z_t ∈ {1, 2, …, K}`` is the **discrete switching state** at time ``t``
- ``x_t ∈ ℝᴰ`` is the **continuous latent state** at time ``t``
- ``y_t ∈ ℝᴾ`` is the **observed data** at time ``t``
- ``π_k`` is the **initial discrete state distribution**
- ``A`` is the **discrete state transition matrix**
- ``F_{z_t}`` is the **state-dependent dynamics matrix** for discrete state ``z_t``
- ``Q_{z_t}`` is the **state-dependent process noise covariance** for discrete state ``z_t``
- ``C_{z_t}`` is the **state-dependent observation matrix** for discrete state ``z_t``
- ``R_{z_t}`` is the **state-dependent observation noise covariance** for discrete state ``z_t``
- ``b_{z_t}`` and ``d_{z_t}`` is the **state-dependent biases** for discrete state ``z_t``

## Implementation Structure

In `StateSpaceDynamics.jl`, an SLDS is represented as:

```julia
mutable struct SLDS{
    T<:Real,
    S<:AbstractStateModel,
    O<:AbstractObservationModel,
    TM<:AbstractMatrix{T},
    ISV<:AbstractVector{T},
} <: AbstractHMM
    A::TM # Transition matrix
    πₖ::ISV # Initial state distribution
    LDSs::Vector{LinearDynamicalSystem{T,S,O}} # Vector of LDS models
end
```

Each mode in the `LDSs` vector contains its own `LinearDynamicalSystem` with:

- **State model**: Defines the continuous latent dynamics $F_k$, $Q_k$
- **Observation model**: Defines the emission process. Currently supports Gaussian and Poisson emission models.

## Sampling from SLDS

You can generate synthetic data from an SLDS to test algorithms or create simulated datasets:

```@docs
rand(rng::AbstractRNG, slds::SLDS, T::Int)
```

The sampling process follows the generative model:

1. **Initialize**: Sample initial discrete state from $\pi_k$ and initial continuous state
2. **For each time step**:
   - Sample next discrete state based on current state and transition matrix $A$
   - Sample continuous state using the dynamics of the current discrete state
   - Generate observation using the observation model of the current discrete state

## Learning in SLDS: Variational Laplace EM (vLEM)

`StateSpaceDynamics.jl` implements a **Variational Laplace Expectation-Maximization (vLEM)** algorithm for parameter estimation in SLDS. This approach efficiently handles the challenging interaction between discrete and continuous latent variables through a structured variational approximation.

```@docs
fit!(slds::AbstractHMM, y::AbstractMatrix{T}; max_iter::Int=1000, tol::Real=1e-3) where {T<:Real}
```

## The vLEM Algorithm

The vLEM algorithm maximizes the **Evidence Lower Bound (ELBO)** instead of the intractable marginal likelihood. The key insight is to use a structured variational approximation that factorizes as:

```math
q(z_{1:T}, x_{1:T}) = q(z_{1:T}) \prod_{k=1}^K q(x_{1:T} | z_{1:T} = k)^{\mathbb{I}[z_{1:T} = k]}
```

This factorization allows efficient inference by alternating between updating discrete and continuous posteriors.

### Variational Laplace Expectation Step

**0. Initialization:**
Initialize with uniform discrete state posteriors and perform an initial smoothing pass using provided parameter values. This establishes the starting point for iterative refinement.

**1. Update Continuous State Posterior ($q(x_{1:T} | z_{1:T})$):**
For each discrete state sequence $k$, run Kalman smoothing weighted by the current discrete posterior:

```math
q(x_{1:T} \mid z_{1:T} = k) = \prod_{t=1}^T \mathcal{N}(x_t; \hat{x}_{t|T}^{(k)}, P_{t|T}^{(k)})
```

To handle expectations efficiently, we use a single Monte Carlo sample from this posterior for subsequent computations.

**2. Update Discrete State Posterior ($q(z_{1:T})$):**
Run forward-backward algorithm with modified observation likelihoods that incorporate the current continuous posterior:

```math
\tilde{p}(y_t | z_t = k) = \int p(y_t | x_t, z_t = k) q(x_t | z_t = k) dx_t
```

This yields the discrete posterior marginals:
```math
q(z_t = k) = \gamma_t(k) = p(z_t = k \mid y_{1:T}, q(x_{1:T}))
```

### Maximization Step

The M-step updates all parameters using expectations from the E-step:

**Discrete State Parameters:**

- Initial distribution: $\pi_k^{(\text{new})} = \gamma_1(k)$
- Transition matrix: $A_{ij}^{(\text{new})} = \frac{\sum_{t=1}^{T-1} \xi_{t,t+1}(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$

where $\xi_{t,t+1}(i,j) = p(z_t = i, z_{t+1} = j | y_{1:T})$ are the two-slice marginals.

**Continuous State Parameters for each mode $k$:**

Using weighted sufficient statistics from the smoothed posteriors:

- Dynamics matrix: $F_k^{(\text{new})}$ from weighted least squares
- Process covariance: $Q_k^{(\text{new})}$ from weighted innovation covariance
- Observation matrix: $C_k^{(\text{new})}$ from weighted observation regression
- Observation covariance: $R_k^{(\text{new})}$ from weighted observation residuals
- Initial parameters: $\mu_0^{(k)}, P_0^{(k)}$ from weighted initial state statistics

The weights are given by the discrete posterior probabilities $\gamma_t(k)$.

## Evidence Lower Bound (ELBO)

The ELBO decomposes into discrete and continuous components:

```math
\mathcal{L}(q) = \underbrace{\mathbb{E}_{q(z_{1:T})}[\log p(z_{1:T})] - \mathbb{E}_{q(z_{1:T})}[\log q(z_{1:T})]}_{\text{Discrete HMM entropy}} + \sum_{k=1}^K \gamma_t(k) \underbrace{\left( \mathbb{E}_{q(x_{1:T}|k)}[\log p(y_{1:T}, x_{1:T} | z_{1:T}=k)] + H[q(x_{1:T}|k)] \right)}_{\text{Weighted LDS contribution for mode } k}
```

## References

For theoretical foundations and algorithmic details:

- **"A general recurrent state space framework for modeling neural dynamics during decision-making"** by **David Zoltowski, Jonathon Pillow, and Scott Linderman** (2020)
- **"Variational Learning for Switching State-Space Models"** by **Zoubin Ghahramani and Geoffrey Hinton** (1998)
- **"Probabilistic Machine Learning: Advanced Topics, Chapter 29"** by **Kevin Murphy**
- **"A Unifying Review of Linear Gaussian Models"** by **Sam Roweis and Zoubin Ghahramani**