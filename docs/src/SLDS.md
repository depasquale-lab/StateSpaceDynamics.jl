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
SwitchingLinearDynamicalSystem
```

## Mathematical Formulation

An SLDS with ``K`` discrete states is defined by the following generative model:

```math
\begin{align*}
    s_1 &\sim \text{Cat}(\pi_k) \\
    z_1 &\sim \mathcal{N}(\mu_{0}, P_{0}) \\
    s_t &\mid s_{t-1} \sim \text{Cat}(A_{s_{t-1}, :}) \\
    z_t &\mid z_{t-1}, s_t \sim \mathcal{N}(F_{s_t} z_{t-1}, Q_{s_t}) \\
    y_t &\mid z_t, s_t \sim \mathcal{N}(C_{s_t} z_t, R_{s_t})
\end{align*}
```

Where:

- ``s_t ∈ {1, 2, …, K}`` is the **discrete switching state** at time ``t``
- ``z_t ∈ ℝᴰ`` is the **continuous latent state** at time ``t``
- ``y_t ∈ ℝᴾ`` is the **observed data** at time ``t``
- ``π_k`` is the **initial discrete state distribution**
- ``A`` is the **discrete state transition matrix**
- ``F_{s_t}`` is the **state-dependent dynamics matrix** for discrete state ``s_t``
- ``Q_{s_t}`` is the **state-dependent process noise covariance** for discrete state ``s_t``
- ``C_{s_t}`` is the **state-dependent observation matrix** for discrete state ``s_t``
- ``R_{s_t}`` is the **state-dependent observation noise covariance** for discrete state ``s_t``

## Implementation Structure

In `StateSpaceDynamics.jl`, an SLDS is represented as:

```julia
mutable struct SwitchingLinearDynamicalSystem <: AbstractHMM
    A::M     # Transition matrix for mode switching (K × K)
    B::VL    # Vector of LinearDynamicalSystem models (length K)
    πₖ::V    # Initial state distribution (length K)
    K::Int   # Number of modes
end
```

Each mode in the `B` vector contains its own `LinearDynamicalSystem` with:

- **State model**: Defines the continuous latent dynamics ``F_k``, ``Q_k``
- **Observation model**: Defines the emission process ``C_k``, ``R_k``

## Sampling from SLDS

You can generate synthetic data from an SLDS to test algorithms or create simulated datasets:

```@docs
rand(rng::AbstractRNG, slds::SwitchingLinearDynamicalSystem, T::Int)
```

The sampling process follows the generative model:

1. **Initialize**: Sample initial discrete state from ``\pi_k`` and initial continuous state
2. **For each time step**:
   - Sample next discrete state based on current state and transition matrix ``A``
   - Sample continuous state using the dynamics of the current discrete state
   - Generate observation using the observation model of the current discrete state

## Learning in SLDS

`StateSpaceDynamics.jl` implements a **Variational Expectation-Maximization (EM)** algorithm for parameter estimation in SLDS. This approach handles the interaction between discrete and continuous latent variables efficiently.

```@docs
fit!(slds::AbstractHMM, y::AbstractMatrix{T}; max_iter::Int=1000, tol::Real=1e-3) where {T<:Real}
```

## Variational EM Algorithm

The variational EM algorithm maximizes the **Evidence Lower Bound (ELBO)** instead of the intractable marginal likelihood. The algorithm alternates between:

### Variational Expectation Step

The E-step iteratively updates the variational distributions until convergence. This involves two coupled updates:

**1. Update continuous state posteriors (``q(z_{1:T})``):**
For each discrete state ``k``, run weighted Kalman smoothing:

```math
q(z_{1:T} \mid s_{1:T} = k) = \prod_{t=1}^T \mathcal{N}(z_t; \hat{z}_{t|T}^{(k)}, P_{t|T}^{(k)})
```

**2. Update discrete state posteriors (``q(s_{1:T})``):**
Run forward-backward algorithm with observation likelihoods computed from current continuous posteriors:

```math
q(s_t = k) = \gamma_t(k) = p(s_t = k \mid y_{1:T}, q(z_{1:T}))
```

The E-step converges when the ELBO stabilizes, ensuring consistency between discrete and continuous posteriors.

### Maximization Step

The M-step updates all model parameters using weighted maximum likelihood:

**Discrete state parameters:**

- Initial distribution: ``\pi_k^{(new)} = \gamma_1(k)``
- Transition matrix: ``A_{ij}^{(new)} = \frac{\sum_{t=1}^{T-1} \xi_{t,t+1}(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}``

**Continuous state parameters for each mode ``k``:**

Using sufficient statistics from weighted Kalman smoothing:

- Dynamics matrix: ``F_k^{(new)}`` from weighted regression
- Process covariance: ``Q_k^{(new)}`` from weighted residuals
- Observation matrix: ``C_k^{(new)}`` from weighted regression
- Initial state parameters: ``\mu_0^{(k)}, P_0^{(k)}``

## Evidence Lower Bound (ELBO)

The ELBO consists of contributions from both discrete and continuous components:

```math
\text{ELBO} = \underbrace{\mathbb{E}_{q(s_{1:T})}[\log p(s_{1:T})] - \mathbb{E}_{q(s_{1:T})}[\log q(s_{1:T})]}_{\text{Discrete HMM contribution}} + \sum_{k=1}^K \underbrace{\mathbb{E}_{q(z_{1:T}|s_{1:T}=k)}[\log p(y_{1:T}, z_{1:T} | s_{1:T}=k)] + H[q(z_{1:T}|s_{1:T}=k)]}_{\text{Continuous LDS contribution for mode } k}
```

# References

For theoretical foundations and algorithmic details:

- **"Learning and Inference in Switching Linear Dynamical Systems"** by **Zoubin Ghahramani and Geoffrey Hinton**
- **"Variational Learning for Switching State-Space Models"** by **Zoubin Ghahramani and Sam Roweis**  
- **"A Unifying Review of Linear Gaussian Models"** by **Sam Roweis and Zoubin Ghahramani**
- **"Probabilistic Machine Learning: Advanced Topics, Chapter 8"** by **Kevin Murphy**