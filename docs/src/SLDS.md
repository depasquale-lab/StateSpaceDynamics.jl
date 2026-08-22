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

An SLDS with ``K`` discrete states is defined by the following generative model:

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

- ``z_t \in \{1, 2, \dots, K\}`` is the **discrete switching state** at time ``t``
- ``x_t \in \mathbb{R}^D`` is the **continuous latent state** at time ``t``
- ``y_t \in \mathbb{R}^P`` is the **observed data** at time ``t``
- ``\pi_k`` is the **initial discrete state distribution**
- ``A`` is the **discrete state transition matrix**
- ``F_{z_t}`` is the **state-dependent dynamics matrix** for discrete state ``z_t``
- ``Q_{z_t}`` is the **state-dependent process noise covariance** for discrete state ``z_t``
- ``C_{z_t}`` is the **state-dependent observation matrix** for discrete state ``z_t``
- ``R_{z_t}`` is the **state-dependent observation noise covariance** for discrete state ``z_t``
- ``b_{z_t}`` and ``d_{z_t}`` are the **state-dependent biases** for discrete state ``z_t``

## Implementation Structure

In `StateSpaceDynamics.jl`, an SLDS is represented as:

```julia
mutable struct SLDS{
    T<:Real,
    S<:AbstractStateModel,
    O<:AbstractObservationModel,
    TM<:AbstractMatrix{T},
    ISV<:AbstractVector{T},
}
    A::TM # Transition matrix
    πₖ::ISV # Initial state distribution
    LDSs::Vector{LinearDynamicalSystem{T,S,O}} # Vector of LDS models
end
```

Each mode in the `LDSs` vector contains its own `LinearDynamicalSystem` with:

- **State model**: Defines the continuous latent dynamics ``F_k``, ``Q_k``
- **Observation model**: Defines the emission process. Currently supports Gaussian and Poisson emission models.

## Sampling from SLDS

You can generate synthetic data from an SLDS to test algorithms or create simulated datasets:

```@docs
Random.rand(rng::AbstractRNG, slds::SLDS{T,S,O}, tsteps::Integer) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
```

The sampling process follows the generative model:

1. **Initialize**: Sample initial discrete state from ``\pi_k`` and initial continuous state
2. **For each time step**:
   - Sample next discrete state based on current state and transition matrix ``A``
   - Sample continuous state using the dynamics of the current discrete state
   - Generate observation using the observation model of the current discrete state

## Learning in SLDS: Variational Laplace EM (vLEM)

`StateSpaceDynamics.jl` implements a **Variational Laplace Expectation-Maximization (vLEM)** algorithm for parameter estimation in SLDS. This approach efficiently handles the challenging interaction between discrete and continuous latent variables through a structured variational approximation.

```@docs
fit!(slds::SLDS{T,S,O}, y::Union{AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}}}; max_iter::Int=50, progress::Bool=true) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
```

Each E-step runs `smoothing_iters` discrete↔continuous alternations before the M-step.
The default of 1 is the standard vLEM update; larger values hand the M-step a
better-converged posterior at proportional cost per iteration.

## The vLEM Algorithm

The vLEM algorithm maximizes the **Evidence Lower Bound (ELBO)** instead of the intractable marginal likelihood. The key insight is to use a structured variational approximation that factorizes as:

```math
q(z_{1:T}, x_{1:T}) = q(z_{1:T}) \prod_{k=1}^K q(x_{1:T} | z_{1:T} = k)^{\mathbb{I}[z_{1:T} = k]}
```

This factorization allows efficient inference by alternating between updating discrete and continuous posteriors.

### Variational Laplace Expectation Step

**0. Initialization:**
Initialize with uniform discrete state posteriors and perform an initial smoothing pass using provided parameter values. This establishes the starting point for iterative refinement.

**1. Update Continuous State Posterior (``q(x_{1:T} | z_{1:T})``):**
For each discrete state sequence ``k``, run Kalman smoothing weighted by the current discrete posterior:

```math
q(x_{1:T} \mid z_{1:T} = k) = \prod_{t=1}^T \mathcal{N}(x_t; \hat{x}_{t|T}^{(k)}, P_{t|T}^{(k)})
```

To handle expectations efficiently, we use a single Monte Carlo sample from this posterior for subsequent computations.

**2. Update Discrete State Posterior (``q(z_{1:T})``):**
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

- Initial distribution: ``\pi_k^{(\text{new})} = \gamma_1(k)``
- Transition matrix: ``A_{ij}^{(\text{new})} = \frac{\sum_{t=1}^{T-1} \xi_{t,t+1}(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}``

where ``\xi_{t,t+1}(i,j) = p(z_t = i, z_{t+1} = j | y_{1:T})`` are the two-slice marginals.

**Continuous State Parameters for each mode ``k``:**

Using weighted sufficient statistics from the smoothed posteriors:

- Dynamics matrix: ``F_k^{(\text{new})}`` from weighted least squares
- Process covariance: ``Q_k^{(\text{new})}`` from weighted innovation covariance
- Observation matrix: ``C_k^{(\text{new})}`` from weighted observation regression
- Observation covariance: ``R_k^{(\text{new})}`` from weighted observation residuals
- Initial parameters: ``\mu_0^{(k)}, P_0^{(k)}`` from weighted initial state statistics

The weights are given by the discrete posterior probabilities ``\gamma_t(k)``.

## Tying the emission across regimes

By default every regime carries its own emission, so `C_k`, `d_k`, `D_k` (and
`R_k`) are fitted from that regime's share of the data. Pass
`tie_emissions=true` to `fit!` for the other common reading of a switching
model: the system's *dynamics* switch while the measurement does not.

```julia
elbos = fit!(slds, y; ux=ux, uy=uy, max_iter=50, tie_emissions=true)
```

The tied update is the ordinary LDS emission M-step. Summing the per-regime
weighted objectives over `k` collapses to the unit-weight one, because the
emission term does not depend on `k` and ``\sum_k \gamma_t(k) = 1`` — so
`[C d D]` is fitted once from the whole trajectory and copied into every regime
(before the first E-step as well, so no regime ever infers through an emission
the model does not have). Combined with `depends_on` the tie is *within* a
group: each session keeps its own emission, shared by every regime. A frozen
emission (`fit_bool`) is left exactly as the caller set it.

This is the usual setup for neural recordings, where the array does not change
when the animal's dynamics do, and it divides the emission's parameter count —
usually the bulk of the model — by `K`.

## Post-fit inference

`fit!` returns the ELBO trace; [`smooth`](@ref) returns the posteriors themselves. Once
an SLDS has been fit, `smooth` infers the full posterior for a dataset with the
parameters held fixed: the continuous states ``q(x)``, the discrete-state
responsibilities ``\gamma_t(k) = q(z_t = k) \approx p(z_t = k \mid y_{1:T})``, and the
ELBO at those posteriors. It alternates the forward-backward pass over the discrete
chain with the Kalman/Laplace smoother over the continuous states, following the classic
coordinate-ascent scheme of Ghahramani & Hinton (1996), stopping once ``\gamma``
converges (`tol`) or after `smoothing_iters` alternations.

Unlike the single-Monte-Carlo-sample E-step that `fit!` uses during learning, the
coupling here is deterministic — the discrete-layer log-likelihoods are evaluated at the
smoothed posterior mean — so the result is reproducible.

Because a converged alternation is expensive, `smooth` returns everything it computed in
one `NamedTuple`; read its `elbo` field rather than calling [`elbo`](@ref) separately.

```@docs; canonical = false
smooth(slds::SLDS{T,S,O}, y::Union{AbstractMatrix{T},AbstractArray{T,3},AbstractVector{<:AbstractMatrix{T}}}) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
```

```julia
post = smooth(fitted, y; ux=ux, uy=uy)
occupancy = post.γ[trial]                    # K × T, columns summing to 1
path = [argmax(view(occupancy, :, t)) for t in axes(occupancy, 2)]
bound = post.elbo                            # scalar ELBO at these posteriors
```

Pass `depends_on` when reading posteriors for a held-out set whose trial count differs
from the one the regimes' stored labels were written for.

[`loglikelihood`](@ref) and [`elbo`](@ref) both return that same ELBO. The exact
marginal ``\log p(y)`` is intractable for a switching model — it requires summing over
all ``K^T`` regime sequences — so `loglikelihood(slds, y)` reports the variational lower
bound.

## Evidence Lower Bound (ELBO)

The ELBO decomposes into discrete and continuous components:

```math
\mathcal{L}(q) =
\underbrace{\mathbb{E}_{q(z_{1:T})}[\log p(z_{1:T})] + H[q(z_{1:T})]}_{\text{discrete HMM terms}}
+ \underbrace{\sum_{t=1}^T \sum_{k=1}^K \gamma_t(k)\, \mathbb{E}_{q(x_{1:T})}\!\left[\log p(y_t, x_t \mid x_{t-1}, z_t = k)\right] + H[q(x_{1:T})]}_{\text{weighted LDS contribution}}
```

where ``H[\cdot]`` denotes entropy and the per-timestep joint terms are weighted by the
discrete posterior marginals ``\gamma_t(k)``.

## References

For theoretical foundations and algorithmic details:

- **"A general recurrent state space framework for modeling neural dynamics during decision-making"** by **David Zoltowski, Jonathon Pillow, and Scott Linderman** (2020)
- **"Variational Learning for Switching State-Space Models"** by **Zoubin Ghahramani and Geoffrey Hinton** (1998)
- **"Probabilistic Machine Learning: Advanced Topics, Chapter 29"** by **Kevin Murphy**
- **"A Unifying Review of Linear Gaussian Models"** by **Sam Roweis and Zoubin Ghahramani**