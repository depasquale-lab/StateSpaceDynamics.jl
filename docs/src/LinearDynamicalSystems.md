# What is a Linear Dynamical System?

```@meta
CollapsedDocStrings = true
```

A **Linear Dynamical System (LDS)** is a mathematical model used to describe how a system evolves over time. These systems are a subset of **state-space models**, where the hidden state dynamics are continuous. What makes these models *linear* is that the latent dynamics evolve according to a linear function of the previous state. The observations, however, can be related to the hidden state through a nonlinear link function.

At its core, an LDS defines:

- **A state transition function**: how the internal state evolves from one time step to the next.
- **An observation function**: how the internal state generates the observed data.

```@docs
LinearDynamicalSystem
```

## The Gaussian Linear Dynamical System

The **Gaussian Linear Dynamical System** — typically just referred to as an LDS — is a specific type of linear dynamical system where both the state transition and observation functions are linear, and all noise is Gaussian.

The generative model is given by:

```math
\begin{aligned}
    x_t &\sim \mathcal{N}(A x_{t-1}, Q) \\
    y_t &\sim \mathcal{N}(C x_t, R)
\end{aligned}
```

Where:

- ``x_t`` is the hidden state at time ``t``
- ``y_t`` is the observed data at time ``t``  
- ``A`` is the state transition matrix
- ``C`` is the observation matrix
- ``Q`` is the process noise covariance
- ``R`` is the observation noise covariance
- ``b`` and ``d`` are bias terms

This can equivalently be written in equation form:

```math
\begin{aligned}
    x_t &= A x_{t-1} + b + \epsilon_t \\
    y_t &= C x_t + d + \eta_t
\end{aligned}
```

Where:

- ``ε_t \sim N(0, Q)`` is the process noise
- ``η_t \sim N(0, R)`` is the observation noise

```@docs
GaussianStateModel
GaussianObservationModel
```

## The Poisson Linear Dynamical System

The **Poisson Linear Dynamical System** is a variant of the LDS where the observations are modeled as counts. This is useful in fields like neuroscience where we are often interested in modeling spike count data. To relate the spiking data to the Gaussian latent variable, we use a nonlinear link function, specifically the exponential function. 

The generative model is given by: 

```math
\begin{aligned}
    x_t &\sim \mathcal{N}(A x_{t-1}, Q) \\
    y_t &\sim \text{Poisson}(\exp(Cx_t + b))
\end{aligned}
```

Where `b` is a bias term.

```@docs
PoissonObservationModel
```

## Sampling from Linear Dynamical Systems

You can generate synthetic data from fitted LDS models:

```@docs
Random.rand(lds::LinearDynamicalSystem; tsteps::Int, ntrials::Int)
```

## Inference in Linear Dynamical Systems

In StateSpaceDynamics.jl, we directly maximize the complete-data log-likelihood function with respect to the latent states given the data and the parameters of the model. In other words, the **maximum a priori** (MAP) estimate of the latent state path is:

```math
\underset{x}{\text{argmax}}  \left[ \log p(x_0) + \sum_{t=2}^T \log p(x_t \mid x_{t-1}) + \sum_{t=1}^T \log p(y_t \mid x_t) \right]
```

This MAP estimation approach has the same computational complexity as traditional Kalman filtering and smoothing — ``\mathcal{O}(T)`` — but is significantly more flexible. Notably, it can handle **nonlinear observations** and **non-Gaussian noise** while still yielding **exact MAP estimates**, unlike approximate techniques such as the Extended Kalman Filter (EKF) or Unscented Kalman Filter (UKF).

## Newton's Method for Latent State Optimization

To find the MAP trajectory, we iteratively optimize the latent states using Newton's method. The update equation at each iteration is:

```math
x^{(i+1)} = x^{(i)} - \left[ \nabla^2 \mathcal{L}(x^{(i)}) \right]^{-1} \nabla \mathcal{L}(x^{(i)})
```

Where:

- ``\mathcal{L}(x)`` is the complete-data log-likelihood:

```math
\mathcal{L}(x) = \log p(x_0) + \sum_{t=2}^T \log p(x_t \mid x_{t-1}) + \sum_{t=1}^T \log p(y_t \mid x_t)
```

- ``\nabla \mathcal{L}(x)`` is the gradient of the full log-likelihood with respect to all latent states
- ``\nabla^2 \mathcal{L}(x)`` is the Hessian of the full log-likelihood

This update is performed over the entire latent state sequence ``x_{1:T}``, and repeated until convergence.

For **Gaussian models**, ``\mathcal{L}(x)`` is quadratic and Newton's method converges in a single step — recovering the exact Kalman smoother solution. For **non-Gaussian models**, the Hessian is not constant and the optimization is more complex. However, the MAP estimate can still be computed efficiently using the same approach as the optimization problem is still convex.

```@docs
smooth
```

## Laplace Approximation of Posterior for Non-Conjugate Observation Models

In the case of non-Gaussian observations, we can use a Laplace approximation to compute the posterior distribution of the latent states. For Gaussian observations (which are conjugate with the Gaussian state model), the posterior is also Gaussian and is the exact posterior. However, for non-Gaussian observations, we can approximate the posterior using a Gaussian distribution centered at the MAP estimate of the latent states. This approximation is given by:

```math  
p(x \mid y) \approx \mathcal{N}(x^{*}, -\left[ \nabla^2 \mathcal{L}(x^{*}) \right]^{-1})
```

Where:

- ``x^{*}`` is the MAP estimate of the latent states
- ``\nabla^2 \mathcal{L}(x^{*})`` is the Hessian of the log-likelihood at the MAP estimate

Despite the requirement of inverting a Hessian of dimension ``(d \times T) \times (d \times T)``, this is still computationally efficient, as the Markov structure of the model renders the Hessian block-tridiagonal, and thus the inversion is tractable.

## Learning in Linear Dynamical Systems

Given the latent structure of state-space models, we must rely on either the Expectation-Maximization (EM) or Variational Inference (VI) approaches to learn the parameters of the model. StateSpaceDynamics.jl supports both EM and VI. For LDS models, we can use Laplace EM, where we approximate the posterior of the latent state path using the Laplace approximation as outlined above. Using these approximate posteriors (or exact ones in the Gaussian case), we can apply closed-form updates for the model parameters.

!!! warning "Identifiability caveats in LDS"
    LDS parameters are **not uniquely identifiable**. For any invertible matrix $$S$$,
    the reparameterization
    ```math
    \begin{aligned}
    x'_t &= S x_t,\\
    A' &= S A S^{-1},\\
    C' &= C S^{-1},\\
    Q' &= S Q S^\top,\\
    R' &= R
    \end{aligned}
    ```
    yields the **same likelihood**. Practical consequences:
    
    - **Scale/rotation ambiguity:** the latent space can be arbitrarily scaled/rotated.
    - **Sign & permutation flips:** columns of $$C$$ (and corresponding rows/cols of $$A$$) can swap or flip signs with no change in fit.
    
    **Common remedies**
    
    - Fix a convention for the latent scale, e.g. set $$Q = I$$ or constrain $$\mathrm{diag}(Q)=1$$.
    - Encourage a canonical orientation, e.g. enforce **orthonormal columns in $$C$$** (up to sign) after each M-step. (Not yet implemented)
    - When comparing fits across runs, align parameters via a **Procrustes** or **Hungarian** matching step.
    
    These issues affect **parameter interpretability** but not **predictive performance**; be cautious when interpreting individual entries of $$A$$, $$C$$, or $$Q$$.

```@docs
fit!(lds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T,3}; max_iter::Int=1000, tol::Float64=1e-12) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
```

## Inverse-Wishart Priors on Covariances (MAP)

To encourage well-conditioned covariance estimates, StateSpaceDynamics.jl supports **Inverse-Wishart (IW)** priors on the covariance matrices of a Linear Dynamical System.  
These can be used to impose **shrinkage** toward a target scale, yielding **maximum a posteriori (MAP)** estimates instead of pure MLEs.

You can attach an IW prior to:

- The **process noise covariance** `Q_prior`
- The **initial state covariance** `P0_prior`
- The **observation noise covariance** `R_prior` (Gaussian models only)

```@docs
IWPrior
```

### Definition

For a covariance matrix ` \Sigma \in \mathbb{R}^{d\times d} `,

```math
\Sigma \sim \text{IW}(\Psi, \nu),
\qquad
p(\Sigma)\propto |\Sigma|^{-(\nu+d+1)/2}\exp\!\Big[-\tfrac12\operatorname{tr}(\Psi\,\Sigma^{-1})\Big].
```

- ` \Psi ` is the **scale matrix** (positive-definite).  
- ` \nu ` is the **degrees of freedom**.  
  A proper mode exists when ` \nu > d+1 `.

Given `n` effective samples and sample statistic `S`,
the posterior is

```math
\Sigma\mid\text{data}\sim\text{IW}(\Psi+S,\;\nu+n)
```

and the **MAP** (mode) update used internally is

```math
\widehat{\Sigma}_{\text{MAP}}
=\frac{\Psi+S}{\nu+n+d+1}.
```

### Example: Adding IW Priors to an LDS

```julia
using LinearAlgebra, Random, StateSpaceDynamics

rng = MersenneTwister(42)
D, P = 3, 4

# Proper SPD scale matrices (avoid UniformScaling)
ΨQ  = diagm(0 => fill(0.01, D))
ΨP0 = diagm(0 => fill(0.01, D))
ΨR  = diagm(0 => fill(0.01, P))

Qprior  = IWPrior(Ψ = ΨQ,  ν = D + 3.0)
P0prior = IWPrior(Ψ = ΨP0, ν = D + 3.0)
Rprior  = IWPrior(Ψ = ΨR,  ν = P + 3.0)

# Gaussian LDS
A = 0.9I + 0.05randn(rng, D, D)
Q = Matrix(I, D, D) .* 0.3
b = zeros(D); x0 = zeros(D); P0 = Matrix(I, D, D) .* 0.8

C = randn(rng, P, D)
R = Matrix(I, P, P) .* 0.25
d = 0.1 .* randn(rng, P)

gsm = GaussianStateModel(A=A, Q=Q, b=b, x0=x0, P0=P0,
                         Q_prior=Qprior, P0_prior=P0prior)
gom = GaussianObservationModel(C=C, R=R, d=d, R_prior=Rprior)
lds = LinearDynamicalSystem(gsm, gom)

X, Y = rand(rng, lds; tsteps=100, ntrials=5)
fit!(lds, Y; max_iter=20, progress=false)
```

### Effect on Learning

- When a prior is present, the M-step uses the MAP mode formula above.  
- The reported **ELBO** includes IW log-prior terms (up to constants), so convergence still tracks the true MAP objective.  
- With small datasets or high-dimensional states, the priors keep `Q`, `P_0`, and `R` **positive-definite and well-conditioned**.

```@docs
calculate_elbo
```

### Choosing `\Psi` and `\nu`

| Goal | Typical choice | Notes |
|------|----------------|-------|
| Mild shrinkage | `\Psi = 0.01I`, `\nu = d+3` | Gentle pull toward small covariances |
| Strong regularization | Larger `\nu` | Acts like adding pseudo-samples |
| Data-scaled prior | `\Psi` ≈ empirical covariance | Centers shrinkage around realistic scale |

> **Tip:** In Julia, `I` is a `UniformScaling` (not a matrix).  
> Use `Matrix(I, d, d)` or `diagm(0 => fill(..., d))` to build `Ψ`.
