"""
ACDCAdapters.jl

Model-specific implementations of the stochastic_drivers interface.

This file provides `stochastic_drivers` implementations for:
- GaussianMixtureModel
- PoissonMixtureModel  
- HiddenMarkovModel (with various emission types)
- ProbabilisticPCA

Requires: ACDCInterface.jl
"""

# =============================================================================
# Helper Functions
# =============================================================================

"""
    _sample_categorical(p::AbstractVector) -> Int

Sample from a categorical distribution with probabilities `p`.

# Arguments
- `p::AbstractVector`: Probability vector (must sum to 1)

# Returns
- Index of sampled category
"""
function _sample_categorical(p::AbstractVector{T}) where {T<:Real}
    u = rand(T)
    cumsum_p = zero(T)
    for (i, pi) in enumerate(p)
        cumsum_p += pi
        if u <= cumsum_p
            return i
        end
    end
    return length(p)
end

"""
    _normal_cdf(x::Real) -> Real

Standard normal CDF ``\\Phi(x) = P(Z \\leq x)`` where ``Z \\sim \\mathcal{N}(0,1)``.

# Arguments
- `x::Real`: Input value

# Returns
- CDF value in ``[0, 1]``
"""
function _normal_cdf(x::T) where {T<:Real}
    return T(cdf(Normal(), x))
end

"""
    _poisson_cdf_randomized(x::Integer, λ::Real) -> Real

Randomized CDF for Poisson distribution (discrete probability integral transform).

Returns ``U \\sim \\text{Uniform}(F(x-1), F(x))`` to ensure exact uniformity under the 
true model, where ``F`` is the Poisson CDF with rate ``\\lambda``.

# Arguments
- `x::Integer`: Observed count
- `λ::Real`: Poisson rate parameter

# Returns
- Randomized PIT value in ``[0, 1]``
"""
function _poisson_cdf_randomized(x::Integer, λ::T) where {T<:Real}
    if λ <= zero(T)
        return rand(T)
    end
    d = Poisson(λ)
    lower = x > 0 ? T(cdf(d, x - 1)) : zero(T)
    upper = T(cdf(d, x))
    return lower + rand(T) * (upper - lower)
end

"""
    _bernoulli_cdf_randomized(x::Real, p::Real) -> Real

Randomized CDF for Bernoulli distribution (discrete probability integral transform).

Returns ``U \\sim \\text{Uniform}(F(x-1), F(x))`` to ensure exact uniformity under the
true model, where ``F`` is the Bernoulli CDF with success probability ``p``.

# Arguments
- `x::Real`: Observed value (0 or 1)
- `p::Real`: Success probability

# Returns
- Randomized PIT value in ``[0, 1]``
"""
function _bernoulli_cdf_randomized(x::Real, p::T) where {T<:Real}
    p = clamp(p, T(1e-10), T(1 - 1e-10))
    if x == 0
        # F(0) = 1 - p, return U(0, 1-p)
        return rand(T) * (one(T) - p)
    else
        # F(1) = 1, return U(1-p, 1)
        return (one(T) - p) + rand(T) * p
    end
end

# =============================================================================
# Gaussian Mixture Model Adapter
# =============================================================================

"""
    stochastic_drivers(model::GaussianMixtureModel, data; n_samples=1) -> StochasticDriverResult

Recover stochastic drivers for a Gaussian Mixture Model.

# Model

For GMM, the generative process is:

```math
z_n \\sim \\text{Categorical}(\\pi)
```
```math
x_n | z_n = k \\sim \\mathcal{N}(\\mu_k, \\Sigma_k)
```

# Driver Recovery

We recover ``\\varepsilon`` by:
1. Sample ``z_n`` from posterior ``P(z_n | x_n)``
2. For active component ``k = z_n``: ``\\varepsilon_{d,k,n} = \\Phi(L_k^{-1}(x_n - \\mu_k))_d``
3. For inactive components: ``\\varepsilon_{d,k,n} \\sim U(0,1)``

where ``L_k`` is the Cholesky factor of ``\\Sigma_k``.

# Arguments
- `model::GaussianMixtureModel`: Fitted GMM
- `data::AbstractMatrix`: Observations, D × N matrix

# Keyword Arguments
- `n_samples::Int=1`: Number of stochastic samples per observation

# Returns
- `StochasticDriverResult` with D × K × N × S array of drivers
"""
function stochastic_drivers(
    model::StateSpaceDynamics.GaussianMixtureModel{T},
    data::AbstractMatrix{T};
    n_samples::Int=1,
) where {T<:Real}
    K = model.k
    D, N = size(data)

    # Compute responsibilities γ_{k,n} = p(z_n=k | x_n, θ)
    γ = StateSpaceDynamics.estep(model, data)  # K × N

    # Usage: expected fraction assigned to each component
    usage = vec(sum(γ; dims=2)) ./ N

    # Precompute Cholesky factors
    L_inv = [inv(cholesky(Symmetric(model.Σₖ[k])).L) for k in 1:K]

    # Output array: D × K × N × S
    ε = Array{T,4}(undef, D, K, N, n_samples)

    for s in 1:n_samples
        for n in 1:N
            # Sample which component generated this observation
            z_n = _sample_categorical(γ[:, n])

            for k in 1:K
                if k == z_n
                    # Active component: compute driver from observation
                    # Whitened residual should be N(0, I)
                    whitened = L_inv[k] * (data[:, n] - model.μₖ[:, k])
                    # Apply normal CDF to each dimension
                    for d in 1:D
                        ε[d, k, n, s] = _normal_cdf(whitened[d])
                    end
                else
                    # Inactive component: sample from prior U(0,1)
                    for d in 1:D
                        ε[d, k, n, s] = rand(T)
                    end
                end
            end
        end
    end

    return StochasticDriverResult(ε, usage)
end

# =============================================================================
# Poisson Mixture Model Adapter
# =============================================================================

"""
    stochastic_drivers(model::PoissonMixtureModel, data; n_samples=1) -> StochasticDriverResult

Recover stochastic drivers for a Poisson Mixture Model.

# Model

For Poisson MM, the generative process is:

```math
z_n \\sim \\text{Categorical}(\\pi)
```
```math
x_{n,d} | z_n = k \\sim \\text{Poisson}(\\lambda_{k,d})
```

# Driver Recovery

Uses randomized probability integral transform (PIT) for discrete observations.
For active component ``k = z_n``, computes ``\\varepsilon_{d,k,n} \\sim U(F(x-1), F(x))``
where ``F`` is the Poisson CDF.

# Arguments
- `model::PoissonMixtureModel`: Fitted Poisson mixture model
- `data::AbstractMatrix{<:Integer}`: Count observations, D × N matrix

# Keyword Arguments
- `n_samples::Int=1`: Number of stochastic samples per observation

# Returns
- `StochasticDriverResult` with D × K × N × S array of drivers
"""
function stochastic_drivers(
    model::StateSpaceDynamics.PoissonMixtureModel{T},
    data::AbstractMatrix{<:Integer};
    n_samples::Int=1,
) where {T<:Real}
    K = model.k
    D, N = size(data)

    # Compute responsibilities γ_{k,n} = p(z_n=k | x_n, θ)
    γ = StateSpaceDynamics.estep(model, data)  # K × N

    # Usage: expected fraction assigned to each component
    usage = vec(sum(γ; dims=2)) ./ N

    # Output array: D × K × N × S
    ε = Array{T,4}(undef, D, K, N, n_samples)

    for s in 1:n_samples
        for n in 1:N
            z_n = _sample_categorical(γ[:, n])

            for k in 1:K
                if k == z_n
                    # Active component: use randomized PIT
                    for d in 1:D
                        ε[d, k, n, s] = _poisson_cdf_randomized(data[d, n], model.λₖ[k])
                    end
                else
                    # Inactive: sample from prior
                    for d in 1:D
                        ε[d, k, n, s] = rand(T)
                    end
                end
            end
        end
    end

    return StochasticDriverResult(ε, usage)
end

# =============================================================================
# Hidden Markov Model Adapter
# =============================================================================

"""
    stochastic_drivers(model::HiddenMarkovModel, data; n_samples=1, X=nothing) -> StochasticDriverResult

Recover stochastic drivers for a Hidden Markov Model.

# Model

For HMM, the generative process is:

```math
z_1 \\sim \\text{Categorical}(\\pi_0)
```
```math
z_t | z_{t-1} \\sim \\text{Categorical}(A_{z_{t-1}, :})
```
```math
x_t | z_t = k \\sim B_k(x_t; \\theta_k)
```

where ``B_k`` is the emission distribution for state ``k``.

# Driver Recovery

We sample states from the forward-backward posterior, then compute emission-specific 
drivers for each state. Inactive states receive samples from ``U(0,1)``.

Supported emission types:
- `GaussianEmission`: Whitened residuals through normal CDF
- `GaussianRegressionEmission`: Regression residuals through normal CDF
- `BernoulliRegressionEmission`: Randomized PIT for binary outcomes
- `PoissonRegressionEmission`: Randomized PIT for count outcomes

# Arguments
- `model::HiddenMarkovModel`: Fitted HMM
- `data::AbstractMatrix`: Observations, D × T matrix

# Keyword Arguments
- `n_samples::Int=1`: Number of stochastic samples per observation
- `X::Union{AbstractMatrix,Nothing}=nothing`: Covariates for regression emissions, D_x × T matrix

# Returns
- `StochasticDriverResult` with D × K × T × S array of drivers
"""
function stochastic_drivers(
    model::StateSpaceDynamics.HiddenMarkovModel{T},
    data::AbstractMatrix{T};
    n_samples::Int=1,
    X::Union{AbstractMatrix{<:Real},Nothing}=nothing,
) where {T<:Real}
    K = model.K

    # Get data dimensions - data is D × T (same format as fit!)
    D = size(data, 1)
    N = size(data, 2)
    Y = Matrix(data')  # Convert to T × D for internal use

    # Get state posteriors via forward-backward
    if X === nothing
        γ = StateSpaceDynamics.class_probabilities(model, data)  # K × T
    else
        γ = StateSpaceDynamics.class_probabilities(model, data, X)
    end

    # Convert X to T × D_x for internal indexing
    X_internal = X === nothing ? nothing : Matrix(X')

    # Usage: time fraction in each state
    usage = vec(sum(γ; dims=2)) ./ N

    # Get output dimension from first emission model
    emission_dim = model.B[1].output_dim

    # Output array: D × K × N × S
    ε = Array{T,4}(undef, emission_dim, K, N, n_samples)

    for s in 1:n_samples
        for t in 1:N
            # Sample state from posterior
            z_t = _sample_categorical(γ[:, t])

            for k in 1:K
                if k == z_t
                    # Active state: compute emission-specific driver
                    _emission_to_driver!(
                        view(ε, :, k, t, s),
                        model.B[k],
                        Y[t, :],
                        X_internal === nothing ? nothing : X_internal[t, :],
                    )
                else
                    # Inactive: sample from prior
                    for d in 1:emission_dim
                        ε[d, k, t, s] = rand(T)
                    end
                end
            end
        end
    end

    return StochasticDriverResult(ε, usage)
end

# =============================================================================
# HMM Emission-Specific Driver Computation
# =============================================================================

"""
    _emission_to_driver!(ε_out, emission::GaussianEmission, y, x)

Compute stochastic drivers for Gaussian emission.

Computes ``\\varepsilon_d = \\Phi(L^{-1}(y - \\mu))_d`` where ``L`` is the 
Cholesky factor of ``\\Sigma``.

# Arguments
- `ε_out::AbstractVector`: Output vector for drivers (modified in-place)
- `emission::GaussianEmission`: Gaussian emission parameters
- `y::AbstractVector`: Observation vector
- `x::Nothing`: Unused (no covariates)
"""
function _emission_to_driver!(
    ε_out::AbstractVector{T},
    emission::StateSpaceDynamics.GaussianEmission{T},
    y::AbstractVector{T},
    x::Nothing,
) where {T<:Real}
    μ = emission.μ
    Σ = emission.Σ
    L_inv = inv(cholesky(Symmetric(Σ)).L)

    whitened = L_inv * (y - μ)
    for d in eachindex(ε_out)
        ε_out[d] = _normal_cdf(whitened[d])
    end
end

"""
    _emission_to_driver!(ε_out, emission::GaussianRegressionEmission, y, x)

Compute stochastic drivers for Gaussian regression emission.

Computes ``\\varepsilon_d = \\Phi(L^{-1}(y - \\beta^\\top x))_d`` where ``L`` is the 
Cholesky factor of ``\\Sigma``.

# Arguments
- `ε_out::AbstractVector`: Output vector for drivers (modified in-place)
- `emission::GaussianRegressionEmission`: Regression emission parameters
- `y::AbstractVector`: Observation vector
- `x::AbstractVector`: Covariate vector
"""
function _emission_to_driver!(
    ε_out::AbstractVector{T},
    emission::StateSpaceDynamics.GaussianRegressionEmission{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
) where {T<:Real}
    β = emission.β
    Σ = emission.Σ

    x_design = emission.include_intercept ? vcat(one(T), x) : x
    μ = β' * x_design

    L_inv = inv(cholesky(Symmetric(Σ)).L)
    whitened = L_inv * (y - μ)

    for d in eachindex(ε_out)
        ε_out[d] = _normal_cdf(whitened[d])
    end
end

"""
    _emission_to_driver!(ε_out, emission::BernoulliRegressionEmission, y, x)

Compute stochastic drivers for Bernoulli regression emission.

Uses randomized PIT: ``\\varepsilon_d \\sim U(F(y_d - 1), F(y_d))`` where
``F`` is the Bernoulli CDF with ``p_d = \\sigma(\\beta^\\top x)_d``.

# Arguments
- `ε_out::AbstractVector`: Output vector for drivers (modified in-place)
- `emission::BernoulliRegressionEmission`: Regression emission parameters
- `y::AbstractVector`: Binary observation vector
- `x::AbstractVector`: Covariate vector
"""
function _emission_to_driver!(
    ε_out::AbstractVector{T},
    emission::StateSpaceDynamics.BernoulliRegressionEmission{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
) where {T<:Real}
    β = emission.β

    x_design = emission.include_intercept ? vcat(one(T), x) : x
    η = β' * x_design
    p = one(T) ./ (one(T) .+ exp.(-η))

    for d in eachindex(ε_out)
        ε_out[d] = _bernoulli_cdf_randomized(y[d], p[d])
    end
end

"""
    _emission_to_driver!(ε_out, emission::PoissonRegressionEmission, y, x)

Compute stochastic drivers for Poisson regression emission.

Uses randomized PIT: ``\\varepsilon_d \\sim U(F(y_d - 1), F(y_d))`` where
``F`` is the Poisson CDF with ``\\lambda_d = \\exp(\\beta^\\top x)_d``.

# Arguments
- `ε_out::AbstractVector`: Output vector for drivers (modified in-place)
- `emission::PoissonRegressionEmission`: Regression emission parameters
- `y::AbstractVector`: Count observation vector
- `x::AbstractVector`: Covariate vector
"""
function _emission_to_driver!(
    ε_out::AbstractVector{T},
    emission::StateSpaceDynamics.PoissonRegressionEmission{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
) where {T<:Real}
    β = emission.β

    x_design = emission.include_intercept ? vcat(one(T), x) : x
    η = β' * x_design
    if any(abs.(η) .> 30)
        @warn "Clamping extreme linear predictor η in Poisson regression" extrema(η)
        η = clamp.(η, -30, 30)
    end
    λ = exp.(η)

    for d in eachindex(ε_out)
        ε_out[d] = _poisson_cdf_randomized(Int(y[d]), λ[d])
    end
end

# =============================================================================
# Probabilistic PCA Adapter
# =============================================================================

"""
    stochastic_drivers(model::ProbabilisticPCA, data; n_samples=1) -> StochasticDriverResult

Recover stochastic drivers for Probabilistic PCA.

# Model

For PPCA, the generative model is:

```math
z_n \\sim \\mathcal{N}(0, I_K)
```
```math
x_n = W z_n + \\mu + \\varepsilon, \\quad \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2 I_D)
```

# Decomposition

We decompose this as ``x_d = \\sum_k y_{dk} + \\mu_d`` where ``y_{dk} \\sim \\mathcal{N}(W_{dk} z_k, \\sigma^2_k(n))``.

The per-component noise variance ``\\sigma^2_k(n)`` is allocated **per-sample** based on the 
activation level: components with stronger activation for that sample get more 
noise budget. This ensures misspecification in component ``k`` is localized to 
samples where ``k`` is strongly activated.

The per-component contributions ``y_{dk}`` are sampled via Gaussian deconvolution,
then transformed to uniform via:

```math
\\varepsilon_{dk} = \\Phi\\left(\\frac{y_{dk} - W_{dk} z_k}{\\sigma_k(n)}\\right)
```

# Arguments
- `model::ProbabilisticPCA`: Fitted PPCA model
- `data::AbstractMatrix`: Observations, D × N matrix

# Keyword Arguments
- `n_samples::Int=1`: Number of stochastic samples per observation

# Returns
- `StochasticDriverResult` with D × K × N × S array of drivers 
  (dimensions × factors × observations × samples)
"""
function stochastic_drivers(
    model::StateSpaceDynamics.ProbabilisticPCA{T}, data::AbstractMatrix{T}; n_samples::Int=1
) where {T<:Real}
    D, N = size(data)
    K = model.k  # number of latent factors

    W = model.W      # D × K
    μ = model.μ      # D
    σ² = model.σ²
    z = model.z      # K × N (posterior means, computed by fit!)

    # Usage: variance explained by each factor (for reporting)
    factor_var = [norm(W[:, k])^2 for k in 1:K]
    total_var = sum(factor_var) + σ² * D
    usage = factor_var ./ total_var

    # Output array: D × K × N × S
    ε = Array{T,4}(undef, D, K, N, n_samples)

    for s in 1:n_samples
        for n in 1:N
            z_n = z[:, n]  # K-vector of latent factors for this sample
            x_n = data[:, n]

            # Per-sample variance allocation based on activation strength
            # Component k's contribution to this sample: ||W[:,k] * z_n[k]||²
            activation_k = [(norm(W[:, k]) * abs(z_n[k]))^2 for k in 1:K]
            total_activation = sum(activation_k)

            # Avoid division by zero
            if total_activation < eps(T)
                total_activation = one(T)
            end

            # Allocate σ² proportionally to each component's activation for THIS sample
            σ²_k_n = [σ² * activation_k[k] / total_activation for k in 1:K]

            # Ensure minimum variance to avoid numerical issues
            σ²_k_n = [max(σ²_k, σ² * eps(T) * K) for σ²_k in σ²_k_n]
            σ_k_n = sqrt.(σ²_k_n)

            # For each dimension, deconvolve the sum into per-component contributions
            for d in 1:D
                # Means for each component: W_{dk} * z_k
                μ_dk = [W[d, k] * z_n[k] for k in 1:K]

                # Sample y_{d1}, ..., y_{dK} given they sum to x_d - μ_d
                # Each y_{dk} ~ N(μ_dk, σ²_k(n)) with per-sample component-specific variance
                y_d = _deconvolve_gaussian_sum(x_n[d] - μ[d], μ_dk, σ_k_n)

                # Transform to uniform: ε_{dk} = Φ((y_{dk} - μ_{dk}) / σ_k(n))
                for k in 1:K
                    ε[d, k, n, s] = _normal_cdf((y_d[k] - μ_dk[k]) / σ_k_n[k])
                end
            end
        end
    end

    return StochasticDriverResult(ε, usage)
end

"""
    _deconvolve_gaussian_sum(x_sum, μs, σs) -> Vector

Sample ``(y_1, \\ldots, y_K)`` given that ``\\sum_k y_k = x_{sum}`` and 
``y_k \\sim \\mathcal{N}(\\mu_k, \\sigma_k^2)`` independently.

Uses sequential conditioning: sample ``y_1 | x_{sum}``, then ``y_2 | x_{sum} - y_1``, etc.
Following the formulation in Li et al. (Appendix I.1, Gaussian Factor Analysis).

# Arguments
- `x_sum::Real`: The observed sum to decompose
- `μs::Vector`: Means for each component
- `σs::Union{Real,Vector}`: Standard deviations for each component (scalar for equal variances)

# Returns
- Vector of sampled component values ``y_1, \\ldots, y_K``
"""
function _deconvolve_gaussian_sum(
    x_sum::T, μs::Vector{T}, σs::Union{T,Vector{T}}
) where {T<:Real}
    K = length(μs)
    ys = Vector{T}(undef, K)

    # Handle scalar σ (equal variances) vs vector σ (per-component variances)
    σ_vec = σs isa Vector ? σs : fill(σs, K)
    σ²_vec = σ_vec .^ 2

    cum_x = x_sum
    cum_μ = sum(μs)
    cum_σ² = sum(σ²_vec)

    for k in 1:(K - 1)
        this_μ = μs[k]
        this_σ² = σ²_vec[k]

        # Remove this component from cumulative
        cum_μ -= this_μ
        cum_σ² -= this_σ²

        # Conditional distribution of y_k given y_k + rest = cum_x
        # where y_k ~ N(this_μ, this_σ²) and rest ~ N(cum_μ, cum_σ²)
        #
        # p(y_k | y_k + rest = cum_x) is Gaussian with:
        #   mean = (σ_k^{-2} μ_k - σ̄_k^{-2}(μ̄_k - x̄_k)) / (σ_k^{-2} + σ̄_k^{-2})
        #   var = σ_k² σ̄_k² / (σ_k² + σ̄_k²)

        total_σ² = this_σ² + cum_σ²
        cond_mean = this_μ + (this_σ² / total_σ²) * (cum_x - this_μ - cum_μ)
        cond_var = this_σ² * cum_σ² / total_σ²
        cond_std = sqrt(max(cond_var, eps(T)))

        # Sample y_k from conditional
        ys[k] = cond_mean + cond_std * randn(T)

        # Update remaining sum
        cum_x -= ys[k]
    end

    # Last component is determined by the constraint
    ys[K] = cum_x

    return ys
end
