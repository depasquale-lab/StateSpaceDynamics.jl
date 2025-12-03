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
Sample from a categorical distribution with probabilities `p`.
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
Standard normal CDF using Distributions.jl.
"""
function _normal_cdf(x::T) where {T<:Real}
    return T(cdf(Normal(), x))
end

"""
Randomized CDF for Poisson distribution (for discrete PIT).
Returns `U(F(x-1), F(x))` to ensure exact uniformity.
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
Randomized CDF for Bernoulli distribution (for discrete PIT).
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

For GMM, the generative process is:

```math
\\begin{aligned}
    z_n &\\sim \\mathrm{Categorical}(\\pi) \\\\
    x_n | z_n=k &\\sim N(\\mu_k, \\Sigma_k)
\\end{aligned}
```


We recover `ε` by:
    1. Sample `z_n` from posterior `P(z_n | x_n)`
    2. For active component `k=z_n: ε_{d,k,n} = Φ(L_k^{-1}(x_n - μ_k))_d`
    3. For inactive components: `ε_{d,k,n} ~ U(0,1)`

# Returns D × K × N × S array of drivers.
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

Uses randomized PIT for discrete Poisson observations.
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

For HMM, we sample states from the forward-backward posterior, then compute
emission-specific drivers for each state.

Note: `X` should be passed in the same format as fit!, i.e., `D_x × T`.
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
Compute stochastic drivers for Gaussian emission.
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
Compute stochastic drivers for Gaussian regression emission.
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
Compute stochastic drivers for Bernoulli regression emission.
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
Compute stochastic drivers for Poisson regression emission.
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

For PPCA, the generative model is:
    z_n ~ N(0, I_K)
    x_n = W z_n + μ + ε,   ε ~ N(0, σ²I_D)

We decompose this as x_d = Σ_k y_{dk} + μ_d where y_{dk} ~ N(W_{dk} z_k, σ²_k(n)).
The per-component noise variance σ²_k(n) is allocated PER-SAMPLE based on the 
activation level: components with stronger activation for that sample get more 
noise budget. This ensures misspecification in component k is localized to 
samples where k is strongly activated.

The per-component contributions y_{dk} are sampled via Gaussian deconvolution,
then transformed to uniform via ε_{dk} = Φ((y_{dk} - W_{dk} z_k) / σ_k(n)).

Returns D × K × N × S array of drivers (dimensions × factors × observations × samples).
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
            # σ²_k_n = [σ² / K for k in 1:K]

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
    _deconvolve_gaussian_sum(x_sum, μs, σs)

Sample (y_1, ..., y_K) given that Σ_k y_k = x_sum and y_k ~ N(μ_k, σ_k²) independently.

Uses sequential conditioning: sample y_1 | x_sum, then y_2 | x_sum - y_1, etc.
Following the formulation in Li et al. (Appendix I.1, Gaussian Factor Analysis).

# Arguments
- `x_sum`: The observed sum to decompose
- `μs`: Vector of means for each component
- `σs`: Vector of standard deviations for each component (or scalar for equal variances)
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
