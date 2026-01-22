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

"""
    _wrapped_cauchy_cdf(θ::Real, μ::Real, ρ::Real) -> Real

Wrapped Cauchy CDF for angle ``θ`` in radians.
"""
function _wrapped_cauchy_cdf(θ::T, μ::T, ρ::T) where {T<:Real}
    eps_T = eps(T)
    ρ = clamp(ρ, zero(T), one(T) - eps_T)

    # Wrap angle difference to (-pi, pi]
    δ = atan(sin(θ - μ), cos(θ - μ))
    scale = (one(T) + ρ) / (one(T) - ρ)
    return T(0.5) + atan(scale * tan(δ / T(2))) / T(pi)
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

Uses probabilistic resampling: for each observation, we sample the component assignment
from the posterior ``P(z_n = k | x_n)``, then only compute and store the driver for that
sampled component. This avoids dilution from uniform samples on inactive components.

For active component ``k = z_n``: ``\\varepsilon_{d,k,n} = \\Phi(L_k^{-1}(x_n - \\mu_k))_d``
where ``L_k`` is the Cholesky factor of ``\\Sigma_k``.

# Arguments
- `model::GaussianMixtureModel`: Fitted GMM
- `data::AbstractMatrix`: Observations, D × N matrix

# Keyword Arguments
- `n_samples::Int=1`: Number of sampling passes over all observations

# Returns
- `StochasticDriverResult` with per-component driver pools
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

    # Collect drivers per component (only for active samples)
    # Using Vector of Vectors for dynamic sizing, will convert to Matrix later
    ε_lists = [Vector{Vector{T}}() for _ in 1:K]

    for s in 1:n_samples
        for n in 1:N
            # Sample which component generated this observation
            z_n = _sample_categorical(γ[:, n])

            # Only compute and store driver for the sampled (active) component
            whitened = L_inv[z_n] * (data[:, n] - model.μₖ[:, z_n])
            ε_sample = [_normal_cdf(whitened[d]) for d in 1:D]
            push!(ε_lists[z_n], ε_sample)
        end
    end

    # Convert to matrices (D × n_k for each component)
    ε_pools = Vector{Matrix{T}}(undef, K)
    for k in 1:K
        n_k = length(ε_lists[k])
        if n_k > 0
            ε_pools[k] = reduce(hcat, ε_lists[k])  # D × n_k
        else
            ε_pools[k] = Matrix{T}(undef, D, 0)  # Empty matrix
        end
    end

    return StochasticDriverResult(ε_pools, usage)
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

Uses probabilistic resampling with randomized probability integral transform (PIT) for 
discrete observations. For each observation, samples the component assignment from the 
posterior, then computes ``\\varepsilon_{d,k,n} \\sim U(F(x-1), F(x))`` where ``F`` is 
the Poisson CDF, only for the sampled component.

# Arguments
- `model::PoissonMixtureModel`: Fitted Poisson mixture model
- `data::AbstractMatrix{<:Integer}`: Count observations, D × N matrix

# Keyword Arguments
- `n_samples::Int=1`: Number of sampling passes over all observations

# Returns
- `StochasticDriverResult` with per-component driver pools
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

    # Collect drivers per component (only for active samples)
    ε_lists = [Vector{Vector{T}}() for _ in 1:K]

    for s in 1:n_samples
        for n in 1:N
            z_n = _sample_categorical(γ[:, n])

            # Only compute and store driver for the sampled (active) component
            ε_sample = [_poisson_cdf_randomized(data[d, n], model.λₖ[z_n]) for d in 1:D]
            push!(ε_lists[z_n], ε_sample)
        end
    end

    # Convert to matrices (D × n_k for each component)
    ε_pools = Vector{Matrix{T}}(undef, K)
    for k in 1:K
        n_k = length(ε_lists[k])
        if n_k > 0
            ε_pools[k] = reduce(hcat, ε_lists[k])
        else
            ε_pools[k] = Matrix{T}(undef, D, 0)
        end
    end

    return StochasticDriverResult(ε_pools, usage)
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

Uses probabilistic resampling: for each time step, we sample the state from the 
forward-backward posterior, then only compute and store the driver for that sampled 
state. This avoids dilution from uniform samples on inactive states.

Supported emission types:
- `GaussianEmission`: Whitened residuals through normal CDF
- `GaussianRegressionEmission`: Regression residuals through normal CDF
- `BernoulliRegressionEmission`: Randomized PIT for binary outcomes
- `PoissonRegressionEmission`: Randomized PIT for count outcomes
- `WrappedCauchyEmission`: Wrapped Cauchy CDF

# Arguments
- `model::HiddenMarkovModel`: Fitted HMM
- `data::AbstractMatrix`: Observations, D × T matrix

# Keyword Arguments
- `n_samples::Int=1`: Number of sampling passes over all observations
- `X::Union{AbstractMatrix,Nothing}=nothing`: Covariates for regression emissions, D_x × T matrix

# Returns
- `StochasticDriverResult` with per-component driver pools
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

    # Collect drivers per state (only for active samples)
    ε_lists = [Vector{Vector{T}}() for _ in 1:K]

    for s in 1:n_samples
        for t in 1:N
            # Sample state from posterior
            z_t = _sample_categorical(γ[:, t])

            # Only compute and store driver for the sampled (active) state
            ε_sample = Vector{T}(undef, emission_dim)
            _emission_to_driver!(
                ε_sample,
                model.B[z_t],
                Y[t, :],
                X_internal === nothing ? nothing : X_internal[t, :],
            )
            push!(ε_lists[z_t], ε_sample)
        end
    end

    # Convert to matrices (D × n_k for each component)
    ε_pools = Vector{Matrix{T}}(undef, K)
    for k in 1:K
        n_k = length(ε_lists[k])
        if n_k > 0
            ε_pools[k] = reduce(hcat, ε_lists[k])
        else
            ε_pools[k] = Matrix{T}(undef, emission_dim, 0)
        end
    end

    return StochasticDriverResult(ε_pools, usage)
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

"""
    _emission_to_driver!(ε_out, emission::WrappedCauchyEmission, y, x)

Compute stochastic drivers for wrapped Cauchy emission.

Uses the wrapped Cauchy CDF: ``\\varepsilon_d = F(\\theta_d)``.

# Arguments
- `ε_out::AbstractVector`: Output vector for drivers (modified in-place)
- `emission::WrappedCauchyEmission`: Wrapped Cauchy emission parameters
- `y::AbstractVector`: Angular observation vector (radians)
- `x::Nothing`: Unused (no covariates)
"""
function _emission_to_driver!(
    ε_out::AbstractVector{T},
    emission::StateSpaceDynamics.WrappedCauchyEmission{T},
    y::AbstractVector{T},
    x::Nothing,
) where {T<:Real}
    for d in eachindex(ε_out)
        ε_out[d] = _wrapped_cauchy_cdf(y[d], emission.μ[d], emission.ρ[d])
    end
end

# =============================================================================
# Probabilistic PCA Adapter
# =============================================================================

"""
    stochastic_drivers(model::ProbabilisticPCA, data; n_samples=1) -> StochasticDriverResult

Recover stochastic drivers for Probabilistic PCA.

# Model

We use a non-standard decomposition where each component has its own noise:

```math
x = \\sum_k y_k, \\quad y_k = W_k z_k + \\varepsilon_k
```

where ``\\varepsilon_k \\sim \\mathcal{N}(0, \\sigma_k^2 I_D)`` with component-specific variance.

# Driver Recovery

Uses variance-based weighting: for each component k, observation n is included with 
probability proportional to ``\\sigma_k^2(n)`` - the variance contribution of component k 
for that observation. This reflects how much information observation n carries about 
component k's specification.

# Arguments
- `model::ProbabilisticPCA`: Fitted PPCA model
- `data::AbstractMatrix`: Observations, D × N matrix

# Keyword Arguments
- `n_samples::Int=1`: Number of sampling passes over all observations

# Returns
- `StochasticDriverResult` with per-component driver pools
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

    # Compute per-component, per-observation variance: σ²_k(n) ∝ ||W_k||² * z_k(n)²
    # This represents how much component k contributes to observation n
    σ²_kn = Matrix{T}(undef, K, N)
    for k in 1:K
        W_k_norm² = norm(W[:, k])^2
        for n in 1:N
            σ²_kn[k, n] = W_k_norm² * z[k, n]^2
        end
    end

    # Usage: average variance contribution per component (for reporting)
    usage = vec(mean(σ²_kn; dims=2))
    total_usage = sum(usage)
    if total_usage > eps(T)
        usage = usage ./ total_usage
    else
        usage = fill(one(T) / K, K)
    end

    # For variance-based rejection sampling, compute max variance per component
    # Accept observation n for component k with prob σ²_k(n) / max_m σ²_k(m)
    max_var_k = vec(maximum(σ²_kn; dims=2))

    # Collect drivers per component
    ε_lists = [Vector{Vector{T}}() for _ in 1:K]

    for s in 1:n_samples
        for n in 1:N
            z_n = z[:, n]
            x_n = data[:, n]

            # Total variance for this observation (for residual allocation)
            total_var_n = sum(σ²_kn[:, n])

            for k in 1:K
                # Rejection sampling: accept with prob σ²_k(n) / max_m σ²_k(m)
                if max_var_k[k] < eps(T)
                    continue  # Component has no variance anywhere
                end

                accept_prob = σ²_kn[k, n] / max_var_k[k]
                if rand(T) > accept_prob
                    continue  # Rejected
                end

                # Accepted - compute driver for component k
                # Component k's share of the noise variance
                if total_var_n < eps(T)
                    continue
                end
                var_share = σ²_kn[k, n] / total_var_n
                σ_k = sqrt(σ² * var_share)
                σ_k = max(σ_k, sqrt(σ² * eps(T)))

                ε_sample = Vector{T}(undef, D)
                for d in 1:D
                    μ_dk = W[d, k] * z_n[k]
                    # Attribute residual proportionally to this component's variance share
                    residual_share = (x_n[d] - μ[d]) * var_share
                    ε_sample[d] = _normal_cdf((residual_share - μ_dk) / σ_k)
                end
                push!(ε_lists[k], ε_sample)
            end
        end
    end

    # Convert to matrices
    ε_pools = Vector{Matrix{T}}(undef, K)
    for k in 1:K
        n_k = length(ε_lists[k])
        if n_k > 0
            ε_pools[k] = reduce(hcat, ε_lists[k])
        else
            ε_pools[k] = Matrix{T}(undef, D, 0)
        end
    end

    return StochasticDriverResult(ε_pools, usage)
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
