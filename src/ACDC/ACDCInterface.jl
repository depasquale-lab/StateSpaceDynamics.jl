"""
ACDCInterface.jl

Core interface for ACDC (Accumulated Cutoff Discrepancy Criterion) model selection.

The interface is built around the concept of "stochastic drivers" - the uniform noise
``\\epsilon_{n,k} ~ U(0,1)`` that drives the generative process for each component's contribution
to each observation.

Generative model:

```math
\\begin{aligned}
    x_n &= \\sum_k y_{n,k} \\\\
    y_{n,k} &= f(z_{n,k}, \\phi_k, \\epsilon_{n,k})\\qquad \\text{where} \\epsilon_{n,k} \\sim U(0,1)
\\end{aligned}
```

If the model is correct, the recovered ``\\epsilon_{n,k}`` should be uniformly distributed.
"""

export ACDCResult, StochasticDriverResult, ComponentDiscrepancy
export stochastic_drivers, component_discrepancies
export acdc_loss, acdc_select
export compute_discrepancy
export KLDiscrepancy, WassersteinDiscrepancy, SquaredErrorDiscrepancy
export KSDiscrepancy, MMDDiscrepancy
export get_critical_rho_values

# =============================================================================
# Result Types
# =============================================================================

"""
    ComponentDiscrepancy

Abstract type for discrepancy measures between samples and a reference distribution.

All discrepancies measure deviation from ``U(0,1)`` in the stochastic drivers framework.
"""
abstract type ComponentDiscrepancy end

"""
    StochasticDriverResult{T<:Real}

Container for stochastic drivers recovered from a fitted model.

# Fields
- `ε::Array{T,4}` - Stochastic drivers, D × K × N × S array where:
  - D = dimension of observations
  - K = number of components
  - N = number of observations
  - S = number of samples per observation
  - All values in ``[0, 1]``; should be ``U(0,1)`` if model is correct
- `usage::Vector{T}` - Component contribution magnitudes (length K)
"""
struct StochasticDriverResult{T<:Real}
    ε::Array{T,4}
    usage::Vector{T}

    function StochasticDriverResult(ε::Array{T,4}, usage::Vector{T}) where {T<:Real}
        K = size(ε, 2)
        @assert length(usage) == K "usage must have length K=$(K)"
        return new{T}(ε, usage)
    end
end

"""
    ACDCResult{T<:Real}

Container for ACDC analysis results.

# Fields
- `K::Int` - Number of components
- `component_discrepancies::Vector{T}` - Per-component discrepancy values D̂ₖ
- `component_usage::Vector{T}` - Component contribution magnitudes
"""
struct ACDCResult{T<:Real}
    K::Int
    component_discrepancies::Vector{T}
    component_usage::Vector{T}

    function ACDCResult(K::Int, discs::Vector{T}, usage::Vector{T}) where {T<:Real}
        @assert length(discs) == K "Must have K discrepancy values"
        @assert length(usage) == K "Must have K usage values"
        return new{T}(K, discs, usage)
    end
end

# =============================================================================
# Core Interface
# =============================================================================

"""
    stochastic_drivers(model, data; n_samples::Int=1) -> StochasticDriverResult

Recover the stochastic drivers ``\\epsilon_{d,n,k}`` for each dimension d, component k, and observation n.

The generative model is:

```math
\\begin{aligned}
    x_n &= \\sum_k y_{n,k} \\\\
    y_{n,k} &= f(z_{n,k}, \\phi_k, \\epsilon_{n,k})\\qquad \\text{where} \\epsilon_{n,k} \\sim U(0,1)
\\end{aligned}
```

This function inverts the generative process to recover ε_{n,k}. Since the inversion
involves posterior uncertainty, multiple samples can be drawn.

# Arguments
- `model` - Fitted model
- `data` - Observed data (D × N matrix)
- `n_samples::Int=1` - Number of samples to draw from ``p(\\epsilon_{n,k} | x_n, model)``

# Returns
- `StochasticDriverResult` containing:
  - `ε`: D × K × N × S array of drivers in `[0, 1]`
  - `usage`: K-vector of component contribution magnitudes
"""
function stochastic_drivers end

"""
    component_discrepancies(model, data, discrepancy; n_samples::Int=1) -> ACDCResult

Compute component-wise discrepancies using the stochastic drivers framework.

For each component `k`, the discrepancy is computed by aggregating across all dimensions.
Each `(d, k)` pair contributes a KL divergence, and the component discrepancy is the 
sum across dimensions.

# Arguments
- `model` - Fitted model
- `data` - Observed data
- `discrepancy::ComponentDiscrepancy` - Discrepancy measure
- `n_samples::Int=1` - Number of samples per observation for driver estimation
- `kwargs...` - Custom keyword arguments to pass down to `stochastic_drivers`

# Returns
- `ACDCResult` containing per-component discrepancies from ``U(0,1)``
"""
function component_discrepancies(
    model, data, discrepancy::ComponentDiscrepancy; n_samples::Int=1, kwargs...
)
    result = stochastic_drivers(model, data; n_samples=n_samples, kwargs...)
    ε = result.ε
    usage = result.usage

    D, K, N, S = size(ε)
    T = eltype(ε)

    discs = Vector{T}(undef, K)

    for k in 1:K
        # Reshape to D × (N * S) matrix for multivariate testing
        # Each column is a D-dimensional sample that should be U([0,1]^D)
        ε_k = reshape(ε[:, k, :, :], D, N * S)
        discs[k] = compute_discrepancy(discrepancy, ε_k)
    end

    return ACDCResult(K, discs, usage)
end

# =============================================================================
# Discrepancy Measures
# =============================================================================

"""
    KLDiscrepancy{T<:Real} <: ComponentDiscrepancy

KL divergence ``\\mathcal{D}_KL(P || U([0,1]^D))`` estimated via k-nearest neighbors density estimation.

Measures how much the empirical distribution of stochastic drivers diverges from 
the uniform distribution on the unit hypercube. Returns 0 when drivers are perfectly
uniform, positive otherwise.

# Fields
- `k_neighbors::Int` - Number of neighbors for kNN density estimation (default: 5)
"""
struct KLDiscrepancy{T<:Real} <: ComponentDiscrepancy
    k_neighbors::Int

    function KLDiscrepancy{T}(; k_neighbors::Int=5) where {T<:Real}
        @assert k_neighbors > 0 "k_neighbors must be positive"
        return new{T}(k_neighbors)
    end
end

KLDiscrepancy(; kwargs...) = KLDiscrepancy{Float64}(; kwargs...)

"""
    KSDiscrepancy{T<:Real} <: ComponentDiscrepancy

Kolmogorov-Smirnov statistic for testing uniformity on ``[0,1]^D``.

For D=1: Standard KS test comparing empirical CDF to F(x) = x.
For D>1: Maximum KS statistic across marginal distributions.

Note: The multivariate version tests marginal uniformity but not independence.
Use KLDiscrepancy for a joint uniformity test.
"""
struct KSDiscrepancy{T<:Real} <: ComponentDiscrepancy
    function KSDiscrepancy{T}() where {T<:Real}
        return new{T}()
    end
end

KSDiscrepancy() = KSDiscrepancy{Float64}()

"""
    WassersteinDiscrepancy{T<:Real} <: ComponentDiscrepancy

Wasserstein-p distance between empirical distribution and ``U([0,1]^D)``.

For `D=1`: Closed-form solution using sorted samples vs uniform quantiles.
For `D>1`: Sliced Wasserstein approximation (average over random 1D projections).

# Fields
- `p::Int` - Order of Wasserstein distance (default: 2)
- `regularization::T` - Unused, kept for API compatibility (default: 0.1)
"""
struct WassersteinDiscrepancy{T<:Real} <: ComponentDiscrepancy
    p::Int
    regularization::T

    function WassersteinDiscrepancy{T}(; p::Int=2, regularization::T=T(0.1)) where {T<:Real}
        @assert p > 0 "p must be positive"
        return new{T}(p, regularization)
    end
end

WassersteinDiscrepancy(; kwargs...) = WassersteinDiscrepancy{Float64}(; kwargs...)

"""
    SquaredErrorDiscrepancy{T<:Real} <: ComponentDiscrepancy

Moment-based discrepancy from ``U([0,1]^D)``.

Tests deviation from expected moments of the uniform distribution:
- Marginal means: ``\\mathbb{E}[\\epsilon_d] = 0.5``
- Marginal variances: ``\\mathrm{Var}[\\epsilon_d] = 1/12``
- Cross-covariances: ``\\mathrm{Cov}[\\epsilon_d, \\epsilon_d'] = 0 for d \\neq d'``

Fast to compute but less sensitive than KL or Wasserstein.
"""
struct SquaredErrorDiscrepancy{T<:Real} <: ComponentDiscrepancy
    function SquaredErrorDiscrepancy{T}() where {T<:Real}
        return new{T}()
    end
end

SquaredErrorDiscrepancy() = SquaredErrorDiscrepancy{Float64}()

"""
    MMDDiscrepancy{T<:Real} <: ComponentDiscrepancy

Unbiased Maximum Mean Discrepancy (MMD) using a Gaussian (RBF) kernel.
Measures the distance between the empirical distribution and the reference ``U([0,1]^D)``
by embedding them into a Reproducing Kernel Hilbert Space (RKHS).

This estimator is:
- **Unbiased**: Uses the U-statistic formulation (removing diagonal bias).
- **Dependence-Aware**: The RBF kernel detects dependencies between dimensions.
- **Scalable**: Uses a block-averaging strategy for large N to maintain performance.

# Fields
- `sigma::T` - Kernel bandwidth (default: 0.5). 
               Rule of thumb: set to median pairwise distance.
- `block_size::Int` - max samples per block for computation (default: 5000).
                      If N > block_size, computes average MMD over blocks.
"""
struct MMDDiscrepancy{T<:Real} <: ComponentDiscrepancy
    sigma::T
    block_size::Int

    function MMDDiscrepancy{T}(; sigma::T=T(0.5), block_size::Int=5000) where {T<:Real}
        @assert sigma > 0 "sigma must be positive"
        @assert block_size > 1 "block_size must be > 1"
        return new{T}(sigma, block_size)
    end
end

MMDDiscrepancy(; kwargs...) = MMDDiscrepancy{Float64}(; kwargs...)

# =============================================================================
# Discrepancy Implementations
# =============================================================================

"""
    compute_discrepancy(d::ComponentDiscrepancy, samples) -> Real

Compute the divergence between the empirical distribution of `samples` and the 
theoretical Uniform distribution ``U([0,1]^D)``.

# Arguments
- `d`: ComponentDiscrepancy configuration.
- `samples`: ``D \\times N`` matrix of samples in \$[0, 1]\$.

# Returns
- The estimated divergence (non-negative).
"""
function compute_discrepancy end

"""
Direct estimation of KL divergence on the bounded hypercube ``[0,1]^D`` suffers from 
severe boundary bias, particularly in high dimensions, leading to underestimated densities. 
We apply the Probit transform to map samples to the unbounded space ``\\mathbb{R}^D``,
transforming the problem into comparing the empirical distribution against a standard Multivariate 
Normal \$\\mathcal{N}(0, I)\$. This eliminates boundary artifacts, enabling consistent k-NN density 
estimation without the negative bias associated with the unit cube.
"""
function compute_discrepancy(
    d::KLDiscrepancy{T}, samples::AbstractMatrix{T}
) where {T<:Real}
    D, N = size(samples)
    k = d.k_neighbors

    if N < k + 1
        @warn "Too few samples ($N) for k-NN KL estimation (k=$k)"
        return T(Inf)
    end

    # Transform to Normal Space (Probit Transform)
    # Clamp to avoid -Inf/Inf at 0.0 and 1.0
    ϵ = eps(T)
    samples_clamped = clamp.(samples, ϵ, one(T) - ϵ)

    # Transform U(0,1) -> N(0,1)
    samples_normal = quantile.(Normal(zero(T), one(T)), samples_clamped)

    # Estimate E[log p_data(x)] using k-NN in R^D
    mean_log_p = _mean_log_pdf_knn_multivariate(samples_normal, k)

    # Compute E[log q_ref(x)] where q_ref is N(0, I)
    # log(N(x; 0, I)) = -D/2 * log(2π) - 0.5 * ||x||^2
    log_2pi = log(T(2) * T(π))

    # We compute the mean log-likelihood of the REFERENCE distribution at the observed points
    # sum(x.^2, dims=1) computes squared norm for each sample
    mean_sq_norm = mean(sum(samples_normal .^ 2; dims=1))
    mean_log_q = -0.5 * (D * log_2pi + mean_sq_norm)

    # KL = E[log p] - E[log q]
    # (Using max(0, ...) for numerical stability)
    return max(zero(T), mean_log_p - mean_log_q)
end

function compute_discrepancy(
    d::KSDiscrepancy{T}, samples::AbstractMatrix{T}
) where {T<:Real}
    D, N = size(samples)

    if D == 1
        # 1D case: standard KS
        x = sort(vec(samples))
        ecdf_vals = collect(T, 1:N) ./ N
        ref_cdf_vals = clamp.(x, T(0), T(1))
        ks_stat = maximum(
            max.(
                abs.(ecdf_vals .- ref_cdf_vals), abs.((ecdf_vals .- 1 / N) .- ref_cdf_vals)
            ),
        )
        return ks_stat
    else
        # Multivariate: use maximum of marginal KS statistics
        # This is a simple extension; more sophisticated options exist
        max_ks = zero(T)
        for dim in 1:D
            x = sort(vec(samples[dim, :]))
            ecdf_vals = collect(T, 1:N) ./ N
            ref_cdf_vals = clamp.(x, T(0), T(1))
            ks_stat = maximum(
                max.(
                    abs.(ecdf_vals .- ref_cdf_vals),
                    abs.((ecdf_vals .- 1 / N) .- ref_cdf_vals),
                ),
            )
            max_ks = max(max_ks, ks_stat)
        end
        return max_ks
    end
end

function compute_discrepancy(
    d::SquaredErrorDiscrepancy{T}, samples::AbstractMatrix{T}
) where {T<:Real}
    D, N = size(samples)

    # For U([0,1]^D): 
    # - Each marginal has mean 0.5, var 1/12
    # - Covariance between dimensions should be 0

    total_err = zero(T)

    # Marginal moments
    for dim in 1:D
        x = samples[dim, :]
        μ_sample = mean(x)
        σ2_sample = var(x)
        total_err += (μ_sample - T(0.5))^2
        total_err += (σ2_sample - T(1 / 12))^2
    end

    # Cross-covariances (should be 0)
    for i in 1:D
        for j in (i + 1):D
            cov_ij = cov(samples[i, :], samples[j, :])
            total_err += cov_ij^2
        end
    end

    return total_err / D  # Normalize by dimension
end

function compute_discrepancy(
    d::WassersteinDiscrepancy{T}, samples::AbstractMatrix{T}
) where {T<:Real}
    D, N = size(samples)

    if D == 1
        # 1D closed-form Wasserstein
        x_sorted = sort(vec(samples))
        quantiles = [(i - T(0.5)) / N for i in 1:N]
        w_dist = zero(T)
        for i in 1:N
            w_dist += abs(x_sorted[i] - quantiles[i])^d.p
        end
        return (w_dist / N)^(one(T) / d.p)
    else
        # Multivariate: use sliced Wasserstein (average over random projections)
        n_projections = 50
        total_w = zero(T)

        for _ in 1:n_projections
            # Random unit direction
            direction = randn(T, D)
            direction ./= norm(direction)

            # Project samples onto direction
            projected = vec(direction' * samples)

            # 1D Wasserstein on projection
            x_sorted = sort(projected)
            # For projection of U([0,1]^D), reference is more complex
            # Approximate: use empirical quantiles of uniform samples
            ref_samples = sort(vec(direction' * rand(T, D, N)))

            w_dist = zero(T)
            for i in 1:N
                w_dist += abs(x_sorted[i] - ref_samples[i])^d.p
            end
            total_w += (w_dist / N)^(one(T) / d.p)
        end

        return total_w / n_projections
    end
end

function compute_discrepancy(
    d::MMDDiscrepancy{T}, samples::AbstractMatrix{T}
) where {T<:Real}
    D, N = size(samples)

    # If dataset is small enough, compute exact full MMD
    if N <= d.block_size
        # Generate reference noise on the fly
        # We use the same number of samples N to minimize variance
        reference_uniform = rand(T, D, N)
        return _compute_mmd_quadratic_unbiased(samples, reference_uniform, d.sigma)
    end

    # Block MMD Strategy for Large N
    # Split N into blocks, compute MMD for each, and average.
    # This maintains O(N) complexity instead of O(N^2).
    n_blocks = div(N, d.block_size)
    total_mmd = zero(T)

    for b in 1:n_blocks
        start_idx = (b - 1) * d.block_size + 1
        end_idx = b * d.block_size

        # View into current block
        block_samples = view(samples, :, start_idx:end_idx)

        # Generate fresh uniform noise for this block
        reference = rand(D, d.block_size)

        total_mmd += _compute_mmd_quadratic_unbiased(block_samples, reference, d.sigma)
    end

    # Average the result
    return max(zero(T), total_mmd / n_blocks)
end

# =============================================================================
# ACDC Loss and Model Selection
# =============================================================================

"""
    acdc_loss(result::ACDCResult, ρ::Real) -> Real

Compute the ACDC loss for a given cutoff threshold ρ.

``
    R^ρ(K) = Σₖ max(0, D̂ₖ - ρ)
``

# Arguments
- `result::ACDCResult` - Result from component_discrepancies
- `ρ::Real` - Cutoff threshold for acceptable misspecification

# Returns
- Loss value (non-negative)
"""
function acdc_loss(result::ACDCResult{T}, ρ::Real) where {T<:Real}
    return sum(max(zero(T), d - ρ) for d in result.component_discrepancies)
end

"""
    acdc_select(results::Vector{ACDCResult}, ρ::Real) -> Int

Select the optimal number of components K for a given ρ.

Chooses K with minimum ACDC loss, breaking ties by preferring smaller K.

# Arguments
- `results::Vector{ACDCResult}` - Results for different K values (must be ordered)
- `ρ::Real` - Cutoff threshold

# Returns
- Optimal K value
"""
function acdc_select(results::Vector{ACDCResult{T}}, ρ::Real) where {T<:Real}
    losses = [acdc_loss(r, ρ) for r in results]
    min_loss = minimum(losses)

    # Return smallest K achieving minimum loss
    for (i, loss) in enumerate(losses)
        if loss ≈ min_loss
            return results[i].K
        end
    end

    return results[argmin(losses)].K
end

"""
    get_critical_rho_values(results::Vector{ACDCResult}) -> Vector

Get all critical ρ values where the ACDC loss function changes slope.

These are exactly the component discrepancy values across all K.
Useful for efficient evaluation of the loss curve.

# Arguments
- `results::Vector{ACDCResult}` - Results for different K values

# Returns
- Sorted vector of unique critical ρ values
"""
function get_critical_rho_values(results::Vector{ACDCResult{T}}) where {T<:Real}
    all_discs = T[]
    for r in results
        append!(all_discs, r.component_discrepancies)
    end

    return unique(sort(all_discs))
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    _mean_log_pdf_knn_multivariate(samples, k)

Estimate E_P[log p(x)] using k-NN with KDTree acceleration and bias correction.
Uses the Kozachenko-Leonenko estimator.
"""
function _mean_log_pdf_knn_multivariate(samples::AbstractMatrix{T}, k::Int) where {T<:Real}
    D, N = size(samples)

    # Build KDTree for O(N log N) search
    # NearestNeighbors expects input as columns
    tree = KDTree(samples)

    # Query k+1 nearest neighbors for all points
    # (k+1 because the 1st neighbor is the point itself)
    # The 'true' argument sorts the results so we can grab the (k+1)-th
    idxs, dists = knn(tree, samples, k + 1, true)

    # Compute Log Volume Constant for D-dimensional Euclidean ball
    # log_c_D = (D/2) * log(π) - loggamma(D/2 + 1)
    log_c_D = (D / 2) * log(T(π)) - loggamma(T(D) / 2 + 1)

    # Compute Kozachenko-Leonenko Estimator
    # Bias-corrected term: digamma(k) - digamma(N)
    # Replaces the naive log(k) - log(N)
    bias_correction = digamma(T(k)) - digamma(T(N))

    sum_log_p = zero(T)

    for i in 1:N
        # Distance to the k-th neighbor (excluding self)
        # knn returns self as dists[i][1], so we want dists[i][k+1]
        rho_k = dists[i][k + 1]
        rho_k = max(rho_k, eps(T)) # Numerical stability

        # log(p(x_i)) ≈ ψ(k) - ψ(N) - log(c_D) - D * log(rho_k)
        sum_log_p += bias_correction - log_c_D - D * log(rho_k)
    end

    return sum_log_p / N
end

"""
    _compute_mmd_quadratic_unbiased(X, Y, sigma)

Computes the unbiased U-statistic MMD^2 between sample matrices X and Y.
Cost is O(N^2) where N is the number of columns.
"""
function _compute_mmd_quadratic_unbiased(
    X::AbstractMatrix{T}, Y::AbstractMatrix{T}, sigma::T
) where {T<:Real}
    D, N = size(X)
    gamma = T(1) / (T(2) * sigma^2)

    sum_xx = zero(T)
    sum_yy = zero(T)
    sum_xy = zero(T)

    # Iterating over upper triangle for symmetric terms
    for i in 1:N
        # Intra-group sums (X vs X) and (Y vs Y)
        # Exclude diagonal (i == j) for unbiased estimator
        for j in (i + 1):N
            dist_sq_xx = zero(T)
            dist_sq_yy = zero(T)

            for d in 1:D
                diff_x = X[d, i] - X[d, j]
                diff_y = Y[d, i] - Y[d, j]
                dist_sq_xx += diff_x^2
                dist_sq_yy += diff_y^2
            end

            sum_xx += exp(-gamma * dist_sq_xx)
            sum_yy += exp(-gamma * dist_sq_yy)
        end

        # Cross-group sum (X vs Y)
        # Must check all pairs
        for j in 1:N
            dist_sq_xy = zero(T)
            for d in 1:D
                diff_xy = X[d, i] - Y[d, j]
                dist_sq_xy += diff_xy^2
            end
            sum_xy += exp(-gamma * dist_sq_xy)
        end
    end

    # Normalization factors for U-statistic
    # 2.0 accounts for the symmetric lower triangle we skipped
    norm_intra = T(2) / (N * (N - 1))
    norm_inter = T(2) / (N * N)

    mmd_sq = (norm_intra * sum_xx) + (norm_intra * sum_yy) - (norm_inter * sum_xy)

    return mmd_sq
end
