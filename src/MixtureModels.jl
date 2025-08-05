# Public API
export GaussianMixtureModel, PoissonMixtureModel, fit!, loglikelihood

"""
    GaussianMixtureModel

A Gaussian Mixture Model for clustering and density estimation.
"""
mutable struct GaussianMixtureModel{
    T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}
} <: MixtureModel
    k::Int
    μₖ::M           # Means of each cluster (D, K)
    Σₖ::Vector{M}     # Covariance matrices of each cluster (D, D)
    πₖ::V             # Mixing coefficients (K)
end

function Base.show(io::IO, gmm::GaussianMixtureModel; gap = "")
    D = size(gmm.μₖ,1)

    println(io, gap, "Gaussian Mixture Model:")
    println(io, gap, " dimension D = $D")

    if D > 4
        println(io, gap, " size(μₖ) = ($(size(gmm.μₖ,1)),)")
        println(io, gap, " size(Σₖ) = ($(size(gmm.Σₖ[1],1)), $(size(gmm.Σₖ[1],2)))")
    end

    println(io, gap, "-----------------------")

    show_all = get(io, :limit, true) == false

    if gmm.k > 4 && !show_all
        for k in 1:3
            println(io, gap, " Gaussian $k:")
            println(io, gap, " -----------")

            if D > 4
                println(io, gap, "  πₖ = $(round(gmm.πₖ[k], sigdigits=3))")
            else
                println(io, gap, "  μₖ = $(round.(gmm.μₖ[:, k], digits=3))")
                println(io, gap, "  Σₖ = $(round.(gmm.Σₖ[k], digits=3))")
                println(io, gap, "  πₖ = $(round(gmm.πₖ[k], sigdigits=3))")
            end
            println(io, gap, " -----------")
        end
        println(io, gap, "  $(gmm.k - 3) more ..., see `print_full()`")
    else
        for k in 1:gmm.k
            println(io, gap, " Gaussian $k:")
            println(io, gap, " -----------")

            if D > 4
                println(io, gap, "  πₖ = $(round(gmm.πₖ[k], sigdigits=3))")
            else
                println(io, gap, "  μₖ = $(round.(gmm.μₖ[:, k], digits=3))")
                println(io, gap, "  Σₖ = $(round.(gmm.Σₖ[k], digits=3))")
                println(io, gap, "  πₖ = $(round(gmm.πₖ[k], sigdigits=3))")
            end

            if k < gmm.k
                println(io, gap, " -----------")
            end
        end
    end

    return nothing
end

"""
    GaussianMixtureModel(k::Int, data_dim::Int)

Constructor for GaussianMixtureModel. Initializes Σₖ's covariance matrices to the identity,
πₖ to a uniform distribution, and μₖ's means to zeros.
"""
function GaussianMixtureModel(k::Int, D::Int)
    Σ = [Matrix{Float64}(I, D, D) for _ in 1:k]
    πs = ones(k) ./ k
    μ = randn(D, k)

    return GaussianMixtureModel(k, μ, Σ, πs)
end

"""
    Random.rand(rng::AbstractRNG, gmm::GaussianMixtureModel, n::Int)

Draw 'n' samples from gmm. Returns a Matrix{Float64}, where each row is a data point.
"""
function Random.rand(rng::AbstractRNG, gmm::GaussianMixtureModel, n::Int)
    counts = rand(rng, Multinomial(n, gmm.πₖ), 1)
    D = size(gmm.μₖ, 1)
    samples = Matrix{Float64}(undef, D, n)
    idx = 1
    for k in 1:gmm.k
        nk = counts[k]
        if nk > 0
            dist = MvNormal(gmm.μₖ[:, k], gmm.Σₖ[k])
            samples[:, idx:idx + nk - 1] .= rand(rng, dist, nk)
            idx += nk
        end
    end

    return samples
end

Random.rand(gmm::GaussianMixtureModel, n::Int) = rand(Random.default_rng(), gmm, n)

"""
    estep(gmm::GaussianMixtureModel, data::AbstractMatrix{<:Real})

Performs the E-step of the Expectation-Maximization Algorithm for a Gaussian Mixture Model.
"""
function estep(gmm::GaussianMixtureModel, data::AbstractMatrix{<:Real})
    D, N = size(data)
    K = gmm.k
    log_γ = zeros(K, N)
    γ = zeros(K, N)

    for n in 1:N
        for k in 1:K
            dist = MvNormal(gmm.μₖ[:, k], gmm.Σₖ[k])
            log_γ[k, n] = log(gmm.πₖ[k]) + logpdf(dist, data[:, n])
        end
        logsum = logsumexp(log_γ[:, n])
        γ[:, n] .= exp.(log_γ[:, n] .- logsum)
    end

    return γ
end

estep(gmm::GaussianMixtureModel, data::AbstractVector{<:Real}) = estep(gmm, reshape(data, :, 1))

"""
    mstep!(
        gmm::GaussianMixtureModel,
        data::AbstractMatrix{<:Real},
        class_probabilities::AbstractMatrix{<:Real}
    )

Performs the M-step of the Expectation-Maximization Algorithm
"""
function mstep!(
    gmm::GaussianMixtureModel, data::AbstractMatrix{<:Real}, γ::AbstractMatrix{<:Real}
)
    D, N = size(data)
    K = gmm.k
    N_k = sum(γ, dims=2)
    μ_new = (data * γ') ./ N_k'
    Σ_new = Matrix{Float64}[]

    for k in 1:K
        x_centered = data .- μ_new[:, k]
        cov = (x_centered .* reshape(γ[k, :], :, 1)') * x_centered' / (N_k[k] + 1e-6) + 1e-6 * I
        push!(Σ_new, Symmetric(cov))
    end

    gmm.μₖ = μ_new
    gmm.Σₖ = Σ_new
    gmm.πₖ .= vec(N_k) / N

    return nothing
end

mstep!(gmm::GaussianMixtureModel, data::AbstractVector{<:Real}, γ::AbstractMatrix{<:Real}) = mstep!(gmm, reshape(data, :, 1), γ)

"""
    loglikelihood(gmm::GaussianMixtureModel, data::AbstractMatrix{<:Real})

Compute the log-likelihood of the data given the Gaussian Mixture Model (GMM). The data
matrix should be of shape (# observations, # features).

# Arguments
- `gmm::GaussianMixtureModel`: The Gaussian Mixture Model instance
- `data::AbstractMatrix{<:Real}`: data matrix to calculate the Log-Likelihood

# Returns
- `Float64`: The log-likelihood of the data given the model.
"""
function loglikelihood(gmm::GaussianMixtureModel, data::AbstractMatrix{<:Real})
    D, N = size(data)
    K = gmm.k
    ll = 0.0

    for n in 1:N
        probs = [gmm.πₖ[k] * pdf(MvNormal(gmm.μₖ[:, k], gmm.Σₖ[k]), data[:, n]) for k in 1:K]
        ll += log(sum(probs))
    end

    return ll
end

loglikelihood(gmm::GaussianMixtureModel, data::AbstractVector{<:Real}) = loglikelihood(gmm, reshape(data, :, 1))

"""
    fit!(gmm::GaussianMixtureModel, data::AbstractMatrix{<:Real}; <keyword arguments>)

Fits a Gaussian Mixture Model (GMM) to the given data using the Expectation-Maximization
(EM) algorithm.

# Arguments
- `gmm::GaussianMixtureModel`: The Gaussian Mixture Model to be fitted.
- `data::AbstractMatrix{<:Real}`: The dataset on which the model will be fitted, where each
    row represents a data point.
- `maxiter::Int=50`: The maximum number of iterations for the EM algorithm (default: 50).
- `tol::Float64=1e-3`: The tolerance for convergence. The algorithm stops if the change in
    log-likelihood between iterations is less than this value (default: 1e-3).
- `initialize_kmeans::Bool=false`: If true, initializes the means of the GMM using K-means++
    initialization (default: false).

# Returns
- `class_probabilities`: A matrix where each entry (i, k) represents the probability of the
    i-th data point belonging to the k-th component of the mixture model.
"""
function fit!(
    gmm::GaussianMixtureModel, data::AbstractMatrix{<:Real};
    maxiter=50, tol=1e-3, initialize_kmeans=false,
)
    prev_ll = -Inf
    lls = Float64[]

    if initialize_kmeans
        gmm.μₖ = kmeanspp_initialization(data, gmm.k)
    end

    for i in 1:maxiter
        γ = estep(gmm, data)
        mstep!(gmm, data, γ)
        ll = loglikelihood(gmm, data)
        push!(lls, ll)
        println("Iteration $i: Log-likelihood = $ll")
        if abs(ll - prev_ll) < tol
            println("Converged at iteration $i")
            break
        end
        prev_ll = ll
    end

    γ = estep(gmm, data)

    return γ, lls
end

fit!(gmm::GaussianMixtureModel, data::AbstractVector{<:Real}; kwargs...) = fit!(gmm, reshape(data, :, 1); kwargs...)

"""
    PoissonMixtureModel

A Poisson Mixture Model for clustering and density estimation.

## Fields
- `k::Int`: Number of poisson-distributed clusters.
- `λₖ::Vector{Float64}`: Means of each cluster.
- `πₖ::Vector{Float64}`: Mixing coefficients for each cluster.
"""
mutable struct PoissonMixtureModel{T<:Real,V<:AbstractVector{T}} <: MixtureModel
    k::Int
    λₖ::V
    πₖ::V
end

function Base.show(io::IO, pmm::PoissonMixtureModel; gap = "")
    println(io, gap, "Poisson Mixture Model:")
    println(io, gap, "----------------------")

    show_all = get(io, :limit, true) == false

    if pmm.k > 4 && !show_all
        for k in 1:3
            println(io, gap, " Poisson $k:")
            println(io, gap, " ----------")
            println(io, gap, "  λₖ = $(round(pmm.λₖ[k], sigdigits=3))")
            println(io, gap, "  πₖ = $(round(pmm.πₖ[k], sigdigits=3))")
            println(io, gap, " ----------")
        end
        println(io, gap, "  $(pmm.k - 3) more ..., see `print_full()`")
    else
        for k in 1:pmm.k
            println(io, gap, " Poisson $k:")
            println(io, gap, " ----------")
            println(io, gap, "  λₖ = $(round(pmm.λₖ[k], sigdigits=3))")
            println(io, gap, "  πₖ = $(round(pmm.πₖ[k], sigdigits=3))")

            if k < pmm.k
                println(io, gap, " ----------")
            end
        end
    end

    return nothing
end

"""
    PoissonMixtureModel(k::Int)

Constructor for PoissonMixtureModel. Initializes λₖ's means to
ones and πₖ to a uniform distribution.
"""
function PoissonMixtureModel(k::Int)
    λ = ones(k)
    π = ones(k) ./ k

    return PoissonMixtureModel(k, λ, π)
end

"""
    estep(pmm::PoissonMixtureModel, data::AbstractMatrix{<:Integer})

Performs the E-step of the Expectation-Maximization Algorithm for Poisson Mixture Model.
"""
function estep(pmm::PoissonMixtureModel, data::AbstractMatrix{<:Integer})
    _, N = size(data)
    K = pmm.k
    log_γ = zeros(K, N)
    γ = zeros(K, N)

    for n in 1:N
        for k in 1:K
            log_γ[k, n] = log(pmm.πₖ[k]) + logpdf(Poisson(pmm.λₖ[k]), data[1, n])
        end
        logsum = logsumexp(log_γ[:, n])
        γ[:, n] .= exp.(log_γ[:, n] .- logsum)
    end

    return γ
end

estep(pmm::PoissonMixtureModel, data::AbstractVector{<:Integer}) = estep(pmm, reshape(data, 1, :))

"""
    mstep!(
        pmm::PoissonMixtureModel,
        data::AbstractMatrix{<:Integer},
        γ::AbstractMatrix{<:Real}
    )

Performs the M-step of the Expectation-Maximization Algorithm for Poisson Mixture Model.
"""
function mstep!(
    pmm::PoissonMixtureModel,
    data::AbstractMatrix{<:Integer},
    γ::AbstractMatrix{<:Real}
)
    _, N = size(data)
    for k in 1:pmm.k
        Nk = sum(γ[k, :])
        pmm.λₖ[k] = sum(γ[k, :] .* data[1, :]) / Nk
        pmm.πₖ[k] = Nk / N
    end

    return nothing
end

mstep!(pmm::PoissonMixtureModel, data::AbstractVector{<:Integer}, γ::AbstractMatrix{<:Real}) = mstep!(pmm, reshape(data, 1, :), γ)

"""
    Random.rand(rng::AbstractRNG, pmm::PoissonMixtureModel, n::Int)

Draw 'n' samples from pmm. Returns a Vector{Int} of length n.
"""
function Random.rand(rng::AbstractRNG, pmm::PoissonMixtureModel, n::Int)
    component_samples = rand(rng, Multinomial(n, pmm.πₖ), 1)
    samples = Vector{Int}(undef, n)
    start_idx = 1

    for i in 1:pmm.k
        num_samples = component_samples[i]
        if num_samples > 0
            λ = pmm.λₖ[i]
            samples[start_idx:(start_idx + num_samples - 1)] .= rand(rng, Poisson(λ), num_samples)
            start_idx += num_samples
        end
    end

    return samples
end

function Random.rand(pmm::PoissonMixtureModel, n::Int)
    return rand(Random.default_rng(), pmm, n)
end

"""
    loglikelihood(pmm::PoissonMixtureModel, data::AbstractMatrix{<:Integer})

Compute the log-likelihood of the data given the Poisson Mixture Model (PMM). The data
matrix should be of shape (# features, # obs).

# Returns
- `Float64`: The log-likelihood of the data given the model.
"""
function loglikelihood(pmm::PoissonMixtureModel, data::AbstractMatrix{<:Integer})
    _, N = size(data)
    K = pmm.k
    ll = 0.0

    for n in 1:N
        probs = [pmm.πₖ[k] * pdf(Poisson(pmm.λₖ[k]), data[1, n]) for k in 1:K]
        ll += log(sum(probs))
    end

    return ll
end

loglikelihood(pmm::PoissonMixtureModel, data::AbstractVector{<:Integer}) = loglikelihood(pmm, reshape(data, 1, :))

"""
    fit!(
        pmm::PoissonMixtureModel, data::AbstractMatrix{<:Integer};
        maxiter::Int=50, tol::Float64=1e-3, initialize_kmeans::Bool=false,
    )

Fits a Poisson Mixture Model (PMM) to the given data using the Expectation-Maximization (EM)
algorithm.

# Arguments
- `pmm::PoissonMixtureModel`: The Poisson Mixture Model to be fitted.
- `data::AbstractMatrix{<:Integer}`: The dataset on which the model will be fitted, where
    each row represents a data point.

# Keyword Arguments
- `maxiter::Int=50`: The maximum number of iterations for the EM algorithm (default: 50).
- `tol::Float64=1e-3`: The tolerance for convergence. The algorithm stops if the change in
    log-likelihood between iterations is less than this value (default: 1e-3).
- `initialize_kmeans::Bool=false`: If true, initializes the means of the PMM using K-means++
    initialization (default: false).

# Returns
- `class_probabilities`: A matrix where each entry (i, k) represents the probability of the
    i-th data point belonging to the k-th component of the mixture model.
"""
function fit!(
    pmm::PoissonMixtureModel, data::AbstractMatrix{<:Integer};
    maxiter=50, tol=1e-3, initialize_kmeans=false,
)
    prev_ll = -Inf
    lls = Float64[]

    if initialize_kmeans
        pmm.λₖ = vec(kmeanspp_initialization(Float64.(data), pmm.k))
    end

    for iter in 1:maxiter
        γ = estep(pmm, data)
        mstep!(pmm, data, γ)
        ll = loglikelihood(pmm, data)
        push!(lls, ll)
        println("Iteration $iter: Log-likelihood = $ll")
        if abs(ll - prev_ll) < tol
            println("Converged at iteration $iter")
            break
        end
        prev_ll = ll
    end

    γ = estep(pmm, data)

    return γ, lls
end

fit!(pmm::PoissonMixtureModel, data::AbstractVector{<:Integer}; kwargs...) = fit!(pmm, reshape(data, 1, :); kwargs...)
