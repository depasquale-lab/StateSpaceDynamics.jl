export GaussianMixtureModel,
    PoissonMixtureModel, fit!, log_likelihood, sample, E_Step, M_Step!, MixtureModel

"""
    GaussianMixtureModel

A Gaussian Mixture Model for clustering and density estimation.

# Fields
- `k::Int`: Number of clusters.
- `μₖ::Matrix{<:Real}`: Means of each cluster (dimensions: data_dim x k).
- `Σₖ::Array{Matrix{<:Real}, 1}`: Covariance matrices of each cluster.
- `πₖ::Vector{Float64}`: Mixing coefficients for each cluster.


# Examples
```julia
gmm = GaussianMixtureModel(3, 2) # Create a Gaussian Mixture Model with 3 clusters and 2-dimensional data
fit!(gmm, data)
```
"""
mutable struct GaussianMixtureModel <: MixtureModel
    k::Int # Number of clusters
    μₖ::Matrix{<:Real} # Means of each cluster
    Σₖ::Array{Matrix{<:Real},1} # Covariance matrices of each cluster
    πₖ::Vector{Float64} # Mixing coefficients
end

"""
    GaussianMixtureModel(k::Int, data_dim::Int)
    
Constructor for GaussianMixtureModel. Initializes Σₖ's covariance matrices to the 
identity, πₖ to a uniform distribution, and μₖ's means to zeros.

"""
function GaussianMixtureModel(k::Int, data_dim::Int)
    Σ = [Matrix{Float64}(I, data_dim, data_dim) for _ in 1:k]
    πs = ones(k) ./ k
    μ = zeros(Float64, k, data_dim)  # Mean of each cluster initialized to zero matrix
    return GaussianMixtureModel(k, μ, Σ, πs)
end

"""
Draw 'n' samples from gmm. Returns a Matrix{<:Real}, where each row is a data point.
"""
function Random.rand(rng::AbstractRNG, gmm::GaussianMixtureModel, n::Int)
    component_samples = rand(rng, Multinomial(n, gmm.πₖ), 1)
    samples = Matrix{Float64}(undef, n, size(gmm.μₖ, 2))
    start_idx = 1

    for i in 1:gmm.k
        num_samples = component_samples[i]
        if num_samples > 0
            dist = MvNormal(gmm.μₖ[i, :], gmm.Σₖ[i])
            samples[start_idx:(start_idx + num_samples - 1), :] .= transpose(
                rand(rng, dist, num_samples)
            )
            start_idx += num_samples
        end
    end
    return samples
end

function Random.rand(gmm::GaussianMixtureModel, n::Int)
    return rand(Random.default_rng(), gmm, n)
end

function E_Step(gmm::GaussianMixtureModel, data::Matrix{<:Real})
    N, _ = size(data)
    K = gmm.k
    γ = zeros(N, K)
    log_γ = zeros(N, K)
    class_probabilities = zeros(N, K)

    @views for n in 1:N
        for k in 1:K
            distribution = MvNormal(gmm.μₖ[k, :], gmm.Σₖ[k])
            log_γ[n, k] = log(gmm.πₖ[k]) + logpdf(distribution, data[n, :])
        end
        logsum = logsumexp(log_γ[n, :])
        γ[n, :] = exp.(log_γ[n, :] .- logsum)
    end
    # Return probabilities of each class for each point
    return γ
end

function M_Step!(
    gmm::GaussianMixtureModel, data::Matrix{<:Real}, class_probabilities::Matrix{<:Real}
)
    N, D = size(data)
    K = gmm.k
    γ = class_probabilities

    N_k = zeros(K)
    μₖ = zeros(K, D)
    Σₖ = zeros(D, D, K)

    @views for k in 1:K
        N_k[k] = sum(γ[:, k])
        μₖ[k, :] = (γ[:, k]' * data) ./ N_k[k]
    end

    @views for k in 1:K
        x_n = data .- μₖ[k, :]'
        Σₖ[:, :, k] = ((γ[:, k] .* x_n)' * x_n ./ (N_k[k] + I * 1e-6)) + (I * 1e-6)
        if !ishermitian(Σₖ[:, :, k])
            Σₖ[:, :, k] = 0.5 * (Σₖ[:, :, k] + Σₖ[:, :, k]')
        end
        gmm.πₖ[k] = N_k[k] / N
    end

    gmm.μₖ = μₖ
    return gmm.Σₖ = [Σₖ[:, :, k] for k in 1:K]
end

"""
    log_likelihood(gmm::GaussianMixtureModel, data::Matrix{<:Real})

Compute the log-likelihood of the data given the Gaussian Mixture Model (GMM). The data matrix should be of shape (# observations, # features).

# Returns
- `Float64`: The log-likelihood of the data given the model.
"""
function log_likelihood(gmm::GaussianMixtureModel, data::Matrix{<:Real})
    N, K = size(data, 1), gmm.k
    ll = 0.0
    @views for n in 1:N
        log_probabilities = [
            log(gmm.πₖ[k]) + logpdf(MvNormal(gmm.μₖ[k, :], gmm.Σₖ[k]), data[n, :]) for
            k in 1:K
        ]
        max_log_prob = maximum(log_probabilities)
        ll_n =
            max_log_prob +
            log(sum(exp(log_prob - max_log_prob) for log_prob in log_probabilities))
        ll += ll_n
    end
    return ll
end

"""
    fit!(gmm::GaussianMixtureModel, data::Matrix{<:Real}; <keyword arguments>)
Fits a Gaussian Mixture Model (GMM) to the given data using the Expectation-Maximization (EM) algorithm.

# Arguments
- `gmm::GaussianMixtureModel`: The Gaussian Mixture Model to be fitted.
- `data::Matrix{<:Real}`: The dataset on which the model will be fitted, where each row represents a data point.
- `maxiter::Int=50`: The maximum number of iterations for the EM algorithm (default: 50).
- `tol::Float64=1e-3`: The tolerance for convergence. The algorithm stops if the change in log-likelihood between iterations is less than this value (default: 1e-3).
- `initialize_kmeans::Bool=false`: If true, initializes the means of the GMM using K-means++ initialization (default: false).

# Returns
- `class_probabilities`: A matrix where each entry (i, k) represents the probability of the i-th data point belonging to the k-th component of the mixture model.

# Example
```julia
data = rand(2, 100)  # Generate some random data
gmm = GaussianMixtureModel(k=3, d=2)  # Initialize a GMM with 3 components and 2-dimensional data
class_probabilities = fit!(gmm, data, maxiter=100, tol=1e-4, initialize_kmeans=true)
```
"""
function fit!(
    gmm::GaussianMixtureModel,
    data::Matrix{<:Real};
    maxiter::Int=50,
    tol::Float64=1e-3,
    initialize_kmeans::Bool=false,
)
    prev_ll = -Inf  # Initialize to negative infinity

    if initialize_kmeans
        gmm.μₖ = permutedims(kmeanspp_initialization(data, gmm.k))
    end

    for i in 1:maxiter
        # E-Step
        class_probabilities = E_Step(gmm, data)
        # M-Step
        M_Step!(gmm, data, class_probabilities)
        # Calculate current log-likelihood
        curr_ll = log_likelihood(gmm, data)
        # Debug: Output log-likelihood
        println("Iteration: $i, Log-likelihood: $curr_ll")
        # Check for convergence
        if abs(curr_ll - prev_ll) < tol
            println("Convergence reached at iteration $i")
            break
        end
        prev_ll = curr_ll
    end

    class_probabilities = E_Step(gmm, data)
    return class_probabilities
end

# Handle vector data by reshaping it into a 2D matrix with a single column
function E_Step(gmm::GaussianMixtureModel, data::Vector{Float64})
    return E_Step(gmm, reshape(data, :, 1))
end

function M_Step!(
    gmm::GaussianMixtureModel, data::Vector{Float64}, class_probabilities::Matrix{<:Real}
)
    return M_Step!(gmm, reshape(data, :, 1), class_probabilities::Matrix{<:Real})
end

function log_likelihood(gmm::GaussianMixtureModel, data::Vector{Float64})
    return log_likelihood(gmm, reshape(data, :, 1))
end

function fit!(
    gmm::GaussianMixtureModel,
    data::Vector{Float64};
    maxiter::Int=50,
    tol::Float64=1e-3,
    initialize_kmeans::Bool=true,
)
    return fit!(
        gmm,
        reshape(data, :, 1);
        maxiter=maxiter,
        tol=tol,
        initialize_kmeans=initialize_kmeans,
    )
end

"""
    PoissonMixtureModel

A Poisson Mixture Model for clustering and density estimation.

## Fields
- `k::Int`: Number of poisson-distributed clusters.
- `λₖ::Vector{Float64}`: Means of each cluster.
- `πₖ::Vector{Float64}`: Mixing coefficients for each cluster.


## Examples
```julia
pmm = PoissonMixtureModel(3) # 3 clusters, 2-dimensional data
fit!(pmm, data)```
"""
mutable struct PoissonMixtureModel <: MixtureModel
    k::Int # Number of clusters
    λₖ::Vector{Float64} # Means of each cluster
    πₖ::Vector{Float64} # Mixing coefficients
end

"""
    PoissonMixtureModel(k::Int)
    
Constructor for PoissonMixtureModel. Initializes λₖ's means to 
ones and πₖ to a uniform distribution.

"""
function PoissonMixtureModel(k::Int)
    λs = ones(k)
    πs = ones(k) ./ k
    return PoissonMixtureModel(k, λs, πs)
end

function E_Step(pmm::PoissonMixtureModel, data::Matrix{Int})
    N, _ = size(data)
    γ = zeros(N, pmm.k)

    @views for n in 1:N
        for k in 1:(pmm.k)
            λk = pmm.λₖ[k]
            log_γnk = log(pmm.πₖ[k]) + logpdf(Poisson(λk), data[n, 1])
            γ[n, k] = exp(log_γnk)  # Direct computation of responsibilities
        end
        γ[n, :] /= sum(γ[n, :])  # Normalize responsibilities
    end

    return γ  # Return the responsibility matrix
end

function M_Step!(pmm::PoissonMixtureModel, data::Matrix{Int}, γ::Matrix{<:Real})
    N, _ = size(data)

    @views for k in 1:(pmm.k)
        Nk = sum(γ[:, k])
        pmm.λₖ[k] = sum(γ[:, k] .* data) / Nk  # Update λk
        pmm.πₖ[k] = Nk / N  # Update mixing coefficient
    end
end

"""
    fit!(pmm::PoissonMixtureModel, data::Matrix{Int}; <keyword arguments>)

Fits a Poisson Mixture Model (PMM) to the given data using the Expectation-Maximization (EM) algorithm.

# Arguments
- `pmm::PoissonMixtureModel`: The Poisson Mixture Model to be fitted.
- `data::Matrix{Int}`: The dataset on which the model will be fitted, where each row represents a data point.
- `maxiter::Int=50`: The maximum number of iterations for the EM algorithm (default: 50).
- `tol::Float64=1e-3`: The tolerance for convergence. The algorithm stops if the change in log-likelihood between iterations is less than this value (default: 1e-3).
- `initialize_kmeans::Bool=false`: If true, initializes the means of the PMM using K-means++ initialization (default: false).

# Returns
- `class_probabilities`: A matrix where each entry (i, k) represents the probability of the i-th data point belonging to the k-th component of the mixture model.

# Example
```julia
data = rand(1:10, 100, 1)  # Generate some random integer data
pmm = PoissonMixtureModel(k=3)  # Initialize a PMM with 3 components
class_probabilities = fit!(pmm, data, maxiter=100, tol=1e-4, initialize_kmeans=true)
```
"""
function fit!(
    pmm::PoissonMixtureModel,
    data::Matrix{Int};
    maxiter::Int=50,
    tol::Float64=1e-3,
    initialize_kmeans::Bool=false,
)
    prev_ll = -Inf  # Initialize previous log likelihood to negative infinity

    if initialize_kmeans
        λₖ_matrix = permutedims(kmeanspp_initialization(Float64.(data), pmm.k))
        pmm.λₖ = vec(λₖ_matrix)
    end

    for iter in 1:maxiter
        γ = E_Step(pmm, data)  # E-Step
        M_Step!(pmm, data, γ)  # M-Step
        curr_ll = log_likelihood(pmm, data)  # Current log likelihood

        println("Iteration: $iter, Log-likelihood: $curr_ll")

        if abs(curr_ll - prev_ll) < tol  # Check for convergence
            println("Convergence reached at iteration $iter")
            break
        end
        prev_ll = curr_ll  # Update previous log likelihood
    end
end

"""
    log_likelihood(pmm::PoissonMixtureModel, data::Matrix{Int})

Compute the log-likelihood of the data given the Poisson Mixture Model (PMM). The data matrix should be of shape (# observations, # features).

# Returns
- `Float64`: The log-likelihood of the data given the model.
"""
function log_likelihood(pmm::PoissonMixtureModel, data::Matrix{Int})
    ll = 0.0
    for n in 1:size(data, 1)
        ll_n = log(
            sum([pmm.πₖ[k] * pdf(Poisson(pmm.λₖ[k]), data[n, 1]) for k in 1:(pmm.k)])
        )
        ll += ll_n
    end
    return ll
end

"""
    sample(pmm::PoissonMixtureModel, n)

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

# Handle vector data by reshaping it into a 2D matrix with a single column
function E_Step(pmm::PoissonMixtureModel, data::Vector{Int})
    return E_Step(pmm, reshape(data, :, 1))
end

function M_Step!(
    pmm::PoissonMixtureModel, data::Vector{Int}, class_probabilities::Matrix{<:Real}
)
    return M_Step!(pmm, reshape(data, :, 1), class_probabilities)
end

function log_likelihood(pmm::PoissonMixtureModel, data::Vector{Int})
    return log_likelihood(pmm, reshape(data, :, 1))
end

function fit!(
    pmm::PoissonMixtureModel,
    data::Vector{Int};
    maxiter::Int=50,
    tol::Float64=1e-3,
    initialize_kmeans::Bool=true,
)
    return fit!(
        pmm,
        reshape(data, :, 1);
        maxiter=maxiter,
        tol=tol,
        initialize_kmeans=initialize_kmeans,
    )
end
