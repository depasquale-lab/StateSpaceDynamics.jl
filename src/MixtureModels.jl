export  GaussianMixtureModel, PoissonMixtureModel, fit!, log_likelihood, sample



"""
GaussianMixtureModel

A Gaussian Mixture Model (GaussianMixtureModel) for clustering and density estimation.

## Fields
- `k::Int`: Number of clusters.
- `μₖ::Matrix{Float64}`: Means of each cluster (dimensions: data_dim x k).
- `Σₖ::Array{Matrix{Float64}, 1}`: Covariance matrices of each cluster.
- `πₖ::Vector{Float64}`: Mixing coefficients for each cluster.


## Examples
```julia
gmm = GaussianMixtureModel(3, 2, data) # 3 clusters, 2-dimensional data
fit!(gmm, data)
"""
mutable struct GaussianMixtureModel <: MixtureModel
    k::Int # Number of clusters
    μₖ::Matrix{Float64} # Means of each cluster
    Σₖ::Array{Matrix{Float64}, 1} # Covariance matrices of each cluster
    πₖ::Vector{Float64} # Mixing coefficients
end


"""GaussianMixtureModel Constructor"""
function GaussianMixtureModel(k::Int, data_dim::Int)
    Σ = [I(data_dim) for _ = 1:k]
    πs = ones(k) ./ k
    μ = zeros(Float64, k, data_dim)  # Mean of each cluster initialized to zero matrix
    return GaussianMixtureModel(k, μ, Σ, πs)
end



"""
sample(gmm::GaussianMixtureModel, n)

Draw 'n' samples from gmm. Returns 'data dim' by n size Matrix{Float64}.

"""
function sample(gmm::GaussianMixtureModel, n::Int)
    # Determine the number of samples from each component
    component_samples = rand(Multinomial(n, gmm.πₖ), 1)
    
    # Initialize a container for all samples
    samples = Matrix{Float64}(undef, n, size(gmm.μₖ, 2))
    start_idx = 1
    
    for i in 1:gmm.k
        num_samples = component_samples[i]
        if num_samples > 0
            # Sample all at once from the i-th Gaussian component
            dist = MvNormal(gmm.μₖ[i, :], gmm.Σₖ[i])
            samples[start_idx:(start_idx + num_samples - 1), :] = transpose(rand(dist, num_samples))
            start_idx += num_samples
        end
    end
    return samples
end





# E-Step (Confirmed in Python)
"""
    Returns 
        - `class_probabilities::Matrix{Float64}`: Probability of each class for each data point.

"""
function EStep(gmm::GaussianMixtureModel, data::Matrix{Float64})
    N, _ = size(data)
    K = gmm.k
    γ = zeros(N, K)
    log_γ = zeros(N, K)
    class_probabilities = zeros(N, K)



    for n in 1:N
        for k in 1:K
            distribution = MvNormal(gmm.μₖ[k, :], gmm.Σₖ[k])
            log_γ[n, k] = log(gmm.πₖ[k]) + logpdf(distribution, data[n, :])
        end
        logsum = logsumexp(log_γ[n, :])
        γ[n, :] = exp.(log_γ[n, :] .- logsum)
    end
    # Return probability of each class for each point
    return γ
end

function MStep!(gmm::GaussianMixtureModel, data::Matrix{Float64}, class_probabilities::Matrix{Float64})
    N, D = size(data)
    K = gmm.k
    γ = class_probabilities  

    N_k = zeros(K)
    μₖ = zeros(K, D)
    Σₖ = zeros(D, D, K)

    for k in 1:K
        N_k[k] = sum(γ[:, k])
        μₖ[k, :] = (γ[:, k]' * data) ./ N_k[k]
    end

    for k in 1:K
        x_n = data .- μₖ[k, :]'
        Σₖ[:,:,k] = ((γ[:, k] .* x_n)' * x_n ./ (N_k[k] + I*1e-6)) + (I * 1e-6)
        if !ishermitian(Σₖ[:, :, k])
            Σₖ[:,:,k] = 0.5 * (Σₖ[:,:,k] + Σₖ[:,:,k]')
        end
        gmm.πₖ[k] = N_k[k] / N
    end

    gmm.μₖ = μₖ
    gmm.Σₖ = [Σₖ[:,:,k] for k in 1:K]
end

function log_likelihood(gmm::GaussianMixtureModel, data::Matrix{Float64})
    N, K = size(data, 1), gmm.k
    ll = 0.0
    for n in 1:N
        log_probabilities = [log(gmm.πₖ[k]) + logpdf(MvNormal(gmm.μₖ[k, :], gmm.Σₖ[k]), data[n, :]) for k in 1:K]
        max_log_prob = maximum(log_probabilities)
        ll_n = max_log_prob + log(sum(exp(log_prob - max_log_prob) for log_prob in log_probabilities))
        ll += ll_n
    end
    return ll
end

# **NOTE** Auto initializes means by default (thus, using repeated maxiter=1 provides unexpected results)
function fit!(gmm::GaussianMixtureModel, data::Matrix{Float64}; maxiter::Int=50, tol::Float64=1e-3, initialize_kmeans::Bool=true)
    prev_ll = -Inf  # Initialize to negative infinity

    if initialize_kmeans
        gmm.μₖ = permutedims(kmeanspp_initialization(data, gmm.k))
    end

    for i = 1:maxiter
        # E-Step
        class_probabilities = EStep(gmm, data)
        # M-Step
        MStep!(gmm, data, class_probabilities)
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

    class_probabilities = EStep(gmm, data)
    return class_probabilities
end

# Handle vector data by reshaping it into a 2D matrix with a single column
function EStep(gmm::GaussianMixtureModel, data::Vector{Float64})
    EStep(gmm, reshape(data, :, 1))
end

function MStep!(gmm::GaussianMixtureModel, data::Vector{Float64}, class_probabilities::Matrix{Float64})
    MStep!(gmm, reshape(data, :, 1), class_probabilities::Matrix{Float64})
end

function log_likelihood(gmm::GaussianMixtureModel, data::Vector{Float64})
    log_likelihood(gmm, reshape(data, :, 1))
end

function fit!(gmm::GaussianMixtureModel, data::Vector{Float64}; maxiter::Int=50, tol::Float64=1e-3, initialize_kmeans::Bool=true)
    fit!(gmm, reshape(data, :, 1); maxiter=maxiter, tol=tol, initialize_kmeans=initialize_kmeans)
end

"""
Poisson Mixture Model.

Args:
    k: number of clusters
    data: data matrix of size (N, D) where N is the number of data points and D is the dimension of the data.
"""

mutable struct PoissonMixtureModel <: MixtureModel
    k::Int # Number of clusters
    λₖ::Vector{Float64} # Means of each cluster
    πₖ::Vector{Float64} # Mixing coefficients
end


"""PoissonMixtureModel Constructor"""
function PoissonMixtureModel(k::Int)
    λs = ones(k)
    πs = ones(k) ./ k
    return PoissonMixtureModel(k, λs, πs)
end

"""E-Step for PMM"""
function EStep(pmm::PoissonMixtureModel, data::Matrix{Int})
    N, _ = size(data)
    γ = zeros(N, pmm.k)
    
    for n in 1:N
        for k in 1:pmm.k
            λk = pmm.λₖ[k]
            log_γnk = log(pmm.πₖ[k]) + logpdf(Poisson(λk), data[n, 1])
            γ[n, k] = exp(log_γnk)  # Direct computation of responsibilities
        end
        γ[n, :] /= sum(γ[n, :])  # Normalize responsibilities
    end
    
    return γ  # Return the responsibility matrix
end

"""M-Step for PMM"""
function MStep!(pmm::PoissonMixtureModel, data::Matrix{Int}, γ::Matrix{Float64})
    N, _ = size(data)
    
    for k in 1:pmm.k
        Nk = sum(γ[:, k])
        pmm.λₖ[k] = sum(γ[:, k] .* data) / Nk  # Update λk
        pmm.πₖ[k] = Nk / N  # Update mixing coefficient
    end
end

"""Fit PMM using the EM algorithm with KMeans initialization and convergence check"""
function fit!(pmm::PoissonMixtureModel, data::Matrix{Int}; maxiter::Int=50, tol::Float64=1e-3, initialize_kmeans::Bool=true)
    prev_ll = -Inf  # Initialize previous log likelihood to negative infinity

    if initialize_kmeans
        λₖ_matrix = permutedims(kmeanspp_initialization(Float64.(data), pmm.k))
        pmm.λₖ = vec(λₖ_matrix)
    end

    for iter in 1:maxiter
        γ = EStep(pmm, data)  # E-Step
        MStep!(pmm, data, γ)  # M-Step
        curr_ll = log_likelihood(pmm, data)  # Current log likelihood

        println("Iteration: $iter, Log-likelihood: $curr_ll")

        if abs(curr_ll - prev_ll) < tol  # Check for convergence
            println("Convergence reached at iteration $iter")
            break
        end
        prev_ll = curr_ll  # Update previous log likelihood
    end
end

"""Log Likelihood for PMM"""
function log_likelihood(pmm::PoissonMixtureModel, data::Matrix{Int})
    ll = 0.0
    for n in 1:size(data, 1)
        ll_n = log(sum([pmm.πₖ[k] * pdf(Poisson(pmm.λₖ[k]), data[n, 1]) for k in 1:pmm.k]))
        ll += ll_n
    end
    return ll
end

"""
sample(pmm::PoissonMixtureModel, n)

Draw 'n' samples from pmm. Returns a Vector{Int} of length n.

"""
function sample(pmm::PoissonMixtureModel, n::Int)
    # Determine the number of samples from each component
    component_samples = rand(Multinomial(n, pmm.πₖ), 1)
    
    # Initialize a container for all samples
    samples = Vector{Int}(undef, n)
    start_idx = 1
    
    for i in 1:pmm.k
        num_samples = component_samples[i]
        if num_samples > 0
            # Sample all at once from the i-th Poisson component
            λ = pmm.λₖ[i] # λ for the i-th component
            samples[start_idx:(start_idx + num_samples - 1)] = rand(Poisson(λ), num_samples)
            start_idx += num_samples
        end
    end
    return samples
end


# Handle vector data by reshaping it into a 2D matrix with a single column
function EStep(pmm::PoissonMixtureModel, data::Vector{Int})
    EStep(pmm, reshape(data, :, 1))
end

function MStep!(pmm::PoissonMixtureModel, data::Vector{Int}, class_probabilities::Matrix{Float64})
    MStep!(pmm, reshape(data, :, 1), class_probabilities)
end

function log_likelihood(pmm::PoissonMixtureModel, data::Vector{Int})
    log_likelihood(pmm, reshape(data, :, 1))
end

function fit!(pmm::PoissonMixtureModel, data::Vector{Int}; maxiter::Int=50, tol::Float64=1e-3, initialize_kmeans::Bool=true)
    fit!(pmm, reshape(data, :, 1); maxiter=maxiter, tol=tol, initialize_kmeans=initialize_kmeans)
end