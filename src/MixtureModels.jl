export  GaussianMixtureModel, MMM, fit!, log_likelihood, sample



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
    μ = zeros(Float64, data_dim, k)  # Mean of each cluster initialized to zero matrix
    return GaussianMixtureModel(k, μ, Σ, πs)
end



"""
rand(gmm::GaussianMixtureModel, n)

Draw 'n' samples from gmm. Returns 'data dim' by n size Matrix{Float64}.

"""
function sample(gmm::GaussianMixtureModel, n::Int)
    # Determine the number of samples from each component
    component_samples = rand(Multinomial(n, gmm.πₖ), 1)
    
    # Initialize a container for all samples
    samples = Matrix{Float64}(undef, size(gmm.μₖ, 1), n)
    start_idx = 1
    
    for i in 1:gmm.k
        num_samples = component_samples[i]
        if num_samples > 0
            # Sample all at once from the i-th Gaussian component
            dist = MvNormal(gmm.μₖ[:, i], gmm.Σₖ[i])
            samples[:, start_idx:(start_idx + num_samples - 1)] = rand(dist, num_samples)
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
            distribution = MvNormal(gmm.μₖ[:, k], gmm.Σₖ[k])
            log_γ[n, k] = log(gmm.πₖ[k]) + logpdf(distribution, data[n, :])
        end
        logsum = logsumexp(log_γ[n, :])
        γ[n, :] = exp.(log_γ[n, :] .- logsum)
    end
    # Update probability of each class for each point
    class_probabilities = γ

    return class_probabilities
end

function MStep!(gmm::GaussianMixtureModel, data::Matrix{Float64}, class_probabilities::Matrix{Float64})
    N, D = size(data)
    K = gmm.k
    γ = class_probabilities  

    N_k = zeros(K)
    μₖ = zeros(D, K)
    Σₖ = zeros(D, D, K)

    for k in 1:K
        N_k[k] = sum(γ[:, k])
        μₖ[:, k] = (γ[:, k]' * data) ./ N_k[k]
    end

    for k in 1:K
        x_n = data .- μₖ[:, k]'
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
        log_probabilities = [log(gmm.πₖ[k]) + logpdf(MvNormal(gmm.μₖ[:, k], gmm.Σₖ[k]), data[n, :]) for k in 1:K]
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
        gmm.μₖ = kmeanspp_initialization(data, gmm.k)
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
    class_probabilities::Matrix{Float64} # Probability of each class for each point
    class_labels::Vector{Int} # Class label for each point based on the class probabilities
end

"""
MMM

A Multinomial Mixture Model (MMM) for clustering and density estimation of categorical data.

## Fields
- `k::Int`: Number of clusters.
- `pₖ::Array{Matrix{Float64}, 1}`: Category probabilities for each cluster (each Matrix is categories x k).
- `πₖ::Vector{Float64}`: Mixing coefficients for each cluster.
"""

mutable struct MMM <: MixtureModel
    k::Int # Number of clusters
    pₖ::Array{Matrix{Float64}, 1} # Category probabilities for each cluster
    πₖ::Vector{Float64} # Mixing coefficients

    # Inner constructor with validation
    function MMM(k::Int, p::Array{Matrix{Float64}, 1}, πs::Vector{Float64})
        # Perform validations here as previously described

        # Check that πs sums to 1
        if abs(sum(πs) - 1) > 1e-6
            error("The mixing coefficients (πₖ) must sum to 1.")
        end

        # Check that k matches the length of πs and p
        if k != length(πs) || k != length(p)
            error("The number of clusters (k) must match the length of πₖ and the number of probability matrices in pₖ.")
        end

        # Validation for each probability matrix in p
        for (i, matrix) in enumerate(p)
            for col in eachcol(matrix)
                if abs(sum(col) - 1) > 1e-6
                    error("Each column of the probability matrix for cluster $i must sum to 1.")
                end
            end
        end

        # If all validations pass, create the struct instance
        new(k, p, πs)
    end
end

"""MMM Constructor"""
function MMM(k::Int, data_dim::Int, num_categories::Int)
    πs = ones(k) ./ k
    p = [rand(num_categories, data_dim) for _ = 1:k] # Randomly initialize category probabilities
    for matrix in p
        for col in eachcol(matrix)
            col ./= sum(col) # Normalize probabilities
        end
    end
    return MMM(k, p, πs)
end


"""
rand(mmm::MMM, n)

Draw 'n' samples from mmm. Returns an array of size n, where each element represents a sampled category.
"""
function sample(mmm::MMM, n::Int)
    # Determine the number of samples from each component
    component_samples = rand(Multinomial(n, mmm.πₖ), 1)
    
    samples = []
    for i in 1:mmm.k
        num_samples = component_samples[i]
        if num_samples > 0
            # Sample all at once from the i-th multinomial component
            for j = 1:num_samples
                push!(samples, rand(Categorical(mmm.pₖ[i][:, j])))
            end
        end
    end
    return samples
end


