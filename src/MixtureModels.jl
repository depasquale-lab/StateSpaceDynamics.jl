export  GMM, fit!, log_likelihood

"""
GMM

A Gaussian Mixture Model (GMM) for clustering and density estimation.

## Fields
- `k::Int`: Number of clusters.
- `μₖ::Matrix{Float64}`: Means of each cluster (dimensions: data_dim x k).
- `Σₖ::Array{Matrix{Float64}, 1}`: Covariance matrices of each cluster.
- `πₖ::Vector{Float64}`: Mixing coefficients for each cluster.
- `class_probabilities::Matrix{Float64}`: Probability of each class for each data point.
- `class_labels::Vector{Int}`: Class label for each data point based on the highest probability.

## Examples
```julia
gmm = GMM(3, 2, data) # 3 clusters, 2-dimensional data
fit!(gmm, data)
"""
mutable struct GMM <: MixtureModel
    k::Int # Number of clusters
    μₖ::Matrix{Float64} # Means of each cluster
    Σₖ::Array{Matrix{Float64}, 1} # Covariance matrices of each cluster
    πₖ::Vector{Float64} # Mixing coefficients
    class_probabilities::Matrix{Float64} # Probability of each class for each point
    class_labels::Vector{Int} # Class label for each point based on the class probabilities
end


"""GMM Constructor"""
function GMM(k::Int, data_dim::Int, data::Union{Vector{Float64}, Matrix{Float64}})
    N = size(data, 1)  # Number of data points
    μ = kmeanspp_initialization(data, k)
    Σ = [I(data_dim) for _ = 1:k]
    πs = ones(k) ./ k
    # Initialize class_probabilities with zeros or equal probabilities
    class_probs = zeros(N, k)
    # Initialize class_labels with zeros
    class_lbls = zeros(Int, N)
    return GMM(k, μ, Σ, πs, class_probs, class_lbls)
end


# E-Step (Confirmed in Python)
function EStep!(gmm::GMM, data::Matrix{Float64})
    N, _ = size(data)
    K = gmm.k
    γ = zeros(N, K)
    log_γ = zeros(N, K)
    for n in 1:N
        for k in 1:K
            distribution = MvNormal(gmm.μₖ[:, k], gmm.Σₖ[k])
            log_γ[n, k] = log(gmm.πₖ[k]) + logpdf(distribution, data[n, :])
        end
        logsum = logsumexp(log_γ[n, :])
        γ[n, :] = exp.(log_γ[n, :] .- logsum)
    end
    gmm.class_probabilities = γ
    # Update the most likely class labels for each data point
    gmm.class_labels = [argmax(γ[n, :]) for n in 1:N]
end

function MStep!(gmm::GMM, data::Matrix{Float64})
    N, D = size(data)
    K = gmm.k
    γ = gmm.class_probabilities  # Use class_probabilities from the GMM struct

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

function log_likelihood(gmm::GMM, data::Matrix{Float64})
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

function fit!(gmm::GMM, data::Matrix{Float64}; maxiter::Int=50, tol::Float64=1e-3)
    prev_ll = -Inf  # Initialize to negative infinity
    for i = 1:maxiter
        # E-Step
        EStep!(gmm, data)
        # M-Step
        MStep!(gmm, data)
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
end

# Handle vector data by reshaping it into a 2D matrix with a single column
function EStep!(gmm::GMM, data::Vector{Float64})
    EStep!(gmm, reshape(data, :, 1))
end

function MStep!(gmm::GMM, data::Vector{Float64})
    MStep!(gmm, reshape(data, :, 1))
end

function log_likelihood(gmm::GMM, data::Vector{Float64})
    log_likelihood(gmm, reshape(data, :, 1))
end

function fit!(gmm::GMM, data::Vector{Float64}; maxiter::Int=50, tol::Float64=1e-3)
    fit!(gmm, reshape(data, :, 1); maxiter=maxiter, tol=tol)
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

