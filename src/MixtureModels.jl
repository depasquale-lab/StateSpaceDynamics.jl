export  GMM, fit!, log_likelihood

#Move general rand() for MM to Global Types?
"""

    rand(d::MixtureModel, n)

Draw `n` samples from `d`.
"""
rand(d::MixtureModel, n::Int)




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
end


"""GMM Constructor"""
function GMM(k::Int, data_dim::Int)
    Σ = [I(data_dim) for _ = 1:k]
    πs = ones(k) ./ k
    μ = zeros(Float64, data_dim, k)  # Mean of each cluster initialized to zero matrix
    return GMM(k, μ, Σ, πs)
end



"""
rand(gmm::GMM, n)

Draw 'n' samples from gmm. Returns 'data dim' by n size Matrix{Float64}.

"""
function Base.rand(gmm::GMM, n::Int)
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
function EStep(gmm::GMM, data::Matrix{Float64})
    N, _ = size(data)
    K = gmm.k
    γ = zeros(N, K)
    log_γ = zeros(N, K)
    class_probabilities = zeros(N, K)
    class_labels = zeros(Int, N)



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

function MStep!(gmm::GMM, data::Matrix{Float64}, class_probabilities::Matrix{Float64})
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

    println("Called w maxiter: $maxiter")

    gmm.μₖ = kmeanspp_initialization(data, gmm.k)

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
end

# Handle vector data by reshaping it into a 2D matrix with a single column
function EStep(gmm::GMM, data::Vector{Float64})
    EStep(gmm, reshape(data, :, 1))
end

function MStep!(gmm::GMM, data::Vector{Float64}, class_probabilities::Matrix{Float64})
    MStep!(gmm, reshape(data, :, 1), class_probabilities::Matrix{Float64})
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

"""
Binomial Mixture Model.

Args:
    k: number of clusters
    data: data matrix of size (N, D) where N is the number of data points and D is the dimension of the data.
"""

mutable struct BinomialMixtureModel <: MixtureModel
    k::Int # Number of clusters
    pₖ::Vector{Float64} # Means of each cluster
    πₖ::Vector{Float64} # Mixing coefficients
    class_probabilities::Matrix{Float64} # Probability of each class for each point
    class_labels::Vector{Int} # Class label for each point based on the class probabilities
end


