export  GMM, fit!, log_likelihood


abstract type MixtureModel end

"""Set of functions for estimating a Gaussian Mixture Model"""

"""GMM Class"""
mutable struct GMM <: MixtureModel
    k_means::Int # Number of clusters
    μ_k::Matrix{Float64} # Means of each cluster
    Σ_k::Array{Matrix{Float64}, 1} # Covariance matrices of each cluster
    π_k::Vector{Float64} # Mixing coefficients
    class_probabilities::Matrix{Float64} # Probability of each class for each point
end

"""GMM Constructor"""
function GMM(k_means::Int, data_dim::Int, data::Union{Vector{Float64}, Matrix{Float64}})
    N = size(data, 1)  # Number of data points
    μ = kmeanspp_initialization(data, k_means)
    Σ = [I(data_dim) for _ = 1:k_means]
    πs = ones(k_means) ./ k_means
    # Initialize class_probabilities with zeros or equal probabilities
    class_probs = zeros(N, k_means)
    return GMM(k_means, μ, Σ, πs, class_probs)
end

# E-Step (Confirmed in Python)
function EStep!(gmm::GMM, data::Matrix{Float64})
    N, _ = size(data)
    K = gmm.k_means
    γ = zeros(N, K)
    log_γ = zeros(N, K)
    for n in 1:N
        for k in 1:K
            distribution = MvNormal(gmm.μ_k[:, k], gmm.Σ_k[k])
            log_γ[n, k] = log(gmm.π_k[k]) + logpdf(distribution, data[n, :])
        end
        logsum = logsumexp(log_γ[n, :])
        γ[n, :] = exp.(log_γ[n, :] .- logsum)
    end
    gmm.class_probabilities = γ
end

function MStep!(gmm::GMM, data::Matrix{Float64})
    N, D = size(data)
    K = gmm.k_means
    γ = gmm.class_probabilities  # Use class_probabilities from the GMM struct

    N_k = zeros(K)
    μ_k = zeros(D, K)
    Σ_k = zeros(D, D, K)

    for k in 1:K
        N_k[k] = sum(γ[:, k])
        μ_k[:, k] = (γ[:, k]' * data) ./ N_k[k]
    end

    for k in 1:K
        x_n = data .- μ_k[:, k]'
        Σ_k[:,:,k] = ((γ[:, k] .* x_n)' * x_n ./ (N_k[k] + I*1e-6)) + (I * 1e-6)
        if !ishermitian(Σ_k[:, :, k])
            Σ_k[:,:,k] = 0.5 * (Σ_k[:,:,k] + Σ_k[:,:,k]')
        end
        gmm.π_k[k] = N_k[k] / N
    end

    gmm.μ_k = μ_k
    gmm.Σ_k = [Σ_k[:,:,k] for k in 1:K]
end

function log_likelihood(gmm::GMM, data::Matrix{Float64})
    N, K = size(data, 1), gmm.k_means
    ll = 0.0
    for n in 1:N
        log_probabilities = [log(gmm.π_k[k]) + logpdf(MvNormal(gmm.μ_k[:, k], gmm.Σ_k[k]), data[n, :]) for k in 1:K]
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

