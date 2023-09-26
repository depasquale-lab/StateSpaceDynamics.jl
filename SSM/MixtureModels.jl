export  GMM, fit!, log_likelihood

mutable struct GMM
    k_means::Int
    μ_k::Matrix{Float64}
    Σ_k::Array{Matrix{Float64}, 1}
    π_k::Vector{Float64}
end

function GMM(k_means::Int, data_dim::Int, data::Matrix{Float64})
    μ = kmeanspp_initialization(data, k_means)
    Σ = [I(data_dim) for _ = 1:k_means]
    πs = ones(k_means) ./ k_means
    return GMM(k_means, μ, Σ, πs)
end

function euclidean_distance(a::Vector{Float64}, b::Vector{Float64})
    return sqrt(sum((a .- b).^2))
end

function kmeanspp_initialization(data::Matrix{Float64}, k_means::Int)
    N, D = size(data)
    centroids = zeros(D, k_means)
    rand_idx = rand(1:N)
    centroids[:, 1] = data[rand_idx, :]

    for k in 2:k_means
        dists = zeros(N)
        for i in 1:N
            dists[i] = minimum([euclidean_distance(data[i, :], centroids[:, j]) for j in 1:(k-1)])
        end
        probs = dists .^ 2
        probs ./= sum(probs)
        next_idx = sample(1:N, Weights(probs))
        centroids[:, k] = data[next_idx, :]
    end
    return centroids
end

function EStep!(gmm::GMM, data::Matrix{Float64})
    N, _ = size(data)
    K = gmm.k_means
    γ = zeros(N, K)
    
    for n in 1:N
        for k in 1:K
            distribution = MvNormal(gmm.μ_k[:, k], gmm.Σ_k[k])
            γ[n, k] = gmm.π_k[k] * pdf(distribution, data[n, :])
        end
    end
    for n in 1:N
        γ[n, :] ./= sum(γ[n, :], dims=1)
    end
    return γ
end


function MStep!(gmm::GMM, data::Matrix{Float64}, γ::Matrix{Float64})
    N, D = size(data)
    K = gmm.k_means

    # Initializing the variables to zero
    N_k = zeros(K)
    μ_k = zeros(D, K)
    Σ_k = zeros(D, D, K)

    for k in 1:K
        N_k[k] = sum(γ[:, k])
        println("Debug MStep: N_k for cluster $k = ", N_k[k])
        μ_k[:, k] = (γ[:, k]' * data) ./ N_k[k]
    end
    for k in 1:K
        x_n = data .- μ_k[:, k]'
        Σ_k[:,:,k] += (((γ[:, k] .* x_n)' * x_n) ./ N_k[k]) + (I * 1e-6)
        # This is a complete hack...for some reason my covariance matrices were not hermitian so i just avergaed...dunno if my calcualtions were wrong or numerical issues
        Σ_k[:,:,k] = 0.5 * (Σ_k[:,:,k] * Σ_k[:,:,k]')
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
        γ_s = EStep!(gmm, data)
        # M-Step
        MStep!(gmm, data, γ_s)
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

using StatsBase
using LinearAlgebra
using Distributions
μ1, μ2 = [1., 1.], [-1., -1.]
σ1, σ2 = [1 0; 0 1], [1 0; 0 1]
x = [1. 2.; 1. 3.;2. 1.; 0. 0.; 0. -1.; -1. -1.]

K = 2

gmm = GMM(K, 2, x)
gmm.μ_k[:, 1] = μ1
gmm.μ_k[:, 2] = μ2
println(log_likelihood(gmm, x))
fit!(gmm, x)