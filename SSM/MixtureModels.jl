using Random
using Statistics
using Distributions
using LinearAlgebra


export  GMM, fit!

mutable struct GMM
    k_means::Int
    μ_k::Matrix{Float64}
    Σ_k::Array{Matrix{Float64}, 1}
    π_k::Vector{Float64}
end

function GMM(k_means::Int, data_dim::Int)
    μ = randn(data_dim, k_means)
    Σ = [I(data_dim) for _ = 1:k_means]
    πs = ones(k_means) ./ k_means
    return GMM(k_means, μ, Σ, πs)
end

function EStep!(gmm::GMM, data::Matrix{Float64})
    N, K = size(data, 1), gmm.k_means
    πs = zeros(N, K)
    for n in 1:N
        for k in 1:K
            πs[n, k] = gmm.π_k[k] * pdf(MvNormal(gmm.μ_k[:, k], gmm.Σ_k[k]), data[n, :])
        end
    end
    πs .= πs ./ sum(πs, dims=2)
    return πs
end

function MStep!(gmm::GMM, data::Matrix{Float64}, πs::Matrix{Float64})
    N, D = size(data)
    K = gmm.k_means

    for k in 1:K
        N_k = sum(πs[:, k])
        μ_k = sum(data .* πs[:, k], dims=1) ./ N_k
        gmm.μ_k[:, k] = μ_k[:]

        Σ_ks = zeros(D, D)
        for n in 1:N
            x_n = data[n, :] .- μ_k
            Σ_ks .+= (πs[n, k] / N_k) * (x_n' * x_n)
        end
        gmm.Σ_k[k] = Σ_ks

        gmm.π_k[k] = N_k / N 
    end
end

function log_likelihood(gmm::GMM, data::Matrix{Float64})
    N, K = size(data, 1), gmm.k_means
    ll = 0.0
    for n in 1:N
        ll_n = 0.0
        for k in 1:K
            ll_n += gmm.π_k[k] * pdf(MvNormal(gmm.μ_k[:, k], gmm.Σ_k[k]), data[n, :])
        end
        ll += log(ll_n)
    end
    return ll
end

function fit!(gmm::GMM, data::Matrix{Float64}; maxiter::Int=50, tol::Float64=1e-3)
    prev_ll = -Inf  # Initialize to negative infinity
    for i = 1:maxiter
        π_hat = EStep!(gmm, data)
        MStep!(gmm, data, π_hat)
        # Calculate current log-likelihood
        curr_ll = log_likelihood(gmm, data)
        # Print and/or store the log-likelihood
        println("Iteration: $i, Log-likelihood: $curr_ll")
        # Check for convergence
        if abs(curr_ll - prev_ll) < tol
            println("Convergence reached at iteration $i")
            break
        end
        prev_ll = curr_ll
    end
end
