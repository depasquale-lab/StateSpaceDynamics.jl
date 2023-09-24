module ssm

using Random
using Statistics
using Distributions
using LinearAlgebra

export GMM, fit!

mutable struct GMM
    k_means::Int
    means::Matrix{Float64}
    covariances::Array{Matrix{Float64}, 1}
    weights::Vector{Float64}
end

"""Initialize a GMM and create initial parameters and return a GMM struct."""
function GMM(k_means::Int, data_dim::Int)
    means = randn(data_dim, k_means)
    covariances = [I(data_dim) for _ = 1:k_means]
    weights = ones(k_means) ./ k_means
    return GMM(k_means, means, covariances, weights)
end

"""E-Step for a GMM"""
function EStep!(gmm::GMM, data::Matrix{Float64})
    # Compute responsibilities and return
    N, K = size(data, 2), gmm.k_means
    responsibilites = zeros(N, K)
    for n in 1:N
        for k in 1:K
            responsibilites[n ,k] = gmm.weights[k] * pdf(MvNormal(gmm.means[:, k], gmm.covariances[k]), data[:, n])
        end
    end
    responsibilites .= responsibilites ./ sum(responsibilites, dim=2)
    return responsibilites
end

"""M-Step for a GMM"""
function MStep!(gmm::GMM, data::Matrix{Float64}, responsibilities::Vector{Float64})
    N, K = size(data, 2), gmm.k_means
    # Update model parameters and return
    for k in 1:K
        #TODO Finish M-step
    end
end

"""Fit the GMM to data"""
function fit!(gmm::GMM, data::Matrix{Float64}, maxiter=50, tol=1e-3)
    for i = 1:maxiter
        EStep!(gmm, data)
        MStep!(gmm, data)
    end
end

end #module 