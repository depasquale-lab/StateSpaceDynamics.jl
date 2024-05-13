export ProbabilisticPCA



"""Probabilistic Principal Component Analysis"""
mutable struct ProbabilisticPCA
    W::Matrix{Float64}
    σ²::Float64
    μ::Vector{Float64}
    k::Int
    D::Int
    z::Matrix{Float64}
end

"""ProbabilisticPCA Constructor"""
function ProbabilisticPCA(X::Matrix{Float64}, k::Int)
    _, D = size(X)
    # Initialize parameters
    W = randn(D, k) / sqrt(D)
    σ² = 1.0
    μ = vec(mean(X, dims=1))
    z = zeros(size(X, 1), k)
    return ProbabilisticPCA(W, σ², μ, k, D, z)
end

"""
E-Step for Probabilistic PCA

Args:
    ppca: ProbabilisticPCA object
    X: Data matrix
"""
function E_step(ppca::ProbabilisticPCA, X::Matrix{Float64})
    # get dims
    N, D = size(X)
    # preallocate E_zz and E_zz
    E_z = zeros(N, ppca.k)
    E_zz = zeros(N, ppca.k, ppca.k)
    # calculate M
    M = ppca.W' * ppca.W + (ppca.σ² * I(ppca.k))
    M_inv = cholesky(M).U \ (cholesky(M).L \ I)
    # calculate E_z and E_zz
    for i in 1:N
        E_z[i, :] = M_inv * ppca.W' * (X[i, :] - ppca.μ)
        E_zz[i, :, :] = (ppca.σ² * M_inv) + (E_z[i, :] * E_z[i, :]')
    end
    return E_z, E_zz
end

"""
M-Step for Probabilistic PCA.

Args:
    ppca: ProbabilisticPCA object
    X: Data matrix
    E_z: E_z matrix from E-Step
    E_zz: E_zz matrix from E-Step
"""
function M_step!(ppca::ProbabilisticPCA, X::Matrix{Float64}, E_z::AbstractArray, E_zz::AbstractArray)
    # get dims
    N, D = size(X)
    # update W and σ²
    running_sum_W = zeros(D, ppca.k)
    running_sum_σ² = 0.0
    WW = ppca.W' * ppca.W
    for i in 1:N
        running_sum_W += (X[i, :] - ppca.μ) * E_z[i, :]'
        running_sum_σ² +=sum((X[i, :] - ppca.μ).^2) - (2 * E_z[i, :]' * ppca.W' * (X[i, :] - ppca.μ)) + tr(E_zz[i, :, :] * WW)
    end
    ppca.z = E_z
    ppca.W = running_sum_W * pinv(sum(E_zz, dims=1)[1, :, :])
    ppca.σ² = running_sum_σ² / (N*D)
end

function loglikelihood(ppca::ProbabilisticPCA, X::Matrix{Float64})
    # get dims
    N, D = size(X)
    # calculate C and S
    C = ppca.W * ppca.W' + (ppca.σ² * I(D))
    S = sum([X[i, :] * X[i, :]' for i in 1:size(X, 1)]) / N
    # calculate log-likelihood
    ll = -(N/2) * (D * log(2*π) +logdet(C) + tr(pinv(C) * S))
    return ll
end

"""
Fits the ProbabilisticPCA model to the data using the EM algorithm for ProbabilisticPCA as discussed in Bishop's Pattern Recognition and Machine Learning.
See Chapter 12.2.2 for more details.

Args:
    ppca: ProbabilisticPCA object
    X: Data matrix
    max_iters: Maximum number of iterations
    tol: Tolerance for convergence of the reconstruction error
"""
function fit!(ppca::ProbabilisticPCA, X::Matrix{Float64}, max_iters::Int=100, tol::Float64=1e-6)
    prev_ll = -Inf  # Initialize to negative infinity
    for i in 1:max_iters
        # E-Step
        E_z, E_zz = E_step(ppca, X)
        # M-Step
        M_step!(ppca, X, E_z, E_zz)
        # Check for convergence
        ll = loglikelihood(ppca, X)
        println("Log-Likelihood at iter $i: $ll")
        if abs(ll - prev_ll) < tol
            # println("Converged after $i iterations")
            break
        end
        prev_ll = ll
    end
end