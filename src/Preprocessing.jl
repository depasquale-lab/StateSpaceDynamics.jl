export ProbabilisticPCA, PoissonPCA

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
    # Create a variable to store the previous log-likelihood
    lls=[]
    # Initialize to negative infinity
    prev_ll = -Inf  
    # init progress bar
    prog = Progress(max_iters; desc="Fitting Probabilistic PCA...")
    for i in 1:max_iters
        # E-Step
        E_z, E_zz = E_step(ppca, X)
        # M-Step
        M_step!(ppca, X, E_z, E_zz)
        # Check for convergence
        ll = loglikelihood(ppca, X)
        push!(lls, ll)
        next!(prog)
        if abs(ll - prev_ll) < tol
            finish!(prog)
            return lls
        end
        prev_ll = ll
    end
    finish!(prog)
    return lls
end

"""
    mutable struct PoissonPCA end

A probabilsitic (PCA?) model for Poisson distributed data.

# Fields
- `C::Matrix{<:Real}`: The loading matrix
- `b::Vector{<:Real}`: The bias vector
- `latent_dim::Int64`: The latent dimensionality
- `obs_dim::Int64`: The observation dimensionality
"""
mutable struct PoissonPCA 
    C::Matrix{<:Real}
    log_b::Vector{<:Real} # log of the bias vector, this way we can ensure that the bias is always positive
    latent_dim::Int64
    obs_dim::Int64
end

"""
    PoissonPCA(;C::Matrix{<:Real}=Matrix{Float64}(undef, 0, 0), b::Vector{<:Real}=Vector{Float64}(undef, 0), latent_dim::Int64, obs_dim::Int64)

Constructor for PoissonPCA

# Arguments
- `C::Matrix{<:Real}`: The loading matrix
- `log_b::Vector{<:Real}`: The log of the bias vector, this way we can ensure that the bias is always positive
- `latent_dim::Int64`: The latent dimensionality
- `obs_dim::Int64`: The observation dimensionality
"""
function PoissonPCA(;C::Matrix{<:Real}=Matrix{Float64}(undef, 0, 0), log_b::Vector{<:Real}=Vector{Float64}(undef, 0), latent_dim::Int64, obs_dim::Int64)
    # initialize C
    if isempty(C)
        C = randn(obs_dim, latent_dim)
    end
    # initialize b
    if isempty(log_b)
        log_b = abs.(randn(obs_dim))
    end
    return PoissonPCA(C, log_b, latent_dim, obs_dim)
end

"""
    loglikelihood(ppca::PoissonPCA, y::Matrix{<:Real}, x::Matrix{<:Real})

Calculates the complete data log-likelihood of the Poisson PCA model.

# Arguments
- `ppca::PoissonPCA`: The Poisson PCA model
- `y::Matrix{<:Real}`: The observed data
- `x::Matrix{<:Real}`: The latent variables
"""
function loglikelihood(ppca::PoissonPCA, y::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    # Calculate λ
    λ = exp.(ppca.C * x' .+ exp.(ppca.log_b))
    # Calculate the log-likelihood
    ll = sum(y' .* log.(λ) .- λ .- loggamma.(y .+ 1)) # drop the constant term
    return ll
end