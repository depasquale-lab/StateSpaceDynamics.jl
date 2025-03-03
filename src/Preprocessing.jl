export ProbabilisticPCA, loglikelihood, fit!

"""
    mutable struct ProbabilisticPCA 

Probabilistic PCA model from Bishop's Pattern Recognition and Machine Learning.

# Fields:
    W: Weight matrix that maps from latent space to data space.
    σ²: Noise variance
    μ: Mean of the data
    k: Number of latent dimensions
    D: Number of features
    z: Latent variables
"""
mutable struct ProbabilisticPCA
    W::Matrix{<:AbstractFloat} # weight matrix
    σ²::AbstractFloat # noise variance
    μ::Matrix{<:AbstractFloat} # mean of the data
    k::Int # number of latent dimensions
    D::Int # dimension of the data
    z::Matrix{<:AbstractFloat} # latent variables
end

"""
#     ProbabilisticPCA(;W::Matrix{<:AbstractFloat}, σ²:: <: AbstractFloat, μ::Matrix{<:AbstractFloat}, k::Int, D::Int)

# Constructor for ProbabilisticPCA model.

# # Args:
# - `W::Matrix{<:AbstractFloat}`: Weight matrix that maps from latent space to data space.
# - `σ²:: <: AbstractFloat`: Noise variance
# - `μ::Matrix{<:AbstractFloat}`: Mean of the data
# - `k::Int`: Number of latent dimensions
# - `D::Int`: Number of features

# # Example:
# ```julia
# # PPCA with unknown parameters
# ppca = ProbabilisticPCA(k=1, D=2)
# # PPCA with known parameters
# ppca = ProbabilisticPCA(W=rand(2, 1), σ²=0.1, μ=rand(2), k=1, D=2)
# ```
# """
function ProbabilisticPCA(;
    W::Matrix{<:AbstractFloat}=Matrix{Float64}(undef, 0, 0),
    μ::Matrix{<:AbstractFloat}=Matrix{Float64}(undef, 0, 0),
    σ²::AbstractFloat=0.0,
    k::Int,
    D::Int,
)
    # if W is not provided, initialize it randomly
    W = isempty(W) ? rand(D, k) / sqrt(k) : W
    # if σ² is not provided, initialize it randomly
    σ² = σ² === 0.0 ? abs(rand()) : σ²
    # add empty z
    z = Matrix{Float64}(undef, 0, 0)
    return ProbabilisticPCA(W, σ², μ, k, D, z)
end

"""
    E_Step(ppca::ProbabilisticPCA, X::Matrix{<:AbstractFloat})

Expectation step of the EM algorithm for PPCA. See Bishop's Pattern Recognition and Machine Learning for more details.

# Args:
- `ppca::ProbabilisticPCA`: PPCA model
- `X::Matrix{<:AbstractFloat}`: Data matrix

# Examples:
```julia
ppca = ProbabilisticPCA(K=1, D=2)
E_Step(ppca, rand(10, 2))
```
"""
function E_Step(ppca::ProbabilisticPCA, X::Matrix{<:Real})
    # get dims
    N, _ = size(X)
    # preallocate E_zz and E_zz
    E_z = zeros(N, ppca.k)
    E_zz = zeros(N, ppca.k, ppca.k)
    # calculate M
    M = ppca.W' * ppca.W + (ppca.σ² * I(ppca.k))
    M_inv = cholesky(M).U \ (cholesky(M).L \ I)
    # calculate E_z and E_zz
    for i in 1:N
        E_z[i, :] = M_inv * ppca.W' * (X[i, :] - ppca.μ')
        E_zz[i, :, :] = (ppca.σ² * M_inv) + (E_z[i, :] * E_z[i, :]')
    end
    return E_z, E_zz
end

"""
    M_Step!(model::ProbabilisticPCA, X::Matrix{<:AbstractFloat}, E_z::Matrix{<:AbstractFloat}, E_zz::Array{<:AbstractFloat, 3}
Maximization step of the EM algorithm for PPCA. See Bishop's Pattern Recognition and Machine Learning for more details.

# Args:
- `model::ProbabilisticPCA`: PPCA model
- `X::Matrix{<:AbstractFloat}`: Data matrix
- `E_z::Matrix{<:AbstractFloat}`: E[z]
- `E_zz::Matrix{<:AbstractFloat}`: E[zz']

# Examples:
```julia
ppca = ProbabilisticPCA(K=1, D=2)
E_z, E_zz = E_Step(ppca, rand(10, 2))
M_Step!(ppca, rand(10, 2), E_z, E_zzᵀ)
```
"""
function M_Step!(
    ppca::ProbabilisticPCA, X::Matrix{<:Real}, E_z::AbstractArray, E_zz::AbstractArray
)
    # get dims
    N, D = size(X)
    # update W and σ²
    running_sum_W = zeros(D, ppca.k)
    running_sum_σ² = 0.0
    WW = ppca.W' * ppca.W
    for i in 1:N
        running_sum_W += (X[i, :] - ppca.μ') * E_z[i, :]'
        running_sum_σ² +=
            sum((X[i, :] - ppca.μ') .^ 2) -
            sum((2 * E_z[i, :]' * ppca.W' * (X[i, :] - ppca.μ'))) + tr(E_zz[i, :, :] * WW)
    end
    ppca.z = E_z
    ppca.W = running_sum_W * pinv(sum(E_zz; dims=1)[1, :, :])
    return ppca.σ² = running_sum_σ² / (N * D)
end

"""
    loglikelihood(model::ProbabilisticPCA, X::Matrix{<:AbstractFloat})
    
Calculate the log-likelihood of the data given the PPCA model.

# Args:
- `model::ProbabilisticPCA`: PPCA model
- `X::Matrix{<:AbstractFloat}`: Data matrix

# Examples:
```julia
ppca = ProbabilisticPCA(K=1, D=2)
loglikelihood(ppca, rand(10, 2))
```
"""
function loglikelihood(ppca::ProbabilisticPCA, X::Matrix{<:Real})
    # get dims
    N, D = size(X)
    # calculate C and S
    C = ppca.W * ppca.W' + (ppca.σ² * I(D))
    # center the data
    X = X .- ppca.μ
    S = sum([X[i, :] * X[i, :]' for i in axes(X, 1)]) / N
    # calculate log-likelihood
    ll = -(N / 2) * (D * log(2 * π) + logdet(C) + tr(pinv(C) * S))
    return ll
end

"""
    fit!(model::ProbabilisticPCA, X::Matrix{<:AbstractFloat}, max_iter::Int=100, tol::AbstractFloat=1e-6)

Fit the PPCA model to the data using the EM algorithm.

# Args:
- `model::ProbabilisticPCA`: PPCA model
- `X::Matrix{<:AbstractFloat}`: Data matrix
- `max_iter::Int`: Maximum number of iterations
- `tol::AbstractFloat`: Tolerance for convergence

# Examples:
```julia
ppca = ProbabilisticPCA(K=1, D=2)
fit!(ppca, rand(10, 2))
```
"""
function fit!(
    ppca::ProbabilisticPCA, X::Matrix{Float64}, max_iters::Int=100, tol::Float64=1e-6
)
    # initialize the μ if not done 
    ppca.μ = mean(X; dims=1)
    # initiliaze the log-likelihood
    lls = []
    prev_ll = -Inf
    prog = Progress(max_iters; desc="Fitting Probabilistic PCA...")

    for i in 1:max_iters
        E_z, E_zz = E_Step(ppca, X)
        M_Step!(ppca, X, E_z, E_zz)

        ll = loglikelihood(ppca, X)
        push!(lls, ll)
        next!(prog)

        if abs(ll - prev_ll) < tol
            finish!(prog)
            return lls
        end

        prev_ll = ll  # Update prev_ll for the next iteration
    end

    finish!(prog)
    return lls
end
