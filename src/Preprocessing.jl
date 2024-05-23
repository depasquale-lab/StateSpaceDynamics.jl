export ProbabilisticPCA

"""
    mutable struct ProbabilisticPCA 

Probabilistic PCA model from Bishop's Pattern Recognition and Machine Learning.

# Fields:
    W: Weight matrix that maps from latent space to data space.
    σ²: Noise variance
    μ: Mean of the data
    k: Number of latent dimensions
    D: Number of features
"""
mutable struct ProbabilisticPCA
    W::Matrix{<:AbstractFloat}
    σ²::AbstractFloat
    μ::Vector{<:AbstractFloat}
    k::Int
    D::Int
end

"""
    ProbabilisticPCA(;W::Matrix{<:AbstractFloat}, σ²:: <: AbstractFloat, μ::Vector{<:AbstractFloat}, k::Int, D::Int)

Constructor for ProbabilisticPCA model.

# Args:
- `W::Matrix{<:AbstractFloat}`: Weight matrix that maps from latent space to data space.
- `σ²:: <: AbstractFloat`: Noise variance
- `μ::Vector{<:AbstractFloat}`: Mean of the data
- `k::Int`: Number of latent dimensions
- `D::Int`: Number of features

# Example:
```julia
# PPCA with unknown parameters
ppca = ProbabilisticPCA(k=1, D=2)
# PPCA with known parameters
ppca = ProbabilisticPCA(W=rand(2, 1), σ²=0.1, μ=rand(2), k=1, D=2)
```
"""
function ProbabilisticPCA(;W::Matrix{<:AbstractFloat}=Matrix{Float64}(undef, 0, 0), σ²::AbstractFloat=0.0, k::Int, D::Int)
    # if W is not provided, initialize it randomly
    W = isempty(W) ? rand(D, k)/sqrt(k) : W
    # if σ² is not provided, initialize it randomly
    σ² = σ² === 0.0 ? rand() : σ²
    # calcualte mu
    μ = mean(X, dims=1)
    return ProbabilisticPCA(W, σ², μ, k, D)
end

"""
    E_Step(ppca::ProbabilisticPCA, X::Matrix{<:AbstractFloat})

Expectation step of the EM algorithm for PPCA. See Bishop's Pattern Recognition and Machine Learning for more details.

# Args:
- `ppca::ProbabilisticPCA`: PPCA model
- `X::Matrix{<:AbstractFloat}`: Data matrix

# Examples:
```julia
ppca = ProbabilisticPCA(k=1, D=2)
E_Step(ppca, rand(10, 2))
```
"""
function E_step!(model::ProbabilisticPCA, X::Matrix{<:AbstractFloat})
    # calculate M
    M = model.W * model.W' + model.σ² * I(model.k)
    M⁻¹ = pinv(M)
    # calculate the sufficient statistics i.e. E[z] and E[zz']
    E_z = M⁻¹ * model.W' * (X .- model.μ)'
    E_zzᵀ = model.σ² * M⁻¹ + E_z * E_z'
    return E_z, E_zzᵀ
end

"""
    M_Step!(model::ProbabilisticPCA, X::Matrix{<:AbstractFloat}, E_z::Matrix{<:AbstractFloat}, E_zzᵀ::Matrix{<:AbstractFloat})

Maximization step of the EM algorithm for PPCA. See Bishop's Pattern Recognition and Machine Learning for more details.

# Args:
- `model::ProbabilisticPCA`: PPCA model
- `X::Matrix{<:AbstractFloat}`: Data matrix
- `E_z::Matrix{<:AbstractFloat}`: E[z]
- `E_zzᵀ::Matrix{<:AbstractFloat}`: E[zz']

# Examples:
```julia
ppca = ProbabilisticPCA(k=1, D=2)
E_z, E_zzᵀ = E_Step(ppca, rand(10, 2))
M_Step!(ppca, rand(10, 2), E_z, E_zzᵀ)
```
"""
function M_Step!(model::ProbabilisticPCA, X::Matrix{<:AbstractFloat}, E_z::Matrix{<:AbstractFloat}, E_zzᵀ::Matrix{<:AbstractFloat})
    N = size(X, 1)
    # update W
    model.W = sum((X .- model.μ) * E_z', dims=1) * pinv(sum(E_zzᵀ, dims=1))
    # update σ²
    model.σ² = (1/N*model.D) * sum(abs(X .- model.μ).^2 - 2*E_z'*model.W'*(X .- model.μ) .+ trace(E_zzᵀ*model.W'*model.W), dims=1)
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
ppca = ProbabilisticPCA(k=1, D=2)
fit!(ppca, rand(10, 2))
```
"""
function fit!()
end