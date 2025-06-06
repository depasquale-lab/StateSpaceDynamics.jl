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
mutable struct ProbabilisticPCA{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
    W::M # weight matrix
    σ²::T # noise variance
    μ::V # mean of the data
    k::Int # number of latent dimensions
    D::Int # dimension of the data
    z::M # latent variables
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
function ProbabilisticPCA(
    ::Type{T} = Float64;
    W::Union{AbstractMatrix{T}, Nothing} = nothing,
    μ::Union{AbstractVector{T}, Nothing} = nothing, 
    σ²::Union{T, Nothing} = nothing,
    k::Int,
    D::Int
) where {T<:Real}
    
    # Initialize W with proper type
    if W === nothing
        W_matrix = randn(T, D, k) / sqrt(T(k))
    else
        @assert size(W) == (D, k) "W must have size ($D, $k)"
        W_matrix = W
    end
    
    # Initialize μ with proper type
    if μ === nothing
        μ_vector = zeros(T, D)  # Will be set during fitting
    else
        @assert length(μ) == D "μ must have length $D"
        μ_vector = μ
    end
    
    # Initialize σ²
    if σ² === nothing
        σ²_val = one(T)  # Default to 1.0 in type T
    else
        @assert σ² > zero(T) "σ² must be positive"
        σ²_val = σ²
    end
    
    # Empty latent variables - will be allocated during fitting
    z_matrix = Matrix{T}(undef, 0, k)
    
    # Infer matrix and vector types from the actual objects
    M = typeof(W_matrix)
    V = typeof(μ_vector)
    
    return ProbabilisticPCA{T, M, V}(W_matrix, σ²_val, μ_vector, k, D, z_matrix)
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
function estep(ppca::ProbabilisticPCA, X::Matrix{T}) where {T <: Real}
    # get dims
    N, D = size(X)
    @assert D == ppca.D "Data dimension mismatch: expected $(ppca.D), got $D"
    
    # preallocate E_z and E_zz
    E_z = zeros(T, N, ppca.k)
    E_zz = zeros(T, N, ppca.k, ppca.k)
    
    # calculate M
    M = ppca.W' * ppca.W + (ppca.σ² * I(ppca.k))
    M_chol = cholesky(Symmetric(M))  # More stable than manual inversion
    
    # calculate E_z and E_zz
    for i in 1:N
        # Center the data point
        centered_xi = X[i, :] - ppca.μ
        
        # E[z_i] = M^{-1} W^T (x_i - μ)
        E_z[i, :] = M_chol \ (ppca.W' * centered_xi)
        
        # E[z_i z_i^T] = σ² M^{-1} + E[z_i] E[z_i]^T
        E_zz[i, :, :] = ppca.σ² * inv(M_chol) + E_z[i, :] * E_z[i, :]'
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
function mstep!(
    ppca::ProbabilisticPCA, X::Matrix{T}, E_z::AbstractArray{T}, E_zz::AbstractArray{T}
) where {T <: Real}
    # get dims
    N, D = size(X)
    
    # Calculate the sum of E[z_i z_i^T] across all samples
    sum_E_zz = sum(E_zz, dims=1)[1, :, :]  # Shape: (k, k)
    
    # Calculate the numerator: Σᵢ (xᵢ - μ) E[zᵢ]ᵀ
    numerator = zeros(T, D, ppca.k)
    for i in 1:N
        centered_xi = X[i, :] - ppca.μ
        numerator += centered_xi * E_z[i, :]'
    end
    
    # Update W: W_new = numerator / sum_E_zz
    ppca.W .= Matrix{eltype(ppca.W)}(numerator / sum_E_zz)
    
    # Update σ²
    running_sum_σ² = zero(T)
    WW = ppca.W' * ppca.W
    
    for i in 1:N
        centered_xi = X[i, :] - ppca.μ
        
        running_sum_σ² += 
            sum(abs2, centered_xi) -  # ||x_i - μ||²
            2 * dot(E_z[i, :], ppca.W' * centered_xi) +  # 2 * E[z_i]^T W^T (x_i - μ)
            tr(E_zz[i, :, :] * WW)  # tr(E[z_i z_i^T] W^T W)
    end
    
    # Update parameters
    ppca.z = E_z
    ppca.σ² = running_sum_σ² / (N * D)
    
    return ppca
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
    @assert D == ppca.D "Data dimension mismatch"
    
    # calculate C
    C = ppca.W * ppca.W' + ppca.σ² * I(D)
    
    X_centered = X .- ppca.μ'  # μ' makes it a row vector for broadcasting
    
    # Calculate sample covariance matrix more efficiently
    S = (X_centered' * X_centered) / N
    
    try
        # Use Cholesky for numerical stability
        C_chol = cholesky(Symmetric(C))
        log_det_C = logdet(C_chol)
        trace_term = tr(C_chol \ S)
        
        # calculate log-likelihood
        ll = -(N / 2) * (D * log(2π) + log_det_C + trace_term)
        return ll
    catch e
        @warn "Covariance matrix is not positive definite" e
        return -Inf
    end
end

"""
    fit!(model::ProbabilisticPCA, X::Matrix{<:Real}, max_iter::Int=100, tol::AbstractFloat=1e-6)

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
    ppca::ProbabilisticPCA, X::Matrix{<:Real}, max_iters::Int=100, tol::Float64=1e-6
)
    if all(iszero, ppca.μ)
        ppca.μ .= vec(mean(X; dims=1))
    end

    # initiliaze the log-likelihood
    lls = []
    prev_ll = -Inf
    prog = Progress(max_iters; desc="Fitting Probabilistic PCA...")

    for i in 1:max_iters
        E_z, E_zz = estep(ppca, X)
        mstep!(ppca, X, E_z, E_zz)

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
