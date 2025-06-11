export ProbabilisticPCA, loglikelihood, fit!

"""
    mutable struct ProbabilisticPCA{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}

Probabilistic PCA model from Bishop's Pattern Recognition and Machine Learning.

# Fields:
    W: Weight matrix that maps from latent space to data space.
    σ²: Noise variance
    μ: Mean of the data 
    k: Number of latent dimensions
    D: Dimension of the data 
    z: Latent variables
"""
mutable struct ProbabilisticPCA{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
    W::M 
    σ²::T 
    μ::V
    k::Int
    D::Int 
    z::M 
end

"""
    E_Step(ppca::ProbabilisticPCA, X::Matrix{<:AbstractFloat})

Expectation step of the EM algorithm for PPCA. See Bishop's Pattern Recognition and Machine Learning for more details.
"""
function estep(ppca::ProbabilisticPCA, X::Matrix{T}) where {T<:Real}
    # get dims
    N, D = size(X)
    @assert D == ppca.D "Data dimension mismatch: expected $(ppca.D), got $D"
    
    # preallocate E_z and E_zz
    E_z = zeros(T, N, ppca.k)
    E_zz = zeros(T, N, ppca.k, ppca.k)
    
    # calculate M
    M = ppca.W' * ppca.W + (ppca.σ² * I(ppca.k))
    M_inv = cholesky(M).U \ (cholesky(M).L \ I(ppca.k))
    # calculate E_z and E_zz
    @views for i in 1:N
        E_z[i, :] = M_inv * ppca.W' * (X[i, :] .- ppca.μ)
        E_zz[i, :, :] = (ppca.σ² * M_inv) + (E_z[i, :]  * E_z[i, :]')
    end
    
    return E_z, E_zz
end


"""
    M_Step!(model::ProbabilisticPCA, X::Matrix{<:AbstractFloat}, E_z::Matrix{<:AbstractFloat}, E_zz::Array{<:AbstractFloat, 3}

Maximization step of the EM algorithm for PPCA. See Bishop's Pattern Recognition and Machine Learning for more details.
"""
function mstep!(
    ppca::ProbabilisticPCA, X::Matrix{T}, E_z::AbstractArray{T}, E_zz::AbstractArray{T}
) where {T<:Real}
    # get dims
    N, D = size(X)
    
    # Calculate the sum of E[z_i z_i^T] across all samples
    sum_E_zz = sum(E_zz, dims=1)[1, :, :]  # Shape: (k, k)
    WW = ppca.W' * ppca.W

    numerator = zeros(T, D, ppca.k)
    running_sum_σ² = zero(T)

    # Calculate the numerator: Σᵢ (xᵢ - μ) E[zᵢ]ᵀ
    for i in 1:N
        centered = @view(X[i, :])  .- ppca.μ
        numerator .+= centered * @view(E_z[i, :])'

        running_sum_σ² += 
            sum(centered .^ 2) -
            sum((2.0 .* (@view(E_z[i, :])' * ppca.W')) .* centered') +
            tr(@view(E_zz[i, :, :])  * WW)
    end
    
    # Update W: W_new = numerator / sum_E_zz
    ppca.W .= Matrix{eltype(ppca.W)}(numerator / sum_E_zz)
    
    # Update σ²
    @views for i in 1:N
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
    loglikelihood(model::ProbabilisticPCA, X::AbstractMatrix{T}) where {T<:Real}
    
Calculate the log-likelihood of the data given the PPCA model.

# Args:
- `model::ProbabilisticPCA`: PPCA model
- `X::AbstractMatrix{T}`: Data matrix

# Returns 
- `ll`: Complete data log-likelihood 
"""
function loglikelihood(ppca::ProbabilisticPCA, X::AbstractMatrix{T}) where {T<:Real}
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

function Random.rand(rng::AbstractRNG, ppca::ProbabilisticPCA, n::Int)
    # z ~ N(0, I) in latent space
    z = rand(rng, MvNormal(zeros(ppca.k), I), n)  # (k, n)

    # noise ε ~ N(0, σ² I) in data space
    ε = rand(rng, MvNormal(zeros(ppca.D), ppca.σ² * I), n)  # (D, n)

    # x = W z + μ + ε
    # Convert μ to (D, 1) to broadcast correctly
    μ = ppca.μ
    μ = size(μ, 2) == 1 ? μ : reshape(μ, ppca.D, 1)

    X = ppca.W * z .+ μ + ε
    return X, z
end

function Random.rand(ppca::ProbabilisticPCA, n::Int)
    return rand(Random.default_rng(), ppca, n)
end


"""
    fit!(ppca::ProbabilisticPCA, X::AbstractMatrix{T}, max_iter::Int=100, tol::AbstractFloat=1e-6)

Fit the PPCA model to the data using the EM algorithm.

# Args:
- `model::ProbabilisticPCA`: PPCA model
- `X::AbstractMatrix{T}`: Data matrix
- `max_iter::Int`: Maximum number of iterations
- `tol::AbstractFloat`: Tolerance for convergence

# Returns
- `lls::Vector{T}`: Vector of log-likelihood values for each iteration.
"""
function fit!(
    ppca::ProbabilisticPCA, X::AbstractMatrix{T}, max_iters::Int=100, tol::Float64=1e-6
) where {T<:Real}
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
