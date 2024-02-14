export kmeanspp_initialization, kmeans_clustering, Autoregression, fit!, loglikelihood, ensure_positive_definite, PPCA, fit!, block_tridgm, interleave_reshape

"""Interleave Reshape"""
function interleave_reshape(data::AbstractArray, t::Int, d::Int)
    # get length of data 
    l = length(data)
    # check if the length of data equal to t * d
    if l != (t * d)
        error("The length of data must be equivalent to  t * d")
    end
    # create a matrix of zeros
    X = zeros(t, d)
    # loop through the data and reshape
    for i in 1:d
        X[:, i] = data[i:d:l]
    end
    # return the reshaped matrix
    return X
end

"""Euclidean Distance for two points"""
function euclidean_distance(a::AbstractVector{Float64}, b::AbstractVector{Float64})
    return sqrt(sum((a .- b).^2))
end

"""KMeans++ Initialization"""
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
        next_idx = StatsBase.sample(1:N, Weights(probs))
        centroids[:, k] = data[next_idx, :]
    end
    return centroids
end

function ensure_positive_definite(A::Matrix{T}) where T
    # Perform eigenvalue decomposition
    eigen_decomp = eigen(Symmetric(A))  # Ensure A is treated as symmetric
    λ, V = eigen_decomp.values, eigen_decomp.vectors
    # Set a threshold for eigenvalues (e.g., a small positive number)
    ε = max(eps(T), 1e-10)
    # Replace any non-positive eigenvalues with ε
    λ_clipped = [max(λi, ε) for λi in λ]
    # Reconstruct the matrix with the clipped eigenvalues
    A_posdef = V * Diagonal(λ_clipped) * V'
    return A_posdef
end

"""K++ Initialization for Vector input"""
function kmeanspp_initialization(data::Vector{Float64}, k_means::Int)
    # reshape data
    data = reshape(data, length(data), 1)
    return kmeanspp_initialization(data, k_means)
end

"""KMeans Clustering Initialization"""
function kmeans_clustering(data::Matrix{Float64}, k_means::Int, max_iters::Int=100, tol::Float64=1e-6)
    N, _ = size(data)
    centroids = kmeanspp_initialization(data, k_means)  # Assuming you have this function defined
    labels = zeros(Int, N)
    for iter in 1:max_iters
        # Assign each data point to the nearest cluster
        labels .= [argmin([euclidean_distance(x, c) for c in eachcol(centroids)]) for x in eachrow(data)]
        # Cache old centroids for convergence check
        old_centroids = centroids
        # Update the centroids
        new_centroids = zeros(size(centroids))
        for k in 1:k_means
            points_in_cluster = data[labels .== k, :]
            if isempty(points_in_cluster)
                # If a cluster has no points, reinitialize its centroid
                new_centroids[:, k] = data[rand(1:N), :]
            else
                new_centroids[:, k] = mean(points_in_cluster, dims=1)
            end
        end
        centroids .= new_centroids
        # Check for convergence
        if maximum([euclidean_distance(centroids[:, i], old_centroids[:, i]) for i in 1:k_means]) < tol
            # println("Converged after $iter iterations")
            break
        end
    end
    return centroids, labels
end

"""KMeans Clustering Initialization for Vector input"""
function kmeans_clustering(data::Vector{Float64}, k_means::Int, max_iters::Int=100, tol::Float64=1e-6)
    # reshape data
    data = reshape(data, length(data), 1)
    return kmeans_clustering(data, k_means, max_iters, tol)
end

"""Convenience fucntion to construct a block tridiagonal amtrix from three vectors of matrices"""
function block_tridgm(main_diag::Vector{Matrix{T}}, upper_diag::Vector{Matrix{T}}, lower_diag::Vector{Matrix{T}}) where T<:Real
   # Check that the vectors have the correct lengths
   if length(upper_diag) != length(main_diag) - 1 || length(lower_diag) != length(main_diag) - 1
    error("The length of upper_diag and lower_diag must be one less than the length of main_diag")
    end

    # Determine the size of the blocks and the total matrix size
    m = size(main_diag[1], 1) # Size of each block
    n = length(main_diag) # Number of blocks
    N = m * n # Total size of the matrix

    # Initialize a sparse matrix
    A = spzeros(N, N)

    # Fill in the main diagonal blocks
    for i in 1:n
        row = (i - 1) * m + 1
        col = row
        A[row:row+m-1, col:col+m-1] = main_diag[i]
    end

    # Fill in the upper diagonal blocks
    for i in 1:n-1
        row = (i - 1) * m + 1
        col = row + m
        A[row:row+m-1, col:col+m-1] = upper_diag[i]
    end

    # Fill in the lower diagonal blocks
    for i in 1:n-1
        row = i * m + 1
        col = (i - 1) * m + 1
        A[row:row+m-1, col:col+m-1] = lower_diag[i]
    end

    return A
end 


"""Probabilistic Principal Component Analysis"""
mutable struct PPCA
    W::Matrix{Float64}
    σ²::Float64
    μ::Vector{Float64}
    k::Int
    D::Int
    z::Matrix{Float64}
end

"""PPCA Constructor"""
function PPCA(X::Matrix{Float64}, k::Int)
    _, D = size(X)
    # Initialize parameters
    W = randn(D, k) / sqrt(D)
    σ² = 1.0
    μ = vec(mean(X, dims=1))
    z = zeros(size(X, 1), k)
    return PPCA(W, σ², μ, k, D, z)
end

"""
E-Step for Probabilistic PCA

Args:
    ppca: PPCA object
    X: Data matrix
"""
function EStep(ppca::PPCA, X::Matrix{Float64})
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
    ppca: PPCA object
    X: Data matrix
    E_z: E_z matrix from E-Step
    E_zz: E_zz matrix from E-Step
"""
function MStep!(ppca::PPCA, X::Matrix{Float64}, E_z::AbstractArray, E_zz::AbstractArray)
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

function loglikelihood(ppca::PPCA, X::Matrix{Float64})
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
Fits the PPCA model to the data using the EM algorithm for PPCA as discussed in Bishop's Pattern Recognition and Machine Learning.
See Chapter 12.2.2 for more details.

Args:
    ppca: PPCA object
    X: Data matrix
    max_iters: Maximum number of iterations
    tol: Tolerance for convergence of the reconstruction error
"""
function fit!(ppca::PPCA, X::Matrix{Float64}, max_iters::Int=100, tol::Float64=1e-6)
    prev_ll = -Inf  # Initialize to negative infinity
    for i in 1:max_iters
        # E-Step
        E_z, E_zz = EStep(ppca, X)
        # M-Step
        MStep!(ppca, X, E_z, E_zz)
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

"""
Autoregressive Model
"""
mutable struct Autoregression{T<:Real}
    X::AbstractArray # Data
    p::Int # order of the autoregressive model
    β::Vector{T} # Coefficients for the autoregressive model
    σ²::T # Variance of the noise
end

"""
Constructor for the autoregressive model.
"""
function Autoregression(X::Vector{T}, p::Int) where T<:Real
    # Initialize parameters
    β = zeros(p + 1)
    σ² = 1.0
    return Autoregression(X, p, β, σ²)
end

"""
Create lagged matrix for autoregressive model.
"""
function create_lagged_matrix(data::Vector{Float64}, p::Int)
    T = length(data)
    backward_lagged = zeros(Float64, T - p, p + 1)
    # create the backward lagged matrix
    for i in 1:p
        backward_lagged[:, i] = data[i:T-p+i-1] # i think this is right
    end
    # add a constant
    backward_lagged[:, end] = ones(T - p)
    # create the forward lagged matrix
    forward_lagged = data[p+1:end, :]
    return forward_lagged, backward_lagged
end

"""
Log-Likelihood of the autoregressive model.
"""
function loglikelihood(model::Autoregression)
    forward_lagged, backward_lagged = create_lagged_matrix(model.X, model.p)
    residuals = Vector(forward_lagged[:, 1]) - (backward_lagged * β)
    ll = logpdf(Normal(0, sqrt(model.σ²)), residuals)
    return ll
end

"""
Update σ² for the autoregressive model.
"""
function update_σ²!(model::Autoregression, residuals::Vector{Float64})
    n = length(residuals)
    model.σ² = sum(residuals.^2) / n
end

"""
Fit the model using MLE and Autodifferentiation.
"""
function fit!(model::Autoregression)
    # create lagged matrix
    forward_lagged, backward_lagged = create_lagged_matrix(model.X, model.p)
    # define loss function i.e. log-likelihood
    function loss(β)
        predictions = backward_lagged * β
        sse = sum((forward_lagged[:, 1] - predictions).^2)
        return sse
    end
    # optimize using BFGS
    result = Optim.optimize(β -> loss(β), model.β, BFGS(), autodiff=:forward)
    # update parameters
    model.β = result.minimizer
    residuals = forward_lagged[:, 1] - backward_lagged * model.β
    update_σ²!(model, residuals)
end

"""
Vectorautoregression Model
"""
mutable struct VectorAutoregression{T<:Real}
    X::AbstractArray # Data
    p::Int # order of the autoregressive model
    β::Matrix{T} # Coefficients for the autoregressive model
    σ²::Matrix{T} # Variance of the noise
end

"""
Constructor for the VAR model.
"""
function VectorAutoregression(X::Matrix{T}, p::Int) where T<:Real
    # get dim of data
    K, _ = size(X)
    # Initialize parameters
    β = zeros(T, K, K * p + 1)
    σ² = zeros(T, K, K)
    return VectorAutoregression(X, p, β, σ²)
end

"""
Helper function to create lagged matrix for VAR model.
"""
function create_lagged_matrix(data::Matrix{Float64}, p::Int)
end

