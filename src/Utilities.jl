export kmeanspp_initialization,
    kmeans_clustering, fit!, block_tridgm, interleave_reshape, block_tridiagonal_inverse
export row_matrix, stabilize_covariance_matrix

# Matrix utilities

"""
    row_matrix(x::AbstractVector)

Convert a vector to a row matrix.

# Arguments
- `x::AbstractVector`: The input vector.

# Returns
- A row matrix (1 × n) containing the elements of `x`.
"""
function row_matrix(x::AbstractVector)
    return reshape(x, 1, length(x))
end

"""
    block_tridiagonal_inverse(A, B, C)

Compute the inverse of a block tridiagonal matrix.

# Arguments
- `A`: Lower diagonal blocks.
- `B`: Main diagonal blocks.
- `C`: Upper diagonal blocks.

# Returns
- `λii`: Diagonal blocks of the inverse.
- `λij`: Off-diagonal blocks of the inverse.

# Notes: This implementation is from the paper:

An Accelerated Lambda Iteration Method for Multilevel Radiative Transfer” Rybicki, G.B., and Hummer, D.G., Astronomy and Astrophysics, 245, 171–181 (1991), Appendix B.
"""
function block_tridiagonal_inverse(A::Vector{Matrix{T}},
                                   B::Vector{Matrix{T}},
                                   C::Vector{Matrix{T}}) where {T<:Real}
    n = length(B)
    block_size = size(B[1], 1)

    # Initialize D and E arrays
    D = Vector{Matrix{T}}(undef, n + 1)
    E = Vector{Matrix{T}}(undef, n + 1)
    D[1] = zeros(T, block_size, block_size)
    E[n + 1] = zeros(T, block_size, block_size)

    # Initialize λii and λij arrays
    λii = Array{T}(undef, block_size, block_size, n)
    λij = Array{T}(undef, block_size, block_size, n - 1)
    identity = Matrix{T}(I, block_size, block_size)

    # Add zero matrices to A and C
    pushfirst!(A, zeros(T, block_size, block_size))
    push!(C, zeros(T, block_size, block_size))

    # Preallocate LU factorization arrays
    lu_D = Vector{LU{T, Matrix{T}}}(undef, n)
    lu_E = Vector{LU{T, Matrix{T}}}(undef, n)
    lu_S = Vector{LU{T, Matrix{T}}}(undef, n)

    # Forward sweep for D
    @inbounds for i in 1:n
        M = B[i] - A[i] * D[i]
        lu_D[i] = lu(M)
        D[i + 1] = lu_D[i] \ C[i]
    end

    # Backward sweep for E
    @inbounds for i in n:-1:1
        M = B[i] - C[i] * E[i + 1]
        lu_E[i] = lu(M)
        E[i] = lu_E[i] \ A[i]
    end

    # Compute λii
    @inbounds for i in 1:n
        term1 = identity - D[i + 1] * E[i + 1]
        term2 = B[i] - A[i] * D[i]
        S = term2 * term1
        lu_S[i] = lu(S)
        λii[:, :, i] = lu_S[i] \ identity
    end

    # Compute λij
    @inbounds for i in 2:n
        λij[:, :, i - 1] = E[i] * λii[:, :, i - 1]
    end

    return λii, -λij
end

"""
    interleave_reshape(data::AbstractArray, t::Int, d::Int)

Reshape a vector into a matrix by interleaving its elements.

# Arguments
- `data::AbstractArray`: The input vector to be reshaped.
- `t::Int`: The number of rows in the output matrix.
- `d::Int`: The number of columns in the output matrix.

# Returns
- A reshaped matrix of size `t × d`.

# Throws
- `ErrorException` if the length of `data` is not equal to `t * d`.
"""
function interleave_reshape(data::AbstractVector, t::Int, d::Int)
    length(data) == t * d || throw(
        DimensionMismatch("Length of data ($(length(data))) must equal t * d ($(t * d))"),
    )
    return permutedims(reshape(data, d, t))
end

"""
    block_tridgm(main_diag::Vector{Matrix{T}}, upper_diag::Vector{Matrix{T}}, lower_diag::Vector{Matrix{T}}) where {T<:Real}

Construct a block tridiagonal matrix from three vectors of matrices.

# Arguments
- `main_diag::Vector{Matrix{T}}`: Vector of matrices for the main diagonal.
- `upper_diag::Vector{Matrix{T}}`: Vector of matrices for the upper diagonal.
- `lower_diag::Vector{Matrix{T}}`: Vector of matrices for the lower diagonal.

# Returns
- A sparse matrix representing the block tridiagonal matrix.

# Throws
- `ErrorException` if the lengths of `upper_diag` and `lower_diag` are not one less than the length of `main_diag`.
"""
function block_tridgm(
    main_diag::Vector{Matrix{T}},
    upper_diag::Vector{Matrix{T}},
    lower_diag::Vector{Matrix{T}},
) where {T<:Real}
    # Input validation
    if length(upper_diag) != length(main_diag) - 1 ||
        length(lower_diag) != length(main_diag) - 1
        throw(DimensionMismatch(
            "The length of upper_diag and lower_diag must be one less than the length of main_diag"
        ))
    end

    # Determine dimensions
    m = size(main_diag[1], 1)  # block size
    n = length(main_diag)      # number of blocks
    N = m * n                  # total matrix size
    
    # Pre-calculate number of non-zero elements for each section
    nnz_main = n * m * m           # main diagonal blocks
    nnz_off = 2 * (n-1) * m * m   # upper and lower diagonal blocks
    total_nnz = nnz_main + nnz_off
    
    # Pre-allocate arrays with exact sizes
    I = Vector{Int}(undef, total_nnz)
    J = Vector{Int}(undef, total_nnz)
    V = Vector{T}(undef, total_nnz)
    
    # Use linear indexing for better performance
    idx = 1
    
    # Fill main diagonal blocks
    @inbounds for block_idx in 1:n
        block = main_diag[block_idx]
        base = (block_idx - 1) * m
        
        # Use linear indexing for the block
        for j in 1:m, i in 1:m
            I[idx] = base + i
            J[idx] = base + j
            V[idx] = block[i, j]
            idx += 1
        end
    end
    
    # Fill upper and lower diagonal blocks simultaneously
    @inbounds for block_idx in 1:(n-1)
        upper_block = upper_diag[block_idx]
        lower_block = lower_diag[block_idx]
        
        base_current = (block_idx - 1) * m
        base_next = block_idx * m
        
        # Upper diagonal block
        for j in 1:m, i in 1:m
            I[idx] = base_current + i
            J[idx] = base_next + j
            V[idx] = upper_block[i, j]
            idx += 1
        end
        
        # Lower diagonal block
        for j in 1:m, i in 1:m
            I[idx] = base_next + i
            J[idx] = base_current + j
            V[idx] = lower_block[i, j]
            idx += 1
        end
    end
    
    # Create sparse matrix optimized for subsequent operations
    return sparse(I, J, V, N, N, +)
end

# Initialization utilities
"""
    euclidean_distance(a::AbstractVector{Float64}, b::AbstractVector{Float64})

Calculate the Euclidean distance between two points.

# Arguments
- `a::AbstractVector{Float64}`: The first point.
- `b::AbstractVector{Float64}`: The second point.

# Returns
- The Euclidean distance between `a` and `b`.
"""
function euclidean_distance(a::AbstractVector{Float64}, b::AbstractVector{Float64})
    return sqrt(sum((a .- b) .^ 2))
end

"""
    kmeanspp_initialization(data::Matrix{<:Real}, k_means::Int)

Perform K-means++ initialization for cluster centroids.

# Arguments
- `data::Matrix{<:Real}`: The input data matrix where each row is a data point.
- `k_means::Int`: The number of clusters.

# Returns
- A matrix of initial centroids for K-means clustering.
"""
function kmeanspp_initialization(data::Matrix{<:Real}, k_means::Int)
    N, D = size(data)
    centroids = zeros(D, k_means)
    rand_idx = rand(1:N)
    centroids[:, 1] = data[rand_idx, :]
    for k in 2:k_means
        dists = zeros(N)
        for i in 1:N
            dists[i] = minimum([
                euclidean_distance(data[i, :], centroids[:, j]) for j in 1:(k - 1)
            ])
        end
        probs = dists .^ 2
        probs ./= sum(probs)
        next_idx = StatsBase.sample(1:N, Weights(probs))
        centroids[:, k] = data[next_idx, :]
    end
    return centroids
end

"""
    kmeanspp_initialization(data::Vector{Float64}, k_means::Int)

Perform K-means++ initialization for cluster centroids on vector data.

# Arguments
- `data::Vector{Float64}`: The input data vector.
- `k_means::Int`: The number of clusters.

# Returns
- A matrix of initial centroids for K-means clustering.
"""
function kmeanspp_initialization(data::Vector{Float64}, k_means::Int)
    # reshape data
    data = reshape(data, length(data), 1)
    return kmeanspp_initialization(data, k_means)
end

"""
    kmeans_clustering(data::Matrix{<:Real}, k_means::Int, max_iters::Int=100, tol::Float64=1e-6)

Perform K-means clustering on the input data.

# Arguments
- `data::Matrix{<:Real}`: The input data matrix where each row is a data point.
- `k_means::Int`: The number of clusters.
- `max_iters::Int=100`: Maximum number of iterations.
- `tol::Float64=1e-6`: Convergence tolerance.

# Returns
- A tuple containing the final centroids and cluster labels for each data point.
"""
function kmeans_clustering(
    data::Matrix{<:Real}, k_means::Int, max_iters::Int=100, tol::Float64=1e-6
)
    N, _ = size(data)
    centroids = kmeanspp_initialization(data, k_means)  # Assuming you have this function defined
    labels = zeros(Int, N)
    for iter in 1:max_iters
        # Assign each data point to the nearest cluster
        labels .= [
            argmin([euclidean_distance(x, c) for c in eachcol(centroids)]) for
            x in eachrow(data)
        ]
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
                new_centroids[:, k] = mean(points_in_cluster; dims=1)
            end
        end
        centroids .= new_centroids
        # Check for convergence
        if maximum([
            euclidean_distance(centroids[:, i], old_centroids[:, i]) for i in 1:k_means
        ]) < tol
            # println("Converged after $iter iterations")
            break
        end
    end
    return centroids, labels
end

"""
    kmeans_clustering(data::Vector{Float64}, k_means::Int, max_iters::Int=100, tol::Float64=1e-6)

Perform K-means clustering on vector data.

# Arguments
- `data::Vector{Float64}`: The input data vector.
- `k_means::Int`: The number of clusters.
- `max_iters::Int=100`: Maximum number of iterations.
- `tol::Float64=1e-6`: Convergence tolerance.

# Returns
- A tuple containing the final centroids and cluster labels for each data point.
"""
function kmeans_clustering(
    data::Vector{Float64}, k_means::Int, max_iters::Int=100, tol::Float64=1e-6
)
    # reshape data
    data = reshape(data, length(data), 1)
    return kmeans_clustering(data, k_means, max_iters, tol)
end

"""
    logistic(x::Real)

Calculate the logistic function in a numerically stable way.

# Arguments
- `x::Real`: The input value.

# Returns
- The result of the logistic function applied to `x`.
"""
function logistic(x::Real)
    if x > 0
        return 1 / (1 + exp(-x))
    else
        exp_x = exp(x)
        return exp_x / (1 + exp_x)
    end
end

"""
    ensure_positive_definite(A::Matrix{T}) where {T}

Ensure that a matrix is positive definite by adjusting its eigenvalues.

# Arguments
- `A::Matrix{T}`: The input matrix.

# Returns
- A positive definite matrix derived from `A`.
"""
function ensure_positive_definite(A::Matrix{T}; min_eigenvalue::Real=1e-6) where {T}
    # Perform eigenvalue decomposition
    eigen_decomp = eigen(Symmetric(A))
    λ, V = eigen_decomp.values, eigen_decomp.vectors

    # Compute the maximum absolute eigenvalue
    λ_max = maximum(abs.(λ))

    # Set a threshold relative to the maximum eigenvalue
    ε = max(min_eigenvalue, eps(T) * λ_max * length(λ))

    # Replace any eigenvalues smaller than ε with ε
    λ_clipped = [max(λi, ε) for λi in λ]

    # Reconstruct the matrix with the clipped eigenvalues
    A_posdef = V * Diagonal(λ_clipped) * V'

    # Ensure perfect symmetry
    A_posdef = (A_posdef + A_posdef') / 2

    return A_posdef
end

"""
    stabilize_covariance_matrix(Σ::Matrix{<:Real})

Stabilize a covariance matrix by ensuring it is symmetric and positive definite.

# Arguments
- `Σ::Matrix{<:Real}`: The input covariance matrix.

# Returns
- A stabilized version of the input covariance matrix.
"""
function stabilize_covariance_matrix(Σ::Matrix{<:Real})
    # check if the covariance is symmetric. If not, make it symmetric
    if !ishermitian(Σ)
        Σ = (Σ + Σ') * 0.5
    end
    # check if matrix is posdef. If not, add a small value to the diagonal (sometimes an emission only models one observation and the covariance matrix is singular)
    if !isposdef(Σ)
        Σ = Σ + 1e-12 * I
    end
    return Σ
end
