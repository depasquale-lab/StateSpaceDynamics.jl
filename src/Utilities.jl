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
function block_tridiagonal_inverse(A, B, C)
    n = length(B)
    block_size = size(B[1], 1)
    # Initialize Di and Ei arrays
    D = Array{AbstractMatrix}(undef, n + 1)
    E = Array{AbstractMatrix}(undef, n + 1)
    λii = Array{Float64}(undef, block_size, block_size, n)
    λij = Array{Float64}(undef, block_size, block_size, n - 1)
    # Add a zero matrix to the subdiagonal and superdiagonal
    pushfirst!(A, zeros(block_size, block_size))
    push!(C, zeros(block_size, block_size))
    # Initial conditions
    D[1] = zeros(block_size, block_size)
    E[n + 1] = zeros(block_size, block_size)
    # Forward sweep for D
    for i in 1:n
        D[i + 1] = (B[i] - A[i] * D[i]) \ C[i]
    end
    # Backward sweep for E
    for i in n:-1:1
        E[i] = (B[i] - C[i] * E[i + 1]) \ A[i]
    end
    # Compute the inverses of the diagonal blocks λii
    for i in 1:n
        term1 = (I - D[i + 1] * E[i + 1])
        term2 = (B[i] - A[i] * D[i])
        λii[:, :, i] = pinv(term1) * pinv(term2)
    end
    # Compute the inverse of the diagonal blocks λij
    for i in 2:n
        λij[:, :, i - 1] = (E[i] * λii[:, :, i - 1])
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
    length(data) == t * d || throw(DimensionMismatch("Length of data ($(length(data))) must equal t * d ($(t * d))"))
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
    # Check that the vectors have the correct lengths
    if length(upper_diag) != length(main_diag) - 1 ||
        length(lower_diag) != length(main_diag) - 1
        error(
            "The length of upper_diag and lower_diag must be one less than the length of main_diag",
        )
    end

    # Determine the size of the blocks and the total matrix size
    m = size(main_diag[1], 1) # Size of each block
    n = length(main_diag) # Number of blocks
    N = m * n # Total size of the matrix

    # Initialize containers for constructing sparse matrix
    row_indices = Int[]
    col_indices = Int[]
    values = T[]

    # Function to add block indices and values to arrays
    function append_block(i_row, i_col, block)
        base_row = (i_row - 1) * m
        base_col = (i_col - 1) * m
        for j in 1:m
            for i in 1:m
                push!(row_indices, base_row + i)
                push!(col_indices, base_col + j)
                push!(values, block[i, j])
            end
        end
    end

    # Fill in the main diagonal blocks
    for i in 1:n
        append_block(i, i, main_diag[i])
    end

    # Fill in the upper diagonal blocks
    for i in 1:(n - 1)
        append_block(i, i + 1, upper_diag[i])
    end

    # Fill in the lower diagonal blocks
    for i in 1:(n - 1)
        append_block(i + 1, i, lower_diag[i])
    end

    # Create sparse matrix from collected indices and values
    A = sparse(row_indices, col_indices, values, N, N)

    return A
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
function ensure_positive_definite(A::Matrix{T}; min_eigenvalue::Real = 1e-6) where {T}
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