export kmeanspp_initialization, kmeans_clustering, PPCA, fit!, block_tridgm, interleave_reshape, block_tridiagonal_inverse

# Matrix utilities

"""Block Tridiagonal Inverse"""
function block_tridiagonal_inverse(A, B, C)
    n = length(B)
    block_size = size(B[1], 1)
    # Initialize Di and Ei arrays
    D = Array{AbstractMatrix}(undef, n+1)
    E = Array{AbstractMatrix}(undef, n+1)
    λii = Array{Float64}(undef, n, block_size, block_size)

    # Add a zero matrix to the subdiagonal and superdiagonal
    pushfirst!(A, zeros(block_size, block_size))
    push!(C, zeros(block_size, block_size))
    
    # Initial conditions
    D[1] = zeros(block_size, block_size)
    E[n+1] = zeros(block_size, block_size)

 
    # Forward sweep for D
    for i in 1:n
        D[i+1] = (B[i] - A[i] * D[i]) \ C[i]
    end
  
    # Backward sweep for E
    for i in n:-1:1
        E[i] = (B[i] - C[i]*E[i+1]) \ A[i]
    end

    # Compute the inverses of the diagonal blocks λii
    for i in 1:n
        term1 = (I - D[i+1]*E[i+1])
        term2 = (B[i] - A[i]*D[i])
        λii[i, :, :] = pinv(term1) * pinv(term2)
    end

    return λii
end

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

"""Convenience function to construct a block tridiagonal matrix from three vectors of matrices"""
function block_tridgm(main_diag::Vector{Matrix{T}}, upper_diag::Vector{Matrix{T}}, lower_diag::Vector{Matrix{T}}) where T<:Real
    # Check that the vectors have the correct lengths
    if length(upper_diag) != length(main_diag) - 1 || length(lower_diag) != length(main_diag) - 1
        error("The length of upper_diag and lower_diag must be one less than the length of main_diag")
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
    for i in 1:n-1
        append_block(i, i + 1, upper_diag[i])
    end

    # Fill in the lower diagonal blocks
    for i in 1:n-1
        append_block(i + 1, i, lower_diag[i])
    end

    # Create sparse matrix from collected indices and values
    A = sparse(row_indices, col_indices, values, N, N)

    return A
end

# Initialization utilities
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

# Miscellaneous utilities
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