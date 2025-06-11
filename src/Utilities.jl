export kmeanspp_initialization,
    kmeans_clustering, fit!, block_tridgm, block_tridiagonal_inverse, block_tridiagonal_inverse_static
export row_matrix, stabilize_covariance_matrix, valid_Σ, make_posdef!, gaussian_entropy
export random_rotation_matrix

# Type checking utilities
"""
    check_same_type(args...)

Utility function to check if n arguments share the same types. 
"""
function check_same_type(args...)
    if length(args) ≤ 1
        return true  # trivial case
    end
    first_type = typeof(args[1])
    all(x -> typeof(x) == first_type, args)
end


# Matrix utilities

"""
    row_matrix(x::AbstractVector)

Convert a vector to a row matrix.
"""
function row_matrix(x::AbstractVector)
    return reshape(x, 1, length(x))
end

"""
    block_tridiagonal_inverse(A, B, C)

Compute the inverse of a block tridiagonal matrix.

# Notes: This implementation is from the paper:
"An Accelerated Lambda Iteration Method for Multilevel Radiative Transfer” Rybicki, G.B., and Hummer, D.G., Astronomy and Astrophysics, 245, 171–181 (1991), Appendix B.
"""
function block_tridiagonal_inverse(A::Vector{AbstractMatrix{T}},
                                   B::Vector{AbstractMatrix{T}},
                                   C::Vector{AbstractMatrix{T}}) where {T<:Real}
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
    for i in 1:n
        M = B[i] - A[i] * D[i]
        lu_D[i] = lu(M)
        D[i + 1] = lu_D[i] \ C[i]
    end

    # Backward sweep for E
    for i in n:-1:1
        M = B[i] - C[i] * E[i + 1]
        lu_E[i] = lu(M)
        E[i] = lu_E[i] \ A[i]
    end

    # Compute λii
    for i in 1:n
        term1 = identity - D[i + 1] * E[i + 1]
        term2 = B[i] - A[i] * D[i]
        S = term2 * term1
        lu_S[i] = lu(S)
        λii[:, :, i] = lu_S[i] \ identity
    end

    # Compute λij
    for i in 2:n
        λij[:, :, i - 1] = E[i] * λii[:, :, i - 1]
    end

    return λii, -λij
end

"""
    block_tridiagonal_inverse_static(A, B, C)

Compute the inverse of a block tridiagonal matrix using static matrices. See `block_tridiagonal_inverse` for details.
"""
function block_tridiagonal_inverse_static(
    A::Vector{AbstractMatrix{T}}, 
    B::Vector{AbstractMatrix{T}},
    C::Vector{AbstractMatrix{T}}
) where {T<:Real}
    n = length(B)
    N = size(B[1], 1)
    N2 = N * N

    # Convert input vectors to static matrices
    A_static = [SMatrix{N,N,T,N2}(A[i]) for i in eachindex(A)]
    B_static = [SMatrix{N,N,T,N2}(B[i]) for i in eachindex(B)]
    C_static = [SMatrix{N,N,T,N2}(C[i]) for i in eachindex(C)]

    # Initialize D and E arrays
    D = Vector{SMatrix{N,N,T,N2}}(undef, n + 1)
    E = Vector{SMatrix{N,N,T,N2}}(undef, n + 1)
    D[1] = @SMatrix zeros(T, N, N)
    E[n + 1] = @SMatrix zeros(T, N, N)

    # Initialize λii and λij arrays
    λii = Array{T}(undef, N, N, n)
    λij = Array{T}(undef, N, N, n - 1)

    # Static identity
    identity_static = SMatrix{N,N,T,N2}(I)

    # Add zero matrices to A and C
    A_extended = vcat([(@SMatrix zeros(T, N, N))], A_static)
    C_extended = vcat(C_static, [(@SMatrix zeros(T, N, N))])

    # Forward sweep for D
    for i in 1:n
        M = B_static[i] - A_extended[i] * D[i]
        lu_M = lu(M)  # LU factorization directly on static matrix
        D[i + 1] = lu_M \ C_extended[i]
    end

    # Backward sweep for E
    for i in n:-1:1
        M = B_static[i] - C_extended[i] * E[i + 1]
        lu_M = lu(M)  # LU factorization directly on static matrix
        E[i] = lu_M \ A_extended[i]
    end

    # Compute λii
    for i in 1:n
        term1 = identity_static - D[i + 1] * E[i + 1]
        term2 = B_static[i] - A_extended[i] * D[i]
        S = term2 * term1
        lu_S = lu(S)  # LU factorization directly on static matrix
        λii[:, :, i] = Matrix(lu_S \ identity_static)  # Convert only final result
    end

    # Compute λij
    for i in 2:n
        λij[:, :, i - 1] = Matrix(E[i] * SMatrix{N,N,T,N2}(λii[:, :, i - 1]))
    end

    return λii, -λij
end

"""
    block_tridgm(main_diag::Vector{Matrix{T}}, upper_diag::Vector{Matrix{T}}, lower_diag::Vector{Matrix{T}}) where {T<:Real}

Construct a block tridiagonal matrix from three vectors of matrices.

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
        throw(
            DimensionMismatch(
                "The length of upper_diag and lower_diag must be one less than the length of main_diag",
            ),
        )
    end

    # Determine dimensions
    m = size(main_diag[1], 1)  # block size
    n = length(main_diag)      # number of blocks
    N = m * n                  # total matrix size

    # Pre-calculate number of non-zero elements for each section
    nnz_main = n * m * m           # main diagonal blocks
    nnz_off = 2 * (n - 1) * m * m   # upper and lower diagonal blocks
    total_nnz = nnz_main + nnz_off

    # Pre-allocate arrays with exact sizes
    I = Vector{Int}(undef, total_nnz)
    J = Vector{Int}(undef, total_nnz)
    V = Vector{T}(undef, total_nnz)

    # Use linear indexing for better performance
    idx = 1

    # Fill main diagonal blocks
    for block_idx in 1:n
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
    for block_idx in 1:(n - 1)
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
"""
function euclidean_distance(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:Real}
    return sqrt(sum((a .- b) .^ 2))
end

"""
    kmeanspp_initialization(data::AbstractMatrix{T}, k_means::Int) where {T<:Real}

Perform K-means++ initialization for cluster centroids.
"""
function kmeanspp_initialization(data::AbstractMatrix{T}, k_means::Int) where {T<:Real}
    N, D = size(data)
    centroids = zeros(D, k_means)
    rand_idx = rand(1:N)
    centroids[:, 1] = data[rand_idx, :]
    for k in 2:k_means
        dists = zeros(N)
        for i in 1:N
            dists[i] = minimum([
                euclidean_distance(@view(data[i, :]), @view(centroids[:, j])) for j in 1:(k - 1)
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
Perform K-means++ initialization for cluster centroids on vector data.
"""
function kmeanspp_initialization(data::AbstractVector{T}, k_means::Int) where {T<:Real}
    # reshape data
    data = reshape(data, length(data), 1)
    return kmeanspp_initialization(data, k_means)
end

"""
    kmeans_clustering(data::Matrix{T}, k_means::Int, max_iters::Int=100, tol::Float64=1e-6) where {T<:Real}

Perform K-means clustering on the input data.

# Arguments
- `data::Matrix{T}`: The input data matrix where each row is a data point.
- `k_means::Int`: The number of clusters.
- `max_iters::Int=100`: Maximum number of iterations.
- `tol::Float64=1e-6`: Convergence tolerance.

# Returns
- A tuple containing the final centroids and cluster labels for each data point.
"""
function kmeans_clustering(
    data::AbstractMatrix{T}, k_means::Int, max_iters::Int=100, tol::Float64=1e-6
) where {T<:Real}
    N, D = size(data)
    centroids = kmeanspp_initialization(data, k_means)
    labels = zeros(Int, N)

    for iter in 1:max_iters
        # Assign each data point to the nearest cluster
        @inbounds for i in 1:N
            x_i = @view data[i, :]
            min_k, min_dist = 1, euclidean_distance(x_i, @view centroids[:, 1])
            for k in 2:k_means
                c_k = @view centroids[:, k]
                dist = euclidean_distance(x_i, c_k)
                if dist < min_dist
                    min_dist = dist
                    min_k = k
                end
            end
            labels[i] = min_k
        end

        # Cache old centroids and initialize new ones
        old_centroids = copy(centroids)
        new_centroids = zeros(Float64, D, k_means)

        # Update each centroid
        for k in 1:k_means
            inds = findall(labels .== k)
            if isempty(inds)
                new_centroids[:, k] .= @view data[rand(1:N), :]
            else
                cluster_data = @view data[inds, :]
                new_centroids[:, k] .= vec(mean(cluster_data; dims=1))
            end
        end

        centroids .= new_centroids

        # Check convergence
        converged = true
        for k in 1:k_means
            if euclidean_distance(@view(centroids[:, k]), @view(old_centroids[:, k])) > tol
                converged = false
                break
            end
        end
        if converged
            break
        end
    end

    return centroids, labels
end

"""
Perform K-means clustering on vector data.
"""
function kmeans_clustering(
    data::AbstractVector{T}, k_means::Int, max_iters::Int=100, tol::Float64=1e-6
) where {T<:Real}
    # reshape data
    data = reshape(data, length(data), 1)
    return kmeans_clustering(data, k_means, max_iters, tol)
end

"""
    logistic(x::Real)

Calculate the logistic function in a numerically stable way.
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
    make_posdef!(A::AbstractMatrix{T}) where {T<:Real}

Ensure that a matrix is positive definite by adjusting its eigenvalues.
"""
function make_posdef!(A::AbstractMatrix{T}; min_eigval::T=convert(T, 1e-6)) where {T<:Real}
    # Work with the symmetric part
    B = Symmetric((A + A') / 2)

    # Get eigendecomposition
    F = eigen(B)

    # Find negative or small eigenvalues
    neg_eigs = F.values .< min_eigval

    # If already positive definite, return early
    if !any(neg_eigs)
        return A
    end

    # Fix negative eigenvalues
    F.values[neg_eigs] .= min_eigval

    # Reconstruct
    A .= F.vectors * Diagonal(F.values) * F.vectors'

    # Ensure symmetry due to numerical errors
    A .= (A + A') / 2

    return A
end

"""
    stabilize_covariance_matrix(Σ::Matrix{<:Real})

Stabilize a covariance matrix by ensuring it is symmetric and positive definite.
"""
function stabilize_covariance_matrix(Σ::AbstractMatrix{T}) where {T<:Real}
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

function valid_Σ(Σ::AbstractMatrix{T}) where {T<:Real}
    return ishermitian(Σ) && isposdef(Σ)
end

# Function for stacking data... in prep for the trialized M_step!()
function stack_tuples(d)
    # Determine the number of tuples and number of elements in each tuple
    num_tuples = length(d)
    num_elements = length(d[1])

    # Initialize an array to store the stacked matrices
    stacked_matrices = Vector{Matrix{Float64}}(undef, num_elements)

    # Stack matrices for each position in the tuple
    for i in 1:num_elements
        # Extract all matrices at the i-th position from each tuple
        matrices_to_stack = [d[j][i] for j in 1:num_tuples]
        # Vertically concatenate the collected matrices
        stacked_matrices[i] = vcat(matrices_to_stack...)
    end

    # Return the stacked matrices as a tuple
    return tuple(stacked_matrices...)
end

"""
    gaussian_entropy(H::Symmetric{T}) where {T<:Real}

Calculate the entropy of a Gaussian distribution with Hessian (i.e. negative precision) matrix `H`.
"""
function gaussian_entropy(H::Symmetric{T}) where {T<:Real}
    n = size(H, 1)
    F = cholesky(-H)
    logdet_H = 2 * sum(log.(diag(F)))
    return 0.5 * (n * log(2π) + logdet_H)
end

"""
    gaussian_entropy(H::Symmetric{BigFloat, <:SparseMatrix})

Specialized method for BigFloat sparse matrices using logdet.
"""
function gaussian_entropy(H::Symmetric{BigFloat, <:AbstractSparseMatrix})
    n = size(H, 1)
    logdet_H = logdet(-H)
    return 0.5 * (n * log(BigFloat(2π)) + logdet_H)
end

"""
    random_rotation_matrix(n)

Generate a random rotation matrix of size `n x n`.
"""
function random_rotation_matrix(n::Int)
    # Generate a random orthogonal matrix using QR decomposition
    Q, _ = qr(randn(n, n))
    return Matrix(Q)
end

"""
    getproperty(model::AutoRegressiveEmission, sym::Symbol)

Get various properties of 'innerGaussianRegression`. 
"""
function Base.getproperty(model::AutoRegressiveEmission, sym::Symbol)
    if sym === :β
        return model.innerGaussianRegression.β
    elseif sym === :Σ
        return model.innerGaussianRegression.Σ
    elseif sym === :include_intercept
        return model.innerGaussianRegression.include_intercept
    elseif sym === :λ
        return model.innerGaussianRegression.λ
    else # fallback to getfield
        return getfield(model, sym)
    end
end

"""
    setproperty!(model::AutoRegressiveEmission, sym::Symbol, value)

Assign to properties of an `AutoRegressiveEmission` by forwarding certain symbols
to its `innerGaussianRegression` field:
"""
# define setters for innerGaussianRegression fields
function Base.setproperty!(model::AutoRegressiveEmission, sym::Symbol, value)
    if sym === :β
        model.innerGaussianRegression.β = value
    elseif sym === :Σ
        model.innerGaussianRegression.Σ = value
    elseif sym === :λ
        model.innerGaussianRegression.λ = value
    else # fallback to setfield!
        setfield!(model, sym, value)
    end
end