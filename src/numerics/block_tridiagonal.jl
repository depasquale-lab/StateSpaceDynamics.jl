# Block-tridiagonal linear algebra kernel.
#
# Self-contained numerical routines for block-tridiagonal systems used by the
# BTD smoother path: dense/sparse assembly (`block_tridgm`, `block_tridgm!`),
# the Thomas-style inverse and log-det (`block_tridiagonal_inverse!`,
# `block_tridiagonal_inverse_logdet!`), banded SPD solves, and back-substitution.
# Extracted from Utilities.jl, which retains only general-purpose helpers.

# Block Tridiagonal Workspace
"""
    BlockTridiagonalWorkspace{T<:Real}

Pre-allocated workspace for block tridiagonal operations in LDS smoothing.
Holds all temporary buffers needed by `hessian!`, `block_tridgm!`, and
`block_tridiagonal_inverse!` to avoid repeated allocations during EM iterations.
"""
struct BlockTridiagonalWorkspace{T<:Real}
    block_size::Int
    n_blocks::Int

    # Hessian block storage
    H_diag::Vector{Matrix{T}}
    H_sub::Vector{Matrix{T}}
    H_super::Vector{Matrix{T}}

    # Sparse matrix with fixed sparsity pattern
    H_sparse::SparseMatrixCSC{T,Int}
    # Precomputed map from block (k, i, j) to nzval index for zero-allocation writes
    nzval_map::Vector{Int}

    # block_tridiagonal_inverse buffers
    D::Vector{Matrix{T}}
    E::Vector{Matrix{T}}
    M::Matrix{T}
    term1::Matrix{T}
    term2::Matrix{T}
    S::Matrix{T}
    Ibs::Matrix{T}
    Z::Matrix{T}

    # Negated blocks
    neg_diag::Vector{Matrix{T}}
    neg_sub::Vector{Matrix{T}}
    neg_super::Vector{Matrix{T}}
    chol_factors::Vector{Matrix{T}}

    #=
    Preallocated ipiv for the per-block LU factorisations inside
    `block_tridiagonal_solve!`. `lu!(M)` would otherwise allocate a
    fresh `Vector{BlasInt}` of length `block_size` on every call, and
    the BT solve runs `n_blocks` LUs per Newton evaluation.
    =#
    lu_ipiv::Vector{LinearAlgebra.BlasInt}

    #=
    Banded-format scratch for the SPD `pbsv`-based fast path used when
    `block_size ≤ 8`. Layout: `(2*block_size, block_size * n_blocks)`
    — `ldab = 2D` (one row past `kd+1 = 2D-1+1 = 2D`), one column per
    global matrix column. `pbsv` overwrites this with the Cholesky
    factor on each call, so it gets refilled from the block storage
    every BT solve.
    =#
    Hb::Matrix{T}
end

"""
    BlockTridiagonalWorkspace(::Type{T}, block_size::Int, n_blocks::Int)

Construct a preallocated workspace for block tridiagonal operations with the
given block size (latent dimension) and number of blocks (timesteps).
"""
function BlockTridiagonalWorkspace(
    ::Type{T}, block_size::Int, n_blocks::Int
) where {T<:Real}
    H_diag = [zeros(T, block_size, block_size) for _ in 1:n_blocks]
    H_sub = [zeros(T, block_size, block_size) for _ in 1:(n_blocks - 1)]
    H_super = [zeros(T, block_size, block_size) for _ in 1:(n_blocks - 1)]

    H_sparse = _build_block_tridiag_pattern(T, block_size, n_blocks)
    nzval_map = _build_nzval_map(H_sparse, block_size, n_blocks)

    D = [zeros(T, block_size, block_size) for _ in 1:(n_blocks + 1)]
    E = [zeros(T, block_size, block_size) for _ in 1:(n_blocks + 1)]
    M = zeros(T, block_size, block_size)
    term1 = zeros(T, block_size, block_size)
    term2 = zeros(T, block_size, block_size)
    S = zeros(T, block_size, block_size)
    Ibs = Matrix{T}(I, block_size, block_size)
    Z = zeros(T, block_size, block_size)

    neg_diag = [zeros(T, block_size, block_size) for _ in 1:n_blocks]
    neg_sub = [zeros(T, block_size, block_size) for _ in 1:(n_blocks - 1)]
    neg_super = [zeros(T, block_size, block_size) for _ in 1:(n_blocks - 1)]

    chol_factors = [zeros(T, block_size, block_size) for _ in 1:n_blocks]
    lu_ipiv = Vector{LinearAlgebra.BlasInt}(undef, block_size)
    Hb = zeros(T, 2 * block_size, block_size * n_blocks)

    return BlockTridiagonalWorkspace{T}(
        block_size,
        n_blocks,
        H_diag,
        H_sub,
        H_super,
        H_sparse,
        nzval_map,
        D,
        E,
        M,
        term1,
        term2,
        S,
        Ibs,
        Z,
        neg_diag,
        neg_sub,
        neg_super,
        chol_factors,
        lu_ipiv,
        Hb,
    )
end

"""
    block_tridgm(
        main_diag::Vector{Matrix{T}},
        upper_diag::Vector{Matrix{T}},
        lower_diag::Vector{Matrix{T}}
    ) where {T<:Real}

Construct a block tridiagonal matrix from three vectors of matrices.

# Throws
- `ErrorException` if the lengths of `upper_diag` and `lower_diag` are not one less than the
    length of `main_diag`.
"""
function block_tridgm(
    main_diag::AbstractVector{<:AbstractMatrix{T}},
    upper_diag::AbstractVector{<:AbstractMatrix{T}},
    lower_diag::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real}
    n = length(main_diag)
    m = size(main_diag[1], 1)
    N = n * m
    total_nnz = n * m * m + 2 * (n - 1) * m * m

    I = Vector{Int}(undef, total_nnz)
    J = Vector{Int}(undef, total_nnz)
    V = Vector{T}(undef, total_nnz)

    idx = 1

    for k in 1:n
        base = (k - 1) * m
        block = main_diag[k]
        for i in 1:m, j in 1:m
            I[idx] = base + i
            J[idx] = base + j
            V[idx] = block[i, j]
            idx += 1
        end
    end

    for k in 1:(n - 1)
        base_k = (k - 1) * m
        base_kp1 = k * m
        block_up = upper_diag[k]
        block_low = lower_diag[k]
        for i in 1:m, j in 1:m
            I[idx] = base_k + i
            J[idx] = base_kp1 + j
            V[idx] = block_up[i, j]
            idx += 1
        end
        for i in 1:m, j in 1:m
            I[idx] = base_kp1 + i
            J[idx] = base_k + j
            V[idx] = block_low[i, j]
            idx += 1
        end
    end

    return sparse(I, J, V, N, N)
end

"""
    _build_nzval_map(H_sparse, bs, n)

Build a mapping vector that encodes, for each logical block entry in the order
that `block_tridgm!` iterates, the corresponding index into `H_sparse.nzval`.
This allows `block_tridgm!` to write directly to nzval without sparse lookups.
"""
function _build_nzval_map(H_sparse::SparseMatrixCSC, bs::Int, n::Int)
    total_entries = n * bs * bs + 2 * (n - 1) * bs * bs
    nzval_map = Vector{Int}(undef, total_entries)

    colptr = H_sparse.colptr
    rowval = H_sparse.rowval

    idx = 1
    # Diagonal blocks: for k=1:n, j=1:bs, i=1:bs
    for k in 1:n
        base = (k - 1) * bs
        for j in 1:bs
            col = base + j
            # Find row (base + i) in this column's rowval range
            for i in 1:bs
                row = base + i
                # Binary search in rowval[colptr[col]:colptr[col+1]-1]
                lo, hi = colptr[col], colptr[col + 1] - 1
                while lo <= hi
                    mid = (lo + hi) >> 1
                    if rowval[mid] == row
                        nzval_map[idx] = mid
                        break
                    elseif rowval[mid] < row
                        lo = mid + 1
                    else
                        hi = mid - 1
                    end
                end
                idx += 1
            end
        end
    end

    # Off-diagonal blocks: for k=1:n-1, j=1:bs, i=1:bs (upper then lower)
    for k in 1:(n - 1)
        base_k = (k - 1) * bs
        base_kp1 = k * bs
        # Upper block: row in base_k+1:base_k+bs, col in base_kp1+1:base_kp1+bs
        for j in 1:bs
            col = base_kp1 + j
            for i in 1:bs
                row = base_k + i
                lo, hi = colptr[col], colptr[col + 1] - 1
                while lo <= hi
                    mid = (lo + hi) >> 1
                    if rowval[mid] == row
                        nzval_map[idx] = mid
                        break
                    elseif rowval[mid] < row
                        lo = mid + 1
                    else
                        hi = mid - 1
                    end
                end
                idx += 1
            end
        end
        # Lower block: row in base_kp1+1:base_kp1+bs, col in base_k+1:base_k+bs
        for j in 1:bs
            col = base_k + j
            for i in 1:bs
                row = base_kp1 + i
                lo, hi = colptr[col], colptr[col + 1] - 1
                while lo <= hi
                    mid = (lo + hi) >> 1
                    if rowval[mid] == row
                        nzval_map[idx] = mid
                        break
                    elseif rowval[mid] < row
                        lo = mid + 1
                    else
                        hi = mid - 1
                    end
                end
                idx += 1
            end
        end
    end

    return nzval_map
end

"""
    _build_block_tridiag_pattern(::Type{T}, bs::Int, n::Int)

Build a sparse matrix with the block tridiagonal sparsity pattern (values zeroed),
so that subsequent calls can update values in-place.
"""
function _build_block_tridiag_pattern(::Type{T}, bs::Int, n::Int) where {T<:Real}
    N = n * bs
    total_nnz = n * bs * bs + 2 * (n - 1) * bs * bs

    I_vec = Vector{Int}(undef, total_nnz)
    J_vec = Vector{Int}(undef, total_nnz)
    # Use ones so sparse() retains the structural entries (zeros get dropped)
    V_vec = ones(T, total_nnz)

    idx = 1
    for k in 1:n
        base = (k - 1) * bs
        for i in 1:bs, j in 1:bs
            I_vec[idx] = base + i
            J_vec[idx] = base + j
            idx += 1
        end
    end
    for k in 1:(n - 1)
        base_k = (k - 1) * bs
        base_kp1 = k * bs
        for i in 1:bs, j in 1:bs
            I_vec[idx] = base_k + i
            J_vec[idx] = base_kp1 + j
            idx += 1
        end
        for i in 1:bs, j in 1:bs
            I_vec[idx] = base_kp1 + i
            J_vec[idx] = base_k + j
            idx += 1
        end
    end

    H = sparse(I_vec, J_vec, V_vec, N, N)
    fill!(H.nzval, zero(T))  # Zero out values but keep the structure
    return H
end

"""
    block_tridgm!(ws::BlockTridiagonalWorkspace{T})

Update the values of the preallocated sparse matrix `ws.H_sparse` from the
current contents of `ws.H_diag`, `ws.H_sub`, `ws.H_super`.
Uses the precomputed `nzval_map` for direct writes — no sparse lookups or allocations.
"""
function block_tridgm!(ws::BlockTridiagonalWorkspace{T}) where {T<:Real}
    bs = ws.block_size
    n = ws.n_blocks
    nzval = ws.H_sparse.nzval
    map = ws.nzval_map

    idx = 1
    # Diagonal blocks
    for k in 1:n
        block = ws.H_diag[k]
        for j in 1:bs, i in 1:bs
            nzval[map[idx]] = block[i, j]
            idx += 1
        end
    end

    # Off-diagonal blocks (upper then lower for each k)
    for k in 1:(n - 1)
        block_up = ws.H_super[k]
        block_low = ws.H_sub[k]
        for j in 1:bs, i in 1:bs
            nzval[map[idx]] = block_up[i, j]
            idx += 1
        end
        for j in 1:bs, i in 1:bs
            nzval[map[idx]] = block_low[i, j]
            idx += 1
        end
    end

    return ws.H_sparse
end

"""
    block_tridiagonal_inverse!(p_smooth, p_smooth_tt1, A, B, C, ws)

Compute the block tridiagonal inverse, writing diagonal blocks into
`p_smooth[:,:,i]` and off-diagonal blocks into `p_smooth_tt1[:,:,i]`.
Uses preallocated buffers from `ws`.
"""
function block_tridiagonal_inverse!(
    p_smooth::AbstractArray{T,3},
    p_smooth_tt1::AbstractArray{T,3},
    A::AbstractVector{<:AbstractMatrix{T}},
    B::AbstractVector{<:AbstractMatrix{T}},
    C::AbstractVector{<:AbstractMatrix{T}},
    ws::BlockTridiagonalWorkspace{T},
) where {T<:Real}
    n = length(B)

    D = ws.D
    E = ws.E
    M = ws.M
    term1 = ws.term1
    term2 = ws.term2
    S = ws.S
    Ibs = ws.Ibs
    Z = ws.Z

    fill!(D[1], zero(T))
    fill!(E[n + 1], zero(T))

    # Forward sweep for D
    for i in 1:n
        Ai = (i == 1) ? Z : A[i - 1]
        Ci = (i <= length(C)) ? C[i] : Z

        copyto!(M, B[i])
        mul!(M, Ai, D[i], -one(T), one(T))
        F = lu!(M)
        ldiv!(D[i + 1], F, Ci)
    end

    # Backward sweep for E
    for i in n:-1:1
        Ci = (i <= length(C)) ? C[i] : Z
        Ai = (i == 1) ? Z : A[i - 1]

        copyto!(M, B[i])
        mul!(M, Ci, E[i + 1], -one(T), one(T))
        F = lu!(M)
        ldiv!(E[i], F, Ai)
    end

    # Compute diagonal blocks -> p_smooth[:,:,i]
    for i in 1:n
        copyto!(term1, Ibs)
        mul!(term1, D[i + 1], E[i + 1], -one(T), one(T))

        Ai = (i == 1) ? Z : A[i - 1]
        copyto!(term2, B[i])
        mul!(term2, Ai, D[i], -one(T), one(T))

        mul!(S, term2, term1)
        F = lu!(S)

        @views ldiv!(p_smooth[:, :, i], F, Ibs)
    end

    # Compute off-diagonal blocks -> p_smooth_tt1[:,:,i] for i=2:n
    for i in 2:n
        @views mul!(p_smooth_tt1[:, :, i], E[i], p_smooth[:, :, i - 1])
        @views p_smooth_tt1[:, :, i] .*= -one(T)
    end

    return nothing
end

"""
    block_tridiagonal_inverse_logdet!(p_smooth, p_smooth_tt1, A, B, C, ws)

Compute the block tridiagonal inverse and log-determinant simultaneously.
Returns the log-determinant of the precision matrix (i.e., logdet of the input matrix).

This is more efficient than calling block_tridiagonal_inverse! and gaussian_entropy
separately, as it computes logdet during the forward sweep without additional
matrix factorizations.
"""
function block_tridiagonal_inverse_logdet!(
    p_smooth::AbstractArray{T,3},
    p_smooth_tt1::AbstractArray{T,3},
    A::AbstractVector{<:AbstractMatrix{T}},
    B::AbstractVector{<:AbstractMatrix{T}},
    C::AbstractVector{<:AbstractMatrix{T}},
    ws::BlockTridiagonalWorkspace{T},
) where {T<:Real}
    n = length(B)
    bs = ws.block_size

    D = ws.D
    E = ws.E
    S = ws.S
    Ibs = ws.Ibs
    Z = ws.Z

    fill!(D[1], zero(T))
    fill!(E[n + 1], zero(T))

    # Accumulate log-determinant during forward sweep
    logdet_val = zero(T)

    #=
    Forward sweep — caches each Schur complement's Cholesky upper-triangle
    factor into `ws.chol_factors[i]` so `block_tridiagonal_backsubst!` can
    reuse them per trial.
    =#
    for i in 1:n
        Ai = (i == 1) ? Z : A[i - 1]
        Ci = (i <= length(C)) ? C[i] : Z

        Mi = ws.chol_factors[i]
        copyto!(Mi, B[i])
        mul!(Mi, Ai, D[i], -one(T), one(T))
        F = cholesky!(Symmetric(Mi, :U))

        # log|det(Mᵢ)| = 2·Σ log U[j,j] for the cached upper factor.
        for j in 1:bs
            logdet_val += 2 * log(Mi[j, j])
        end

        ldiv!(D[i + 1], F, Ci)
    end

    # Backward sweep — uses `ws.M` as scratch; not cached (per-trial backsubst
    # only needs the forward-sweep factors + D arrays).
    M = ws.M
    for i in n:-1:1
        Ci = (i <= length(C)) ? C[i] : Z
        Ai = (i == 1) ? Z : A[i - 1]

        copyto!(M, B[i])
        mul!(M, Ci, E[i + 1], -one(T), one(T))
        F = cholesky!(Symmetric(M, :U))
        ldiv!(E[i], F, Ai)
    end

    # Diagonal blocks -> p_smooth[:,:,i] via the SPD closed form
    # `Σᵢ = (Bᵢ - Aᵢ₋₁·Dᵢ - Cᵢ·Eᵢ₊₁)⁻¹`. 
    for i in 1:n
        Ai = (i == 1) ? Z : A[i - 1]
        Ci = (i <= length(C)) ? C[i] : Z

        copyto!(S, B[i])
        mul!(S, Ai, D[i], -one(T), one(T))         # S -= Aᵢ₋₁·Dᵢ
        mul!(S, Ci, E[i + 1], -one(T), one(T))     # S -= Cᵢ·Eᵢ₊₁
        F = cholesky!(Symmetric(S, :U))
        @views ldiv!(p_smooth[:, :, i], F, Ibs)
    end

    # Compute off-diagonal blocks -> p_smooth_tt1[:,:,i] for i=2:n
    for i in 2:n
        @views mul!(p_smooth_tt1[:, :, i], E[i], p_smooth[:, :, i - 1])
        @views p_smooth_tt1[:, :, i] .*= -one(T)
    end

    return logdet_val
end

"""
    gaussian_entropy_from_logdet(logdet_precision::T, n::Int) where {T<:Real}

Compute Gaussian entropy from the log-determinant of the precision matrix.
`n` is the dimensionality (number of variables).
"""
function gaussian_entropy_from_logdet(logdet_precision::T, n::Int) where {T<:Real}
    return T(0.5) * (n * (1 + log(2π)) - logdet_precision)
end

"""
    block_tridiagonal_backsubst!(x, A, b, ws, n)

Solve `H x = b` for the block tridiagonal `H` with lower off-diagonals `A`,
**using the Cholesky factors cached in `ws` by a prior
`block_tridiagonal_inverse_logdet!` call**. No factorization is performed
here — just the per-trial RHS forward elimination and the back-substitution
against `ws.D` (modified upper diagonals) and `ws.chol_factors` (the
per-block Cholesky factor of each SPD modified diagonal).

`x` and `b` are length `n * bs` vectors. The cached factors must correspond to
the same `H` blocks the caller passed to `block_tridiagonal_inverse_logdet!`;
calling this with a stale cache produces silently wrong results.

This is the per-trial half of the cov-cache fast path: the cov pass runs once
per E-step on a single workspace (filling the Cholesky cache); every
per-trial Newton solve then calls back into this function instead of
`block_tridiagonal_solve!`, saving an `O(T · D³)` factorization per trial.
"""
function block_tridiagonal_backsubst!(
    x::AbstractVector{T},
    A::AbstractVector{<:AbstractMatrix{T}},
    b::AbstractVector{T},
    ws::BlockTridiagonalWorkspace{T},
    n::Int,
) where {T<:Real}
    bs = ws.block_size
    D = ws.D

    # Block 1: x₁ = F₁ \ b₁  (Cholesky solve against cached upper factor)
    @views copyto!(x[1:bs], b[1:bs])
    F1 = LinearAlgebra.Cholesky{T,Matrix{T}}(ws.chol_factors[1], 'U', 0)
    @views ldiv!(F1, x[1:bs])

    # Forward elim of the RHS only — the modified diagonals are already
    # factored in ws.chol_factors[i].
    for i in 2:n
        idx_start = (i - 1) * bs + 1
        idx_end = i * bs
        idx_prev_start = (i - 2) * bs + 1
        idx_prev_end = (i - 1) * bs

        @views copyto!(x[idx_start:idx_end], b[idx_start:idx_end])
        # b̃ᵢ = bᵢ - Aᵢ₋₁ · x[i-1]
        @views mul!(
            x[idx_start:idx_end], A[i - 1], x[idx_prev_start:idx_prev_end], -one(T), one(T)
        )
        Fi = LinearAlgebra.Cholesky{T,Matrix{T}}(ws.chol_factors[i], 'U', 0)
        @views ldiv!(Fi, x[idx_start:idx_end])
    end

    #=
    Back-substitution: x[i] -= D[i+1] · x[i+1]. `D[i+1]` was filled by
    `block_tridiagonal_inverse_logdet!` during its forward sweep as
    `B̃ᵢ⁻¹ · Cᵢ`, which is exactly the modified upper diagonal needed here.
    =#
    for i in (n - 1):-1:1
        idx_start = (i - 1) * bs + 1
        idx_end = i * bs
        idx_next_start = i * bs + 1
        idx_next_end = (i + 1) * bs

        @views mul!(
            x[idx_start:idx_end], D[i + 1], x[idx_next_start:idx_next_end], -one(T), one(T)
        )
    end

    return x
end

#=
Matrix-RHS overload: each column is an independent system sharing the same
cached Cholesky factors. With `N` columns, every `ldiv!` becomes a triangular
solve with `bs × N` RHS, and every `mul!` is a `bs × bs × bs × N` matmul —
i.e. BLAS-3 instead of BLAS-2. This is the entry point for the batched
multi-trial mean pass in the cov-cache fast path.
=#
function block_tridiagonal_backsubst!(
    x::AbstractMatrix{T},
    A::AbstractVector{<:AbstractMatrix{T}},
    b::AbstractMatrix{T},
    ws::BlockTridiagonalWorkspace{T},
    n::Int,
) where {T<:Real}
    bs = ws.block_size
    D = ws.D

    # Block 1
    @views copyto!(x[1:bs, :], b[1:bs, :])
    F1 = LinearAlgebra.Cholesky{T,Matrix{T}}(ws.chol_factors[1], 'U', 0)
    @views ldiv!(F1, x[1:bs, :])

    for i in 2:n
        idx_start = (i - 1) * bs + 1
        idx_end = i * bs
        idx_prev_start = (i - 2) * bs + 1
        idx_prev_end = (i - 1) * bs

        @views copyto!(x[idx_start:idx_end, :], b[idx_start:idx_end, :])
        @views mul!(
            x[idx_start:idx_end, :],
            A[i - 1],
            x[idx_prev_start:idx_prev_end, :],
            -one(T),
            one(T),
        )
        Fi = LinearAlgebra.Cholesky{T,Matrix{T}}(ws.chol_factors[i], 'U', 0)
        @views ldiv!(Fi, x[idx_start:idx_end, :])
    end

    for i in (n - 1):-1:1
        idx_start = (i - 1) * bs + 1
        idx_end = i * bs
        idx_next_start = i * bs + 1
        idx_next_end = (i + 1) * bs

        @views mul!(
            x[idx_start:idx_end, :],
            D[i + 1],
            x[idx_next_start:idx_next_end, :],
            -one(T),
            one(T),
        )
    end

    return x
end

"""
    block_tridiagonal_solve!(x, A, B, C, b, ws)

Solve a block tridiagonal system `H * x = b` where H has:
- Lower off-diagonal blocks A[i] (size bs×bs, i=1:n-1)
- Main diagonal blocks B[i] (size bs×bs, i=1:n)
- Upper off-diagonal blocks C[i] (size bs×bs, i=1:n-1)

Uses block LU decomposition (Thomas algorithm for blocks).
The solution overwrites `x` (which should be length n*bs).

This is much more efficient than sparse LU for block tridiagonal matrices
as it only requires O(n) block factorizations instead of filling in a
potentially large sparse LU.

# Arguments
- `x::AbstractVector{T}`: Output vector (length n*bs), will be overwritten with solution
- `A::Vector{Matrix{T}}`: Lower off-diagonal blocks (length n-1)
- `B::Vector{Matrix{T}}`: Main diagonal blocks (length n)
- `C::Vector{Matrix{T}}`: Upper off-diagonal blocks (length n-1)
- `b::AbstractVector{T}`: Right-hand side vector (length n*bs)
- `ws::BlockTridiagonalWorkspace{T}`: Workspace with temp buffers
"""
#=
Pack the upper triangle of a symmetric block-tridiagonal matrix into
LAPACK banded storage (`uplo='U'`, bandwidth `kd = 2·bs - 1`).
`AB[kd+1+i-j, j] = H[i, j]` for `max(1, j-kd) ≤ i ≤ j`. Only the first
`bs*n_blocks` columns and `kd+1` rows of `AB` are touched.
=#
@inline function _pack_block_tridiag_banded!(
    AB::AbstractMatrix{T},
    diag_blocks::AbstractVector{<:AbstractMatrix{T}},
    super_blocks::AbstractVector{<:AbstractMatrix{T}},
) where {T}
    n_blocks = length(diag_blocks)
    bs = size(diag_blocks[1], 1)
    kd = 2 * bs - 1
    n = bs * n_blocks
    #=
    Zero the active region. Most of these slots get overwritten by the
    block fills below; the diagonal-block lower triangle (which we
    never touch) needs to stay zero so LAPACK doesn't read garbage.
    =#
    @inbounds for j in 1:n, r in 1:(kd + 1)
        AB[r, j] = zero(T)
    end
    @inbounds for k in 1:n_blocks
        Bk = diag_blocks[k]
        base = (k - 1) * bs
        for jj in 1:bs
            j_global = base + jj
            for ii in 1:jj
                AB[kd + 1 + (base + ii) - j_global, j_global] = Bk[ii, jj]
            end
        end
    end
    @inbounds for k in 1:(n_blocks - 1)
        Ck = super_blocks[k]
        row_base = (k - 1) * bs
        col_base = k * bs
        for jj in 1:bs, ii in 1:bs
            j_global = col_base + jj
            AB[kd + 1 + (row_base + ii) - j_global, j_global] = Ck[ii, jj]
        end
    end
    return AB
end

#=
pbsv-based fast path for SPD block-tridiagonal solves at small block
size. At `bs ≤ 8` the per-block BLAS dispatch overhead in the generic
block-Thomas path dominates over the tiny arithmetic; one packed
`pbsv` call to LAPACK amortises that overhead and is 30-60× faster.
=#
function _block_tridiagonal_solve_pbsv!(
    x::AbstractVector{T},
    diag_blocks::AbstractVector{<:AbstractMatrix{T}},
    super_blocks::AbstractVector{<:AbstractMatrix{T}},
    b::AbstractVector{T},
    ws::BlockTridiagonalWorkspace{T},
) where {T<:Union{Float32,Float64}}
    bs = size(diag_blocks[1], 1)
    n_blocks = length(diag_blocks)
    n = bs * n_blocks
    kd = 2 * bs - 1
    _pack_block_tridiag_banded!(ws.Hb, diag_blocks, super_blocks)
    @views copyto!(x[1:n], b[1:n])
    @views _pbsv_inplace!('U', n, kd, ws.Hb, x[1:n])
    return x
end

"""
    block_tridiagonal_solve_spd!(x, A, B, C, b, ws)

SPD-specialised solve for symmetric block-tridiagonal systems
`H · x = b` where the lower off-diagonal blocks `A[i]` equal
`C[i]'` (symmetric) and `H` is positive definite (Hessian-style
matrices at the smoother MAP). Same signature as
`block_tridiagonal_solve!` so callers can swap in.

At small block sizes (`bs ≤ 8`) and BlasFloat eltypes, packs the upper
triangle into LAPACK banded format and calls `pbsv` directly — 30-60×
faster than the general block-Thomas code at that size, because one
LAPACK call amortises the per-block BLAS dispatch overhead. For
larger `bs` (where blocked BLAS-3 already efficiently overlaps with
the arithmetic), or non-BlasFloat eltypes, falls back to the general
`block_tridiagonal_solve!`.

`A` is accepted for signature parity but only used on the fallback
branch. The pbsv path consults only `B` (diagonal) and `C` (upper
off-diagonal).
"""
function block_tridiagonal_solve_spd!(
    x::AbstractVector{T},
    A::AbstractVector{<:AbstractMatrix{T}},
    B::AbstractVector{<:AbstractMatrix{T}},
    C::AbstractVector{<:AbstractMatrix{T}},
    b::AbstractVector{T},
    ws::BlockTridiagonalWorkspace{T},
) where {T<:Real}
    bs = size(B[1], 1)
    if T <: Union{Float32,Float64} && bs <= 8
        return _block_tridiagonal_solve_pbsv!(x, B, C, b, ws)
    end
    return block_tridiagonal_solve!(x, A, B, C, b, ws)
end

function block_tridiagonal_solve!(
    x::AbstractVector{T},
    A::AbstractVector{<:AbstractMatrix{T}},
    B::AbstractVector{<:AbstractMatrix{T}},
    C::AbstractVector{<:AbstractMatrix{T}},
    b::AbstractVector{T},
    ws::BlockTridiagonalWorkspace{T},
) where {T<:Real}
    n = length(B)
    bs = size(B[1], 1)

    #=
    Reuse workspace buffers
    We need: modified diagonal blocks (stored in D), modified RHS (stored temporarily)
    D[i] will store the modified upper off-diagonal after forward elimination
    =#
    D = ws.D
    M = ws.M  # Temp for LU factorization

    # Forward elimination (modify diagonal and upper off-diagonal, and RHS)
    # Store modified C[i] in D[i+1] and modified b[i] in x[block i] temporarily

    # Per-block LU pivots are reused across iterations (the backward sweep
    # only consumes `D[i+1]`, not the factorisations themselves).
    ipiv = ws.lu_ipiv

    # First block: just copy b₁ to x₁ block and store C₁' in D[2]
    @views copyto!(x[1:bs], b[1:bs])

    # Process first block specially (no A[0])
    copyto!(M, B[1])
    _getrf_inplace!(M, ipiv)
    @views LinearAlgebra.LAPACK.getrs!('N', M, ipiv, x[1:bs])   # x₁ = B₁⁻¹ b₁
    copyto!(D[2], C[1])
    LinearAlgebra.LAPACK.getrs!('N', M, ipiv, D[2])             # D[2] = B₁⁻¹ C₁

    # Forward sweep for blocks 2 to n
    for i in 2:n
        # Get block indices
        idx_start = (i - 1) * bs + 1
        idx_end = i * bs
        idx_prev_start = (i - 2) * bs + 1
        idx_prev_end = (i - 1) * bs

        # Copy b[i] to x[i] block
        @views copyto!(x[idx_start:idx_end], b[idx_start:idx_end])

        # Modify diagonal: B̃ᵢ = Bᵢ - Aᵢ₋₁ * D[i]
        copyto!(M, B[i])
        mul!(M, A[i - 1], D[i], -one(T), one(T))

        # Modify RHS: b̃ᵢ = bᵢ - Aᵢ₋₁ * x[i-1]
        @views mul!(
            x[idx_start:idx_end], A[i - 1], x[idx_prev_start:idx_prev_end], -one(T), one(T)
        )

        # Factor modified diagonal (in-place, reusing `ipiv`).
        _getrf_inplace!(M, ipiv)

        # Solve for x[i] block
        @views LinearAlgebra.LAPACK.getrs!('N', M, ipiv, x[idx_start:idx_end])

        # Compute modified upper off-diagonal for next iteration (if not last block)
        if i < n
            copyto!(D[i + 1], C[i])
            LinearAlgebra.LAPACK.getrs!('N', M, ipiv, D[i + 1])
        end
    end

    # Backward substitution
    for i in (n - 1):-1:1
        idx_start = (i - 1) * bs + 1
        idx_end = i * bs
        idx_next_start = i * bs + 1
        idx_next_end = (i + 1) * bs

        # x[i] = x[i] - D[i+1] * x[i+1]
        @views mul!(
            x[idx_start:idx_end], D[i + 1], x[idx_next_start:idx_next_end], -one(T), one(T)
        )
    end

    return x
end

"""
    _negate_blocks!(ws::BlockTridiagonalWorkspace{T}, n_active::Int=ws.n_blocks)

Copy negated H_diag/H_sub/H_super into neg_diag/neg_sub/neg_super in-place
for the first `n_active` diagonal blocks (and `n_active - 1` off-diagonal blocks).
"""
function _negate_blocks!(
    ws::BlockTridiagonalWorkspace{T}, n_active::Int=ws.n_blocks
) where {T<:Real}
    for i in 1:n_active
        ws.neg_diag[i] .= .-ws.H_diag[i]
    end
    for i in 1:(n_active - 1)
        ws.neg_sub[i] .= .-ws.H_sub[i]
        ws.neg_super[i] .= .-ws.H_super[i]
    end
    return nothing
end
