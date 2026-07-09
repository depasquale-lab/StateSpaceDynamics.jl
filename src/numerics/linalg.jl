#=
In-place LU helpers — `LinearAlgebra.lu!(A)` allocates a fresh
`Vector{BlasInt}` pivot vector on every call. These wrappers reuse a
caller-provided `ipiv`, so a tight inner loop (e.g. the per-block LUs in
`block_tridiagonal_solve!`) can run allocation-free for `Float64`/`Float32`.
=#
for (gtrf, elty) in ((:dgetrf_, :Float64), (:sgetrf_, :Float32))
    @eval function _getrf_inplace!(A::Matrix{$elty}, ipiv::Vector{LinearAlgebra.BlasInt})
        m, n = size(A)
        @boundscheck length(ipiv) >= min(m, n) ||
            throw(ArgumentError("ipiv too small for getrf!"))
        info = Ref{LinearAlgebra.BlasInt}(0)
        ccall(
            (LinearAlgebra.BLAS.@blasfunc($gtrf), LinearAlgebra.libblastrampoline),
            Cvoid,
            (
                Ref{LinearAlgebra.BlasInt},
                Ref{LinearAlgebra.BlasInt},
                Ptr{$elty},
                Ref{LinearAlgebra.BlasInt},
                Ptr{LinearAlgebra.BlasInt},
                Ref{LinearAlgebra.BlasInt},
            ),
            m,
            n,
            A,
            max(1, stride(A, 2)),
            ipiv,
            info,
        )
        LinearAlgebra.LAPACK.chklapackerror(info[])
        return A
    end
end

#=
Generic fallback for non-BLAS element types (e.g. BigFloat). The pivot
vector is sized large enough at the workspace level, so `copyto!`
can't underflow.
=#
function _getrf_inplace!(A::AbstractMatrix, ipiv::Vector{LinearAlgebra.BlasInt})
    F = LinearAlgebra.lu!(A)
    n = length(F.ipiv)
    @boundscheck length(ipiv) >= n ||
        throw(ArgumentError("ipiv too small for getrf! fallback"))
    @inbounds for i in 1:n
        ipiv[i] = F.ipiv[i]
    end
    return A
end

#=
In-place SPD banded solver. Wraps LAPACK's `?pbsv` (not exposed in
`LinearAlgebra.LAPACK`). Solves A·X = B where A is SPD with
bandwidth `kd`, stored in upper-banded format (`uplo='U'`):
A[i,j] = AB[kd+1+i-j, j] for max(1, j-kd) ≤ i ≤ j.
Both `AB` (Cholesky factor on return) and `B` (solution on return)
are overwritten. Falls back to the dense path for non-BLAS eltypes.
=#
for (pbsv, elty) in ((:dpbsv_, :Float64), (:spbsv_, :Float32))
    @eval function _pbsv_inplace!(
        uplo::Char, n::Int, kd::Int, AB::Matrix{$elty}, B::AbstractVecOrMat{$elty}
    )
        ldab = stride(AB, 2)
        @boundscheck ldab >= kd + 1 ||
            throw(ArgumentError("AB row stride too small for pbsv"))
        @boundscheck size(AB, 2) >= n || throw(ArgumentError("AB has fewer than n columns"))
        nrhs = B isa AbstractVector ? 1 : size(B, 2)
        ldb = B isa AbstractVector ? n : stride(B, 2)
        info = Ref{LinearAlgebra.BlasInt}(0)
        ccall(
            (LinearAlgebra.BLAS.@blasfunc($pbsv), LinearAlgebra.libblastrampoline),
            Cvoid,
            (
                Ref{UInt8},
                Ref{LinearAlgebra.BlasInt},
                Ref{LinearAlgebra.BlasInt},
                Ref{LinearAlgebra.BlasInt},
                Ptr{$elty},
                Ref{LinearAlgebra.BlasInt},
                Ptr{$elty},
                Ref{LinearAlgebra.BlasInt},
                Ref{LinearAlgebra.BlasInt},
            ),
            uplo,
            n,
            kd,
            nrhs,
            AB,
            ldab,
            B,
            ldb,
            info,
        )
        LinearAlgebra.LAPACK.chklapackerror(info[])
        return B
    end
end

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
    return all(x -> typeof(x) == first_type, args)
end

# Matrix utilities
"""
    Symmetrize!(A::AbstractMatrix{T})::Symmetric{T} where {T<:Real}

In-place symmetrization of a square matrix `A` via averaging with its transpose:
A <- 0.5*(A + A').
"""
function Symmetrize!(A::AbstractMatrix{T})::Symmetric{T} where {T<:Real}
    n, m = size(A)
    @boundscheck n == m || throw(
        DimensionMismatch("Matrix must be square for symmetrization, got $(n)×$(m)")
    )
    half = T(0.5)

    for j in 1:n
        for i in 1:(j - 1)
            avg = (A[i, j] + A[j, i]) * half
            A[i, j] = avg
            A[j, i] = avg
        end
    end

    return Symmetric(A)
end

"""
    valid_Σ(Σ) -> Bool

Check whether `Σ` is a valid covariance matrix, i.e. Hermitian and positive
definite.
"""
function valid_Σ(Σ::AbstractMatrix{T}) where {T<:Real}
    return ishermitian(Σ) && isposdef(Σ)
end

"""
    tol_PD(A; tol=1e-6) -> PDMat

Eigen-floor stabilization for covariance matrices used along the Kalman path. All
eigenvalues below `tol * λ_max` are raised to `tol * λ_max`, preserving the overall
scale/conditioning of the matrix; the result is rewrapped as a `PDMat` so downstream
code can reuse the cached Cholesky. The matrix is symmetrized (via `hermitianpart`) if
passed as a plain `Matrix`.

Used by the Kalman filter/smoother to keep predicted and filtered covariances strictly
positive definite in the presence of floating-point noise. Ported from
StateSpaceAnalysis.
"""
function tol_PD(
    A_sym::Union{Symmetric{T},Hermitian{T}}; tol::T=1e-6
)::PDMat{T,Matrix{T}} where {T<:Real}
    F = eigen!(A_sym)
    λ_max = F.values[end]
    scale = λ_max * tol
    slope = λ_max - scale        # = λ_max * (1 - tol)
    for i in eachindex(F.values)
        r = max(F.values[i] / λ_max, zero(T))
        F.values[i] = slope * r + scale
    end
    return PDMat(X_A_Xt(PDiagMat(F.values), F.vectors))
end

tol_PD(A::Matrix; tol::Real=1e-6)::PDMat = tol_PD(hermitianpart(A); tol=tol)
tol_PD(A::PDMat; tol::Real=1e-6)::PDMat = tol_PD(Hermitian(Matrix(A)); tol=tol)
id_PD(A::Matrix; tol::Real=1e-6)::PDMat =
    PDMat(hermitianpart(A + (tol * tr(A) / size(A, 1)) * I))

"""
    gaussian_entropy(H::Symmetric{T}) where {T<:Real}

Entropy (in nats) of a Gaussian whose log-posterior Hessian at the MAP is `H`
(i.e., `H = ∇² log p` so the precision is `Λ = -H`).
"""
function gaussian_entropy(H::Symmetric{T}) where {T<:Real}
    n = size(H, 1)
    F = cholesky(-H)                    # factorize Λ = -H (SPD)
    logdetΛ = 2 * sum(log, diag(F))     # logdet precision
    return 0.5 * (n * (1 + log(2π)) - logdetΛ)
end

"""
    random_rotation_matrix(n)

Generate a random rotation matrix of size `n x n`.
"""
function random_rotation_matrix(n::Int, rng::AbstractRNG=Random.default_rng())
    # Generate a random orthogonal matrix using QR decomposition
    Q, _ = qr(randn(rng, n, n))

    return Matrix(Q)
end
