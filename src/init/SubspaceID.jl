# =============================================================================
# SSID.jl — subspace-identification (SSID) initializer
#
# Ports the HallM_NeSS `subspaceid_orig` driver and its `*_hr` helpers
# (`find_PK_hr`, `find_BD_hr`, `reflectd`) — itself adapted from
# ControlSystemIdentification.jl — into StateSpaceDynamics as an *initialization*
# method for the Gaussian LTI LDS, invoked via `fit!(lds, y, SSID(); …)`.
#
# Recommended Algorithm: Canonical Variate Analysis (CVA) 
# Produces a deterministic, data-driven estimate of (A, C, Q, R, b, d, x0, P0, B, D) 
# that seeds EM. It does NOT run EM. Scope is intentionally limited to 
# `GaussianStateModel` + `GaussianObservationModel`; other model types hit a 
# fallback method that throws.
#
# Near dependency-free port of ControlSystemsIdentification.jl: the reference
# uses ControlSystems (`lsim`/`ss`) and MatrixEquations (`ared`). The LTI
# simulation is replaced with the pure-Julia kernel `_ltisim` (discrete-time
# LTI simulation); the stationary state covariance used as `P0` is solved with
# `MatrixEquations.lyapd` (Bartels–Stewart, O(n³)), which scales to the large
# latent dimensions (n up to ~192) this initializer targets.
# =============================================================================

"""
    SSID(; r=0, s1=0, s2=0, W=:CVA, zeroD=true, stable=true, scaleU=false,
          cross_trial=true, new_init=false, ridge=true, λ=1e-3, order=0,
          set_all=true, jitter=1e-8, verbose=false)

Options controlling Canonical Variate Analysis (CVA) subspace identification used
to initialize a Gaussian LTI `LinearDynamicalSystem`. Pass an instance as the
algorithm argument to [`fit!`](@ref):

```julia
diag = fit!(lds, y, SSID())                 # autonomous LDS
diag = fit!(lds, y, SSID(); latent_inputs=u)  # with dynamics inputs (estimates B)
```

`fit!` mutates `lds` in place and returns a `NamedTuple` `(; S, fve)` with the
subspace singular values `S` and the fraction of variance explained `fve`.

# Keyword arguments
- `r`: prediction horizon (block-Hankel depth). `0` ⇒ data-adaptive.
- `s1`, `s2`: past-output / past-input horizons. `0` ⇒ equal to `r`.
- `W`: weighting scheme, one of `:CVA` (default), `:MOESP`, `:N4SID`, `:IVM`.
- `zeroD`: force the observation feed-through `D` to zero.
- `stable`: reflect unstable eigenvalues of `A` into the unit disk (`reflectd`).
- `scaleU`: rescale dynamics-input channels by their std for the subspace step.
- `cross_trial`: if `true`, concatenate trials into one sequence so the delay
    embedding spans epoch boundaries (matches HallM `subspaceid_orig`); if
    `false`, build per-trial Hankel blocks and pool their columns.
- `new_init`: if `false` (default), estimate `B, D, x0` by regression
    (`find_BD_hr` with Kalman gain `K=0`); if `true`, use HallM placeholder
    `B = ones(n,m)/sum`, `x0 = ones(n)/n`, `D = 0`.
- `ridge`, `λ`: use a ridge estimator with strength `λ` for the `A`/`B`/`D`
    regressions (else plain least squares `\\`).
- `order`: subspace model order; `0` ⇒ `lds.latent_dim`. Must equal
    `lds.latent_dim` (the result is written into the model's fields).
- `set_all`: if `true`, write every Gaussian parameter regardless of
    `lds.fit_bool`; if `false`, only write parameter groups whose `fit_bool`
    entry is `true` (Gaussian layout `[x0, P0, A&b&B, Q, C&d&D, R]`).
- `jitter`: eigenvalue floor used when projecting `Q`, `R`, `P0` to be
    symmetric positive-definite.
- `verbose`: print progress information.

# Notes
- Identified `A`, `C` are only defined up to a similarity transform; compare
    invariants (eigenvalues of `A`, Markov parameters `C Aᵏ B`) rather than raw
    matrices.
- Supports `Float32`/`Float64` (uses LAPACK `lq!`/`svd!`).
- Observations are demeaned internally; the empirical mean is placed in the
    observation bias `d`, with state bias `b = 0`.
"""
Base.@kwdef struct SSID
    r::Int = 0
    s1::Int = 0
    s2::Int = 0
    W::Symbol = :CVA
    zeroD::Bool = true
    stable::Bool = true
    scaleU::Bool = false
    cross_trial::Bool = true
    new_init::Bool = false
    ridge::Bool = true
    λ::Float64 = 1e-3
    order::Int = 0
    set_all::Bool = true
    jitter::Float64 = 1e-8
    verbose::Bool = false
end

# -----------------------------------------------------------------------------
# Low-level numerical helpers (pure Julia)
# -----------------------------------------------------------------------------

# Per-column standard deviation over rows (dim 1), avoiding a Statistics.std import.
function _colstd(X::AbstractMatrix{T}) where {T}
    μ = mean(X; dims=1)
    nrow = size(X, 1)
    denom = max(nrow - 1, 1)
    return max(sqrt.(vec(sum(abs2, X .- μ; dims=1)) ./ denom), eps(T(1.0)))
end

# Ridge / Tikhonov least squares: argmin ||x*β - y||² + λ||β||², whose normal
# equations are (XᵀX + λI) β = Xᵀy. For the tall time-series regressors here
# (rows N ≫ columns k) forming the k×k Gram matrix and taking a Cholesky factor
# is far cheaper in both time and memory than a QR of the (N+k)×k augmented
# system. The Gram matrix is positive-definite for λ>0, so Cholesky succeeds; we
# keep the augmented-QR path only as a numerical safety net.
function _ridge(x::AbstractMatrix{T}, y::AbstractVecOrMat{T}, λ::T) where {T}
    G = Symmetric(x' * x + λ * I)
    rhs = x' * y
    C = cholesky(G; check=false)
    issuccess(C) && return C \ rhs
    return _ridge_qr(x, y, λ)
end

# Augmented-system fallback (numerically robust QR) for the rare case where the
# regularized Gram matrix is not numerically positive-definite.
function _ridge_qr(x::AbstractMatrix{T}, y::AbstractVector{T}, λ::T) where {T}
    k = size(x, 2)
    xr = vcat(x, sqrt(λ) * Matrix{T}(I, k, k))
    yr = vcat(y, zeros(T, k))
    return xr \ yr
end
function _ridge_qr(x::AbstractMatrix{T}, y::AbstractMatrix{T}, λ::T) where {T}
    k = size(x, 2)
    xr = vcat(x, sqrt(λ) * Matrix{T}(I, k, k))
    yr = vcat(y, zeros(T, k, size(y, 2)))
    return xr \ yr
end

# Reflect a scalar eigenvalue with |λ|>~1 back inside the unit circle. Returns a
# complex value (the matrix variant takes `real` once at the end); `thr` is typed
# from `abs(x)` so element types (e.g. Float32) are preserved.
function _reflectd(x::Number)
    a = abs(x)
    thr = oftype(a, 0.9999)
    a < thr && return oftype(cis(angle(x)), x)
    return (thr / a) * cis(angle(x))
end

# Stabilize a matrix by reflecting its unstable eigenvalues (HallM `reflectd`).
function _reflectd(A::AbstractMatrix)
    vals, vecs = eigen(A)
    vals = _reflectd.(vals)
    A2 = vecs * Diagonal(vals) / vecs
    return eltype(A) <: Real ? real(A2) : A2
end

# Symmetrize and project a matrix to be symmetric positive-definite. The
# `Symmetric` wrapper reads a single triangle, so it implicitly symmetrizes `M`
# with no extra allocation. The returned matrix satisfies `issymmetric` exactly
# (via the final (P+P')/2 idiom, which is exactly symmetric in IEEE arithmetic)
# and `isposdef` (eigenvalue floor).
function _make_pd(M::AbstractMatrix, ::Type{T}; jitter) where {T}
    E = eigen(Symmetric(M))
    λmax = isempty(E.values) ? one(real(T)) : maximum(E.values)
    floorλ = max(T(jitter), T(jitter) * (λmax > 0 ? λmax : one(real(T))))
    vals = max.(E.values, floorλ)
    P = E.vectors * Diagonal(vals) * E.vectors'
    P = (P .+ P') ./ 2
    return Matrix{T}(P)
end

# Discrete-time LTI simulation: x_{t+1} = A x_t + E u_t,  y_t = C x_t  (D = 0).
# Replaces `lsim(ss(A,E,C,0,1), u; x0)`. `E` may have zero columns (free run).
function _ltisim(
    A::AbstractMatrix{T},
    E::AbstractMatrix{T},
    C::AbstractMatrix{T},
    u::AbstractMatrix{T},
    x0::AbstractVector{T},
) where {T}
    n = size(A, 1)
    p = size(C, 1)
    N = size(u, 2)
    Y = Matrix{T}(undef, p, N)
    x = Vector{T}(undef, n)
    copyto!(x, x0)
    xn = Vector{T}(undef, n)
    has_input = size(E, 2) > 0
    @inbounds for t in 1:N
        mul!(view(Y, :, t)::SubArray{T}, C, x)
        mul!(xn, A, x)
        if has_input
            mul!(xn, E, view(u, :, t)::SubArray{T}, one(T), one(T))
        end
        x, xn = xn, x
    end
    return Y
end

# Batched deterministic LTI responses for the `_find_BD_ssid` regressor. In a
# single time sweep this simulates *every* column of the B/x0 design block at
# once by stacking their states into one n × (m*n + n) matrix `Z`:
#   - B columns  (m*n): response of (A, C) to a unit input on dynamics channel k
#     entering state j — state IC 0, driven by `U[k, :]`. Column (k, j) sits at
#     index (k-1)*n + j, matching the per-column loop it replaces.
#   - x0 columns (n): free decay of (A, C) from each canonical initial state e_j.
# The state is propagated with a BLAS-3 `mul!(Znext, A, Z)` (one dense gemm per
# step) plus a cheap diagonal input injection, rather than m*n + n independent
# matrix-vector simulations; the vectorized outputs are written straight into
# the first m*n + n columns of `Φ` (layout `Φ[(t-1)*p+1:t*p, col] = C Z_t`).
# Mathematically identical to calling `_ltisim` once per column, but far faster
# and far lighter on allocations at the large latent dimensions this targets.
function _ltisim_batch!(
    Φ::AbstractMatrix{T},
    A::AbstractMatrix{T},
    C::AbstractMatrix{T},
    U::AbstractMatrix{T},
    n::Int,
    p::Int,
    m::Int,
    N::Int,
) where {T}
    ncolB = m * n
    ncol = ncolB + n
    Z = zeros(T, n, ncol)
    @inbounds for j in 1:n
        Z[j, ncolB + j] = one(T)              # x0 initial conditions e_j
    end
    Znext = Matrix{T}(undef, n, ncol)
    @inbounds for t in 1:N
        mul!(view(Φ, ((t - 1) * p + 1):(t * p), 1:ncol)::SubArray{T}, C, Z)   # y_t = C Z_t
        if t < N
            mul!(Znext, A, Z)                                    # Z_{t+1} = A Z_t + inj
            for k in 1:m
                uk = U[k, t]
                base = (k - 1) * n
                for j in 1:n
                    Znext[j, base + j] += uk
                end
            end
            Z, Znext = Znext, Z
        end
    end
    return Φ
end

# Discrete Lyapunov solve A P A' + Q = P for the stationary state covariance,
# used as P0. Replaces the `ared`-based P in HallM `find_PK_hr`. Uses
# `MatrixEquations.lyapd` (Bartels–Stewart via Schur decomposition, O(n³)),
# which scales to the large latent dimensions (n up to ~192) this initializer
# targets — a Kronecker `(I - A⊗A) \ vec(Q)` solve would be O(n⁶) in time and
# O(n⁴) in memory. Falls back to Q if the solve fails (e.g. A not stable, so no
# unique/finite solution). Always returned symmetric positive-definite.
function _dlyap(A::AbstractMatrix{T}, Q::AbstractMatrix{T}; jitter, stable::Bool) where {T}
    P = copy(Q)
    if stable
        local Pc
        solved = true
        try
            Pc = lyapd(A, Q)
        catch
            solved = false
        end
        if solved && all(isfinite, Pc)
            P = Pc
        end
    end
    return _make_pd(P, T; jitter=jitter)
end

# -----------------------------------------------------------------------------
# SSID subspace machinery (block-Hankel construction + weighting + recovery)
# -----------------------------------------------------------------------------

# Forward block-Hankel matrix (r*d × N) from time-major data (t × d). Every
# element is written, so `undef` avoids a wasted zero-fill. The column index Ni
# is the outer loop and the row-block ri the inner one, so writes march down each
# contiguous column (column-major) instead of striding across columns.
function _ssid_hankel(data::AbstractMatrix{T}, t0::Int, r::Int, N::Int) where {T}
    d = size(data, 2)
    H = Matrix{T}(undef, r * d, N)
    @inbounds for Ni in 1:N, ri in 1:r
        H[((ri - 1) * d + 1):(ri * d), Ni] .= view(data, t0 + ri + Ni - 2, :)::SubArray{T}
    end
    return H
end

# Past regressor Φ (s × N), s = s1*p + s2*m. Block layout [past-outputs;
# past-inputs], each block column-major over (lag, channel) to match the
# reference `vec(y[t-1:-1:t-s1, :])`: within a channel column c the lags run
# t-1, t-2, …, t-s1, then the next channel follows. Filling element-by-element
# keeps the writes marching down Φ's contiguous columns and avoids the
# temporary that `vec(@view y[(t-1):-1:(t-s1), :])` allocates from a
# reversed (negative-stride) SubArray.
function _ssid_phi(
    y::AbstractMatrix{T},
    u::AbstractMatrix{T},
    t0::Int,
    N::Int,
    s1::Int,
    s2::Int,
    p::Int,
    m::Int,
) where {T}
    s = s1 * p + s2 * m
    Φ = Matrix{T}(undef, s, N)
    @inbounds for (idx, t) in enumerate(t0:(t0 + N - 1))
        for c in 1:p, ℓ in 1:s1
            Φ[(c - 1) * s1 + ℓ, idx] = y[t - ℓ, c]
        end
        if m > 0 && s2 > 0
            off = s1 * p
            for c in 1:m, ℓ in 1:s2
                Φ[off + (c - 1) * s2 + ℓ, idx] = u[t - ℓ, c]
            end
        end
    end
    return Φ
end

# Oblique projection helper used by the IVM weighting (HallM `proj_hr`).
function _proj_hr(UY::AbstractMatrix, Yinds)
    fact = lq!(copy(UY))
    L = fact.L
    Qm = Matrix(fact.Q)
    return L[Yinds, Yinds] * Qm[Yinds, :]
end

# Form the extended observability matrix `Or`, singular values `sv`, fraction of
# variance explained `fve`, and the L-blocks (L1, L2) needed by `_find_PK_ssid`.
function _ssid_subspace(
    Y::AbstractMatrix{T},
    U::AbstractMatrix{T},
    Φ::AbstractMatrix{T},
    r::Int,
    s1::Int,
    s2::Int,
    p::Int,
    m::Int,
    W::Symbol,
    n::Int,
) where {T}
    nU, nΦ, nY = size(U, 1), size(Φ, 1), size(Y, 1)
    fact = lq!(vcat(U, Φ, Y))
    L = fact.L

    Uinds = 1:nU
    Φinds = (1:nΦ) .+ nU
    Yinds = (1:nY) .+ (nU + nΦ)

    L1 = L[Uinds, Uinds]
    L2 = L[(s1 * p + (r + s2) * m + 1):end, 1:(s1 * p + (r + s2) * m + p)]

    local Or, sv
    if W === :MOESP || W === :N4SID
        Qm = Matrix(fact.Q)
        L21 = L[Φinds, Uinds]
        L22 = L[Φinds, Φinds]
        L32 = L[Yinds, Φinds]
        Q1 = Qm[Uinds, :]
        Q2 = Qm[Φinds, :]
        Ĝ = L32 * (L22 \ hcat(L21, L22)) * vcat(Q1, Q2)
        G = W === :MOESP ? L32 * Q2 : Ĝ
        sv = svd(G)
        Rsv = Diagonal(sqrt.(sv.S[1:n]))
        Or = sv.U[:, 1:n] * Rsv            # W1 = I for both
    elseif W === :IVM
        Ncols = size(Y, 2)
        UY = vcat(U, Y)
        Yi = (1:nY) .+ nU
        YΠUt = _proj_hr(UY, Yi)
        G = YΠUt * Φ'
        W1 = real(sqrt(Symmetric(pinv((one(T) / Ncols) * (YΠUt * Y')))))
        W2 = real(sqrt(Symmetric(pinv((one(T) / Ncols) * (Φ * Φ')))))
        G = W1 * G * W2
        sv = svd(G)
        Rsv = Diagonal(sqrt.(sv.S[1:n]))
        Or = W1 \ (sv.U[:, 1:n] * Rsv)
    elseif W === :CVA
        L32 = L[Yinds, Φinds]
        W1 = L[Yinds, vcat(Φinds, Yinds)]
        ull1, sll1 = svd(W1)
        sll1d = Diagonal(sll1[1:(r * p)])
        cva = svd(sll1d \ (ull1' * L32))
        Or = ull1 * sll1d * cva.U
        sv = svd(L32)
    else
        throw(
            ArgumentError(
                "SSID: unknown weighting W=:$(W); expected :CVA, :MOESP, :N4SID, or :IVM"
            ),
        )
    end

    fve = sum(sv.S[1:n]) / sum(sv.S)
    return Or, sv, fve, L1, L2
end

# Recover C (first block-row of Or) and A (shift-invariance least squares),
# stabilizing A by eigenvalue reflection when requested.
function _ssid_AC(Or, p::Int, r::Int, n::Int, Aestimator, stable::Bool)
    C = Or[1:p, 1:n]
    A = Aestimator(Or[1:(p * (r - 1)), 1:n], Or[(p + 1):(p * r), 1:n])
    if !all(e -> abs(e) <= 1, eigvals(A)) && stable
        A = _reflectd(A)
    end
    return Matrix(A), Matrix(C)
end

# Process / observation noise covariances from the residual of the one-step
# predictor regression (HallM `find_PK_hr` / CSI `find_PK`, normalized by the
# effective sample count). The cross-covariance S and Kalman gain K are not
# needed (SSD has no field for them) and are discarded.
function _find_PK_ssid(
    L1, L2, Or, n::Int, p::Int, m::Int, r::Int, s1::Int, s2::Int, ::Type{T}; jitter
) where {T}
    X1 = L2[(p + 1):(r * p), 1:(m * (s2 + r) + p * s1 + p)]
    X2 = hcat(L2[1:(r * p), 1:(m * (s2 + r) + p * s1)], zeros(T, r * p, p))
    vl = vcat(Or[1:((r - 1) * p), 1:n] \ X1, L2[1:p, 1:(m * (s2 + r) + p * s1 + p)])
    hl = vcat(Or[:, 1:n] \ X2, hcat(L1, zeros(T, m * r, (m * s2 + p * s1) + p)))

    K0 = vl * pinv(hl)
    resid = vl - K0 * hl
    dof = max(size(vl, 2) - size(K0, 2), 1)
    Wcov = (resid * resid') ./ dof

    Q = _make_pd(Wcov[1:n, 1:n], T; jitter=jitter)
    R = _make_pd(Wcov[(n + 1):(n + p), (n + 1):(n + p)], T; jitter=jitter)
    return Q, R
end

# Estimate B, D, x0 by regressing the (demeaned) output onto the simulated
# responses of the deterministic system, using Kalman gain K = 0 (so the
# innovation sequence vanishes and this reduces to ordinary least squares).
# Port of HallM `find_BD_hr`, with `lsim` replaced by `_ltisim`. `U` (m × N)
# carries the dynamics inputs (→ B); `V` (mD × N) the observation inputs (→ D).
function _find_BD_ssid(
    A::AbstractMatrix{T},
    C::AbstractMatrix{T},
    U::AbstractMatrix{T},
    V::AbstractMatrix{T},
    Y::AbstractMatrix{T},
    zeroD::Bool,
    estimator,
    ::Type{T},
) where {T}
    n = size(A, 1)
    p = size(C, 1)
    N = size(Y, 2)
    m = size(U, 1)
    mD = size(V, 1)

    ncolB = m * n
    ncolD = zeroD ? 0 : p * mD
    ncol = ncolB + n + ncolD
    Φ = Matrix{T}(undef, p * N, ncol)

    # B blocks (response to a unit input on channel k entering state j) and x0
    # blocks (free decay from each canonical initial state) are simulated in a
    # single batched LTI sweep filling the first ncolB + n columns of Φ.
    _ltisim_batch!(Φ, A, C, U, n, p, m, N)

    # D blocks: static feed-through of obs-input channel k into output row j.
    # `vec(block)` places V[k, t] at row (t-1)*p + j, so the only nonzero rows of
    # this column are the strided rows j, j+p, …, j+(N-1)*p (everything else 0).
    if !zeroD
        col = ncolB + n
        @inbounds for k in 1:mD, j in 1:p
            col += 1
            colv = view(Φ, :, col)
            fill!(colv, zero(T))
            for t in 1:N
                colv[(t - 1) * p + j] = V[k, t]
            end
        end
    end

    BD = estimator(Φ, vec(Y))
    B = m == 0 ? zeros(T, n, 0) : copy(reshape(BD[1:ncolB], n, m))
    x0 = copy(BD[ncolB .+ (1:n)])
    D = zeroD ? zeros(T, p, mD) : copy(reshape(BD[(ncolB + n + 1):end], p, mD))
    return B, D, x0
end

# Data-adaptive prediction horizon when `r == 0`. Keeps r below half the record
# length (with s1=s2=r the effective sample count is Ttot - 2r) and at least n+1.
function _auto_r(Ttot::Int, n::Int)
    rmax = max(1, (Ttot - 2) ÷ 4)
    return clamp(min(Ttot ÷ 20, 50), n + 1, max(n + 1, rmax))
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

function _ssid_fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}},
    u_seq::AbstractVector{<:AbstractMatrix{T}},
    v_seq::AbstractVector{<:AbstractMatrix{T}},
    alg::SSID,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    p = lds.obs_dim
    m = lds.state_input_dim
    mD = lds.obs_input_dim
    zeroD = alg.zeroD || mD == 0

    n = alg.order == 0 ? lds.latent_dim : alg.order
    n == lds.latent_dim || throw(
        ArgumentError("SSID: order ($n) must equal lds.latent_dim ($(lds.latent_dim))")
    )

    # Concatenate + demean observations.
    Ycat = reduce(hcat, y)
    Ttot = size(Ycat, 2)
    ȳ = vec(mean(Ycat; dims=2))
    Ydm = Ycat .- ȳ

    # Resolve horizons.
    r = alg.r == 0 ? _auto_r(Ttot, n) : alg.r
    s1 = alg.s1 == 0 ? r : alg.s1
    s2 = alg.s2 == 0 ? r : alg.s2
    r >= n + 1 || throw(
        ArgumentError("SSID: prediction horizon r=$r must be ≥ model order n+1=$(n + 1)"),
    )
    alg.verbose &&
        @info "SSID: r=$r, s1=$s1, s2=$s2, W=:$(alg.W), n=$n, cross_trial=$(alg.cross_trial)"

    t0 = max(s1, s2) + 1

    # Build block-Hankel matrices Y, U and past regressor Φ.
    local Yh, Uh, Φh
    if alg.cross_trial
        ytm = permutedims(Ydm)                                   # Ttot × p
        utm = m > 0 ? permutedims(reduce(hcat, u_seq)) : zeros(T, Ttot, 0)
        if alg.scaleU && m > 0
            utm = utm ./ _colstd(utm)'
        end
        N = Ttot - r + 1 - t0
        N > 0 || throw(
            ArgumentError(
                "SSID: not enough data (Ttot=$Ttot) for r=$r; need Ttot > $(r - 1 + t0)"
            ),
        )
        Yh = _ssid_hankel(ytm, t0, r, N)
        Uh = _ssid_hankel(utm, t0, r, N)
        Φh = _ssid_phi(ytm, utm, t0, N, s1, s2, p, m)
    else
        Yblocks = Matrix{T}[]
        Ublocks = Matrix{T}[]
        Φblocks = Matrix{T}[]
        col0 = 0
        for (i, yi) in enumerate(y)
            Ti = size(yi, 2)
            Ni = Ti - r + 1 - t0
            Ni > 0 || throw(
                ArgumentError(
                    "SSID: trial $i too short (T=$Ti) for r=$r with cross_trial=false; need T > $(r - 1 + t0)",
                ),
            )
            ytm = permutedims(yi .- ȳ)
            utm = m > 0 ? permutedims(u_seq[i]) : zeros(T, Ti, 0)
            if alg.scaleU && m > 0
                utm = utm ./ _colstd(utm)'
            end
            push!(Yblocks, _ssid_hankel(ytm, t0, r, Ni))
            push!(Ublocks, _ssid_hankel(utm, t0, r, Ni))
            push!(Φblocks, _ssid_phi(ytm, utm, t0, Ni, s1, s2, p, m))
            col0 += Ni
        end
        col0 > 0 || throw(ArgumentError("SSID: no usable Hankel columns (data too short)"))
        Yh = reduce(hcat, Yblocks)
        Uh = reduce(hcat, Ublocks)
        Φh = reduce(hcat, Φblocks)
    end

    # The LQ factor is square only when there are at least as many Hankel
    # columns as stacked block-rows; otherwise the L-block indexing is invalid.
    block_rows = size(Uh, 1) + size(Φh, 1) + size(Yh, 1)
    size(Yh, 2) >= block_rows || throw(
        ArgumentError(
            "SSID: too few Hankel columns ($(size(Yh, 2))) for r=$r, s1=$s1, s2=$s2; " *
            "need ≥ $block_rows. Reduce r/s1/s2 or provide more data.",
        ),
    )

    # Subspace identification → Or, A, C, Q, R.
    Or, sv, fve, L1, L2 = _ssid_subspace(Yh, Uh, Φh, r, s1, s2, p, m, alg.W, n)
    n <= size(Or, 2) || throw(
        ArgumentError(
            "SSID: model order n=$n exceeds identifiable subspace dimension $(size(Or, 2))",
        ),
    )

    Aestimator = alg.ridge ? ((x, yy) -> _ridge(x, yy, T(alg.λ))) : ((x, yy) -> x \ yy)
    A, C = _ssid_AC(Or, p, r, n, Aestimator, alg.stable)
    Q, R = _find_PK_ssid(L1, L2, Or, n, p, m, r, s1, s2, T; jitter=T(alg.jitter))

    # Inputs / initial state.
    local B, D, x0
    if alg.new_init
        B = m == 0 ? zeros(T, n, 0) : fill(one(T) / (n * m), n, m)
        x0 = fill(one(T) / n, n)
        D = zeros(T, p, mD)
    else
        Uorig = m > 0 ? reduce(hcat, u_seq) : zeros(T, 0, Ttot)
        Vorig = mD > 0 ? reduce(hcat, v_seq) : zeros(T, 0, Ttot)
        BDest = alg.ridge ? ((x, yy) -> _ridge(x, yy, T(alg.λ))) : ((x, yy) -> x \ yy)
        B, D, x0 = _find_BD_ssid(A, C, Uorig, Vorig, Ydm, zeroD, BDest, T)
    end

    # Initial-state covariance (stationary) and biases.
    P0 = _dlyap(A, Q; jitter=T(alg.jitter), stable=alg.stable)
    b = zeros(T, n)
    d = Vector{T}(ȳ)

    # Write parameters in place (respecting set_all / fit_bool groups).
    fb = lds.fit_bool
    sm = lds.state_model
    om = lds.obs_model
    if alg.set_all || fb[3]
        sm.A .= A
        sm.b .= b
        m > 0 && (sm.B .= B)
    end
    (alg.set_all || fb[4]) && (sm.Q .= Q)
    (alg.set_all || fb[1]) && (sm.x0 .= x0)
    (alg.set_all || fb[2]) && (sm.P0 .= P0)
    if alg.set_all || fb[5]
        om.C .= C
        om.d .= d
        (!zeroD && mD > 0) ? (om.D .= D) : (om.D .= zeros(T, p, mD))
    end
    (alg.set_all || fb[6]) && (om.R .= R)

    validate_LDS(lds)
    return (; S=sv.S, fve=fve)
end

# -----------------------------------------------------------------------------
# `fit!` methods
# -----------------------------------------------------------------------------

"""
    fit!(lds, y, alg::SSID; latent_inputs=nothing, obs_inputs=nothing)

Initialize a Gaussian LTI `LinearDynamicalSystem` in place via Canonical Variate
Analysis subspace identification. `y` may be a vector of per-trial
`(obs_dim, Tᵢ)` matrices, a single `(obs_dim, T)` matrix, or an
`(obs_dim, T, ntrials)` array. Returns `(; S, fve)`.
"""
function fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}},
    alg::SSID;
    latent_inputs::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    obs_inputs::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps_per_trial = [size(yt, 2) for yt in y]
    u_seq = _normalize_multitrial_latent_inputs(
        latent_inputs, lds.state_input_dim, tsteps_per_trial, T, "latent_inputs"
    )
    v_seq = _normalize_multitrial_obs_inputs(
        obs_inputs, lds.obs_input_dim, tsteps_per_trial, T, lds.obs_model
    )
    return _ssid_fit!(lds, y, u_seq, v_seq, alg)
end

function fit!(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T}, alg::SSID; kwargs...
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    if ndims(y) == 3
        return fit!(
            lds, [view(y, :, :, i)::SubArray{T} for i in 1:size(y, 3)], alg; kwargs...
        )
    elseif ndims(y) == 2
        return fit!(lds, [y], alg; kwargs...)
    else
        throw(ArgumentError("SSID: input array y must be 2D or 3D, got ndims=$(ndims(y))."))
    end
end

# Fallback for unsupported model types (Poisson, stitched, …) — narrower Gaussian
# methods above take precedence when applicable.
function fit!(lds::LinearDynamicalSystem, y, alg::SSID; kwargs...)
    throw(
        ArgumentError(
            "SSID initialization currently supports only Gaussian LTI LDS " *
            "(GaussianStateModel + GaussianObservationModel); got " *
            "state=$(typeof(lds.state_model)), obs=$(typeof(lds.obs_model)).",
        ),
    )
end
