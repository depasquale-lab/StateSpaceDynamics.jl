"""
Linear Dynamical System (LDS) implementation with Gaussian state and observation models.
This module defines functions specific to the Gaussian observation model: log-likelihoods,
gradients, Hessians, smoothing, the ELBO/sufficient-statistics machinery, and the EM
fit driver. Model-agnostic helpers (parameter extraction, `initialize_FilterSmooth`)
live in `common.jl`; sampling lives in `simulate.jl`. The code is optimized for
performance, with careful attention to memory allocation and multi-threading.
"""

"""
    joint_loglikelihood!(ws, x, lds, y)

In-place version of `joint_loglikelihood` that uses pre-computed Cholesky factors from
`ws::SmoothWorkspace` and writes into `ws.ll_vec`. Returns the sum of log-likelihoods.
"""
function joint_loglikelihood!(
    ws::SmoothWorkspace{T},
    x::AbstractMatrix{T},
    lds::LinearDynamicalSystem{T0,S,O},
    y::AbstractMatrix{T0},
    u::AbstractMatrix{T0},
    v::AbstractMatrix{T0},
) where {T<:Real,T0<:Real,S<:GaussianStateModel{T0},O<:GaussianObservationModel{T0}}
    tsteps = size(y, 2)

    A = lds.state_model.A
    b = lds.state_model.b
    B = lds.state_model.B
    x0 = lds.state_model.x0
    d = lds.obs_model.d
    D_obs = lds.obs_model.D

    R_U = ws.R_PD[].chol.U
    Q_U = ws.Q_PD[].chol.U
    P0_U = ws.P0_PD[].chol.U

    ll_vec = ws.ll_vec
    temp_dx = ws.temp_dx
    temp_dy = ws.temp_dy
    temp_solve_Q = ws.temp_solve_Q
    temp_solve_R = ws.temp_solve_R

    latent_dim = lds.latent_dim
    obs_dim = lds.obs_dim

    cP0 = -T(0.5) * (T(latent_dim) * log(T(2π)) + logdet(ws.P0_PD[]))
    cQ = -T(0.5) * (T(latent_dim) * log(T(2π)) + logdet(ws.Q_PD[]))
    cR = -T(0.5) * (T(obs_dim) * log(T(2π)) + logdet(ws.R_PD[]))

    for t in 1:tsteps
        ll_t = zero(T)

        # Initial state (t=1): log p(x1)
        if t == 1
            @views temp_dx .= x[:, 1] .- x0
            ldiv!(temp_solve_Q, P0_U, temp_dx)
            ll_t += cP0 - T(0.5) * sum(abs2, temp_solve_Q)
        end

        # Dynamics (t>1): log p(x_t | x_{t-1}, u_{t-1}) where mean = A x_{t-1} + b + B u_{t-1}.
        # `B*u` is a no-op when `u` has zero rows (size(B,2) == size(u,1) == 0).
        if t > 1
            @views mul!(temp_dx, A, x[:, t - 1])
            @views mul!(temp_dx, B, u[:, t - 1], one(T0), one(T0))
            @views temp_dx .= x[:, t] .- temp_dx .- b
            ldiv!(temp_solve_Q, Q_U, temp_dx)
            ll_t += cQ - T(0.5) * sum(abs2, temp_solve_Q)
        end

        # Emission: log p(y_t | x_t, v_t) where mean = C x_t + d + D v_t.
        @views mul!(temp_dy, lds.obs_model.C, x[:, t])
        @views mul!(temp_dy, D_obs, v[:, t], one(T0), one(T0))
        @views temp_dy .= y[:, t] .- temp_dy .- d
        ldiv!(temp_solve_R, R_U, temp_dy)
        ll_t += cR - T(0.5) * sum(abs2, temp_solve_R)

        ll_vec[t] = ll_t
    end

    return ll_vec
end

# Backward-compatible 4-arg overload: no inputs. Forwards to the 6-arg form
# with zero-row u/v matrices, so callers that don't use controls don't have
# to pass them.
function joint_loglikelihood!(
    ws::SmoothWorkspace{T},
    x::AbstractMatrix{T},
    lds::LinearDynamicalSystem{T0,S,O},
    y::AbstractMatrix{T0},
) where {T<:Real,T0<:Real,S<:GaussianStateModel{T0},O<:GaussianObservationModel{T0}}
    tsteps = size(y, 2)
    u = zeros(T0, 0, tsteps)
    v = zeros(T0, 0, tsteps)
    return joint_loglikelihood!(ws, x, lds, y, u, v)
end

"""
    joint_loglikelihood!(ws, x, lds, y)

Compute per-timestep complete-data log-likelihood contributions for a Gaussian LDS:

- `ll[1]` includes: log p(x₁) + log p(y₁ | x₁)
- `ll[t]` for t≥2 includes: log p(x_t | x_{t-1}) + log p(y_t | x_t)

Writes into `ws.ll_vec` and returns it.

Notes:
- Normalization terms (logdet + log(2π)) are included. These are constant w.r.t. `x`,
  but **not** constant across SLDS discrete states when `Q`/`R` differ by state.
"""
function joint_loglikelihood!(
    ll::AbstractVector{T},
    ws::SLDSSmoothWorkspace{T},
    cc::LDSConstantCache{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    @assert length(ll) == tsteps

    A = lds.state_model.A
    b = lds.state_model.b
    x0 = lds.state_model.x0
    C = lds.obs_model.C
    d = lds.obs_model.d

    Q_U = cc.Q_PD[].chol.U
    P0_U = cc.P0_PD[].chol.U
    R_U = cc.R_PD[].chol.U

    dxt = ws.dxt
    dyt = ws.dyt
    tmp = ws.tmp1  # latent_dim work vector (used as transition residual)

    for t in 1:tsteps
        ll_t = zero(T)

        # emission: cR - 0.5*||R^{-1/2}(y_t - Cx_t - d)||^2
        @views mul!(dyt, C, x[:, t])
        @views dyt .= y[:, t] .- dyt .- d
        ldiv!(dyt, R_U, dyt)
        ll_t += cc.cR - T(0.5) * sum(abs2, dyt)

        if t == 1
            # prior: cP0 - 0.5*||P0^{-1/2}(x1 - x0)||^2
            @views dxt .= x[:, 1] .- x0
            ldiv!(dxt, P0_U, dxt)
            ll_t += cc.cP0 - T(0.5) * sum(abs2, dxt)
        else
            # transition: cQ - 0.5*||Q^{-1/2}(x_t - A x_{t-1} - b)||^2
            @views mul!(tmp, A, x[:, t - 1])
            @views tmp .= x[:, t] .- tmp .- b
            ldiv!(tmp, Q_U, tmp)
            ll_t += cc.cQ - T(0.5) * sum(abs2, tmp)
        end

        ll[t] = ll_t
    end

    return ll
end

"""
    Gradient!(ws, lds, y, x)

In-place version of `Gradient` that uses pre-computed Cholesky-derived terms from
`ws::SmoothWorkspace` and writes the result into `ws.grad_buf`.
Returns `ws.grad_buf`.
"""
function Gradient!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
    u::AbstractMatrix{T},
    v::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    latent_dim, tsteps = size(x)
    A = lds.state_model.A
    b = lds.state_model.b
    B = lds.state_model.B
    x0 = lds.state_model.x0
    C = lds.obs_model.C
    d_obs = lds.obs_model.d
    D_obs = lds.obs_model.D

    C_inv_R = ws.C_inv_R
    A_inv_Q = ws.A_inv_Q
    # ws.x_t = -P0^{-1}, ws.xt_given_xt_1 = -Q^{-1}
    neg_P0_inv = ws.x_t         # = -P0^{-1}
    neg_Q_inv = ws.xt_given_xt_1  # = -Q^{-1}

    grad = ws.grad_buf
    dxt = ws.dxt
    dxt_next = ws.dxt_next
    dyt = ws.dyt
    tmp1 = ws.tmp1
    tmp2 = ws.tmp2
    tmp3 = ws.tmp3

    # Helper macros (inlined): residual `x_t - A x_{t-1} - b - B u_{t-1}` and
    # `y_t - C x_t - d - D v_t`. The `B*u` / `D*v` updates are no-ops when
    # `u` / `v` have zero rows.

    # First time step
    @views dxt .= x[:, 1] .- x0
    @views mul!(dxt_next, A, x[:, 1])
    @views mul!(dxt_next, B, u[:, 1], one(T), one(T))
    @views dxt_next .= x[:, 2] .- dxt_next .- b
    @views mul!(dyt, C, x[:, 1])
    @views mul!(dyt, D_obs, v[:, 1], one(T), one(T))
    @views dyt .= y[:, 1] .- dyt .- d_obs

    mul!(tmp1, C_inv_R, dyt)
    mul!(tmp2, A_inv_Q, dxt_next)
    mul!(tmp3, neg_P0_inv, dxt)
    grad[:, 1] .= tmp1 .+ tmp2 .+ tmp3

    # Middle steps
    @views for t in 2:(tsteps - 1)
        mul!(dxt, A, x[:, t - 1])
        mul!(dxt, B, u[:, t - 1], one(T), one(T))
        dxt .= x[:, t] .- dxt .- b

        mul!(dxt_next, A, x[:, t])
        mul!(dxt_next, B, u[:, t], one(T), one(T))
        dxt_next .= x[:, t + 1] .- dxt_next .- b

        mul!(dyt, C, x[:, t])
        mul!(dyt, D_obs, v[:, t], one(T), one(T))
        dyt .= y[:, t] .- dyt .- d_obs

        mul!(tmp1, C_inv_R, dyt)
        mul!(tmp2, A_inv_Q, dxt_next)
        mul!(tmp3, neg_Q_inv, dxt)

        grad[:, t] .= tmp1 .+ tmp3 .+ tmp2
    end

    # Last time step
    @views begin
        mul!(dxt, A, x[:, tsteps - 1])
        mul!(dxt, B, u[:, tsteps - 1], one(T), one(T))
        dxt .= x[:, tsteps] .- dxt .- b
        mul!(dyt, C, x[:, tsteps])
        mul!(dyt, D_obs, v[:, tsteps], one(T), one(T))
        dyt .= y[:, tsteps] .- dyt .- d_obs

        mul!(tmp1, C_inv_R, dyt)
        mul!(tmp3, neg_Q_inv, dxt)

        grad[:, tsteps] .= tmp1 .+ tmp3
    end

    return grad
end

# Backward-compatible no-input overload.
function Gradient!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(x, 2)
    u = zeros(T, 0, tsteps)
    v = zeros(T, 0, tsteps)
    return Gradient!(ws, lds, y, x, u, v)
end

"""
    Hessian!(sws, lds, y, x)

Fill `sws.btd.H_diag`, `H_sub`, `H_super` with the log-likelihood Hessian blocks for
the active trial (length derived from `size(y, 2)`). Returns nothing — the sparse form
is **not** built here because the Newton solver consumes blocks directly.
Workspace buffers may be sized for a longer trial; only the first `tsteps` blocks are
written, which keeps this hot path safe for ragged-length fitting.
"""
function Hessian!(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    return _fill_hessian_blocks!(sws, size(y, 2))
end

# Length-only Hessian assembly. The BT Hessian for a Gaussian LDS is
# observation-independent — its blocks depend only on `A, Q, C, R, P0`
# (already cached in `sws` by `compute_smooth_constants!`) and the trial
# length. Factored out so the equal-length multi-trial fast path can fill
# blocks without constructing a dummy `y` matrix.
function _fill_hessian_blocks!(sws::SmoothWorkspace{T}, tsteps::Int) where {T<:Real}
    btd = sws.btd

    for i in 1:(tsteps - 1)
        copyto!(btd.H_sub[i], sws.H_sub_entry)
        copyto!(btd.H_super[i], sws.H_super_entry)
    end

    btd.H_diag[1] .= sws.yt_given_xt .+ sws.xt1_given_xt .+ sws.x_t
    for i in 2:(tsteps - 1)
        btd.H_diag[i] .= sws.yt_given_xt .+ sws.xt_given_xt_1 .+ sws.xt1_given_xt
    end
    btd.H_diag[tsteps] .= sws.yt_given_xt .+ sws.xt_given_xt_1

    return nothing
end

"""
    smooth(lds, y::AbstractMatrix)

Direct smoothing for a single trial.

# Arguments
- `lds::LinearDynamicalSystem`: The model.
- `y::AbstractMatrix`: Observations (obs_dim × tsteps).

# Returns
- `x_smooth::AbstractMatrix`: Smoothed latent means (latent_dim × tsteps).
- `p_smooth::Array{T,3}`: Smoothed latent covariances (latent_dim × latent_dim × tsteps).
"""
function smooth(lds::LinearDynamicalSystem, y::AbstractMatrix{T}) where {T}
    # Type assertion narrows the union for JET; runtime no-op since dispatch on
    # `size(y, 2)::Int` already lands in the FilterSmooth-returning method.
    fs = initialize_FilterSmooth(lds, size(y, 2))::FilterSmooth{T}
    sws = SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, size(y, 2))
    smooth!(lds, fs, y, sws)
    return fs.x_smooth, fs.p_smooth
end

function smooth(
    lds::LinearDynamicalSystem, y::AbstractVector{<:AbstractMatrix{T}}
) where {T}
    tsteps_per_trial = [size(yt, 2) for yt in y]
    T_max = maximum(tsteps_per_trial)
    tfs = initialize_FilterSmooth(lds, tsteps_per_trial)::TrialFilterSmooth{T}
    sws_pool = [
        SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, T_max) for
        _ in 1:Threads.maxthreadid()
    ]
    smooth!(lds, tfs, y, sws_pool)

    N = length(y)
    xs = Vector{Matrix{T}}(undef, N)
    Ps = Vector{Array{T,3}}(undef, N)
    for n in 1:N
        fs = tfs.FilterSmooths[n]
        xs[n] = copy(fs.x_smooth)
        Ps[n] = copy(fs.p_smooth)
    end
    return xs, Ps
end

"""
    smooth!(lds, fs, y, sws::SmoothWorkspace)

Low-allocation smoothing using `SmoothWorkspace`. Uses a direct single-step
Newton solver (since the Gaussian LDS has a quadratic log-likelihood),
exploiting the block tridiagonal structure of the Hessian for efficient solving.
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    fs::FilterSmooth{T},
    y::AbstractMatrix{T},
    sws::SmoothWorkspace{T},
    u::AbstractMatrix{T},
    v::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps, D = size(y, 2), lds.latent_dim
    n_active = D * tsteps
    btd = sws.btd

    compute_smooth_constants!(sws, lds)

    # Workspace buffers may be sized for a longer trial; take active-length views.
    X0 = view(sws.X₀, 1:n_active)
    grad_vec = view(sws.grad_vec, 1:n_active)
    neg_diag_v = view(btd.neg_diag, 1:tsteps)
    neg_sub_v = view(btd.neg_sub, 1:(tsteps - 1))
    neg_super_v = view(btd.neg_super, 1:(tsteps - 1))

    # Warm-start from previous E[z].
    copyto!(X0, fs.E_z)

    x_mat = reshape(X0, D, tsteps)
    Gradient!(sws, lds, y, x_mat, u, v)
    # grad_vec = -gradient (minimize negative log-likelihood)
    for t in 1:tsteps, i in 1:D
        sws.grad_vec[(t - 1) * D + i] = -sws.grad_buf[i, t]
    end

    # Hessian is independent of `u`/`v` (linear-Gaussian model has identical
    # precision blocks regardless of input means).
    Hessian!(sws, lds, y, x_mat)
    _negate_blocks!(btd, tsteps)

    # Save x_old in fs.x_smooth before we overwrite sws.X₀ with the Newton step.
    fs.x_smooth .= x_mat

    # SPD path: smoother's negated Hessian is PSD at the MAP, and the
    # sub/super blocks are transposes of each other (Hessian is
    # symmetric). At small `latent_dim` (≤ 8) this routes to LAPACK's
    # `pbsv` which is 30-60× faster than the general block-Thomas code.
    block_tridiagonal_solve_spd!(X0, neg_sub_v, neg_diag_v, neg_super_v, grad_vec, btd)

    step_mat = reshape(X0, D, tsteps)
    fs.x_smooth .-= step_mat

    logdet_precision = block_tridiagonal_inverse_logdet!(
        fs.p_smooth, fs.p_smooth_tt1, neg_sub_v, neg_diag_v, neg_super_v, btd
    )

    fs.entropy = gaussian_entropy_from_logdet(logdet_precision, n_active)

    @views for i in 1:tsteps
        Symmetrize!(fs.p_smooth[:, :, i])
    end

    return fs
end

# Backward-compatible no-input overload (zero-row u/v).
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    fs::FilterSmooth{T},
    y::AbstractMatrix{T},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    u = zeros(T, 0, tsteps)
    v = zeros(T, 0, tsteps)
    return smooth!(lds, fs, y, sws, u, v)
end

"""
    smooth!(lds, tfs, y::AbstractVector{<:AbstractMatrix}, sws_pool)

Low-allocation multi-trial smoothing. Each task in `sws_pool` owns one workspace;
trials are partitioned across tasks via `@spawn` / `fetch` (see
https://julialang.org/blog/2023/07/PSA-dont-use-threadid/).

# Arguments
- `lds::LinearDynamicalSystem`
- `tfs::TrialFilterSmooth`: one `FilterSmooth` per trial, sized at each trial's length
- `y::AbstractVector{<:AbstractMatrix}`: one `obs_dim × T_i` matrix per trial
- `sws_pool::Vector{SmoothWorkspace{T}}`: task-local workspaces, each sized at
  `max(T_i)`
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}},
    u_seq::AbstractVector{<:AbstractMatrix{T}},
    v_seq::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    ntrials = length(y)

    if ntrials == 1
        smooth!(lds, tfs[1], y[1], sws_pool[1], u_seq[1], v_seq[1])
        return tfs
    end

    # Equal-length fast path: the BT Hessian (and its inverse) is observation-
    # independent, so the smoothed covariance is identical across trials. Run
    # the cov pass once on `sws_pool[1]`, alias each FilterSmooth's
    # `p_smooth` / `p_smooth_tt1` to the shared storage, then do gradient-and-
    # solve per trial in parallel.
    T1 = size(y[1], 2)
    all_equal = all(yt -> size(yt, 2) == T1, y)

    if all_equal
        # `_precompute_shared_cov!` populates `sws_pool[1]`'s smoothing constants
        # (`R_PD`/`Q_PD`/`P0_PD`/`C_inv_R`/`A_inv_Q`/…), the `btd.neg_*` blocks,
        # and the BT forward-sweep LU cache (`btd.LU_factors`/`LU_ipivs`/`D`).
        # Per-task back-subs and gradient evaluations read from that same
        # workspace (no mutation), so it's safe to share across `@spawn`'d tasks.
        shared_entropy = _precompute_shared_cov!(sws_pool[1], lds, T1)
        source_sws = sws_pool[1]
        for trial in 1:ntrials
            tfs[trial].p_smooth = source_sws.p_smooth_shared
            tfs[trial].p_smooth_tt1 = source_sws.p_smooth_tt1_shared
            tfs[trial].entropy = shared_entropy
        end

        # Batched mean pass: when `sws_pool[1]` was constructed with the right
        # `ntrials`, every per-trial Newton step collapses into a single
        # `(D*T) × N` matrix-RHS backsubst, doing the same total math as the
        # per-trial loop below but with BLAS-3 dispatch (matches the Kalman
        # path's batched-trial efficiency).
        if size(source_sws.batched_x_mat, 3) == ntrials && ntrials > 1
            if !source_sws.batched_data_valid[]
                _populate_batched_data!(source_sws, y, u_seq, v_seq)
            end
            _smooth_mean_only_batched!(lds, tfs, source_sws)
            return tfs
        end

        ntasks = min(ntrials, length(sws_pool))
        chunksize = cld(ntrials, ntasks)
        @sync for i in 1:ntasks
            lo = (i - 1) * chunksize + 1
            hi = min(i * chunksize, ntrials)
            lo > hi && continue
            @spawn begin
                sws = sws_pool[i]
                for trial in lo:hi
                    _smooth_mean_only!(
                        lds,
                        tfs[trial],
                        y[trial],
                        sws,
                        u_seq[trial],
                        v_seq[trial],
                        source_sws,
                    )
                end
            end
        end
        return tfs
    end

    # Variable-length fallback: per-trial smoothing (each trial gets its own
    # Hessian, cov, and mean pass on the assigned worker workspace).
    ntasks = min(ntrials, length(sws_pool))
    chunksize = cld(ntrials, ntasks)

    @sync for i in 1:ntasks
        lo = (i - 1) * chunksize + 1
        hi = min(i * chunksize, ntrials)
        lo > hi && continue
        @spawn begin
            sws = sws_pool[i]
            for trial in lo:hi
                smooth!(lds, tfs[trial], y[trial], sws, u_seq[trial], v_seq[trial])
            end
        end
    end

    return tfs
end

"""
    _precompute_shared_cov!(sws, lds, tsteps)

Fill `sws.p_smooth_shared` and `sws.p_smooth_tt1_shared` with the smoothed
covariance and lag-1 cross-covariance for a single trial of length `tsteps`
(any trial; the result is shared across all equal-length trials). Returns
the per-trial Gaussian entropy contribution `H[q(x_{1:T} | y)]`, which is
identical for every trial because it depends only on the covariances.
"""
function _precompute_shared_cov!(
    sws::SmoothWorkspace{T}, lds::LinearDynamicalSystem{T,S,O}, tsteps::Int
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    D = lds.latent_dim
    btd = sws.btd
    # Hoist `p_smooth_shared` with a concrete eltype so the `Symmetrize!`
    # call below stays in JET's typed union branch (cf. `backwards_cov!`).
    p_smooth_shared = sws.p_smooth_shared::Array{T,3}

    compute_smooth_constants!(sws, lds)
    _fill_hessian_blocks!(sws, tsteps)
    _negate_blocks!(btd, tsteps)

    neg_diag_v = view(btd.neg_diag, 1:tsteps)
    neg_sub_v = view(btd.neg_sub, 1:(tsteps - 1))
    neg_super_v = view(btd.neg_super, 1:(tsteps - 1))

    p_smooth_v = view(p_smooth_shared,:,:,(1:tsteps))
    p_smooth_tt1_v = view((sws.p_smooth_tt1_shared::Array{T,3}),:,:,(1:tsteps))

    logdet_precision = block_tridiagonal_inverse_logdet!(
        p_smooth_v, p_smooth_tt1_v, neg_sub_v, neg_diag_v, neg_super_v, btd
    )

    @views for i in 1:tsteps
        Symmetrize!(p_smooth_shared[:, :, i])
    end

    return gaussian_entropy_from_logdet(logdet_precision, D * tsteps)
end

"""
    _smooth_mean_only!(lds, fs, y, sws, u, v, source_sws)

Per-trial Newton step that **assumes**:

- `fs.p_smooth` / `fs.p_smooth_tt1` are already filled (by
  `_precompute_shared_cov!`),
- `source_sws.btd` contains the **forward-sweep LU cache**, the modified
  upper diagonals `D[i+1]`, and the negated Hessian sub-diagonal blocks
  (also produced by the same `_precompute_shared_cov!` call), and
- `source_sws` itself holds the Cholesky factors and derived gradient
  constants — `compute_smooth_constants!` is **not** called per trial.

Per-task workspaces copy the constants from `source_sws` (cheap fixed-size
`copyto!`s) instead of redoing the Cholesky factorizations. When `sws ===
source_sws` (the task running on the designated workspace), even the copy
is skipped.

Computes the gradient (per-trial), then runs `block_tridiagonal_backsubst!`
against the shared LU cache. No `lu!` and no Cholesky calls happen here —
those are amortized across all equal-length trials in a single E-step.
"""
function _smooth_mean_only!(
    lds::LinearDynamicalSystem{T,S,O},
    fs::FilterSmooth{T},
    y::AbstractMatrix{T},
    sws::SmoothWorkspace{T},
    u::AbstractMatrix{T},
    v::AbstractMatrix{T},
    source_sws::SmoothWorkspace{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps, D = size(y, 2), lds.latent_dim
    n_active = D * tsteps

    # Cholesky factors and derived gradient terms were filled on `source_sws`
    # by `_precompute_shared_cov!` already; just mirror them into the local
    # task workspace. No-op when `sws === source_sws`.
    if sws !== source_sws
        _copy_smooth_constants!(sws, source_sws)
    end

    shared_btd = source_sws.btd
    X0 = view(sws.X₀, 1:n_active)
    grad_vec = view(sws.grad_vec, 1:n_active)
    neg_sub_v = view(shared_btd.neg_sub, 1:(tsteps - 1))

    copyto!(X0, fs.E_z)

    x_mat = reshape(X0, D, tsteps)
    Gradient!(sws, lds, y, x_mat, u, v)
    for t in 1:tsteps, i in 1:D
        sws.grad_vec[(t - 1) * D + i] = -sws.grad_buf[i, t]
    end

    fs.x_smooth .= x_mat
    block_tridiagonal_backsubst!(X0, neg_sub_v, grad_vec, shared_btd, tsteps)
    step_mat = reshape(X0, D, tsteps)
    fs.x_smooth .-= step_mat

    return fs
end

"""
    Gradient_batched!(ws, lds, y_batched, x_batched, u_batched, v_batched)

Batched form of `Gradient!`: every `mul!` is promoted from BLAS-2
(`bs × bs × bs`) to BLAS-3 (`bs × bs × bs × N`) by stacking the trial axis as
the trailing matrix dimension. The shared-cov fast path only ever needs
gradient evaluation at the *current iterate* across all trials, so the work
is structurally identical to N independent per-trial gradients — but BLAS
dispatch overhead is paid once instead of N times.

Writes the result into `ws.batched_grad_buf` (shape `(D, T, N)`); the affine
bias subtractions (`-b`, `-d_obs`, `-x0`) broadcast across the trial axis.
"""
function Gradient_batched!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3},
    x::AbstractArray{T,3},
    u::AbstractArray{T,3},
    v::AbstractArray{T,3},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(x, 2)
    A = lds.state_model.A
    b = lds.state_model.b
    B = lds.state_model.B
    x0 = lds.state_model.x0
    C = lds.obs_model.C
    d_obs = lds.obs_model.d
    D_obs = lds.obs_model.D

    C_inv_R = ws.C_inv_R
    A_inv_Q = ws.A_inv_Q
    neg_P0_inv = ws.x_t            # = -P0^{-1}
    neg_Q_inv = ws.xt_given_xt_1   # = -Q^{-1}

    grad = ws.batched_grad_buf
    dxt = ws.batched_dxt
    dxt_next = ws.batched_dxt_next
    dyt = ws.batched_dyt
    tmp1 = ws.batched_tmp1
    tmp2 = ws.batched_tmp2
    tmp3 = ws.batched_tmp3

    # First time step
    @views begin
        dxt .= x[:, 1, :] .- x0
        mul!(dxt_next, A, x[:, 1, :])
        mul!(dxt_next, B, u[:, 1, :], one(T), one(T))
        dxt_next .= x[:, 2, :] .- dxt_next .- b
        mul!(dyt, C, x[:, 1, :])
        mul!(dyt, D_obs, v[:, 1, :], one(T), one(T))
        dyt .= y[:, 1, :] .- dyt .- d_obs
    end

    mul!(tmp1, C_inv_R, dyt)
    mul!(tmp2, A_inv_Q, dxt_next)
    mul!(tmp3, neg_P0_inv, dxt)
    @views grad[:, 1, :] .= tmp1 .+ tmp2 .+ tmp3

    # Middle steps
    @views for t in 2:(tsteps - 1)
        mul!(dxt, A, x[:, t - 1, :])
        mul!(dxt, B, u[:, t - 1, :], one(T), one(T))
        dxt .= x[:, t, :] .- dxt .- b

        mul!(dxt_next, A, x[:, t, :])
        mul!(dxt_next, B, u[:, t, :], one(T), one(T))
        dxt_next .= x[:, t + 1, :] .- dxt_next .- b

        mul!(dyt, C, x[:, t, :])
        mul!(dyt, D_obs, v[:, t, :], one(T), one(T))
        dyt .= y[:, t, :] .- dyt .- d_obs

        mul!(tmp1, C_inv_R, dyt)
        mul!(tmp2, A_inv_Q, dxt_next)
        mul!(tmp3, neg_Q_inv, dxt)

        grad[:, t, :] .= tmp1 .+ tmp3 .+ tmp2
    end

    # Last time step
    @views begin
        mul!(dxt, A, x[:, tsteps - 1, :])
        mul!(dxt, B, u[:, tsteps - 1, :], one(T), one(T))
        dxt .= x[:, tsteps, :] .- dxt .- b
        mul!(dyt, C, x[:, tsteps, :])
        mul!(dyt, D_obs, v[:, tsteps, :], one(T), one(T))
        dyt .= y[:, tsteps, :] .- dyt .- d_obs

        mul!(tmp1, C_inv_R, dyt)
        mul!(tmp3, neg_Q_inv, dxt)

        grad[:, tsteps, :] .= tmp1 .+ tmp3
    end

    return grad
end

"""
    _populate_batched_data!(sws, y, u, v)

Stack the per-trial `y`/`u`/`v` `Vector{Matrix}` inputs into the contiguous
`(p, T, N)` / `(u_dim, T, N)` / `(d_dim, T, N)` tensors used by the batched
mean pass. Called once per fit (data is constant across EM iterations).
"""
function _populate_batched_data!(
    sws::SmoothWorkspace{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    u::AbstractVector{<:AbstractMatrix{T}},
    v::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real}
    @views for trial in eachindex(y)
        sws.batched_y[:, :, trial] .= y[trial]
    end
    if size(sws.batched_u, 1) > 0
        @views for trial in eachindex(u)
            sws.batched_u[:, :, trial] .= u[trial]
        end
    end
    if size(sws.batched_v, 1) > 0
        @views for trial in eachindex(v)
            sws.batched_v[:, :, trial] .= v[trial]
        end
    end
    sws.batched_data_valid[] = true
    return sws
end

"""
    _smooth_mean_only_batched!(lds, tfs, sws)

Batched form of `_smooth_mean_only!`: runs one Newton step for *all trials at
once* by stacking the per-trial iterate / gradient / RHS into `(D, T, N)`
tensors and performing a single BLAS-3 backsubst.

Assumes `_precompute_shared_cov!` has already populated `sws.btd`'s Cholesky
cache and `sws.batched_data_valid[] == true` (data was stacked at fit entry).
"""
function _smooth_mean_only_batched!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    ntrials = length(tfs)
    D = lds.latent_dim
    tsteps = size(tfs[1].x_smooth, 2)

    # Stage previous-iter smoothed means into the batched iterate buffer.
    @views for trial in 1:ntrials
        sws.batched_x_mat[:, :, trial] .= tfs[trial].E_z
    end

    Gradient_batched!(
        sws, lds, sws.batched_y, sws.batched_x_mat, sws.batched_u, sws.batched_v
    )

    # Pack negated gradient into the (D*T, N) matrix RHS layout.
    n_active = D * tsteps
    grad_flat = reshape(sws.batched_grad_buf, n_active, ntrials)
    x_flat = reshape(sws.batched_x_mat, n_active, ntrials)
    @. grad_flat = -grad_flat

    neg_sub_v = view(sws.btd.neg_sub, 1:(tsteps - 1))
    block_tridiagonal_backsubst!(x_flat, neg_sub_v, grad_flat, sws.btd, tsteps)

    # x_flat now holds the Newton step. Update each tfs[trial].x_smooth.
    @views for trial in 1:ntrials
        tfs[trial].x_smooth .= tfs[trial].E_z .- sws.batched_x_mat[:, :, trial]
    end

    return tfs
end

# Backward-compatible no-input overload.
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    u_seq = [zeros(T, 0, size(yt, 2)) for yt in y]
    v_seq = [zeros(T, 0, size(yt, 2)) for yt in y]
    return smooth!(lds, tfs, y, sws_pool, u_seq, v_seq)
end

"""
    Q_obs!(C, d, E_z, E_zz, y)

Single time-step observation component of the Q-function for
y_t ~ 𝓝(C x_t + d, R), before applying R^{-1} and constants.
"""
function Q_obs!(
    result::AbstractMatrix{T},
    C::AbstractMatrix{T},
    d::AbstractVector{T},
    E_z::AbstractVector{T},
    E_zz::AbstractMatrix{T},
    y::AbstractVector{T},
    buffers,
) where {T<:Real}

    # Unpack buffers
    ytil, sum_yy, sum_yz, work1, work2 = buffers

    # Residualize: ytil = y - d (pre-allocated buffer)
    ytil .= y .- d

    # All operations use pre-allocated buffers
    mul!(sum_yy, ytil, ytil')

    # Efficient outer product: sum_yz = ytil * E_z'
    fill!(sum_yz, zero(T))
    BLAS.ger!(one(T), ytil, E_z, sum_yz)

    # Build result using buffers
    copyto!(result, sum_yy)
    mul!(result, C, sum_yz', -one(T), one(T))   # result -= C * sum_yz'
    mul!(work1, sum_yz, C')                      # work1 = sum_yz * C'  
    result .-= work1                             # result -= work1
    mul!(work2, E_zz, C')                        # work2 = E_zz * C'
    mul!(result, C, work2, one(T), one(T))       # result += C * work2

    return result
end

"""
    Q_obs!(ws, lds, E_z, E_zz, y, v)

Full observation Q-term for Gaussian LDS over all time steps with affine
observation `y_t ~ N(C x_t + d + D v_t, R)`. `v` is the per-trial obs-control
matrix `(d_dim, T_i)`; pass a `0×T_i` matrix when no obs inputs.
"""
function Q_obs!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    y::AbstractMatrix{T},
    v::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    obs_dim = lds.obs_dim
    tsteps = size(y, 2)
    C = lds.obs_model.C
    d = lds.obs_model.d
    D_obs = lds.obs_model.D
    d_dim = size(v, 1)

    R_U = ws.R_PD[].chol.U
    log_det_R = logdet(ws.R_PD[])
    const_term = obs_dim * log(T(2π))

    temp = ws.elbo_obs_temp
    work_matrix = ws.elbo_obs_work
    ytil = ws.elbo_ytil
    sum_yy = ws.elbo_sum_yy
    sum_yz = ws.elbo_sum_yz
    work1 = ws.elbo_obs_work1
    work2 = ws.elbo_obs_work2

    fill!(temp, zero(T))

    @views for t in axes(y, 2)
        # Residualize: ytil = y[:,t] - d - D · v[:,t]
        ytil .= y[:, t] .- d
        if d_dim > 0
            mul!(ytil, D_obs, v[:, t], -one(T), one(T))
        end

        mul!(sum_yy, ytil, ytil')

        fill!(sum_yz, zero(T))
        BLAS.ger!(one(T), ytil, E_z[:, t], sum_yz)

        # work_matrix = sum_yy - C·sum_yz' - sum_yz·C' + C·E_zz·C'
        copyto!(work_matrix, sum_yy)
        mul!(work_matrix, C, sum_yz', -one(T), one(T))
        mul!(work1, sum_yz, C')
        work_matrix .-= work1
        mul!(work2, E_zz[:, :, t], C')
        mul!(work_matrix, C, work2, one(T), one(T))

        temp .+= work_matrix
    end

    ldiv!(R_U', temp)
    ldiv!(R_U, temp)
    return T(-0.5) * (tsteps * (const_term + log_det_R) + tr(temp))
end

# Backward-compatible no-input overload.
function Q_obs!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    y::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    v = zeros(T, 0, size(y, 2))
    return Q_obs!(ws, lds, E_z, E_zz, y, v)
end

"""
    Q_obs!(sws, lds, suf)

Total log-likelihood Q-obs term across all trials and time, computed from
the aggregated sufficient statistics in `suf`. Replaces the per-trial,
per-timestep loop of the legacy `Q_obs!(sws, lds, E_z, E_zz, y, v)`.
"""
function Q_obs!(
    sws::SmoothWorkspace{T}, lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    D = lds.latent_dim
    p = lds.obs_dim
    d_dim = lds.obs_input_dim
    obs_reg_dim = D + 1 + d_dim
    C = lds.obs_model.C
    d = lds.obs_model.d
    D_obs = lds.obs_model.D

    R_U = sws.R_PD[].chol.U
    log_det_R = logdet(sws.R_PD[])
    const_term = p * log(T(2π))

    obs_n = suf.obs_n

    # V = [C d D_obs] (p × obs_reg_dim)
    V = view(sws.CD, :, 1:obs_reg_dim)
    copyto!(view(V, :, 1:D), C)
    copyto!(view(V, :, D + 1), d)
    if d_dim > 0
        copyto!(view(V, :, (D + 2):obs_reg_dim), D_obs)
    end

    # S_obs = obs_yy - V·obs_xy - obs_xy'·V' + V·obs_xx·V'
    S_obs = sws.elbo_obs_temp
    copyto!(S_obs, suf.obs_yy[].mat)
    mul!(S_obs, V, suf.obs_xy, -one(T), one(T))
    mul!(S_obs, transpose(suf.obs_xy), transpose(V), -one(T), one(T))
    V_XX = view(sws.Syz, :, 1:obs_reg_dim)
    mul!(V_XX, V, suf.obs_xx[].mat)
    mul!(S_obs, V_XX, transpose(V), one(T), one(T))

    ldiv!(R_U', S_obs)
    ldiv!(R_U, S_obs)
    return T(-0.5) * (T(obs_n) * (const_term + log_det_R) + tr(S_obs))
end

"""
    calculate_elbo(lds, suf, sws)

Total ELBO from aggregated sufficient statistics. Computes the same quantity
as the legacy `calculate_elbo(lds, tfs, y, sws_pool, ...)` but in
O(D³ + p²·D) instead of O(N·T·p²·D). The Gaussian-posterior entropy comes
from each trial's `fs.entropy` (filled by the smoother) and is summed by
the caller before this function is invoked.
"""
function calculate_elbo(
    lds::LinearDynamicalSystem{T,S,O},
    suf::SufficientStatistics{T},
    sws::SmoothWorkspace{T},
    total_entropy::T,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    Q_total = Q_state!(sws, lds, suf) + Q_obs!(sws, lds, suf)

    prior_term = zero(T)
    if lds.state_model.Q_prior !== nothing
        prior_term += iw_logprior_term(lds.state_model.Q, lds.state_model.Q_prior)
    end
    if lds.state_model.P0_prior !== nothing
        prior_term += iw_logprior_term(lds.state_model.P0, lds.state_model.P0_prior)
    end
    if lds.obs_model.R_prior !== nothing
        prior_term += iw_logprior_term(lds.obs_model.R, lds.obs_model.R_prior)
    end

    # MN-prior log-prior contributions for [A b B] (dynamics) and [C d D] (obs).
    # Required for ELBO monotonicity under MN priors — the M-step's `mn_map`
    # update + the IW posterior scale modification together maximize the
    # MAP objective, but without this term the displayed ELBO drops the
    # MN-quadratic piece and can appear non-monotone.
    if lds.state_model.AB_prior !== nothing
        D = lds.latent_dim
        u_dim = lds.state_input_dim
        W_ab = view(sws.AB, :, 1:(D + 1 + u_dim))
        copyto!(view(W_ab, :, 1:D), lds.state_model.A)
        copyto!(view(W_ab, :, D + 1), lds.state_model.b)
        if u_dim > 0
            copyto!(view(W_ab, :, (D + 2):(D + 1 + u_dim)), lds.state_model.B)
        end
        prior_term += mn_logprior_term(W_ab, lds.state_model.Q, lds.state_model.AB_prior)
    end
    if lds.obs_model.CD_prior !== nothing
        D = lds.latent_dim
        d_dim = lds.obs_input_dim
        W_cd = view(sws.CD, :, 1:(D + 1 + d_dim))
        copyto!(view(W_cd, :, 1:D), lds.obs_model.C)
        copyto!(view(W_cd, :, D + 1), lds.obs_model.d)
        if d_dim > 0
            copyto!(view(W_cd, :, (D + 2):(D + 1 + d_dim)), lds.obs_model.D)
        end
        prior_term += mn_logprior_term(W_cd, lds.obs_model.R, lds.obs_model.CD_prior)
    end

    return Q_total + prior_term + total_entropy
end

function update_C_d!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    lds.fit_bool[5] || return nothing
    D = lds.latent_dim
    d_dim = lds.obs_input_dim
    CD_prior = lds.obs_model.CD_prior

    if CD_prior === nothing
        # Zero-alloc OLS fast path. `sws.Syz` is exactly (p × obs_reg_dim);
        # its transpose is the (obs_reg_dim × p) view we ldiv! into. After
        # the in-place solve, `sws.Syz` itself holds V = [C d D].
        Syz_T = transpose(sws.Syz)
        copyto!(Syz_T, suf.obs_xy)
        ldiv!(suf.obs_xx[].chol, Syz_T)
        V = sws.Syz
    else
        V = mn_map(suf.obs_xx[], suf.obs_xy, CD_prior)
    end

    copyto!(lds.obs_model.C, view(V, :, 1:D))
    copyto!(lds.obs_model.d, view(V, :, D + 1))
    if d_dim > 0
        copyto!(lds.obs_model.D, view(V, :, (D + 2):(D + 1 + d_dim)))
    end
    return nothing
end

function update_R!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    lds.fit_bool[6] || return nothing
    p = lds.obs_dim
    D = lds.latent_dim
    d_dim = lds.obs_input_dim

    # sws.CD is exactly (p × obs_reg_dim); no view needed.
    V = sws.CD
    copyto!(view(V, :, 1:D), lds.obs_model.C)
    copyto!(view(V, :, D + 1), lds.obs_model.d)
    if d_dim > 0
        copyto!(view(V, :, (D + 2):(D + 1 + d_dim)), lds.obs_model.D)
    end

    # Residual scatter S = obs_yy - V·obs_xy - obs_xy'·V' + V·obs_xx·V'
    Vxy = sws.elbo_obs_temp                    # p × p scratch (free post-Q_obs!)
    mul!(Vxy, V, suf.obs_xy)

    S_res = sws.elbo_obs_work                  # p × p scratch
    copyto!(S_res, suf.obs_yy[].mat)
    S_res .-= Vxy
    S_res .-= Vxy'
    # In-place X_A_Xt = V · obs_xx · V'. Mirror PDMats' X_A_Xt: compute
    # `VL = V · L` (obs_xx = L·L' via the cached Cholesky) and add
    # `VL · VL'` to the upper triangle via a symmetric rank-k BLAS call,
    # then reflect upper → lower for exact symmetry. (`mul!` + `Symmetrize!`
    # would halve the off-diagonal contribution because gemm can produce
    # 1-ULP-asymmetric output that averaging then collapses.)
    VL = sws.Syz                               # (p × obs_reg_dim) scratch
    # See `update_Q!`: `BLAS.trmm!` on the raw upper-stored
    # `chol.factors` (with transa='T' since L = U') avoids the
    # `LowerTriangular(...)` wrapper alloc that `mul!(VL, V, chol.L)`
    # would do.
    copyto!(VL, V)
    BLAS.trmm!('R', 'U', 'T', 'N', one(T), suf.obs_xx[].chol.factors, VL)
    mul!(S_res, VL, transpose(VL), one(T), one(T))

    CD_prior = lds.obs_model.CD_prior
    if CD_prior !== nothing
        Wm = V .- CD_prior.M₀
        S_res .+= Wm * CD_prior.Λ * Wm'
    end

    for j in 2:p, i in 1:(j - 1)
        S_res[j, i] = S_res[i, j]
    end

    if lds.obs_model.R_prior === nothing
        S_res ./= T(suf.obs_n)
    else
        Ψ, ν = lds.obs_model.R_prior.Ψ, lds.obs_model.R_prior.ν
        S_res .= iw_map(Ψ, ν, S_res, T(suf.obs_n), p)
    end
    copyto!(lds.obs_model.R, S_res)
    return nothing
end

"""
    mstep!(lds, suf::SufficientStatistics, sws)

Aggregated M-step: runs the six suf-based `update_*!` overloads in sequence
(`x0`, `P0`, `A&b&B`, `Q`, `C&d&D`, `R`). Each respects the corresponding
`lds.fit_bool` flag. The fit hot path in `_fit_tridiag!` calls this once per
EM iteration after `_aggregate_td_suff_stats!`.
"""
function mstep!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    update_initial_state_mean!(lds, suf)
    update_initial_state_covariance!(lds, suf, sws)
    update_A_b!(lds, suf, sws)
    update_Q!(lds, suf, sws)
    update_C_d!(lds, suf, sws)
    update_R!(lds, suf, sws)
    return nothing
end

"""
    fit!(lds, y; max_iter=100, tol=1e-6, progress=true,
         control_seq=nothing, obs_control_seq=nothing)

Fit a Gaussian Linear Dynamical System via Expectation-Maximization.

# Arguments
- `lds::LinearDynamicalSystem{T,S,O}`: model to fit in place
- `y`: observations. Two shapes accepted:
    * `AbstractMatrix{T}` of size `(obs_dim, T)` — single trial
    * `AbstractVector{<:AbstractMatrix{T}}` — multi-trial, each `(obs_dim, T_i)`,
      trial lengths may differ

# Keywords
- `max_iter::Int=100`: maximum EM iterations
- `tol::Float64=1e-6`: convergence tolerance on ELBO change
- `progress::Bool=true`: show progress bar
- `control_seq`: optional dynamics-input sequence. `Vector{<:AbstractMatrix}`
  for multi-trial (each `(u_dim, T_i)`); required when `size(state_model.B, 2) > 0`.
- `obs_control_seq`: optional observation-input sequence (same shape) for the
  obs-side input matrix `D`. Required when `size(obs_model.D, 2) > 0`.

Returns a `Vector{T}` of ELBO values, one per iteration.
"""
function fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}};
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
    control_seq::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    obs_control_seq::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps_per_trial = [size(yt, 2) for yt in y]
    u_seq = _normalize_multitrial_control(
        control_seq, lds.state_input_dim, tsteps_per_trial, T, "control_seq"
    )
    v_seq = _normalize_multitrial_obs_control(
        obs_control_seq, lds.obs_input_dim, tsteps_per_trial, T, lds.obs_model
    )
    return _fit_tridiag!(
        lds,
        y;
        control_seq=u_seq,
        obs_control_seq=v_seq,
        max_iter=max_iter,
        tol=tol,
        progress=progress,
    )
end

function _fit_tridiag!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}};
    control_seq::AbstractVector{<:AbstractMatrix{T}},
    obs_control_seq::AbstractVector{<:AbstractMatrix{T}},
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps_per_trial = [size(yt, 2) for yt in y]
    T_max = maximum(tsteps_per_trial)

    prev_elbo = -T(Inf)
    elbos = Vector{T}()
    sizehint!(elbos, max_iter)

    # Opt in to the cov-alias stub for `p_smooth` / `p_smooth_tt1` when the
    # cov-cache fast path is going to fire (equal-length multi-trial). The
    # smoother aliases them to shared storage on every E-step, so per-trial
    # allocations of `(D, D, T)` are pure waste at large `N`.
    ntrials_total = length(y)
    cov_alias = ntrials_total > 1 && all(t -> t == tsteps_per_trial[1], tsteps_per_trial)
    tfs = initialize_FilterSmooth(
        lds, tsteps_per_trial; cov_alias=cov_alias
    )::TrialFilterSmooth{T}

    u_dim = lds.state_input_dim
    d_dim = lds.obs_input_dim
    # Only `sws_pool[1]` needs the batched mean-pass buffers (used by the
    # equal-length cov-cache fast path); the other workspaces back the
    # per-trial fallback / @spawn'd tasks and stay at `ntrials = 1`.
    pool_size = Threads.maxthreadid()
    sws_pool = Vector{SmoothWorkspace{T}}(undef, pool_size)
    sws_pool[1] = SmoothWorkspace(
        T,
        lds.latent_dim,
        lds.obs_dim,
        T_max;
        u_dim=u_dim,
        d_dim=d_dim,
        ntrials=ntrials_total,
    )
    for i in 2:pool_size
        sws_pool[i] = SmoothWorkspace(
            T, lds.latent_dim, lds.obs_dim, T_max; u_dim=u_dim, d_dim=d_dim
        )
    end

    # Sufficient-statistics aggregator: allocated once, mutated each E-step.
    # Data-only constants (Σ y y', Σ y, Σ u u' …) are precomputed here once
    # and reused across iterations.
    suf = _initialize_td_sufficient_statistics(T, lds, tsteps_per_trial)
    _td_init_const_blocks!(
        sws_pool[1], lds, tsteps_per_trial, y, control_seq, obs_control_seq
    )

    prog = if progress
        Progress(max_iter; desc="Fitting LDS via EM...", barlen=50, showspeed=true)
    else
        nothing
    end

    for _ in 1:max_iter
        # E-step: smooth + aggregate sufficient stats from x_smooth / p_smooth /
        # p_smooth_tt1. We skip the legacy `sufficient_statistics!(tfs)` call —
        # the new aggregator reads smoother outputs directly, so the per-trial
        # E_zz / E_zz_prev arrays are no longer populated on the hot path.
        smooth!(lds, tfs, y, sws_pool, control_seq, obs_control_seq)
        _aggregate_td_suff_stats!(
            suf, tfs, lds, control_seq, obs_control_seq, y, sws_pool[1]
        )

        # ELBO uses the same Cholesky factors as the smoother just filled.
        total_entropy = zero(T)
        for fs in tfs.FilterSmooths
            total_entropy += fs.entropy
        end
        elbo = calculate_elbo(lds, suf, sws_pool[1], total_entropy)
        push!(elbos, elbo)

        # M-step: regression + IW MAP from the aggregated stats. No tfs needed.
        mstep!(lds, suf, sws_pool[1])

        prog !== nothing && next!(prog)

        if abs(elbo - prev_elbo) < tol
            prog !== nothing && finish!(prog)
            return elbos
        end
        prev_elbo = elbo
    end

    prog !== nothing && finish!(prog)
    return elbos
end

function fit!(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T}; kwargs...
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    # reshape y from [obs_dim, tsteps, trials] to vector of matrices if needed
    if ndims(y) == 3
        obs_dim, tsteps, ntrials = size(y)
        y_vec = [view(y,:,:,i) for i in 1:ntrials]
        return fit!(lds, y_vec; kwargs...)
    elseif ndims(y) == 2
        # single trial case, wrap in vector
        return fit!(lds, [y]; kwargs...)
    else
        throw(ArgumentError("Input array y must be 2D or 3D."))
    end
end

function Gradient(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    ws = SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, tsteps)
    compute_smooth_constants!(ws, lds)
    return copy(Gradient!(ws, lds, y, x))
end

function Gradient(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    compute_smooth_constants!(ws, lds)
    return copy(Gradient!(ws, lds, y, x))
end

function Hessian(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    ws = SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, tsteps)
    compute_smooth_constants!(ws, lds)
    Hessian!(ws, lds, y, x)
    block_tridgm!(ws.btd)
    return copy(ws.btd.H_sparse)
end

function Hessian(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    compute_smooth_constants!(ws, lds)
    Hessian!(ws, lds, y, x)
    block_tridgm!(ws.btd)
    return copy(ws.btd.H_sparse)
end

function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    T_max = maximum(size(yt, 2) for yt in y)
    npool = Threads.maxthreadid()
    sws_pool = [SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, T_max) for _ in 1:npool]
    return smooth!(lds, tfs, y, sws_pool)
end

function joint_loglikelihood(
    x::AbstractMatrix{XT}, lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{YT}
) where {T<:Real,YT<:Real,XT<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    WT = promote_type(T, YT, XT)
    ws = SmoothWorkspace(WT, lds.latent_dim, lds.obs_dim, tsteps)
    compute_smooth_constants!(ws, lds)
    return joint_loglikelihood!(ws, x, lds, y)
end

"""
    loglikelihood(lds, y)

Marginal (observed-data) log-likelihood `∑_{t,n} log p(y_t^n | y_{1:t-1}^n)` of a
Gaussian `LinearDynamicalSystem`, computed by running the Kalman filter and summing
the one-step-ahead predictive densities (latent states integrated out). Valid for
any fitted model with a `GaussianObservationModel`.

This is the `StatsAPI.loglikelihood` method for the LDS; for the complete-data
log-likelihood `log p(x, y)` given a trajectory `x`, see `joint_loglikelihood`.

Returns the **total** log-likelihood. Divide by `obs_dim * tsteps * ntrials` for a
per-observation score that is comparable across configurations.
"""
function loglikelihood(
    lds::LinearDynamicalSystem{T,SM,OM}, y::AbstractArray{T,3}
) where {T<:Real,SM<:GaussianStateModel{T},OM<:GaussianObservationModel{T}}
    A = lds.state_model.A
    Q = lds.state_model.Q
    b = lds.state_model.b
    x0 = lds.state_model.x0
    P0 = lds.state_model.P0
    C = lds.obs_model.C
    R = lds.obs_model.R
    d = lds.obs_model.d

    obs_dim, tsteps, ntrials = size(y)
    D = lds.latent_dim

    total_ll = zero(T)
    log2πp = T(obs_dim) * log(T(2π))

    # Pre-allocate buffers (reused across trials and time steps)
    x_p = Vector{T}(undef, D)
    x_f = Vector{T}(undef, D)
    P_p = Matrix{T}(undef, D, D)
    P_f = Matrix{T}(undef, D, D)
    tmp_DD = Matrix{T}(undef, D, D)
    innov = Vector{T}(undef, obs_dim)
    Si_e = Vector{T}(undef, obs_dim)
    Smat = Matrix{T}(undef, obs_dim, obs_dim)
    PCt = Matrix{T}(undef, D, obs_dim)
    SiPCt = Matrix{T}(undef, obs_dim, D)

    for n in 1:ntrials
        x_f .= x0
        P_f .= P0

        for t in 1:tsteps
            # Prediction (t=1: prediction == prior)
            if t == 1
                x_p .= x0
                P_p .= P0
            else
                mul!(x_p, A, x_f)
                x_p .+= b
                mul!(tmp_DD, A, P_f)
                mul!(P_p, tmp_DD, A')
                P_p .+= Q
                Symmetrize!(P_p)
            end

            # Innovation: e = y_t - C x_p - d
            mul!(innov, C, x_p)
            @views innov .= y[:, t, n] .- innov .- d

            # Innovation covariance: S = C P_p C' + R
            mul!(PCt, P_p, C')
            mul!(Smat, C, PCt)
            Smat .+= R
            Symmetrize!(Smat)

            # One-step predictive log-likelihood: log N(e; 0, S)
            S_ch = cholesky(Hermitian(Smat))
            Si_e .= innov
            ldiv!(S_ch, Si_e)           # Si_e ← S⁻¹ e
            total_ll -= T(0.5) * (log2πp + logdet(S_ch) + dot(innov, Si_e))

            # Update: x_f = x_p + K e  where K = PCt S⁻¹
            mul!(x_f, PCt, Si_e)        # x_f = PCt (S⁻¹ e)
            x_f .+= x_p

            # P_f = P_p - PCt S⁻¹ PCt'
            SiPCt .= PCt'
            ldiv!(S_ch, SiPCt)          # SiPCt ← S⁻¹ PCt'
            mul!(P_f, PCt, SiPCt)
            P_f .= P_p .- P_f
            Symmetrize!(P_f)
        end
    end

    return total_ll
end

# Alternative observation methods
# vector of matrices (e.g., for ragged multi-trial observation)
function loglikelihood(
    lds::LinearDynamicalSystem{T,SM,OM}, y::AbstractVector{<:AbstractMatrix{T}}
) where {T<:Real,SM<:GaussianStateModel{T},OM<:GaussianObservationModel{T}}
    y_comb = cat(y...; dims=3)
    return loglikelihood(lds, y_comb)
end

# single-trial observation (Matrix)
function loglikelihood(
    lds::LinearDynamicalSystem{T,SM,OM}, y::AbstractMatrix{T}
) where {T<:Real,SM<:GaussianStateModel{T},OM<:GaussianObservationModel{T}}
    y_comb = reshape(y, size(y, 1), size(y, 2), 1)  # add singleton trial dimension if missing
    return loglikelihood(lds, y_comb)
end
