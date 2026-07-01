# =============================================================================
# Information-form Kalman filter + RTS smoother for Gaussian LDS.
#
# Retained for the marginal log-likelihood (`loglikelihood(lds, y)`) and future
# particle-filter use. No longer wired into `fit!` as a selectable E-step
# backend — all Gaussian fitting goes through the block-tridiagonal path.
# Ported from StateSpaceAnalysis.
#
# Key design choices:
#
# * The covariance forward-backward pass does **not** depend on observations.
#   It is run **once per E-step** and its output (predicted/filtered/smoothed
#   covariances, gains, and per-step entropy contribution) is shared across all
#   trials. This is the main speedup over the block-tridiagonal path when
#   `ntrials` is large.
#
# * Optional input support: `x_{t+1} = A·x_t + b + B·ux_t + ε` and
#   `x_1 ~ N(x0, P0)`. Dynamics-input support activated when `state_model.B`
#   is supplied; the per-step control sequence flows in through kwarg `u`.
#
# * Covariances are stored as `PDMat` so Cholesky factors stay cached; all
#   stabilization goes through `tol_PD` (eigen-floor) rather than
#   `make_posdef!` / `stabilize_covariance_matrix` (absolute floor), which
#   preserves conditioning when the covariance has large dynamic range.
#
# * Reference-sharing: each trial's `FilterSmooth.p_smooth` /
#   `FilterSmooth.p_smooth_tt1` is set to alias the shared 3-D arrays in
#   `KalmanWorkspace`. `sufficient_statistics!` and `Q_state!`/`Q_obs!` only
#   **read** these fields, so aliasing is safe — but mutating through any one
#   trial's view would corrupt all trials.
# =============================================================================

"""
    _fit_kalman!(lds, y; control_seq, obs_control_seq, max_iter, tol, progress)

Kalman-path EM driver. Retained but no longer invoked by `fit!` (the Kalman
E-step backend was removed); kept for reference / future use. `control_seq`
carries dynamics inputs (`B*ux_t`),
`obs_control_seq` carries observation inputs (`D*uy_t`). Both are 3-D arrays
`(input_dim, tsteps, ntrials)` or `nothing` for no inputs.
"""
function _fit_kalman!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3};
    control_seq::Union{Nothing,AbstractArray{T,3}}=nothing,
    obs_control_seq::Union{Nothing,AbstractArray{T,3}}=nothing,
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
    monotonicity_check::Bool=true,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    eltype(y) === T || error("Observed data must be of type $(T); got $(eltype(y))")

    tsteps = size(y, 2)
    ntrials = size(y, 3)

    # format inputs and preallocate workspace + sufficient statistics
    data = format_kf_data!(lds, y, control_seq, obs_control_seq, tsteps, ntrials)
    validate_kalman_inputs(lds, data, ntrials, tsteps)
    kws = KalmanWorkspace(lds, tsteps, ntrials)
    suf = initialize_SufficientStatistics(lds, data, kws) # reset sufficient statistics

    # preallocate elbo
    prev_elbo = -T(Inf)
    elbos = Vector{T}()
    sizehint!(elbos, max_iter)

    prog = if progress
        Progress(max_iter; desc="Fitting LDS (Kalman) via EM...", barlen=50, showspeed=true)
    else
        nothing
    end

    for iter in 1:max_iter
        estep!(lds, suf, kws, data)
        # elbo = marginal_loglikelihood(lds, kws)
        # elbo = compute_elbo(lds, suf, kws)        
        mstep!(lds, suf, kws)
        elbo = compute_elbo(lds, suf, kws)

        # report progress
        push!(elbos, elbo)
        progress && prog !== nothing && next!(prog)

        if monotonicity_check && (elbo - prev_elbo) < 0
            @warn "ELBO decreased from $(prev_elbo) to $(elbo) at iteration $(iter); this should not happen with a correct implementation. Consider reducing `tol` or checking for numerical issues."
        elseif (elbo - prev_elbo) < tol && (elbo - prev_elbo) > 0
            progress && prog !== nothing && finish!(prog)
            return elbos
        end
        prev_elbo = elbo
    end
    progress && prog !== nothing && finish!(prog)
    return elbos
end

function format_kf_data!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3},
    ux::Union{Nothing,AbstractArray{T,3}},
    uy::Union{Nothing,AbstractArray{T,3}},
    tsteps::Int,
    ntrials::Int,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}

    # `data.ux` / `data.uy` carry user-supplied controls only. When the user
    # doesn't supply them, we use 0-column arrays — there's no longer any
    # folding of `b` / `obs_model.d` into `B` / `D`. The bias terms are
    # added explicitly in `precompute_kalman_constants!` and the regression
    # in `mstep!` fits them via a dedicated constant-1 column in the Gram
    # matrix.
    ux_formatted = ux === nothing ? zeros(T, 0, tsteps, ntrials) : ux
    uy_formatted = uy === nothing ? zeros(T, 0, tsteps, ntrials) : uy

    return Data(; y=y, ux=ux_formatted, uy=uy_formatted)
end

@views function initialize_SufficientStatistics(
    model::LinearDynamicalSystem{T,S,O}, data::Data{T}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    latent_dim = model.latent_dim
    obs_dim = model.obs_dim
    tsteps = size(data.y, 2)
    ntrials = size(data.y, 3)
    ux_dim = model.state_input_dim     # user-supplied input dim (0 when no controls)
    uy_dim = model.obs_input_dim       # user-supplied obs-control dim

    # Regression-augmented column counts: D + 1 (bias) + user_input_dim.
    dyn_reg_dim = latent_dim + 1 + ux_dim
    obs_reg_dim = latent_dim + 1 + uy_dim

    y_wide = reshape(data.y, size(data.y, 1), size(data.y, 2) * size(data.y, 3))
    ux_wide = reshape(
        data.ux[:, 1:(end - 1), :], ux_dim, (size(data.ux, 2) - 1) * size(data.ux, 3)
    )
    uy_wide = reshape(data.uy, uy_dim, size(data.uy, 2) * size(data.uy, 3))

    # Matrix{T}(I, dim, dim) is fully type-stable; `diagm(ones(T, dim))`
    # leaves JET seeing a `Union{Array{T,3}, Matrix}` from `diagm`'s signature.
    PD_init(T, dim) = PDMat(Matrix{T}(I, dim, dim))

    # initial conditions — `x0` is estimated as a per-trial mean.
    # The 1×1 `init_xx` scaffold stores `ntrials` so `regress`/`est_cov` still
    # express the math as a regression of `x_init` on a constant.
    init_xx = fill(T(ntrials), 1, 1)

    # Dynamics Gram matrix layout, columns/rows:
    #   1:D                 — x_prev (filled per E-step)
    #   D+1                 — constant-1 column (for the bias `b`)
    #   D+2 : D+1+ux_dim     — user input ux_prev (filled with constants below)
    #
    # Of these, only the bias and ux_prev blocks are observation-independent
    # and so can be filled once at init. The cross blocks involving x_prev
    # are overwritten each iteration in `sufficient_statistics!`.
    dyn_n_T = T((tsteps - 1) * ntrials)
    fill!(kws.dyn_xx_buf, zero(T))
    kws.dyn_xx_buf[latent_dim + 1, latent_dim + 1] = dyn_n_T            # 1ᵀ 1
    if ux_dim > 0
        # 1ᵀ u and uᵀ u blocks
        sum_ux = sum(ux_wide; dims=2)                                     # ux_dim × 1
        kws.dyn_xx_buf[latent_dim + 1, (latent_dim + 2):end] .= vec(sum_ux)
        kws.dyn_xx_buf[(latent_dim + 2):end, latent_dim + 1] .= vec(sum_ux)
        kws.dyn_xx_buf[(latent_dim + 2):end, (latent_dim + 2):end] .=
            tol_PD(ux_wide * ux_wide').mat
    end

    # Observation Gram matrix layout (same pattern as dynamics).
    obs_n_T = T(tsteps * ntrials)
    fill!(kws.obs_xx_buf, zero(T))
    kws.obs_xx_buf[latent_dim + 1, latent_dim + 1] = obs_n_T            # 1ᵀ 1
    if uy_dim > 0
        sum_d = sum(uy_wide; dims=2)                                     # uy_dim × 1
        kws.obs_xx_buf[latent_dim + 1, (latent_dim + 2):end] .= vec(sum_d)
        kws.obs_xx_buf[(latent_dim + 2):end, latent_dim + 1] .= vec(sum_d)
        kws.obs_xx_buf[(latent_dim + 2):end, (latent_dim + 2):end] .=
            tol_PD(uy_wide * uy_wide').mat
    end

    # obs_xy: regress y on [x; 1; d]. The constant and d rows are observation-
    # independent (depend only on data.y and data.uy), so they can be filled
    # once at init.
    obs_xy = zeros(T, obs_reg_dim, obs_dim)
    sum_y = sum(y_wide; dims=2)                                          # obs_dim × 1
    obs_xy[latent_dim + 1, :] .= vec(sum_y)
    if uy_dim > 0
        mul!(obs_xy[(latent_dim + 2):end, :], uy_wide, y_wide', one(T), zero(T))
    end

    obs_yy = zeros(T, obs_dim, obs_dim)
    mul!(obs_yy, y_wide, y_wide', one(T), zero(T))

    # Seed PDMat refs with valid initial values; sufficient_statistics! rewraps
    # them each E-step from the workspace buffers.
    init_dyn_xx = copy(kws.dyn_xx_buf)
    init_dyn_xx[1:latent_dim, 1:latent_dim] .= Matrix{T}(I, latent_dim, latent_dim)
    init_obs_xx = copy(kws.obs_xx_buf)
    init_obs_xx[1:latent_dim, 1:latent_dim] .= Matrix{T}(I, latent_dim, latent_dim)

    return SufficientStatistics{T}(
        # initial conditions
        T(ntrials),                                     # init_n
        _pd_ref(tol_PD(init_xx)),                           # init_xx (1×1, holds ntrials)
        zeros(T, 1, latent_dim),                        # init_xy (1×D, holds Σ x_init)
        _pd_ref(PD_init(T, latent_dim)),                    # init_yy

        # dynamics model
        T((tsteps - 1) * ntrials),                      # dyn_n
        _pd_ref(PDMat(init_dyn_xx)),                        # dyn_xx
        zeros(T, dyn_reg_dim, latent_dim),              # dyn_xy
        _pd_ref(PD_init(T, latent_dim)),                    # dyn_yy

        # observation model
        T(tsteps * ntrials),                            # obs_n
        _pd_ref(PDMat(init_obs_xx)),                        # obs_xx
        obs_xy,                                         # obs_xy
        _pd_ref(tol_PD(obs_yy)),                            # obs_yy
    )
end

# ==== E-STEP =============================================================================

"""
    estep!(lds, suf, kws::KalmanWorkspace, data::Data)

Kalman-path E-step. Runs `smooth!` via the Kalman/RTS backend, computes
sufficient statistics, and calls `calculate_elbo` using the companion
`SmoothWorkspace` pool (which still owns the Cholesky buffers used by
`Q_state!`/`Q_obs!`).
"""
function estep!(
    lds::LinearDynamicalSystem{T,S,O},
    suf::SufficientStatistics{T},
    kws::KalmanWorkspace{T},
    data::Data{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    precompute_kalman_constants!(kws, lds, data)

    smooth_cov!(lds, kws)

    smooth_mean!(lds, kws)

    return sufficient_statistics!(suf, kws, data)
end

"""
    validate_kalman_inputs(lds, data, ntrials, tsteps)

Validate input/parameter dimensional consistency for the Kalman path. Called
**once** at fit entry (not per E-step). Throws `DimensionMismatchError` or
`ArgumentError` on any mismatch between `B`, `D` and the supplied `u`,
`d` arrays in `data`.
"""
function validate_kalman_inputs(
    lds::LinearDynamicalSystem{T,S,O}, data::Data{T}, ntrials::Int, tsteps::Int
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    # Dynamics input matrix `B`: zero-column means "no inputs", so `data.ux`
    # must also have zero rows. Non-zero columns require `data.ux` of matching
    # shape `(ux_dim, tsteps, ntrials)`.
    ux_dim = size(lds.state_model.B, 2)
    if size(data.ux, 1) != ux_dim
        throw(DimensionMismatchError("ux rows vs B cols", ux_dim, size(data.ux, 1)))
    end
    if ux_dim > 0 && (size(data.ux, 2) != tsteps || size(data.ux, 3) != ntrials)
        throw(
            DimensionMismatchError(
                "ux shape (ux_dim, T, ntrials)", (ux_dim, tsteps, ntrials), size(data.ux)
            ),
        )
    end

    uy_dim = size(lds.obs_model.D, 2)
    if size(data.uy, 1) != uy_dim
        throw(DimensionMismatchError("d rows vs D cols", uy_dim, size(data.uy, 1)))
    end
    if uy_dim > 0 && (size(data.uy, 2) != tsteps || size(data.uy, 3) != ntrials)
        throw(
            DimensionMismatchError(
                "d shape (uy_dim, T, ntrials)", (uy_dim, tsteps, ntrials), size(data.uy)
            ),
        )
    end

    return nothing
end

"""
    precompute_kalman_constants!(kws::KalmanWorkspace, lds; tol=1e-6)

Refresh the cached `PDMat` wrappers of `Q`, `R`, `P0` (applying `tol_PD`) and
the derived constants `CiR = C' R^{-1}` (D × p) and `CiRC = C' R^{-1} C` (D × D).
Called once at the start of each E-step. Assumes inputs have already been
validated by `validate_kalman_inputs` at fit entry.
"""
function precompute_kalman_constants!(
    kws::KalmanWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T};
    tol::Real=T(1e-6),
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    kws.P0_PD[] = tol_PD(lds.state_model.P0; tol=tol)
    kws.Q_PD[] = tol_PD(lds.state_model.Q; tol=tol)
    kws.R_PD[] = tol_PD(lds.obs_model.R; tol=tol)

    C = lds.obs_model.C
    copyto!(kws.CiR, C' / kws.R_PD[])
    kws.CiRC[] = tol_PD(Xt_invA_X(kws.R_PD[], C))

    # Initial-state mean is `x0` directly (no `B0·u0` term). Broadcast it
    # into each trial column of `pred_mean[:, 1, :]`.
    @views for n in axes(kws.pred_mean, 3)
        kws.pred_mean[:, 1, n] .= lds.state_model.x0
    end

    # Dynamics offset per timestep/trial: `Bu[:, t, n] = b + B · u[:, t, n]`.
    # The `b` term is present always; `B · u` is added only when user inputs
    # are supplied (state_input_dim > 0). `forwards_mean!` consumes `Bu` as
    # the pre-filled non-`A` part of pred_mean[:, t+1, :].
    b = lds.state_model.b
    @views for n in axes(kws.Bu, 3), t in axes(kws.Bu, 2)
        kws.Bu[:, t, n] .= b
    end
    if kws.state_input_dim > 0
        B = lds.state_model.B
        @views for n in axes(data.ux, 3)
            mul!(kws.Bu[:, :, n], B, data.ux[:, :, n], one(T), one(T))  # Bu += B·u
        end
    end

    # Observation offset: `y_minus_d[:, t, n] = y[:, t, n] - d - D · data_d[:, t, n]`.
    obs_d = lds.obs_model.d
    @views for n in axes(kws.y_minus_d, 3), t in axes(kws.y_minus_d, 2)
        kws.y_minus_d[:, t, n] .= data.y[:, t, n] .- obs_d
    end
    if kws.obs_input_dim > 0
        D = lds.obs_model.D
        @views for n in axes(data.uy, 3)
            mul!(kws.y_minus_d[:, :, n], D, data.uy[:, :, n], -one(T), one(T))
        end
    end

    @views for n in axes(data.y, 3)
        mul!(kws.CiRY[:, :, n], kws.CiR, kws.y_minus_d[:, :, n])
    end
    return kws
end

# ==== SMOOTH COVARIANCE =============================================================================

function smooth_cov!(
    lds::LinearDynamicalSystem{T,S,O}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    @inline forwards_cov!(lds, kws)

    @inline backwards_cov!(lds, kws)
end

@views function forwards_cov!(
    lds::LinearDynamicalSystem{T,S,O}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    # tt = 1: pred_cov[1] is just the (toleranced) P0; pred_icov[1] is its
    # inverse; filt_cov[1] = inv(inv(P0) + CiRC) via the in-place info_update.
    kws.pred_cov[1] = kws.P0_PD[]
    kws.pred_icov[1] = inv(kws.pred_cov[1])
    info_update!(kws.filt_cov[1], kws.cov_tmp2, kws.pred_cov[1], kws.CiRC[])

    for tt in eachindex(kws.filt_cov)[2:end]
        # pred_cov[tt] = A·filt_cov[tt-1]·A' + Q
        kws.cov_tmp1 .= X_A_Xt(kws.filt_cov[tt - 1], lds.state_model.A)
        kws.cov_tmp1 .+= kws.Q_PD[].mat
        Symmetrize!(kws.cov_tmp1)
        kws.pred_cov[tt] = PDMat(kws.cov_tmp1)
        kws.pred_icov[tt] = inv(kws.pred_cov[tt])

        # filt_cov[tt] = inv(inv(pred_cov[tt]) + CiRC), in place via cached Cholesky.
        info_update!(kws.filt_cov[tt], kws.cov_tmp2, kws.pred_cov[tt], kws.CiRC[])
    end

    return kws
end

function backwards_cov!(
    lds::LinearDynamicalSystem{T,S,O}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    # Hoist workspace fields into locals with concrete element types so
    # `@views` indexing below stays inside the typed branch of JET's
    # union-split analysis. `KalmanWorkspace{T}` declares these fields as
    # `Vector{PDMat{T,Matrix{T}}}` / `Array{T,3}`, but JET can't propagate
    # `T` through `maybeview` without the assertion.
    smooth_cov = kws.smooth_cov::Vector{PDMat{T,Matrix{T}}}
    filt_cov = kws.filt_cov::Vector{PDMat{T,Matrix{T}}}
    pred_cov = kws.pred_cov::Vector{PDMat{T,Matrix{T}}}
    Q_PD = kws.Q_PD::Base.RefValue{PDMat{T,Matrix{T}}}
    G = kws.G::Array{T,3}
    A = lds.state_model.A

    # init smoothed cov & accumulators
    smooth_cov[end] = filt_cov[end]
    kws.sum_smooth_cov_all .= smooth_cov[end].mat
    kws.sum_smooth_cov_prev .= zeros(T, kws.latent_dim, kws.latent_dim)
    kws.sum_smooth_cov_next .= smooth_cov[end].mat
    kws.sum_smooth_xcov .= zeros(T, kws.latent_dim, kws.latent_dim)

    # smooth covariance + joint Gaussian entropy ================================
    # H(x_{1:T}|y) = H(x_T|y) + Σ_{t=1}^{T-1} H(x_t | x_{t+1}, y)
    # backward-conditional cov: Σ_{t|x_{t+1},y} = filt_cov[t] - G[t]*pred_cov[t+1]*G[t]'
    #                                             = filt_cov[t] - filt_cov[t]*(G[t]*A)'
    ent_logdet = logdet(smooth_cov[end].mat)

    @views for tt in eachindex(filt_cov)[(end - 1):-1:1]

        # reverse kalman gain G[t] = filt_cov[t] · A' · pred_cov[t+1]^{-1}
        mul!(G[:, :, tt], filt_cov[tt], A', one(T), zero(T))
        G[:, :, tt] /= pred_cov[tt + 1]

        # smoothed covariance — Joseph-style form, algebraically equivalent to
        # the standard RTS update `P_s[t] = P_f[t] + G[t]·(P_s[t+1] - P_p[t+1])·G[t]'`
        # but expanded so every term is manifestly PSD:
        #
        #   smooth_cov[t] = G·(smooth_cov[t+1] + Q)·G' + (I - G·A)·filt_cov[t]·(I - G·A)'
        #
        # The two summands are each `X·A·X'` of a PD matrix, so their sum is
        # symmetric and PSD up to floating-point roundoff. We Symmetrize!
        # defensively before wrapping as a PDMat — an earlier attempt that
        # built `smooth_cov[t+1] + Q` into a separate `Ref{PDMat}` first
        # tripped the strict cholesky in `PDMat(::Matrix)` from accumulated
        # asymmetry; folding the sum into the X_A_Xt argument avoids that.
        mul!(kws.cov_tmp1, G[:, :, tt], A, one(T), zero(T))
        kws.cov_tmp2 .=
            X_A_Xt(smooth_cov[tt + 1] + Q_PD[], G[:, :, tt]) .+
            X_A_Xt(filt_cov[tt], I - kws.cov_tmp1)
        Symmetrize!(kws.cov_tmp2)
        smooth_cov[tt] = PDMat(kws.cov_tmp2)

        # accumulate smoothed covs
        kws.sum_smooth_cov_all .+= smooth_cov[tt].mat
        kws.sum_smooth_cov_prev .+= smooth_cov[tt].mat
        if tt > 1
            kws.sum_smooth_cov_next .+= smooth_cov[tt].mat
        end

        mul!(kws.sum_smooth_xcov, G[:, :, tt], smooth_cov[tt + 1].mat, one(T), one(T))

        # entropy contribution of backward-conditional cov
        ent_logdet += logdet(filt_cov[tt]) .- logdet(pred_cov[tt + 1])
    end

    kws.shared_entropy[] =
        T(0.5) * (
            kws.tsteps * kws.latent_dim * (one(T) + log(T(2π))) +
            ent_logdet +
            (kws.tsteps - 1) * logdet(Q_PD[])
        )

    return kws
end

# ==== SMOOTH MEAN =============================================================================

function smooth_mean!(
    lds::LinearDynamicalSystem{T,S,O}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    forwards_mean!(lds, kws)
    return backwards_mean!(kws)
end

function forwards_mean!(
    lds::LinearDynamicalSystem{T,S,O}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    @views begin
        # filter initial mean
        mul!(kws.mean_tmp, kws.pred_icov[1], kws.pred_mean[:, 1, :], one(T), zero(T))
        kws.mean_tmp .+= kws.CiRY[:, 1, :]
        mul!(kws.filt_mean[:, 1, :], kws.filt_cov[1], kws.mean_tmp, one(T), zero(T))

        # pre-fill predicted means
        kws.pred_mean[:, 2:end, :] .= kws.Bu[:, 1:(end - 1), :]
    end

    @views for tt in axes(kws.pred_mean, 2)[2:end]
        mul!(
            kws.pred_mean[:, tt, :],
            lds.state_model.A,
            kws.filt_mean[:, tt - 1, :],
            one(T),
            one(T),
        )

        mul!(kws.mean_tmp, kws.pred_icov[tt].mat, kws.pred_mean[:, tt, :], one(T), zero(T))
        kws.mean_tmp .+= kws.CiRY[:, tt, :]

        mul!(kws.filt_mean[:, tt, :], kws.filt_cov[tt].mat, kws.mean_tmp, one(T), zero(T))
    end

    return kws
end

function backwards_mean!(kws::KalmanWorkspace{T}) where {T<:Real}
    kws.smooth_mean[:, end, :] .= kws.filt_mean[:, end, :]

    @views for tt in eachindex(kws.pred_icov)[(end - 1):-1:1]
        kws.mean_tmp .= kws.smooth_mean[:, tt + 1, :] .- kws.pred_mean[:, tt + 1, :]
        mul!(kws.smooth_mean[:, tt, :], kws.G[:, :, tt], kws.mean_tmp, one(T), zero(T))
        kws.smooth_mean[:, tt, :] .+= kws.filt_mean[:, tt, :]
    end

    return kws
end

# ==== SUFFICIENT STATISTICS =============================================================================

function sym_syrk!(out, x::Matrix{T}) where {T<:Real}
    BLAS.syrk!('U', 'N', one(T), x, one(T), out)
    LinearAlgebra.copytri!(out, 'U')
    return out
end

@inline function aggregate_xx(
    smooth_mean::Matrix{T},
    # smooth_cov::PDMat{T,Matrix{T}}, 
    smooth_cov::Matrix{T},
    ntrials::Int,
)::PDMat{T,Matrix{T}} where {T<:Real}
    xx = smooth_cov * ntrials
    sym_syrk!(xx, smooth_mean)
    # mul!(xx, smooth_mean, smooth_mean', 1.0, 1.0)

    return PDMat(xx)
end

@views @inline function sufficient_statistics!(
    suf::SufficientStatistics{T}, kws::KalmanWorkspace{T}, data::Data{T}
) where {T<:Real}
    # Hoist workspace fields into concretely-typed locals — see
    # `backwards_cov!` for the same JET-vs-`@views` interaction.
    smooth_mean = kws.smooth_mean::Array{T,3}
    smooth_cov = kws.smooth_cov::Vector{PDMat{T,Matrix{T}}}
    x_init = kws.x_init::Matrix{T}
    x_prev = kws.x_prev::Matrix{T}
    x_next = kws.x_next::Matrix{T}
    x_cur = kws.x_cur::Matrix{T}
    dyn_xx_buf = kws.dyn_xx_buf::Matrix{T}
    obs_xx_buf = kws.obs_xx_buf::Matrix{T}

    # initial conditions -------
    x_init .= smooth_mean[:, 1, :]
    suf.init_n = T(kws.ntrials)
    # init_xx is preset to [ntrials] (1×1) by initialize_SufficientStatistics.
    # init_xy: row vector Σ_n x_init[:, n], shape (1, D).
    fill!(suf.init_xy, zero(T))
    @inbounds for n in axes(x_init, 2), i in axes(x_init, 1)
        suf.init_xy[1, i] += x_init[i, n]
    end
    # init_yy
    suf.init_yy[] = aggregate_xx(x_init, smooth_cov[1].mat, kws.ntrials)

    # transitions -------
    # Regression layout: regress x_next on [x_prev; 1; ux_prev] to fit [A b B].
    # The constant-1 column sits at index `latent_dim + 1` of the Gram matrix;
    # user inputs (if any) occupy `latent_dim + 2 : end`.
    D = kws.latent_dim
    ux_dim = kws.state_input_dim
    dyn_n_int = (kws.tsteps - 1) * kws.ntrials
    suf.dyn_n = T(dyn_n_int)
    x_prev .= reshape(smooth_mean[:, 1:(end - 1), :], D, dyn_n_int)
    x_next .= reshape(smooth_mean[:, 2:end, :], D, dyn_n_int)

    # Reuse the preallocated workspace buffer; the constant blocks at the bias
    # row/col and uᵀu sub-matrix are populated once in
    # `initialize_SufficientStatistics` and not mutated here. PDMat's
    # `cholesky()` makes its own copy of the factors, and downstream readers
    # (mstep!, compute_elbo) operate via `XX + prior` / `X_A_Xt(XX, W)` which
    # produce fresh PDMats — they never write into .mat.
    dyn_xx = dyn_xx_buf
    # Top-left x_prev block: smooth_cov_prev*N + x_prev x_prevᵀ
    dyn_xx[1:D, 1:D] .= kws.sum_smooth_cov_prev .* kws.ntrials
    BLAS.syrk!('U', 'N', one(T), x_prev, one(T), dyn_xx[1:D, 1:D])
    # Top-middle: x_prev · 1 = Σ_n,t x_prev[:, n, t] (column sum). Filling row
    # D+1 of the upper triangle; copytri! mirrors to the symmetric position.
    fill!(view(dyn_xx, 1:D, D + 1), zero(T))
    @inbounds for n in axes(x_prev, 2), i in 1:D
        dyn_xx[i, D + 1] += x_prev[i, n]
    end
    # Top-right user-input block: x_prev · u_prevᵀ (only if user inputs present)
    if ux_dim > 0
        ux_prev = reshape(data.ux[:, 1:(end - 1), :], ux_dim, dyn_n_int)
        mul!(dyn_xx[1:D, (D + 2):end], x_prev, ux_prev', one(T), zero(T))
    end
    LinearAlgebra.copytri!(dyn_xx, 'U')
    suf.dyn_xx[] = PDMat(dyn_xx)

    # dyn_xy: [x_prev; 1; ux_prev] x_nextᵀ. Row D+1 (bias) is Σ x_next.
    fill!(view(suf.dyn_xy, 1:D, :), zero(T))
    suf.dyn_xy[1:D, :] .= kws.sum_smooth_xcov .* kws.ntrials
    mul!(suf.dyn_xy[1:D, :], x_prev, x_next', one(T), one(T))
    fill!(view(suf.dyn_xy, D + 1, :), zero(T))
    @inbounds for n in axes(x_next, 2), j in 1:D
        suf.dyn_xy[D + 1, j] += x_next[j, n]
    end
    if ux_dim > 0
        ux_prev = reshape(data.ux[:, 1:(end - 1), :], ux_dim, dyn_n_int)
        mul!(suf.dyn_xy[(D + 2):end, :], ux_prev, x_next', one(T), zero(T))
    end
    # dyn_yy
    suf.dyn_yy[] = aggregate_xx(x_next, kws.sum_smooth_cov_next, kws.ntrials)

    # observations -------
    # Same layout: regress y on [x; 1; d] to fit [C d_bias D].
    uy_dim = kws.obs_input_dim
    obs_n_int = kws.tsteps * kws.ntrials
    suf.obs_n = T(obs_n_int)
    x_cur .= reshape(smooth_mean, D, obs_n_int)
    y_cur = reshape(data.y, kws.obs_dim, obs_n_int)

    obs_xx = obs_xx_buf
    obs_xx[1:D, 1:D] .= kws.sum_smooth_cov_all .* kws.ntrials
    BLAS.syrk!('U', 'N', one(T), x_cur, one(T), obs_xx[1:D, 1:D])
    # Bias column: Σ x_cur
    fill!(view(obs_xx, 1:D, D + 1), zero(T))
    @inbounds for n in axes(x_cur, 2), i in 1:D
        obs_xx[i, D + 1] += x_cur[i, n]
    end
    if uy_dim > 0
        d_cur = reshape(data.uy, uy_dim, obs_n_int)
        mul!(obs_xx[1:D, (D + 2):end], x_cur, d_cur', one(T), zero(T))
    end
    LinearAlgebra.copytri!(obs_xx, 'U')
    suf.obs_xx[] = PDMat(obs_xx)

    # obs_xy: row D+1 (bias for obs) is constant (= Σ y) and was preset.
    mul!(suf.obs_xy[1:D, :], x_cur, y_cur', one(T), zero(T))
    return suf
    # obs_yy (preset)

end

# ==== M-STEP =============================================================================
# Two regression overloads, dispatched on the prior:
#   * `nothing`  — OLS (no shrinkage, no shift).
#   * `MNPrior`  — matrix-normal MAP via `mn_map`.
# Sharing `mn_map` with `core/priors.jl` means the same math will eventually
# back the tridiag and Poisson M-steps after their Phase 2 migration.

function regress(XX::PDMat{T,Matrix{T}}, XY::AbstractMatrix{T}, ::Nothing) where {T<:Real}
    return transpose(XX \ XY)
end

function regress(
    XX::PDMat{T,Matrix{T}}, XY::AbstractMatrix{T}, prior::MNPrior{T}
) where {T<:Real}
    return mn_map(XX, XY, prior)
end

# Posterior IW scale matrix for Σ given the regression MAP W. The MN-prior
# contribution is `(W - M₀) Λ (W - M₀)'` (reduces to `W Λ W'` when `M₀ = 0`).

function est_cov(
    W::AbstractMatrix{T},
    XX::PDMat{T,Matrix{T}},
    XY::AbstractMatrix{T},
    YY::PDMat{T,Matrix{T}},
    N::Real,
    ::Nothing,
    prior_df::Int,
    prior_mu::AbstractMatrix{T},
)::Matrix{T} where {T<:Real}
    Wxy = W * XY
    Cov = (YY .- Wxy .- Wxy' .+ X_A_Xt(XX, W) .+ (prior_df * prior_mu)) / (N + prior_df)
    return Cov
end

function est_cov(
    W::AbstractMatrix{T},
    XX::PDMat{T,Matrix{T}},
    XY::AbstractMatrix{T},
    YY::PDMat{T,Matrix{T}},
    N::Real,
    prior::MNPrior{T},
    prior_df::Int,
    prior_mu::AbstractMatrix{T},
)::Matrix{T} where {T<:Real}
    Wxy = W * XY
    Wm = W .- prior.M₀
    # MN-prior contribution: (W - M₀) Λ (W - M₀)'. Expanded explicitly because
    # PDMats only exports `X_A_Xt(::AbstractPDMat, ::AbstractMatrix)` and
    # `prior.Λ` is a plain `Matrix`. Matrix-of-matrix triple product is a
    # one-liner here and avoids a cholesky-of-Λ wrap per E-step.
    WmΛWmT = Wm * prior.Λ * Wm'
    Cov =
        (YY .- Wxy .- Wxy' .+ X_A_Xt(XX, W) .+ WmΛWmT .+ (prior_df * prior_mu)) /
        (N + prior_df)
    return Cov
end

function mstep!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    D_lat = kws.latent_dim

    # initials =================================================
    # fit_bool[1] gates x0, fit_bool[2] gates P0. They share the same scatter,
    # so we run the regression only if at least one flag is set, then assign
    # selectively. (P0 uses the freshly-updated x0; if x0 is frozen, the
    # existing model x0 is used in the scatter.)
    if lds.fit_bool[1] || lds.fit_bool[2]
        x0_mat = regress(suf.init_xx[], suf.init_xy, nothing)  # D × 1
        if lds.fit_bool[2]
            x0_used = lds.fit_bool[1] ? x0_mat : reshape(lds.state_model.x0, D_lat, 1)
            P0 = est_cov(
                x0_used,
                suf.init_xx[],
                suf.init_xy,
                suf.init_yy[],
                suf.init_n,
                nothing,
                kws.P0_df,
                kws.P0_mu,
            )
            lds.state_model.P0 .= P0
        end
        if lds.fit_bool[1]
            lds.state_model.x0 .= vec(x0_mat)
        end
    end

    # dynamics =================================================
    # fit_bool[3] gates the joint [A b B] regression; fit_bool[4] gates Q.
    # Q's residual scatter depends on the regression coefficient, so when Q
    # is fit but [A b B] is frozen, use the existing state-model values.
    if lds.fit_bool[3] || lds.fit_bool[4]
        AbB = if lds.fit_bool[3]
            regress(suf.dyn_xx[], suf.dyn_xy, kws.AB_prior)
        else
            # Reassemble current model [A b B] for Q's scatter; never written back.
            buf = Matrix{T}(undef, D_lat, D_lat + 1 + kws.state_input_dim)
            buf[:, 1:D_lat] .= lds.state_model.A
            buf[:, D_lat + 1] .= lds.state_model.b
            if kws.state_input_dim > 0
                buf[:, (D_lat + 2):end] .= lds.state_model.B
            end
            buf
        end

        if lds.fit_bool[4]
            Q = est_cov(
                AbB,
                suf.dyn_xx[],
                suf.dyn_xy,
                suf.dyn_yy[],
                suf.dyn_n,
                kws.AB_prior,
                kws.Q_df,
                kws.Q_mu,
            )
            lds.state_model.Q .= Q
        end

        if lds.fit_bool[3]
            lds.state_model.A .= view(AbB, :, 1:D_lat)
            lds.state_model.b .= view(AbB, :, D_lat + 1)
            if kws.state_input_dim > 0
                lds.state_model.B .= view(AbB, :, (D_lat + 2):size(AbB, 2))
            end
        end
    end

    # observations =============================================
    # fit_bool[5] gates the joint [C d D] regression; fit_bool[6] gates R.
    # Same residual-scatter dependency as Q above.
    if lds.fit_bool[5] || lds.fit_bool[6]
        CdD = if lds.fit_bool[5]
            regress(suf.obs_xx[], suf.obs_xy, kws.CD_prior)
        else
            buf = Matrix{T}(undef, kws.obs_dim, D_lat + 1 + kws.obs_input_dim)
            buf[:, 1:D_lat] .= lds.obs_model.C
            buf[:, D_lat + 1] .= lds.obs_model.d
            if kws.obs_input_dim > 0
                buf[:, (D_lat + 2):end] .= lds.obs_model.D
            end
            buf
        end

        if lds.fit_bool[6]
            R = est_cov(
                CdD,
                suf.obs_xx[],
                suf.obs_xy,
                suf.obs_yy[],
                suf.obs_n,
                kws.CD_prior,
                kws.R_df,
                kws.R_mu,
            )
            lds.obs_model.R .= R
        end

        if lds.fit_bool[5]
            lds.obs_model.C .= view(CdD, :, 1:D_lat)
            lds.obs_model.d .= view(CdD, :, D_lat + 1)
            if kws.obs_input_dim > 0
                lds.obs_model.D .= view(CdD, :, (D_lat + 2):size(CdD, 2))
            end
        end
    end

    return lds
end

# ==== COMPUTE ELBO =============================================================================

# full priors. `n` / `vN` are `Real` rather than `Int` so the SLDS / weighted
# aggregator path (where effective sample sizes are Σₙ w[n,1] ∈ ℝ) flows
# through unchanged. Integer values still hit these signatures via implicit
# Int <: Real subtyping.
function log_post(
    n::Real,
    v::Int,
    v0::Int,
    vN::Real,
    lam0::PDMat{T,Matrix{T}},
    lamN::PDMat{T,Matrix{T}},
    Sig0::Matrix{T},
    SigN::PDMat{T,Matrix{T}},
) where {T<:Real}
    return -0.5 * n * v * log(2pi) .+ 0.5 * v * logdet(lam0) .+ -0.5 * v * logdet(lamN) .+
           0.5 * v0 * logdet(0.5 .* Sig0) .+ -0.5 * vN * logdet(0.5 .* SigN) .+
           -loggamma(0.5 .* v0) .+ loggamma(0.5 .* vN)
end;

# no beta prior
function log_post(
    n::Real,
    v::Int,
    v0::Int,
    vN::Real,
    lam0::Nothing,
    lamN::PDMat{T,Matrix{T}},
    Sig0::Matrix{T},
    SigN::PDMat{T,Matrix{T}},
) where {T<:Real}
    return -0.5 * n * v * log(2pi) .+ -0.5 * v * logdet(lamN) .+
           0.5 * v0 * logdet(0.5 .* Sig0) .+ -0.5 * vN * logdet(0.5 .* SigN) .+
           -loggamma(0.5 .* v0) .+ loggamma(0.5 .* vN)
end;

# no cov prior
function log_post(
    n::Real,
    v::Int,
    vN::Real,
    lam0::PDMat{T,Matrix{T}},
    lamN::PDMat{T,Matrix{T}},
    SigN::PDMat{T,Matrix{T}},
) where {T<:Real}
    return -0.5 * n * v * log(2pi) .+ 0.5 * v * logdet(lam0) .+ -0.5 * v * logdet(lamN) .+
           -0.5 * vN * logdet(0.5 .* SigN) .+ loggamma(0.5 .* vN)
end;

# no prior
function log_post(
    n::Real,
    v::Int,
    vN::Real,
    lam0::Nothing,
    lamN::PDMat{T,Matrix{T}},
    SigN::PDMat{T,Matrix{T}},
) where {T<:Real}
    return -0.5 * n * v * log(2pi) .+ -0.5 * v * logdet(lamN) .+
           -0.5 * vN * logdet(0.5 .* SigN) .+ loggamma(0.5 .* vN)
end;

"""
    _prior_Λ_PD(prior) -> Union{Nothing, PDMat}

Extract the matrix-normal precision Λ from an `MNPrior` and wrap it as a
`PDMat` (so the `log_post` overloads' cached-Cholesky path works unchanged).
Returns `nothing` when the prior is absent.
"""
@inline _prior_Λ_PD(::Nothing) = nothing
@inline _prior_Λ_PD(prior::MNPrior{T}) where {T} = PDMat(prior.Λ)

function compute_elbo(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    elbo = 0.0

    P0_PD = tol_PD(lds.state_model.P0)
    Q_PD = tol_PD(lds.state_model.Q)
    R_PD = tol_PD(lds.obs_model.R)

    # Initial Conditions --------------------------------------
    # The init "regression" has 1 parameter (x0), so the NIW posterior df
    # correction is `n - 1` rather than `n - u0_dim`.
    n = suf.init_n
    v = kws.latent_dim
    v0 = kws.P0_df
    vN = v0 + (n - 1)
    lam0 = nothing
    lamN = suf.init_xx[]
    Sig0 = kws.P0_mu * kws.P0_df
    SigN = P0_PD * vN

    if v0 > 0
        elbo += log_post(n, v, v0, vN, lam0, lamN, Sig0, SigN)
    else
        elbo += log_post(n, v, vN, lam0, lamN, SigN)
    end

    isfinite(elbo) || throw(
        NumericalStabilityError("ELBO (initial conditions)", "non-finite log-posterior")
    )

    # Dynamics --------------------------------------
    # The dynamics regression has (latent_dim + 1 + state_input_dim) parameters
    # per output dim: A (D), bias `b` (1), and B (state_input_dim, possibly 0).
    n = suf.dyn_n
    v = kws.latent_dim
    v0 = kws.Q_df
    vN = v0 + (n - (kws.latent_dim + 1 + kws.state_input_dim))
    lam0 = _prior_Λ_PD(kws.AB_prior)
    lamN = lam0 === nothing ? suf.dyn_xx[] : lam0 + suf.dyn_xx[]
    Sig0 = kws.Q_mu * kws.Q_df
    SigN = Q_PD * vN

    if v0 > 0
        elbo += log_post(n, v, v0, vN, lam0, lamN, Sig0, SigN)
    else
        elbo += log_post(n, v, vN, lam0, lamN, SigN)
    end

    isfinite(elbo) ||
        throw(NumericalStabilityError("ELBO (dynamics)", "non-finite log-posterior"))

    # Observations --------------------------------------
    # Same correction as dynamics: regression fits C, obs-bias `d`, and D.
    n = suf.obs_n
    v = kws.obs_dim
    v0 = kws.R_df
    vN = v0 + (n - (kws.latent_dim + 1 + kws.obs_input_dim))
    lam0 = _prior_Λ_PD(kws.CD_prior)
    lamN = lam0 === nothing ? suf.obs_xx[] : lam0 + suf.obs_xx[]
    Sig0 = kws.R_mu * kws.R_df
    SigN = R_PD * vN

    if v0 > 0
        elbo += log_post(n, v, v0, vN, lam0, lamN, Sig0, SigN)
    else
        elbo += log_post(n, v, vN, lam0, lamN, SigN)
    end
    isfinite(elbo) ||
        throw(NumericalStabilityError("ELBO (observations)", "non-finite log-posterior"))

    # Gaussian posterior entropy H[q(x_{1:T}|y)], shared across trials
    # Required for ELBO monotonicity: ELBO = NIW_marginal + H[q(x)]
    elbo += kws.ntrials * kws.shared_entropy[]

    return elbo
end

function marginal_loglikelihood(
    lds::LinearDynamicalSystem{T,S,O}, kws::KalmanWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    total_ll = zero(T)

    # Hoist workspace fields with concrete eltype for JET; see backwards_cov!.
    innovation = kws.innovation::Array{T,3}
    pred_cov = kws.pred_cov::Vector{PDMat{T,Matrix{T}}}

    Cmu = zeros(T, lds.obs_dim, kws.tsteps * kws.ntrials)
    mul!(
        Cmu,
        lds.obs_model.C,
        reshape(kws.pred_mean, kws.latent_dim, kws.tsteps * kws.ntrials),
    )
    innovation .= kws.y_minus_d .- reshape(Cmu, kws.obs_dim, kws.tsteps, kws.ntrials)

    @views for t in eachindex(pred_cov)
        kws.obs_pd_tmp[] = tol_PD(X_A_Xt(pred_cov[t], lds.obs_model.C) .+ lds.obs_model.R)
        MV = MvNormal(kws.obs_pd_tmp[])

        for n in axes(innovation, 3)
            total_ll += Distributions.logpdf(MV, innovation[:, t, n])
        end
    end

    return total_ll
end

#=
    marginal_loglikelihood(lds, y)

Specific-name companion to [`loglikelihood`](@ref): the marginal (observed-data)
log-likelihood of `lds` with the latent states integrated out. Identical in value
to `loglikelihood(lds, y)`; this name exists so internal call sites can disambiguate 
from the complete-data[`joint_loglikelihood`](@ref). Accepts the same observation 
forms (3-D array, vector-of-matrices, single matrix).
=#
function marginal_loglikelihood(lds::LinearDynamicalSystem, y::AbstractArray{<:Real,3})
    return loglikelihood(lds, y)
end
function marginal_loglikelihood(
    lds::LinearDynamicalSystem, y::AbstractVector{<:AbstractMatrix{<:Real}}
)
    return loglikelihood(lds, y)
end
function marginal_loglikelihood(lds::LinearDynamicalSystem, y::AbstractMatrix{<:Real})
    return loglikelihood(lds, y)
end
