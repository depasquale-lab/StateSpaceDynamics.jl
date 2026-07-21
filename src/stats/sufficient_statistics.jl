
"""
    sufficient_statistics(x_smooth, p_smooth, p_smooth_t1)

Compute sufficient statistics for the EM algorithm in a Linear Dynamical System.

# Note
- The function computes the expected values for all trials.
- For single-trial data, use inputs with ntrials = 1.
"""
function sufficient_statistics!(fs::FilterSmooth{T}) where {T<:Real}
    latent_dim, tsteps = size(fs.x_smooth)

    # `initialize_FilterSmooth` leaves these as `(0, 0, 0)` stubs — the TD
    # aggregator never reads them. Materialize on demand for legacy callers.
    if size(fs.E_zz, 1) != latent_dim || size(fs.E_zz, 3) != tsteps
        fs.E_zz = zeros(T, latent_dim, latent_dim, tsteps)
        fs.E_zz_prev = zeros(T, latent_dim, latent_dim, tsteps)
    end

    # E_z is just a copy of x_smooth
    fs.E_z .= fs.x_smooth

    # Compute E_zz and E_zz_prev in-place
    @views for t in 1:tsteps
        # E_zz[:,:,t] = p_smooth[:,:,t] + x_smooth[:,t] * x_smooth[:,t]'
        mul!(fs.E_zz[:, :, t], fs.x_smooth[:, t:t], fs.x_smooth[:, t:t]')
        fs.E_zz[:, :, t] .+= fs.p_smooth[:, :, t]

        if t > 1
            # E_zz_prev[:,:,t] = p_smooth_tt1[:,:,t] + x_smooth[:,t] * x_smooth[:,t-1]'
            mul!(
                fs.E_zz_prev[:, :, t], fs.x_smooth[:, t:t], fs.x_smooth[:, (t - 1):(t - 1)]'
            )
            fs.E_zz_prev[:, :, t] .+= fs.p_smooth_tt1[:, :, t]
        else
            fs.E_zz_prev[:, :, 1] .= 0
        end
    end
end

function sufficient_statistics!(tfs::TrialFilterSmooth{T}) where {T<:Real}
    ntrials = length(tfs.FilterSmooths)

    if ntrials == 1
        sufficient_statistics!(tfs[1])
    else
        tforeach(1:ntrials) do i
            sufficient_statistics!(tfs[i])
        end
    end
end

"""
    _td_init_const_blocks!(sws, lds, tsteps_per_trial, y, ux_seq, uy_seq)

Fill the data-only constant blocks of the sufficient-statistics buffers
(`sws.agg.obs_yy_const` / `obs_xy_const` / `obs_xx_const` / `dyn_xx_const`)
once at fit entry. These are observation-independent: they depend only on
the raw inputs, not on smoother output.
"""
function _td_init_const_blocks!(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    tsteps_per_trial::AbstractVector{Int},
    y::AbstractVector{<:AbstractMatrix{T}},
    ux_seq::AbstractVector{<:AbstractMatrix{T}},
    uy_seq::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    D = lds.latent_dim
    p = lds.obs_dim
    ux_dim = lds.ux_dim
    uy_dim = lds.uy_dim
    ntrials = length(y)
    dyn_reg_dim = D + 1 + ux_dim
    obs_reg_dim = D + 1 + uy_dim
    total_obs = sum(tsteps_per_trial)
    total_dyn = total_obs - ntrials

    # Hoist workspace fields with concrete eltype to clear JET union-split
    # false positives on the syrk!/copytri! callsites below.
    td_obs_yy_const = sws.agg.obs_yy_const::Matrix{T}
    td_obs_xy_const = sws.agg.obs_xy_const::Matrix{T}
    td_obs_xx_const = sws.agg.obs_xx_const::Matrix{T}
    td_dyn_xx_const = sws.agg.dyn_xx_const::Matrix{T}

    fill!(td_obs_yy_const, zero(T))
    fill!(td_obs_xy_const, zero(T))
    fill!(td_obs_xx_const, zero(T))
    fill!(td_dyn_xx_const, zero(T))

    # obs_yy = Σ_n Σ_t y_t y_t'
    for trial in 1:ntrials
        BLAS.syrk!('U', 'N', one(T), y[trial], one(T), td_obs_yy_const)
    end
    LinearAlgebra.copytri!(td_obs_yy_const, 'U')

    # obs_xy[D+1, :] = Σ_n Σ_t y_t   (bias row sum)
    for trial in 1:ntrials
        y_trial = y[trial]
        for t in axes(y_trial, 2), j in 1:p
            td_obs_xy_const[D + 1, j] += y_trial[j, t]
        end
    end

    # obs_xx bias-bias entry
    td_obs_xx_const[D + 1, D + 1] = T(total_obs)

    if uy_dim > 0
        for trial in 1:ntrials
            uy_t = uy_seq[trial]
            y_t = y[trial]
            # obs_xy[D+2:end, :] += Σ_t uy_t y_t'
            mul!(view(td_obs_xy_const, (D + 2):obs_reg_dim, :), uy_t, y_t', one(T), one(T))
            # obs_xx[D+2:end, D+2:end] += Σ_t uy_t uy_t'  (upper tri)
            BLAS.syrk!(
                'U',
                'N',
                one(T),
                uy_t,
                one(T),
                tview(td_obs_xx_const, (D + 2):obs_reg_dim, (D + 2):obs_reg_dim),
            )
        end
        LinearAlgebra.copytri!(
            view(td_obs_xx_const, (D + 2):obs_reg_dim, (D + 2):obs_reg_dim), 'U'
        )
        # obs_xx[D+1, D+2:end] / [D+2:end, D+1] = Σ_t uy_t   (bias × v cross)
        for trial in 1:ntrials
            uy_trial = uy_seq[trial]
            for t in axes(uy_trial, 2), k in 1:uy_dim
                td_obs_xx_const[D + 1, D + 1 + k] += uy_trial[k, t]
            end
        end
        @views td_obs_xx_const[(D + 2):obs_reg_dim, D + 1] .= td_obs_xx_const[
            D + 1, (D + 2):obs_reg_dim
        ]
    end

    td_dyn_xx_const[D + 1, D + 1] = T(total_dyn)

    if ux_dim > 0
        for trial in 1:ntrials
            ux_trial = ux_seq[trial]
            T_n = size(ux_trial, 2)
            # Convention (matches existing update_A_b!): we use u[:, 1:T_n-1]
            # as `u_{t-1}` for t = 2:T_n.
            u_used = view(ux_trial, :, 1:(T_n - 1))
            BLAS.syrk!(
                'U',
                'N',
                one(T),
                u_used,
                one(T),
                tview(td_dyn_xx_const, (D + 2):dyn_reg_dim, (D + 2):dyn_reg_dim),
            )
        end
        LinearAlgebra.copytri!(
            view(td_dyn_xx_const, (D + 2):dyn_reg_dim, (D + 2):dyn_reg_dim), 'U'
        )
        # bias × u cross
        for trial in 1:ntrials
            ux_trial = ux_seq[trial]
            T_n = size(ux_trial, 2)
            for t in 1:(T_n - 1), k in 1:ux_dim
                td_dyn_xx_const[D + 1, D + 1 + k] += ux_trial[k, t]
            end
        end
        @views td_dyn_xx_const[(D + 2):dyn_reg_dim, D + 1] .= td_dyn_xx_const[
            D + 1, (D + 2):dyn_reg_dim
        ]
    end

    return nothing
end

"""
    _initialize_td_sufficient_statistics(T, lds, tsteps_per_trial)

Allocate a `SufficientStatistics{T}` with the right shapes for the TD path.
The PDMat refs are wrapped around identity placeholders; the aggregator
overwrites them each E-step.
"""
function _initialize_td_sufficient_statistics(
    ::Type{T}, lds::LinearDynamicalSystem{T,S,O}, tsteps_per_trial::AbstractVector{Int}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    D = lds.latent_dim
    p = lds.obs_dim
    ux_dim = lds.ux_dim
    uy_dim = lds.uy_dim
    ntrials = length(tsteps_per_trial)
    dyn_reg_dim = D + 1 + ux_dim
    obs_reg_dim = D + 1 + uy_dim
    total_obs = sum(tsteps_per_trial)
    total_dyn = total_obs - ntrials

    PD_init(d) = PDMat(Matrix{T}(I, d, d))

    return SufficientStatistics{T}(
        T(ntrials),
        zeros(T, 1, D),                            # init_xy
        Base.RefValue{Matrix{T}}(zeros(T, D, D)),  # this does not need to be PD as we do not compute a cholesky
        T(total_dyn),
        _pd_ref(PD_init(dyn_reg_dim)),                 # dyn_xx
        zeros(T, dyn_reg_dim, D),                  # dyn_xy
        _pd_ref(PD_init(D)),                           # dyn_yy
        T(total_obs),
        _pd_ref(PD_init(obs_reg_dim)),                 # obs_xx
        zeros(T, obs_reg_dim, p),                  # obs_xy
        _pd_ref(PD_init(p)),                           # obs_yy
    )
end

"""
    _aggregate_td_suff_stats!(suf, tfs, lds, ux_seq, uy_seq, sws)

Aggregate per-trial smoother output (`x_smooth`, `p_smooth`, `p_smooth_tt1`)
into `suf` using per-trial GEMM/SYRK. Replaces the per-timestep, per-trial
loops formerly done inside `Q_state!`, `Q_obs!`, and the `update_*!`
functions.

Uses the cov-cache fast-path shortcut when all trials' `p_smooth` arrays
are aliased to the same shared storage (equal-length multi-trial fit).
"""
function _aggregate_td_suff_stats!(
    suf::SufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    lds::LinearDynamicalSystem{T,S,O},
    ux_seq::AbstractVector{<:AbstractMatrix{T}},
    uy_seq::AbstractVector{<:AbstractMatrix{T}},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    D = lds.latent_dim
    p = lds.obs_dim
    ux_dim = lds.ux_dim
    uy_dim = lds.uy_dim
    ntrials = length(tfs)
    dyn_reg_dim = D + 1 + ux_dim
    obs_reg_dim = D + 1 + uy_dim

    # Hoist workspace fields with concrete eltype so JET's union-split
    # analysis keeps the callsites below in the typed branch.
    Szz_Ab = sws.reg.Szz_Ab::Matrix{T}
    Szz_Cd = sws.reg.Szz_Cd::Matrix{T}
    Q_sum = sws.reg.Q_sum::Matrix{T}
    R_sum = sws.reg.R_sum::Matrix{T}
    S0_sum = sws.reg.S0_sum::Matrix{T}
    td_init_xy = sws.agg.init_xy::Matrix{T}
    td_dyn_xy = sws.agg.dyn_xy::Matrix{T}
    td_obs_xy = sws.agg.obs_xy::Matrix{T}
    td_obs_xy_const = sws.agg.obs_xy_const::Matrix{T}
    td_obs_xx_const = sws.agg.obs_xx_const::Matrix{T}
    td_dyn_xx_const = sws.agg.dyn_xx_const::Matrix{T}
    td_obs_yy_const = sws.agg.obs_yy_const::Matrix{T}
    sum_cov_prev = sws.agg.sum_smooth_cov_prev::Matrix{T}
    sum_cov_next = sws.agg.sum_smooth_cov_next::Matrix{T}
    sum_cov_all = sws.agg.sum_smooth_cov_all::Matrix{T}
    sum_xcov = sws.agg.sum_smooth_xcov::Matrix{T}

    # Detect cov-cache fast path (equal-length trials share p_smooth storage).
    cov_cache = ntrials > 1 && tfs[1].p_smooth === tfs[2].p_smooth

    fill!(td_init_xy, zero(T))
    fill!(S0_sum, zero(T))                  # init_yy
    fill!(td_dyn_xy, zero(T))               # dyn_xy
    fill!(Q_sum, zero(T))                   # dyn_yy
    fill!(td_obs_xy, zero(T))               # obs_xy (will copy const next)
    fill!(sum_cov_prev, zero(T))
    fill!(sum_cov_next, zero(T))
    fill!(sum_cov_all, zero(T))
    fill!(sum_xcov, zero(T))

    # Seed xx/yy/xy buffers with the precomputed data-only constants.
    copyto!(Szz_Ab, td_dyn_xx_const)
    copyto!(Szz_Cd, td_obs_xx_const)
    copyto!(R_sum, td_obs_yy_const)
    copyto!(td_obs_xy, td_obs_xy_const)

    if cov_cache
        fs1 = tfs[1]
        T_shared = size(fs1.x_smooth, 2)
        p_smooth1 = fs1.p_smooth::Array{T,3}
        p_smooth_tt11 = fs1.p_smooth_tt1::Array{T,3}
        @views for t in 1:T_shared
            sum_cov_all .+= p_smooth1[:, :, t]
            if t < T_shared
                sum_cov_prev .+= p_smooth1[:, :, t]
            end
            if t > 1
                sum_cov_next .+= p_smooth1[:, :, t]
                sum_xcov .+= p_smooth_tt11[:, :, t]
            end
        end
        # Scale to total across N trials.
        N_T = T(ntrials)
        sum_cov_all .*= N_T
        sum_cov_prev .*= N_T
        sum_cov_next .*= N_T
        sum_xcov .*= N_T
    end

    for trial in 1:ntrials
        fs = tfs[trial]
        x = fs.x_smooth::Matrix{T}
        p_smooth = fs.p_smooth::Array{T,3}
        p_smooth_tt1 = fs.p_smooth_tt1::Array{T,3}
        T_n = size(x, 2)

        # Per-trial cov sums when not on the cov-cache fast path.
        if !cov_cache
            @views for t in 1:T_n
                sum_cov_all .+= p_smooth[:, :, t]
                if t < T_n
                    sum_cov_prev .+= p_smooth[:, :, t]
                end
                if t > 1
                    sum_cov_next .+= p_smooth[:, :, t]
                    sum_xcov .+= p_smooth_tt1[:, :, t]
                end
            end
        end

        # init_xy[1, :] += x[:, 1];   init_yy += x[:, 1] x[:, 1]'
        for j in 1:D
            td_init_xy[1, j] += x[j, 1]
        end
        x1 = tview(x, :, 1)
        BLAS.ger!(one(T), x1, x1, S0_sum)

        x_prev = tview(x, :, 1:(T_n - 1))
        x_next = tview(x, :, 2:T_n)

        # dyn_xx[1:D, 1:D] += x_prev x_prev'   (upper triangle via syrk)
        BLAS.syrk!('U', 'N', one(T), x_prev, one(T), tview(Szz_Ab, 1:D, 1:D))
        # obs_xx[1:D, 1:D] += x x'             (upper triangle via syrk)
        BLAS.syrk!('U', 'N', one(T), x, one(T), tview(Szz_Cd, 1:D, 1:D))

        # dyn_xx[1:D, D+1] += Σ x_prev   (column-sum into upper-only bias col)
        for t in 1:(T_n - 1), i in 1:D
            Szz_Ab[i, D + 1] += x_prev[i, t]
        end
        # obs_xx[1:D, D+1] += Σ x
        for t in 1:T_n, i in 1:D
            Szz_Cd[i, D + 1] += x[i, t]
        end

        # dyn_xy[1:D, :] += x_prev x_next'
        mul!(view(td_dyn_xy, 1:D, :), x_prev, x_next', one(T), one(T))
        # dyn_xy[D+1, :] += Σ x_next
        for t in 1:(T_n - 1), j in 1:D
            td_dyn_xy[D + 1, j] += x_next[j, t]
        end

        # dyn_yy += x_next x_next'  (upper tri)
        BLAS.syrk!('U', 'N', one(T), x_next, one(T), Q_sum)

        # obs_xy[1:D, :] += x y'
        mul!(view(td_obs_xy, 1:D, :), x, y[trial]', one(T), one(T))

        # Input-side cross blocks (x × u, u × x).
        if ux_dim > 0
            ux_trial = ux_seq[trial]
            ux_prev = view(ux_trial, :, 1:(T_n - 1))
            mul!(view(Szz_Ab, 1:D, (D + 2):dyn_reg_dim), x_prev, ux_prev', one(T), one(T))
            mul!(view(td_dyn_xy, (D + 2):dyn_reg_dim, :), ux_prev, x_next', one(T), one(T))
        end
        if uy_dim > 0
            uy_trial = uy_seq[trial]
            mul!(view(Szz_Cd, 1:D, (D + 2):obs_reg_dim), x, uy_trial', one(T), one(T))
        end
    end

    # init_yy: need Σ_n P_smooth[n,:,:,1].
    if cov_cache
        @views S0_sum .+= T(ntrials) .* (tfs[1].p_smooth::Array{T,3})[:, :, 1]
    else
        @views for trial in 1:ntrials
            S0_sum .+= (tfs[trial].p_smooth::Array{T,3})[:, :, 1]
        end
    end
    @views Szz_Ab[1:D, 1:D] .+= sum_cov_prev
    @views Szz_Cd[1:D, 1:D] .+= sum_cov_all
    Q_sum .+= sum_cov_next
    # dyn_xy[1:D, :] += (Σ p_smooth_tt1)'   — adjoint because cov(x_{t-1}, x_t) = p_smooth_tt1'.
    @views td_dyn_xy[1:D, :] .+= adjoint(sum_xcov)

    LinearAlgebra.copytri!(Szz_Ab, 'U')
    LinearAlgebra.copytri!(Szz_Cd, 'U')
    LinearAlgebra.copytri!(Q_sum, 'U')
    Symmetrize!(S0_sum)

    # backing storage; each E-step rewraps so the cached Cholesky reflects
    # the latest aggregate.
    suf.init_n = T(ntrials)
    suf.dyn_n = T(sum(size(tfs[trial].x_smooth, 2) for trial in 1:ntrials) - ntrials)
    suf.obs_n = T(sum(size(tfs[trial].x_smooth, 2) for trial in 1:ntrials))

    copyto!(suf.init_xy, td_init_xy)
    copyto!(suf.dyn_xy, td_dyn_xy)
    copyto!(suf.obs_xy, td_obs_xy)

    suf.init_yy[] = copy(S0_sum)                # see above; not needed to be PDMat
    suf.dyn_xx[] = PDMat(copy(Szz_Ab))
    suf.dyn_yy[] = PDMat(copy(Q_sum))
    suf.obs_xx[] = PDMat(copy(Szz_Cd))
    suf.obs_yy[] = PDMat(copy(R_sum))

    return suf
end

"""
    _aggregate_td_suff_stats_weighted!(suf, tfs, lds, ux_seq, uy_seq, y, weights, sws)

Weighted variant of `_aggregate_td_suff_stats!`. Each per-timestep
accumulation is scaled by `weights[trial][t]`, which carries the
responsibility γₖ,ₜ in the SLDS context (`q(zₜ = k)`).

The weighted form cannot reuse the precomputed `td_*_const` blocks —
weights change every E-step, so the data-side sums must be rebuilt fresh.
Likewise, the cov-cache fast path is skipped (responsibilities vary across
trials so `P_smooth[t]` is not shared in any useful way).

Conventions, mirroring the legacy weighted M-step:
- init terms use `weights[trial][1]` (responsibility at t=1)
- dynamics factor at time t uses `weights[trial][t]` (couples xₜ₋₁ and xₜ)
- emission at time t uses `weights[trial][t]`
"""
function _aggregate_td_suff_stats_weighted!(
    suf::SufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    lds::LinearDynamicalSystem{T,S,O},
    ux_seq::AbstractVector{<:AbstractMatrix{T}},
    uy_seq::AbstractVector{<:AbstractMatrix{T}},
    y::AbstractVector{<:AbstractMatrix{T}},
    weights::AbstractVector{<:AbstractVector{T}},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    D = lds.latent_dim
    p = lds.obs_dim
    ux_dim = lds.ux_dim
    uy_dim = lds.uy_dim
    ntrials = length(tfs)
    dyn_reg_dim = D + 1 + ux_dim
    obs_reg_dim = D + 1 + uy_dim

    #=
    Clear the accumulators we'll write into. Each field is hoisted with a
    concrete `Matrix{T}` annotation so the BLAS.ger!/syrk! callsites below
    stay in JET's typed union branch.
    =#
    init_xy = sws.agg.init_xy::Matrix{T}
    fill!(init_xy, zero(T))
    init_yy = sws.reg.S0_sum::Matrix{T}
    fill!(init_yy, zero(T))
    dyn_xx = sws.reg.Szz_Ab::Matrix{T}
    fill!(dyn_xx, zero(T))
    dyn_xy = sws.agg.dyn_xy::Matrix{T}
    fill!(dyn_xy, zero(T))
    dyn_yy = sws.reg.Q_sum::Matrix{T}
    fill!(dyn_yy, zero(T))
    obs_xx = sws.reg.Szz_Cd::Matrix{T}
    fill!(obs_xx, zero(T))
    obs_xy = sws.agg.obs_xy::Matrix{T}
    fill!(obs_xy, zero(T))
    obs_yy = sws.reg.R_sum::Matrix{T}
    fill!(obs_yy, zero(T))

    init_n_acc = zero(T)
    dyn_n_acc = zero(T)
    obs_n_acc = zero(T)

    for trial in 1:ntrials
        fs = tfs[trial]
        x_smooth = fs.x_smooth::Matrix{T}
        P_smooth = fs.p_smooth::Array{T,3}
        P_smooth_tt1 = fs.p_smooth_tt1::Array{T,3}
        y_trial = y[trial]
        T_n = size(x_smooth, 2)
        w = weights[trial]

        # Initial term — weighted by w[1].
        w1 = w[1]
        @views begin
            x1 = tview(x_smooth, :, 1)
            for i in 1:D
                init_xy[1, i] += w1 * x1[i]
            end
            # init_yy += w1 * (x1 x1' + P_smooth[:, :, 1])
            BLAS.ger!(w1, x1, x1, init_yy)
            init_yy .+= w1 .* P_smooth[:, :, 1]
        end
        init_n_acc += w1

        # Dynamics factors at t = 2..T_n.
        @views for t in 2:T_n
            wt = w[t]
            x_prev = tview(x_smooth, :, t - 1)
            x_next = tview(x_smooth, :, t)

            # dyn_xx[1:D, 1:D] += wt * (x_prev x_prev' + P_smooth[t-1])
            BLAS.ger!(wt, x_prev, x_prev, tview(dyn_xx, 1:D, 1:D))
            view(dyn_xx, 1:D, 1:D) .+= wt .* P_smooth[:, :, t - 1]
            # dyn_xx bias col / row
            for i in 1:D
                dyn_xx[i, D + 1] += wt * x_prev[i]
                dyn_xx[D + 1, i] += wt * x_prev[i]
            end
            dyn_xx[D + 1, D + 1] += wt

            # dyn_xy[1:D, :] += wt * (x_prev x_next' + P_smooth_tt1[t]')
            # cov(xₜ₋₁, xₜ) = P_smooth_tt1[t]'  (cf. unweighted aggregator).
            BLAS.ger!(wt, x_prev, x_next, tview(dyn_xy, 1:D, :))
            view(dyn_xy, 1:D, :) .+= wt .* transpose(P_smooth_tt1[:, :, t])
            for j in 1:D
                dyn_xy[D + 1, j] += wt * x_next[j]
            end

            # dyn_yy += wt * (x_next x_next' + P_smooth[t])
            BLAS.ger!(wt, x_next, x_next, dyn_yy)
            dyn_yy .+= wt .* P_smooth[:, :, t]

            #=
            User-input cross blocks (only when ux_dim > 0). The lower-tri
            mirror of the off-diagonal x_prev·ux_prev' block is filled
            once at the end of the function via `copytri!(dyn_xx, 'U')`.
            =#
            if ux_dim > 0
                ux_trial = ux_seq[trial]
                ux_prev = ux_trial[:, t - 1]
                # dyn_xx[1:D, D+2:end] += wt * x_prev ux_prev'
                BLAS.ger!(wt, x_prev, ux_prev, tview(dyn_xx, 1:D, (D + 2):dyn_reg_dim))
                # dyn_xx[D+1, D+2:end] += wt * ux_prev   (bias × u cross; mirrored later)
                for k in 1:ux_dim
                    dyn_xx[D + 1, D + 1 + k] += wt * ux_prev[k]
                end
                # dyn_xx[D+2:end, D+2:end] += wt * ux_prev ux_prev'
                BLAS.ger!(
                    wt,
                    ux_prev,
                    ux_prev,
                    tview(dyn_xx, (D + 2):dyn_reg_dim, (D + 2):dyn_reg_dim),
                )
                # dyn_xy[D+2:end, :] += wt * ux_prev x_next'
                BLAS.ger!(wt, ux_prev, x_next, tview(dyn_xy, (D + 2):dyn_reg_dim, :))
            end

            dyn_n_acc += wt
        end

        # Emissions at t = 1..T_n.
        @views for t in 1:T_n
            wt = w[t]
            x_t = tview(x_smooth, :, t)
            y_t = tview(y_trial, :, t)

            # obs_xx[1:D, 1:D] += wt * (x_t x_t' + P_smooth[t])
            BLAS.ger!(wt, x_t, x_t, tview(obs_xx, 1:D, 1:D))
            obs_xx[1:D, 1:D] .+= wt .* P_smooth[:, :, t]
            for i in 1:D
                obs_xx[i, D + 1] += wt * x_t[i]
                obs_xx[D + 1, i] += wt * x_t[i]
            end
            obs_xx[D + 1, D + 1] += wt

            # obs_xy[1:D, :] += wt * x_t y_t'
            BLAS.ger!(wt, x_t, y_t, tview(obs_xy, 1:D, :))
            for j in 1:p
                obs_xy[D + 1, j] += wt * y_t[j]
            end

            # obs_yy += wt * y_t y_t'
            BLAS.ger!(wt, y_t, y_t, obs_yy)

            # Obs-input cross blocks.
            if uy_dim > 0
                uy_trial = uy_seq[trial]
                uy_t = uy_trial[:, t]
                # obs_xx[1:D, D+2:end] += wt * x_t uy_t'
                BLAS.ger!(wt, x_t, uy_t, tview(obs_xx, 1:D, (D + 2):obs_reg_dim))
                # obs_xx[D+1, D+2:end] / [D+2:end, D+1] += wt * uy_t
                for k in 1:uy_dim
                    obs_xx[D + 1, D + 1 + k] += wt * uy_t[k]
                    obs_xx[D + 1 + k, D + 1] += wt * uy_t[k]
                end
                # obs_xx[D+2:end, D+2:end] += wt * uy_t uy_t'
                BLAS.ger!(
                    wt, uy_t, uy_t, tview(obs_xx, (D + 2):obs_reg_dim, (D + 2):obs_reg_dim)
                )
                # obs_xy[D+2:end, :] += wt * uy_t y_t'
                BLAS.ger!(wt, uy_t, y_t, tview(obs_xy, (D + 2):obs_reg_dim, :))
            end

            obs_n_acc += wt
        end
    end

    if ux_dim > 0
        @views dyn_xx[(D + 2):dyn_reg_dim, 1:D] .= transpose(
            dyn_xx[1:D, (D + 2):dyn_reg_dim]
        )
    end
    if uy_dim > 0
        @views obs_xx[(D + 2):obs_reg_dim, 1:D] .= transpose(
            obs_xx[1:D, (D + 2):obs_reg_dim]
        )
    end

    # Symmetrize PD blocks (BLAS.ger! is not symmetric and we touched the
    # bias row/col by hand; round-trip via Symmetrize! keeps PDMat happy).
    Symmetrize!(init_yy)
    LinearAlgebra.copytri!(dyn_xx, 'U')   # use upper to mirror everything to lower
    LinearAlgebra.copytri!(dyn_yy, 'U')
    LinearAlgebra.copytri!(obs_xx, 'U')
    LinearAlgebra.copytri!(obs_yy, 'U')

    suf.init_n = init_n_acc
    suf.dyn_n = dyn_n_acc
    suf.obs_n = obs_n_acc

    copyto!(suf.init_xy, init_xy)
    copyto!(suf.dyn_xy, dyn_xy)
    copyto!(suf.obs_xy, obs_xy)

    suf.init_yy[] = copy(init_yy)               # see above
    suf.dyn_xx[] = PDMat(copy(dyn_xx))
    suf.dyn_yy[] = PDMat(copy(dyn_yy))
    suf.obs_xx[] = PDMat(copy(obs_xx))
    suf.obs_yy[] = PDMat(copy(obs_yy))

    return suf
end
