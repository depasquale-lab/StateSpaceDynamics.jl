#=============================================================================
Grouped M-step and ELBO for models with `depends_on` set.

The E-step needs no group-specific code: the trials of one cell share every
parameter, so the ordinary smoother + sufficient-statistics aggregator runs on
each cell's sub-`Data` unchanged (see `parameter_groups.jl`). What is left is

  * pooling the per-cell sufficient statistics per parameter *version*, and
  * evaluating the prior terms once per parameter version rather than once per
    cell.

Both are written against a flat list of "units". A unit is one thing that
produced a `SufficientStatistics`: a cell for a `LinearDynamicalSystem`, or a
(regime, cell) pair for an `SLDS`. Each unit carries the `LinearDynamicalSystem`
holding its parameter arrays and, per parameter group, the slot index it uses.
Parameters shared by several units are the *same array object*, so writing
through any one of those units updates all of them.
=============================================================================#

"""
    GroupedSufBuffers{T}

Scratch for pooling per-cell sufficient statistics into one parameter version.
Only the blocks the regression updates read are pooled (`init_n`/`init_xy`,
`dyn_xx`/`dyn_xy`, `obs_xx`/`obs_xy`); the covariance updates never need a
pooled `*_yy` because they accumulate the residual scatter cell by cell, each
with its own regression matrix. That also avoids taking a Cholesky of a pooled
`obs_yy`, which can be rank-deficient for a small group.
"""
struct GroupedSufBuffers{T<:Real}
    suf::SufficientStatistics{T}
    init_yy::Matrix{T}
    dyn_xx::Matrix{T}
    obs_xx::Matrix{T}
end

function GroupedSufBuffers(
    ::Type{T}, lds::LinearDynamicalSystem{T}, tsteps_per_trial::AbstractVector{Int}
) where {T<:Real}
    D = lds.latent_dim
    dyn_reg_dim = D + 1 + lds.ux_dim
    obs_reg_dim = D + 1 + lds.uy_dim
    return GroupedSufBuffers{T}(
        _initialize_td_sufficient_statistics(T, lds, tsteps_per_trial),
        zeros(T, D, D),
        zeros(T, dyn_reg_dim, dyn_reg_dim),
        zeros(T, obs_reg_dim, obs_reg_dim),
    )
end

#=
Pooling. A version backed by a single unit reuses that unit's statistics
verbatim, which keeps a fit whose groups happen to be singletons numerically
identical to the corresponding ungrouped fit.
=#
function _pool_init!(
    bufs::GroupedSufBuffers{T}, sufs::AbstractVector, units::AbstractVector{Int}
) where {T<:Real}
    length(units) == 1 && return sufs[units[1]]
    suf = bufs.suf
    n = zero(T)
    fill!(suf.init_xy, zero(T))
    M = bufs.init_yy
    fill!(M, zero(T))
    for u in units
        src = sufs[u]
        n += T(src.init_n)
        suf.init_xy .+= src.init_xy
        M .+= src.init_yy[]
    end
    suf.init_n = n
    suf.init_yy[] = M
    return suf
end

function _pool_dyn!(
    bufs::GroupedSufBuffers{T}, sufs::AbstractVector, units::AbstractVector{Int}
) where {T<:Real}
    length(units) == 1 && return sufs[units[1]]
    suf = bufs.suf
    n = zero(T)
    fill!(suf.dyn_xy, zero(T))
    XX = bufs.dyn_xx
    fill!(XX, zero(T))
    for u in units
        src = sufs[u]
        n += T(src.dyn_n)
        suf.dyn_xy .+= src.dyn_xy
        XX .+= src.dyn_xx[].mat
    end
    suf.dyn_n = n
    Symmetrize!(XX)
    suf.dyn_xx[] = PDMat(copy(XX))
    return suf
end

function _pool_obs!(
    bufs::GroupedSufBuffers{T}, sufs::AbstractVector, units::AbstractVector{Int}
) where {T<:Real}
    length(units) == 1 && return sufs[units[1]]
    suf = bufs.suf
    n = zero(T)
    fill!(suf.obs_xy, zero(T))
    XX = bufs.obs_xx
    fill!(XX, zero(T))
    for u in units
        src = sufs[u]
        n += T(src.obs_n)
        suf.obs_xy .+= src.obs_xy
        XX .+= src.obs_xx[].mat
    end
    suf.obs_n = n
    Symmetrize!(XX)
    suf.obs_xx[] = PDMat(copy(XX))
    return suf
end

"""
    CellConstBlocks{T}

The data-only aggregator constants (`Σ y y'`, the bias/`uy` blocks of `obs_xx`
and `obs_xy`, the bias/`ux` blocks of `dyn_xx`) for one cell.

A grouped fit shares a single `SmoothWorkspace` pool across cells — the O(D²·T)
block-tridiagonal and shared-covariance storage is the expensive part and is
safe to reuse, because a cell's smoothed covariances are consumed by its own
aggregation before the next cell overwrites them. These blocks, by contrast, are
per-cell and must survive between iterations, so they are cached here (all of
them are small: no `tsteps` dimension) and copied back in before each cell's
aggregation instead of being recomputed every E-step.
"""
struct CellConstBlocks{T<:Real}
    obs_yy::Matrix{T}
    obs_xy::Matrix{T}
    obs_xx::Matrix{T}
    dyn_xx::Matrix{T}
end

"""
    _cache_const_blocks!(sws, lds, data) -> CellConstBlocks

Run `_td_init_const_blocks!` for one cell and take a copy of the result.
"""
function _cache_const_blocks!(
    sws::SmoothWorkspace{T}, lds::LinearDynamicalSystem{T}, data::Data{T}
) where {T<:Real}
    _td_init_const_blocks!(sws, lds, data)
    return CellConstBlocks{T}(
        copy(sws.agg.obs_yy_const),
        copy(sws.agg.obs_xy_const),
        copy(sws.agg.obs_xx_const),
        copy(sws.agg.dyn_xx_const),
    )
end

"""
    _restore_const_blocks!(sws, blocks)

Copy a cell's cached aggregator constants back into the shared workspace.
"""
function _restore_const_blocks!(
    sws::SmoothWorkspace{T}, blocks::CellConstBlocks{T}
) where {T<:Real}
    copyto!(sws.agg.obs_yy_const, blocks.obs_yy)
    copyto!(sws.agg.obs_xy_const, blocks.obs_xy)
    copyto!(sws.agg.obs_xx_const, blocks.obs_xx)
    copyto!(sws.agg.dyn_xx_const, blocks.dyn_xx)
    return sws
end

"""
    GroupedFitState{T,L,DT}

Everything a grouped EM fit needs per cell: the cell's `LinearDynamicalSystem`
view, its sub-`Data`, its slice of the per-trial smoother output, its
sufficient statistics, its cached aggregator constants, and (Gaussian only) its
batched mean-pass buffers.

`tfs_all` holds the same `FilterSmooth` objects as `cell_tfs`, in original trial
order, for the callers that need results per trial rather than per cell.
"""
struct GroupedFitState{T<:Real,L,DT}
    cell_lds::Vector{L}
    cell_data::Vector{DT}
    cell_tfs::Vector{TrialFilterSmooth{T}}
    tfs_all::TrialFilterSmooth{T}
    sufs::Vector{SufficientStatistics{T}}
    cell_consts::Vector{CellConstBlocks{T}}
    cell_batched::Vector{Union{Nothing,BatchedBuffers{T}}}
    bufs::GroupedSufBuffers{T}
end

"""
    _grouped_fit_state(lds, data, grp, sws; batched=false)

Build the per-cell fit state. `batched=true` additionally allocates each cell's
BLAS-3 mean-pass buffers (Gaussian path only, and only for cells whose trials
are equal-length and plural); their total footprint is proportional to the
number of trials, so it matches an ungrouped fit's.
"""
function _grouped_fit_state(
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T},
    grp::ParameterGrouping,
    sws::SmoothWorkspace{T};
    batched::Bool=false,
) where {T<:Real,S<:AbstractStateModel{T},O<:AbstractObservationModel{T}}
    ncells = grp.ncells
    ntrials = length(data.y)
    cell_lds = _cell_ldss(lds, grp)
    cell_data = [_subset_data(data, grp.cell_trials[c]) for c in 1:ncells]

    #=
    Per-trial smoother storage. `cov_alias` is decided per cell: the smoother
    aliases every equal-length trial's `p_smooth` to the shared cache, so those
    trials would otherwise each carry a dead (D, D, T) allocation.
    =#
    fs_all = Vector{FilterSmooth{T}}(undef, ntrials)
    cell_batched = Vector{Union{Nothing,BatchedBuffers{T}}}(undef, ncells)
    for c in 1:ncells
        trials = grp.cell_trials[c]
        tsteps = data.tsteps[trials]
        equal_len = length(trials) > 1 && all(t -> t == tsteps[1], tsteps)
        alias = batched && equal_len
        for (i, n) in enumerate(trials)
            fs_all[n] = initialize_FilterSmooth(lds, tsteps[i]; cov_alias=alias)
        end
        cell_batched[c] = if batched && equal_len
            BatchedBuffers(
                T,
                lds.latent_dim,
                lds.obs_dim,
                tsteps[1],
                length(trials);
                ux_dim=lds.ux_dim,
                uy_dim=lds.uy_dim,
            )
        else
            nothing
        end
    end

    cell_tfs = [
        TrialFilterSmooth([fs_all[n] for n in grp.cell_trials[c]]) for c in 1:ncells
    ]
    sufs = [
        _initialize_td_sufficient_statistics(T, cell_lds[c], cell_data[c].tsteps) for
        c in 1:ncells
    ]
    cell_consts = [_cache_const_blocks!(sws, cell_lds[c], cell_data[c]) for c in 1:ncells]

    return GroupedFitState(
        cell_lds,
        cell_data,
        cell_tfs,
        TrialFilterSmooth(fs_all),
        sufs,
        cell_consts,
        cell_batched,
        GroupedSufBuffers(T, lds, data.tsteps),
    )
end

"""
    _prepare_cell!(sws, state, cell)

Point the shared workspace at one cell: swap in that cell's batched buffers and
restore its aggregator constants.
"""
function _prepare_cell!(sws::SmoothWorkspace, state::GroupedFitState, cell::Int)
    sws.batched = state.cell_batched[cell]
    _restore_const_blocks!(sws, state.cell_consts[cell])
    return sws
end

"""
    _grouped_sws_pool(lds, data)

Workspace pool shared by every cell of a grouped fit. Sized at the longest trial
across all cells; `batched` stays `nothing` here because each cell's BLAS-3
buffers are swapped in from the `GroupedFitState` by `_prepare_cell!`.

Sharing the pool is what keeps a grouped fit's memory at the ungrouped fit's
level: the O(D²·T) block-tridiagonal and shared-covariance storage is allocated
once rather than once per group, which matters when the groups are recording
sessions and there are dozens of them.
"""
function _grouped_sws_pool(lds::LinearDynamicalSystem{T}, data::Data{T}) where {T<:Real}
    T_max = maximum(data.tsteps)
    npool = min(Threads.maxthreadid(), length(data.y))
    return [
        SmoothWorkspace(
            T, lds.latent_dim, lds.obs_dim, T_max; ux_dim=lds.ux_dim, uy_dim=lds.uy_dim
        ) for _ in 1:npool
    ]
end

"""
    _grouped_smooth(lds, data, grp, y)

Smooth each cell with its own parameters and reassemble the results in the
original trial order. Results are copied out per cell before the next one runs:
the equal-length fast path aliases every trial's `p_smooth` to the workspace's
shared covariance cache, which the next cell overwrites.
"""
function _grouped_smooth(
    lds::LinearDynamicalSystem{T}, data::Data{T}, grp::ParameterGrouping, y
) where {T<:Real}
    ntrials = length(data.y)
    xs = Vector{Matrix{T}}(undef, ntrials)
    Ps = Vector{Array{T,3}}(undef, ntrials)
    sws_pool = _grouped_sws_pool(lds, data)

    for c in 1:(grp.ncells)
        trials = grp.cell_trials[c]
        cell_data = _subset_data(data, trials)
        tfs = initialize_FilterSmooth(lds, cell_data.tsteps)::TrialFilterSmooth{T}
        smooth!(_cell_lds(lds, grp, c), tfs, cell_data, sws_pool)
        for (i, n) in enumerate(trials)
            xs[n] = copy(tfs[i].x_smooth)
            Ps[n] = copy(tfs[i].p_smooth)
        end
    end

    return _grouped_smooth_output(xs, Ps, y)
end

# Public-shape return convention, matching `_collect_smooth_output`.
_grouped_smooth_output(xs, Ps, ::AbstractMatrix) = (xs[1], Ps[1])
_grouped_smooth_output(xs, Ps, _) = (xs, Ps)

"""
    _units_by_slot(slots) -> Vector{Vector{Int}}

Group unit indices by the parameter version they use, one entry per occupied
version. A `depends_on` label that no unit uses is skipped rather than fitted
from an empty sufficient statistic.
"""
function _units_by_slot(slots::AbstractVector{Int})
    out = Vector{Int}[]
    seen = Int[]
    for (u, s) in enumerate(slots)
        i = findfirst(isequal(s), seen)
        if i === nothing
            push!(seen, s)
            push!(out, [u])
        else
            push!(out[i], u)
        end
    end
    return out
end

"""
    _distinct_by_slot(slots, units) -> Vector{Int}

One representative unit per distinct value of `slots` within `units`. Used to
add a prior term once per distinct parameter version when the prior's group is
finer than the group being updated (e.g. several `[A b B]` sharing one `Q`).
"""
function _distinct_by_slot(slots::AbstractVector{Int}, units::AbstractVector{Int})
    reps = Int[]
    seen = Int[]
    for u in units
        s = slots[u]
        if !(s in seen)
            push!(seen, s)
            push!(reps, u)
        end
    end
    return reps
end

"""
    _shared_noise(noise_slots, units) -> Bool

Whether every unit of a regression slot draws on the same noise version. When
it does, that covariance is a common factor of the score and divides out, so
pooling the units' statistics and solving the ordinary normal equations is
exact. When it does not, the rows couple and the estimate has to come from
[`_tied_gls_regression`](@ref).
"""
function _shared_noise(noise_slots::AbstractVector{Int}, units::AbstractVector{Int})
    length(units) == 1 && return true
    s = noise_slots[units[1]]
    return all(u -> noise_slots[u] == s, units)
end

"""
    _tied_gls_regression(Szz, Szy, Sigma, prior, Sigma_prior) -> W

Fit one regression matrix `W` (`p x m`) shared by several units whose residual
covariances `Sigma[u]` (`p x p`) differ. `Szz[u]` is the unit's `m x m`
regressor Gram matrix and `Szy[u]` its `m x p` cross-product — the `*_xx` /
`*_xy` blocks of a `SufficientStatistics`.

Pooling the units' statistics and solving the ordinary normal equations
maximizes the objective only when the units also share `Sigma`: held fixed, it
divides out of the score. When it does not, the output rows couple,

    sum_u Sigma_u^-1 (Szy_u' - W Szz_u) = 0

which vectorizes to the `(p*m)`-square system

    [sum_u (Szz_u kron Sigma_u^-1)] vec(W) = vec(sum_u Sigma_u^-1 Szy_u')

A matrix-normal `prior` on `W` contributes `Lambda kron Sigma_prior^-1` on the
left and `Sigma_prior^-1 M0 Lambda` on the right. Its row scale is a residual
covariance and there is no single one here, so the caller passes the
representative it stores the fitted `W` on.

With every `Sigma[u]` equal this returns exactly [`mn_map`](@ref)'s answer — the
common factor cancels from both sides — so the cheap pooled path is a special
case of this one rather than a second estimator.

Costs `O((p*m)^3)` time and `O((p*m)^2)` memory. For a wide emission that
dominates the whole M-step, so callers reach for it only when the residual
covariance genuinely is not shared.
"""
function _tied_gls_regression(
    Szz::AbstractVector{<:AbstractMatrix{T}},
    Szy::AbstractVector{<:AbstractMatrix{T}},
    Sigma::AbstractVector{<:AbstractMatrix{T}},
    prior::Union{Nothing,MNPrior},
    Sigma_prior::AbstractMatrix{T},
) where {T<:Real}
    m = size(Szz[1], 1)
    p = size(Szy[1], 2)

    lhs = zeros(T, m * p, m * p)
    rhs = zeros(T, p, m)
    for u in eachindex(Szz)
        Sinv = inv(cholesky(Symmetric(Sigma[u])))
        # vec(Sigma^-1 W Szz) = (Szz kron Sigma^-1) vec(W), Szz symmetric.
        lhs .+= kron(Szz[u], Sinv)
        mul!(rhs, Sinv, transpose(Szy[u]), one(T), one(T))
    end

    if prior !== nothing
        Sinv0 = inv(cholesky(Symmetric(Sigma_prior)))
        lhs .+= kron(prior.Λ, Sinv0)
        mul!(rhs, Sinv0, prior.M₀ * prior.Λ, one(T), one(T))
    end

    return reshape(Symmetric(lhs) \ vec(rhs), p, m)
end

# ============================================================================
# Grouped parameter updates
# ============================================================================

function _grouped_update_x0!(
    ldss::AbstractVector,
    sufs::AbstractVector,
    slots::AbstractVector{Int},
    bufs::GroupedSufBuffers,
)
    for units in _units_by_slot(slots)
        update_initial_state_mean!(ldss[units[1]], _pool_init!(bufs, sufs, units))
    end
    return nothing
end

function _grouped_update_P0!(
    ldss::AbstractVector,
    sufs::AbstractVector,
    slots::AbstractVector{Int},
    slots_x0::AbstractVector{Int},
    sws::SmoothWorkspace{T},
) where {T<:Real}
    ldss[1].fit_bool[_G_P0] || return nothing
    S0 = sws.reg.S0_sum
    for units in _units_by_slot(slots)
        fill!(S0, zero(T))
        N = zero(T)
        for u in units
            _accumulate_init_scatter!(S0, ldss[u], sufs[u])
            N += T(sufs[u].init_n)
        end
        for u in _distinct_by_slot(slots_x0, units)
            _accumulate_x0_prior_scatter!(S0, ldss[u])
        end
        Symmetrize!(S0)
        _finalize_P0!(ldss[units[1]], S0, N)
    end
    return nothing
end

function _grouped_update_A_b!(
    ldss::AbstractVector,
    sufs::AbstractVector,
    slots::AbstractVector{Int},
    slots_q::AbstractVector{Int},
    sws::SmoothWorkspace,
    bufs::GroupedSufBuffers,
)
    for units in _units_by_slot(slots)
        lds = ldss[units[1]]
        if _shared_noise(slots_q, units)
            # One `Q` over these units, so it divides out and pooled OLS is exact.
            update_A_b!(lds, _pool_dyn!(bufs, sufs, units), sws)
        else
            lds.fit_bool[_G_AB] || continue
            W = _tied_gls_regression(
                [sufs[u].dyn_xx[].mat for u in units],
                [sufs[u].dyn_xy for u in units],
                [ldss[u].state_model.Q for u in units],
                lds.state_model.AB_prior,
                lds.state_model.Q,
            )
            _unpack_dyn_W!(lds, W)
        end
    end
    return nothing
end

function _grouped_update_Q!(
    ldss::AbstractVector,
    sufs::AbstractVector,
    slots::AbstractVector{Int},
    slots_ab::AbstractVector{Int},
    sws::SmoothWorkspace{T},
) where {T<:Real}
    ldss[1].fit_bool[_G_Q] || return nothing
    S_res = sws.reg.Q_sum
    for units in _units_by_slot(slots)
        fill!(S_res, zero(T))
        N = zero(T)
        for u in units
            _accumulate_dyn_scatter!(S_res, ldss[u], sufs[u], sws)
            N += T(sufs[u].dyn_n)
        end
        for u in _distinct_by_slot(slots_ab, units)
            _accumulate_ab_prior_scatter!(S_res, ldss[u], sws)
        end
        _finalize_Q!(ldss[units[1]], S_res, N)
    end
    return nothing
end

function _grouped_update_C_d!(
    ldss::AbstractVector,
    sufs::AbstractVector,
    slots::AbstractVector{Int},
    slots_r::AbstractVector{Int},
    sws::SmoothWorkspace,
    bufs::GroupedSufBuffers,
)
    for units in _units_by_slot(slots)
        lds = ldss[units[1]]
        if _shared_noise(slots_r, units)
            # One `R` over these units, so it divides out and pooled OLS is exact.
            update_C_d!(lds, _pool_obs!(bufs, sufs, units), sws)
        else
            lds.fit_bool[_G_CD] || continue
            V = _tied_gls_regression(
                [sufs[u].obs_xx[].mat for u in units],
                [sufs[u].obs_xy for u in units],
                [ldss[u].obs_model.R for u in units],
                lds.obs_model.CD_prior,
                lds.obs_model.R,
            )
            _unpack_obs_V!(lds, V)
        end
    end
    return nothing
end

function _grouped_update_R!(
    ldss::AbstractVector,
    sufs::AbstractVector,
    slots::AbstractVector{Int},
    slots_cd::AbstractVector{Int},
    sws::SmoothWorkspace{T},
) where {T<:Real}
    ldss[1].fit_bool[_G_R] || return nothing
    S_res = sws.elbo.obs_work
    for units in _units_by_slot(slots)
        fill!(S_res, zero(T))
        N = zero(T)
        for u in units
            _accumulate_obs_scatter!(S_res, ldss[u], sufs[u], sws)
            N += T(sufs[u].obs_n)
        end
        for u in _distinct_by_slot(slots_cd, units)
            _accumulate_cd_prior_scatter!(S_res, ldss[u], sws)
        end
        _finalize_R!(ldss[units[1]], S_res, N)
    end
    return nothing
end

"""
    _grouped_state_mstep!(ldss, sufs, slots, sws, bufs)

The four state-side updates (`x0`, `P0`, `[A b B]`, `Q`) over a flat unit list.
`slots[g][unit]` is the version of parameter group `g` that a unit uses. Shared
by the Gaussian LDS, the Poisson LDS, and the SLDS.
"""
function _grouped_state_mstep!(
    ldss::AbstractVector,
    sufs::AbstractVector,
    slots::AbstractVector{Vector{Int}},
    sws::SmoothWorkspace,
    bufs::GroupedSufBuffers,
)
    _grouped_update_x0!(ldss, sufs, slots[_G_X0], bufs)
    _grouped_update_P0!(ldss, sufs, slots[_G_P0], slots[_G_X0], sws)
    _grouped_update_A_b!(ldss, sufs, slots[_G_AB], slots[_G_Q], sws, bufs)
    _grouped_update_Q!(ldss, sufs, slots[_G_Q], slots[_G_AB], sws)
    return nothing
end

"""
    _grouped_gaussian_obs_mstep!(ldss, sufs, slots, sws, bufs)

The two Gaussian emission updates (`[C d D]`, `R`) over a flat unit list.
"""
function _grouped_gaussian_obs_mstep!(
    ldss::AbstractVector,
    sufs::AbstractVector,
    slots::AbstractVector{Vector{Int}},
    sws::SmoothWorkspace,
    bufs::GroupedSufBuffers,
)
    _grouped_update_C_d!(ldss, sufs, slots[_G_CD], slots[_G_R], sws, bufs)
    _grouped_update_R!(ldss, sufs, slots[_G_R], slots[_G_CD], sws)
    return nothing
end

# ============================================================================
# Grouped prior log-density (the ELBO's `log p(θ)` term)
# ============================================================================

"""
    _slot_representatives(slots) -> Vector{Int}

One representative unit per occupied parameter version.
"""
_slot_representatives(slots::AbstractVector{Int}) = [u[1] for u in _units_by_slot(slots)]

"""
    _pair_slot_representatives(slots_a, slots_b) -> Vector{Int}

One representative unit per distinct `(slots_a, slots_b)` pair. The MN priors
couple a regression with its noise covariance (`x0` with `P0`, `[A b B]` with
`Q`, `[C d D]` with `R`), so each distinct pair is one instance of that prior —
which stays consistent with the M-step, where the same pairs decide how often
the `Wm Λ Wm'` term is folded into the IW scale.
"""
function _pair_slot_representatives(
    slots_a::AbstractVector{Int}, slots_b::AbstractVector{Int}
)
    seen = Tuple{Int,Int}[]
    reps = Int[]
    for u in eachindex(slots_a)
        key = (slots_a[u], slots_b[u])
        if !(key in seen)
            push!(seen, key)
            push!(reps, u)
        end
    end
    return reps
end

"""
    _grouped_state_prior_logdensity(ldss, slots)

`log p(θ)` for the state-side parameters of a grouped model: the IW terms once
per covariance version, the MN terms once per distinct regression/covariance
pair.
"""
function _grouped_state_prior_logdensity(
    ldss::AbstractVector, slots::AbstractVector{Vector{Int}}, ::Type{T}
) where {T<:Real}
    total = zero(T)
    for u in _slot_representatives(slots[_G_P0])
        sm = ldss[u].state_model
        sm.P0_prior === nothing || (total += iw_logprior_term(sm.P0, sm.P0_prior))
    end
    for u in _slot_representatives(slots[_G_Q])
        sm = ldss[u].state_model
        sm.Q_prior === nothing || (total += iw_logprior_term(sm.Q, sm.Q_prior))
    end
    for u in _pair_slot_representatives(slots[_G_X0], slots[_G_P0])
        sm = ldss[u].state_model
        sm.x0_prior === nothing && continue
        total += mn_logprior_term(reshape(sm.x0, :, 1), sm.P0, sm.x0_prior)
    end
    for u in _pair_slot_representatives(slots[_G_AB], slots[_G_Q])
        lds = ldss[u]
        sm = lds.state_model
        sm.AB_prior === nothing && continue
        D = lds.latent_dim
        W_ab = _pack_dyn_W!(Matrix{T}(undef, D, D + 1 + lds.ux_dim), lds)
        total += mn_logprior_term(W_ab, sm.Q, sm.AB_prior)
    end
    return total
end

"""
    _grouped_gaussian_obs_prior_logdensity(ldss, slots)

`log p(θ)` for the Gaussian emission parameters of a grouped model.
"""
function _grouped_gaussian_obs_prior_logdensity(
    ldss::AbstractVector, slots::AbstractVector{Vector{Int}}, ::Type{T}
) where {T<:Real}
    total = zero(T)
    for u in _slot_representatives(slots[_G_R])
        om = ldss[u].obs_model
        om.R_prior === nothing || (total += iw_logprior_term(om.R, om.R_prior))
    end
    for u in _pair_slot_representatives(slots[_G_CD], slots[_G_R])
        lds = ldss[u]
        om = lds.obs_model
        om.CD_prior === nothing && continue
        D = lds.latent_dim
        W_cd = _pack_obs_V!(Matrix{T}(undef, lds.obs_dim, D + 1 + lds.uy_dim), lds)
        total += mn_logprior_term(W_cd, om.R, om.CD_prior)
    end
    return total
end
