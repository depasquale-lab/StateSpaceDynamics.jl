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
    #=
    Pooling writes `obs_xy`, whose column count is the slot's channel count. A
    stitching fit has more than one of those, so the odd-width scratch is built
    on first use and cached for the rest of the fit — bounded by the number of
    distinct widths, and never re-allocated per iteration. Stays empty whenever
    every slot has the model's own `obs_dim`.
    =#
    obs_alt::Dict{Int,SufficientStatistics{T}}
    proto::LinearDynamicalSystem{T}
    tsteps::Vector{Int}
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
        Dict{Int,SufficientStatistics{T}}(),
        lds,
        collect(tsteps_per_trial),
    )
end

#=
The pooling target for a slot of width `p`: the preallocated one when it
already matches, otherwise this fit's cached scratch for that width.
=#
function _obs_pool_target(bufs::GroupedSufBuffers{T}, p::Int) where {T<:Real}
    size(bufs.suf.obs_xy, 2) == p && return bufs.suf
    return get!(bufs.obs_alt, p) do
        lds = bufs.proto
        om = lds.obs_model
        wide = LinearDynamicalSystem{T,typeof(lds.state_model),typeof(om)}(
            lds.state_model, om, lds.latent_dim, p, lds.ux_dim, lds.uy_dim, lds.fit_bool
        )
        _initialize_td_sufficient_statistics(T, wide, bufs.tsteps)
    end
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
    suf = _obs_pool_target(bufs, size(sufs[units[1]].obs_xy, 2))
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
    cell_sws::Vector{Vector{SmoothWorkspace{T}}}
    bufs::GroupedSufBuffers{T}
end

"""
    _grouped_fit_state(lds, data, grp, sws_pool; batched=false)

Build the per-cell fit state. `batched=true` additionally allocates each cell's
BLAS-3 mean-pass buffers (Gaussian path only, and only for cells whose trials
are equal-length and plural); their total footprint is proportional to the
number of trials, so it matches an ungrouped fit's.
"""
function _grouped_fit_state(
    lds::LinearDynamicalSystem{T,S,O},
    data::Data{T},
    grp::ParameterGrouping,
    sws_pool::Vector{SmoothWorkspace{T}};
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
            #=
            The cell's own channel count, not the parent's: these buffers hold
            that cell's `y`, which under stitching is one session's channels.
            Identical to `lds.obs_dim` whenever every session has the same
            width, so the uniform case is unchanged.
            =#
            BatchedBuffers(
                T,
                lds.latent_dim,
                cell_lds[c].obs_dim,
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
    #=
    Each cell's constants are cached from that cell's own workspace, whose
    aggregator blocks are already the right width. With uniform `obs_dim` every
    cell aliases the shared pool, so this is the previous behaviour exactly.
    =#
    cell_sws = _cell_sws_pools(lds, data, grp, cell_lds, sws_pool)
    cell_consts = [
        _cache_const_blocks!(cell_sws[c][1], cell_lds[c], cell_data[c]) for c in 1:ncells
    ]

    return GroupedFitState(
        cell_lds,
        cell_data,
        cell_tfs,
        TrialFilterSmooth(fs_all),
        sufs,
        cell_consts,
        cell_batched,
        cell_sws,
        GroupedSufBuffers(T, lds, data.tsteps),
    )
end

"""
    _cell_workspace(base, latent_dim, obs_dim, tsteps; ux_dim, uy_dim)

A `SmoothWorkspace` for one cell of a stitching fit, reusing every expensive
buffer in `base` and allocating only the ones whose shape depends on the cell's
channel count.

The shared storage is the O(D²·T) part — the block-tridiagonal solver and the
two smoothed-covariance caches — and it is safe to share for exactly the reason
the pool itself is: a cell's covariances are consumed by its own aggregation
before the next cell runs. What cannot be shared is anything shaped by
`obs_dim`, since under stitching that differs per session.

These are built once per fit, not per iteration, so a long fit allocates no
more than a short one.
"""
function _cell_workspace(
    base::SmoothWorkspace{T},
    latent_dim::Int,
    obs_dim::Int,
    tsteps::Int;
    ux_dim::Int=0,
    uy_dim::Int=0,
) where {T<:Real}
    dyn_reg_dim = latent_dim + 1 + ux_dim
    obs_reg_dim = latent_dim + 1 + uy_dim
    agg = TDAggBuffers{T}(
        base.agg.p_smooth_shared,                  # shared (D, D, T)
        base.agg.p_smooth_tt1_shared,              # shared (D, D, T)
        zeros(T, 1, latent_dim),                   # init_xy
        zeros(T, dyn_reg_dim, latent_dim),         # dyn_xy
        zeros(T, obs_reg_dim, obs_dim),            # obs_xy
        zeros(T, latent_dim, latent_dim),          # sum_smooth_cov_prev
        zeros(T, latent_dim, latent_dim),          # sum_smooth_cov_next
        zeros(T, latent_dim, latent_dim),          # sum_smooth_cov_all
        zeros(T, latent_dim, latent_dim),          # sum_smooth_xcov
        zeros(T, obs_dim, obs_dim),                # obs_yy_const
        zeros(T, obs_reg_dim, obs_dim),            # obs_xy_const
        zeros(T, obs_reg_dim, obs_reg_dim),        # obs_xx_const
        zeros(T, dyn_reg_dim, dyn_reg_dim),        # dyn_xx_const
    )
    return SmoothWorkspace{T}(
        base.btd,                                  # shared block-tridiagonal storage
        SmoothConstants(T, latent_dim, obs_dim),
        NewtonBuffers(T, latent_dim, obs_dim, tsteps),
        RegressionBuffers(T, latent_dim, obs_dim; ux_dim=ux_dim, uy_dim=uy_dim),
        ElboBuffers(T, latent_dim, obs_dim),
        agg,
        nothing,                                   # set by `_prepare_cell!`
    )
end

"""
    _cell_sws_pools(lds, data, grp, cell_lds, sws_pool)

One workspace pool per cell, parallel to `sws_pool`.

When every cell has the parent's `obs_dim` — which includes every fit that does
not use `depends_on` at all, and every grouped fit whose sessions have the same
width — each cell simply aliases `sws_pool`, so nothing extra is allocated and
the code path is the one that existed before stitching. Only a genuinely ragged
fit builds per-cell workspaces, and even then the O(D²·T) storage is shared.
"""
function _cell_sws_pools(
    lds::LinearDynamicalSystem{T},
    data::Data{T},
    grp::ParameterGrouping,
    cell_lds::AbstractVector,
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real}
    all(c -> cell_lds[c].obs_dim == lds.obs_dim, 1:(grp.ncells)) &&
        return [sws_pool for _ in 1:(grp.ncells)]

    T_max = maximum(data.tsteps)
    return [
        [
            _cell_workspace(
                base,
                lds.latent_dim,
                cell_lds[c].obs_dim,
                T_max;
                ux_dim=lds.ux_dim,
                uy_dim=lds.uy_dim,
            ) for base in sws_pool
        ] for c in 1:(grp.ncells)
    ]
end

"""
    _prepare_cell!(sws_pool, state, cell) -> Vector{SmoothWorkspace}

Point the workspaces at one cell: swap in that cell's batched buffers and
restore its aggregator constants. Returns the pool the cell's E-step should
run on, which is `sws_pool` itself unless the fit stitches sessions of
differing width.
"""
function _prepare_cell!(
    sws_pool::Vector{SmoothWorkspace{T}}, state::GroupedFitState, cell::Int
) where {T<:Real}
    pool = state.cell_sws[cell]
    pool[1].batched = state.cell_batched[cell]
    _restore_const_blocks!(pool[1], state.cell_consts[cell])
    return pool
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
    #=
    Sized at the widest session, not at the parent's `obs_dim`: under stitching
    each cell has its own channel count and the parent's is only the template's.
    Cells narrower than the maximum use the leading rows and columns.
    =#
    obs_max = maximum(size(yt, 1) for yt in data.y)
    return [
        SmoothWorkspace(
            T, lds.latent_dim, obs_max, T_max; ux_dim=lds.ux_dim, uy_dim=lds.uy_dim
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
    cell_lds = _cell_ldss(lds, grp)
    cell_pools = _cell_sws_pools(lds, data, grp, cell_lds, sws_pool)

    for c in 1:(grp.ncells)
        trials = grp.cell_trials[c]
        cell_data = _subset_data(data, trials)
        tfs = initialize_FilterSmooth(cell_lds[c], cell_data.tsteps)::TrialFilterSmooth{T}
        smooth!(cell_lds[c], tfs, cell_data, cell_pools[c])
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
    sws::SmoothWorkspace,
    bufs::GroupedSufBuffers,
)
    for units in _units_by_slot(slots)
        update_A_b!(ldss[units[1]], _pool_dyn!(bufs, sufs, units), sws)
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

#=
Which workspace a unit's emission regression runs on. Ungrouped and
equal-width fits pass `nothing` and share one; a stitching fit passes one
workspace per unit, since `[C d D]` and `R` are shaped by the unit's width.
=#
_unit_ws(::Nothing, sws::SmoothWorkspace, ::Int) = sws
_unit_ws(v::AbstractVector, ::SmoothWorkspace, u::Int) = v[u]

function _grouped_update_C_d!(
    ldss::AbstractVector,
    sufs::AbstractVector,
    slots::AbstractVector{Int},
    sws::SmoothWorkspace,
    bufs::GroupedSufBuffers;
    unit_sws::Union{Nothing,AbstractVector}=nothing,
)
    for units in _units_by_slot(slots)
        update_C_d!(
            ldss[units[1]], _pool_obs!(bufs, sufs, units), _unit_ws(unit_sws, sws, units[1])
        )
    end
    return nothing
end

function _grouped_update_R!(
    ldss::AbstractVector,
    sufs::AbstractVector,
    slots::AbstractVector{Int},
    slots_cd::AbstractVector{Int},
    sws::SmoothWorkspace{T};
    unit_sws::Union{Nothing,AbstractVector}=nothing,
) where {T<:Real}
    ldss[1].fit_bool[_G_R] || return nothing
    for units in _units_by_slot(slots)
        # `obs_work` is (p, p), so the scratch has to be this slot's width.
        ws = _unit_ws(unit_sws, sws, units[1])
        S_res = ws.elbo.obs_work
        fill!(S_res, zero(T))
        N = zero(T)
        for u in units
            _accumulate_obs_scatter!(S_res, ldss[u], sufs[u], _unit_ws(unit_sws, sws, u))
            N += T(sufs[u].obs_n)
        end
        for u in _distinct_by_slot(slots_cd, units)
            _accumulate_cd_prior_scatter!(S_res, ldss[u], _unit_ws(unit_sws, sws, u))
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
    _grouped_update_A_b!(ldss, sufs, slots[_G_AB], sws, bufs)
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
    bufs::GroupedSufBuffers;
    unit_sws::Union{Nothing,AbstractVector}=nothing,
)
    _grouped_update_C_d!(ldss, sufs, slots[_G_CD], sws, bufs; unit_sws=unit_sws)
    _grouped_update_R!(
        ldss, sufs, slots[_G_R], slots[_G_CD], sws; unit_sws=unit_sws
    )
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

"""
    _grouped_poisson_obs_prior_logdensity(ldss, slots)

`log p(θ)` for the Poisson emission parameters of a grouped model. Poisson has
no observation-noise covariance, so the MN prior on `[C d D]` contributes the
bare quadratic that the LBFGS emission objective penalizes, once per version.
"""
function _grouped_poisson_obs_prior_logdensity(
    ldss::AbstractVector, slots::AbstractVector{Vector{Int}}, ::Type{T}
) where {T<:Real}
    total = zero(T)
    for u in _slot_representatives(slots[_G_CD])
        lds = ldss[u]
        om = lds.obs_model
        om.CD_prior === nothing && continue
        D = lds.latent_dim
        W_cd = _pack_obs_V!(Matrix{T}(undef, lds.obs_dim, D + 1 + lds.uy_dim), lds)
        Wm = W_cd .- om.CD_prior.M₀
        total -= T(0.5) * sum(Wm .* (Wm * om.CD_prior.Λ))
    end
    return total
end
