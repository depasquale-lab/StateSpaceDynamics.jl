#=============================================================================
Stitching: sessions that observe different numbers of channels

These reuse the `pd_*` helpers from `ParameterDependencies.jl`, which is
included first. The latent dimension is shared across sessions throughout;
only `obs_dim` varies.
=============================================================================#

const ST_LATENT_DIM = 2

# A Gaussian emission of width `p`, deterministic so failures are reproducible.
function st_obs_model(p::Int, ::Type{T}=Float64; seed::Int=0) where {T<:Real}
    rng = StableRNG(1234 + seed)
    C = T.(randn(rng, p, ST_LATENT_DIM))
    R = Matrix{T}(0.1I, p, p)
    d = zeros(T, p)
    return GaussianObservationModel(C, R, d)
end

#=
`pd_lds` fixes `obs_dim` at `PD_OBS_DIM`, which is exactly the assumption this
feature removes, so these build the model with `obs_dim` taken from the
emission itself.
=#
function st_build_lds(sm, om)
    return LinearDynamicalSystem(;
        state_model=sm,
        obs_model=om,
        latent_dim=ST_LATENT_DIM,
        obs_dim=size(om.C, 1),
        fit_bool=fill(true, 6),
    )
end

function st_lds(p::Int; seed::Int=0)
    return st_build_lds(pd_state_model(), st_obs_model(p; seed=seed))
end

#=
Two sessions of differing width sharing one latent process: draw each session
from its own model, then hand both to a single model whose emission is
session-dependent.
=#
function st_two_session_data(; p1::Int=3, p2::Int=5, ntrials::Int=4, tsteps::Int=25)
    lds1 = st_lds(p1; seed=1)
    lds2 = st_lds(p2; seed=2)
    _, y1 = rand(StableRNG(11), lds1, fill(tsteps, ntrials))
    _, y2 = rand(StableRNG(22), lds2, fill(tsteps, ntrials))
    y = vcat(y1, y2)
    session = vcat(fill(:a, ntrials), fill(:b, ntrials))
    return y, session, p1, p2
end

function st_grouped_lds(p_template::Int)
    lds = st_lds(p_template)
    return lds
end

"""
Per-slot shapes come from the data, and each cell's LDS reports its own width.
"""
function test_stitching_variant_shapes()
    y, session, p1, p2 = st_two_session_data()
    lds = st_grouped_lds(p1)
    lds.obs_model.depends_on = (C=session, R=session)

    grp = SSD.parameter_grouping(lds, length(y); y=y)
    @test grp !== nothing
    @test grp.ncells == 2

    #=
    `variants` is indexed by the Cartesian product of every group's slots, so a
    two-session model with both `:C` and `:R` varying stores four entries. Only
    the combinations an actual cell uses are meaningful; the cross terms pair
    one session's `C` with another's `R` and are never reached, since
    `_cell_lds` indexes `grp.cell_obs`.
    =#
    variants = lds.obs_model.variants
    occupied = unique(grp.cell_obs)
    @test length(occupied) == 2
    widths = sort([size(variants[i].C, 1) for i in occupied])
    @test widths == sort([p1, p2])

    for i in occupied
        v = variants[i]
        p = size(v.C, 1)
        @test size(v.C, 2) == ST_LATENT_DIM
        @test length(v.d) == p
        @test size(v.R) == (p, p)
        @test size(v.D, 1) == p
    end

    # Each cell's LDS carries its own obs_dim, not the parent's template.
    cell_widths = sort([SSD._cell_lds(lds, grp, c).obs_dim for c in 1:(grp.ncells)])
    @test cell_widths == sort([p1, p2])
    return nothing
end

"""
`obs_dim` belongs to the emission group, so a shared `:R` beside a
session-dependent `:C` has no well-defined size.
"""
function test_stitching_requires_grouped_R()
    y, session, p1, _ = st_two_session_data()
    lds = st_grouped_lds(p1)
    lds.obs_model.depends_on = (C=session,)
    @test_throws ArgumentError SSD.parameter_grouping(lds, length(y); y=y)
    return nothing
end

"""
One parameter version is one matrix, so trials sharing a label must agree.
"""
function test_stitching_rejects_mixed_widths_in_slot()
    y, session, p1, _ = st_two_session_data()
    lds = st_grouped_lds(p1)
    # Collapse both sessions onto one label while the data still has two widths.
    same = fill(:only, length(y))
    lds.obs_model.depends_on = (C=same, R=same)
    @test_throws ArgumentError SSD.parameter_grouping(lds, length(y); y=y)
    return nothing
end

"""
Ragged channel counts are accepted only when the emission is group-dependent.
"""
function test_stitching_data_validation()
    y, session, p1, _ = st_two_session_data()

    plain = st_grouped_lds(p1)
    @test_throws SSD.DimensionMismatchError SSD.Data(plain, y)

    grouped = st_grouped_lds(p1)
    grouped.obs_model.depends_on = (C=session, R=session)
    data = SSD.Data(grouped, y)
    @test length(data.y) == length(y)
    return nothing
end

"""
Nothing about the uniform-`obs_dim` path changes: no specs are computed, slot 1
still aliases the model's own arrays, and the cell pools are the shared pool.
"""
function test_uniform_obs_dim_is_unchanged()
    ntrials = 4
    lds = st_lds(3)
    _, y = rand(StableRNG(7), lds, fill(20, ntrials))
    session = [:a, :a, :b, :b]
    lds.obs_model.depends_on = (C=session, R=session)

    dep = SSD._resolve_dependence(lds.obs_model)
    labels = [dep.trial_labels[g] for g in eachindex(dep.names)]
    @test SSD._obs_slot_specs(lds.obs_model, dep, labels, ntrials, y) === (nothing, nothing)

    grp = SSD.parameter_grouping(lds, ntrials; y=y)
    variants = lds.obs_model.variants
    @test all(v -> size(v.C, 1) == 3, variants)
    # Slot 1 shares storage with the parent, as before this feature.
    @test any(v -> v.C === lds.obs_model.C, variants)

    data = SSD.Data(lds, y)
    pool = SSD._grouped_sws_pool(lds, data)
    cell_lds = SSD._cell_ldss(lds, grp)
    pools = SSD._cell_sws_pools(lds, data, grp, cell_lds, pool)
    @test all(p -> p === pool, pools)
    return nothing
end

"""
A cell workspace reuses the O(D²·T) storage and allocates only what `obs_dim`
shapes — the property that keeps a many-session fit's memory bounded.
"""
function test_cell_workspace_shares_big_buffers()
    D, tsteps = ST_LATENT_DIM, 30
    base = SSD.SmoothWorkspace(Float64, D, 6, tsteps)
    cell = SSD._cell_workspace(base, D, 3, tsteps)

    # Shared: the expensive buffers are the same objects.
    @test cell.btd === base.btd
    @test cell.agg.p_smooth_shared === base.agg.p_smooth_shared
    @test cell.agg.p_smooth_tt1_shared === base.agg.p_smooth_tt1_shared

    # Private and correctly sized: everything shaped by obs_dim.
    @test cell.agg.obs_yy_const !== base.agg.obs_yy_const
    @test size(cell.agg.obs_yy_const) == (3, 3)
    @test size(cell.agg.obs_xy, 2) == 3
    @test size(cell.consts.tmp_RC, 1) == 3
    @test size(cell.elbo.sum_yy) == (3, 3)
    @test cell.batched === nothing

    # The latent-shaped aggregates keep the shared latent dimension.
    @test size(cell.agg.sum_smooth_cov_all) == (D, D)
    return nothing
end

"""
The grouped pool is sized at the widest session, so every cell fits in it.
"""
function test_grouped_pool_sized_at_widest_session()
    y, session, p1, p2 = st_two_session_data()
    lds = st_grouped_lds(p1)
    lds.obs_model.depends_on = (C=session, R=session)
    data = SSD.Data(lds, y)
    pool = SSD._grouped_sws_pool(lds, data)
    @test size(pool[1].agg.obs_yy_const, 1) == max(p1, p2)
    return nothing
end

"""
End-to-end: a stitched fit runs, improves monotonically, and leaves each
session's emission at its own width with the shared dynamics still shared.
"""
function test_stitching_fit_runs_and_improves()
    y, session, p1, p2 = st_two_session_data(; ntrials=4, tsteps=30)
    lds = st_grouped_lds(p1)
    lds.obs_model.depends_on = (C=session, R=session)

    elbos = fit!(lds, y; max_iter=8, progress=false)
    @test all(isfinite, elbos)
    @test pd_is_monotone(elbos)

    grp = SSD.parameter_grouping(lds, length(y); y=y)
    variants = lds.obs_model.variants
    occupied = unique(grp.cell_obs)
    @test sort([size(variants[i].C, 1) for i in occupied]) == sort([p1, p2])
    for i in occupied
        v = variants[i]
        @test size(v.R, 1) == size(v.C, 1)
        @test all(isfinite, v.C)
        @test all(isfinite, v.R)
        @test isposdef(Symmetric(v.R))
    end

    # Latent dynamics stay shared: one version, and it is the model's own array.
    @test lds.state_model.variants === nothing ||
        all(v -> v.A === lds.state_model.A, lds.state_model.variants)
    return nothing
end

"""
Smoothing a stitched model returns per-trial latents of the shared latent
dimension regardless of each trial's channel count.
"""
function test_stitching_smooth_shapes()
    y, session, _, _ = st_two_session_data(; ntrials=3, tsteps=20)
    lds = st_grouped_lds(3)
    lds.obs_model.depends_on = (C=session, R=session)

    xs, Ps = smooth(lds, y)
    @test length(xs) == length(y)
    for (n, x) in enumerate(xs)
        @test size(x, 1) == ST_LATENT_DIM
        @test size(x, 2) == size(y[n], 2)
        @test size(Ps[n], 1) == ST_LATENT_DIM
    end
    return nothing
end

#=============================================================================
SLDS
=============================================================================#

# An SLDS whose regimes share the latent chain and each carry a session-dependent
# emission of the template width; the per-session widths come from the data.
function st_slds(labels; p_template::Int=3, K::Int=2)
    ldss = map(1:K) do k
        sm = pd_state_model()
        sm.A .= k == 1 ? [0.95 0.05; -0.05 0.95] : [0.60 0.30; -0.30 0.60]
        om = st_obs_model(p_template; seed=10 + k)
        # `nothing` builds the ungrouped model each session is sampled from.
        labels === nothing || (om.depends_on = (C=labels, R=labels))
        return st_build_lds(sm, om)
    end
    return SLDS(; A=[0.9 0.1; 0.1 0.9], πₖ=[0.5, 0.5], LDSs=ldss)
end

function st_slds_two_session_data(; p1::Int=3, p2::Int=4, ntrials::Int=3, tsteps::Int=30)
    labels = vcat(fill(:s1, ntrials), fill(:s2, ntrials))
    t1 = st_slds(nothing; p_template=p1)
    t2 = st_slds(nothing; p_template=p2)
    _, _, y1 = rand(StableRNG(31), t1, fill(tsteps, ntrials))
    _, _, y2 = rand(StableRNG(32), t2, fill(tsteps, ntrials))
    return vcat(y1, y2), labels, p1, p2
end

"""
Every regime of a stitched SLDS gets that session's width, and the trial
partition is still shared across regimes.
"""
function test_stitching_slds_shapes()
    y, labels, p1, p2 = st_slds_two_session_data()
    slds = st_slds(labels; p_template=p1)

    grp = SSD._slds_parameter_grouping(slds, length(y); y=y)
    @test grp !== nothing
    @test grp.ncells == 2

    cells = SSD._slds_cell_sldss(slds, grp)
    for sc in cells
        widths = unique([lds.obs_dim for lds in sc.LDSs])
        # All regimes of one cell observe the same session, hence one width.
        @test length(widths) == 1
    end
    @test sort([sc.LDSs[1].obs_dim for sc in cells]) == sort([p1, p2])
    return nothing
end

"""
The per-cell SLDS workspace shares the expensive storage and sizes the rest to
the cell; an equal-width model gets the base workspace itself.
"""
function test_slds_cell_workspace_sharing()
    y, labels, p1, p2 = st_slds_two_session_data()
    slds = st_slds(labels; p_template=p1)
    grp = SSD._slds_parameter_grouping(slds, length(y); y=y)
    cells = SSD._slds_cell_sldss(slds, grp)

    base = SSD.SLDSSmoothWorkspace(Float64, slds, 30)
    ws = SSD._slds_cell_workspaces(slds, cells, base, 30)
    @test ws !== nothing
    for (c, w) in enumerate(ws)
        @test w.btd === base.btd          # shared O(D²·T)
        @test w.ll_tmp === base.ll_tmp    # shared O(T)
        p = cells[c].LDSs[1].obs_dim
        @test all(cc -> size(cc.tmp_RC, 1) == p, w.consts)
    end

    # Equal widths: the base workspace is handed back unchanged.
    uniform = SSD._slds_cell_workspaces(slds, [cells[1], cells[1]], base, 30)
    @test all(w -> w === base, uniform)
    return nothing
end

"""
End-to-end stitched SLDS fit: finite ELBOs and per-session emission widths.
"""
function test_stitching_slds_fit()
    y, labels, p1, p2 = st_slds_two_session_data(; ntrials=3, tsteps=30)
    slds = st_slds(labels; p_template=p1)

    elbos = fit!(slds, y; max_iter=5, progress=false, rng=StableRNG(99))
    @test length(elbos) == 5
    @test all(isfinite, elbos)

    grp = SSD._slds_parameter_grouping(slds, length(y); y=y)
    occupied = unique(grp.cell_obs)
    for k in 1:2
        om = slds.LDSs[k].obs_model
        widths = sort([size(om.variants[i].C, 1) for i in occupied])
        @test widths == sort([p1, p2])
        for i in occupied
            v = om.variants[i]
            @test size(v.R, 1) == size(v.C, 1)
            @test all(isfinite, v.C)
            @test isposdef(Symmetric(v.R))
        end
    end

    # The discrete chain and the initial state stay shared.
    @test slds.LDSs[2].state_model.x0 ≈ slds.LDSs[1].state_model.x0
    @test size(slds.A) == (2, 2)
    return nothing
end
