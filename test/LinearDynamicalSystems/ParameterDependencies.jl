# Ancillary parameter dependencies (`depends_on`): declaring that some
# parameters are estimated separately per group of trials.
#
# This file covers the declaration side — validation, the resolved trial
# partition, and the accessors that read a fitted version back. The grouped
# E/M-step that consumes the partition lands in a follow-up; until it does,
# every entry point refuses a model that declares `depends_on` rather than
# silently pooling it (`test_depends_on_rejected_until_supported`).

const PD_LATENT_DIM = 2
const PD_OBS_DIM = 2

function pd_state_model(::Type{T}=Float64) where {T<:Real}
    return GaussianStateModel(;
        A=T[0.92 0.10; -0.10 0.92],
        Q=Matrix{T}(0.01 * I(PD_LATENT_DIM)),
        b=zeros(T, PD_LATENT_DIM),
        x0=zeros(T, PD_LATENT_DIM),
        P0=Matrix{T}(0.10 * I(PD_LATENT_DIM)),
    )
end

function pd_obs_model(::Type{T}=Float64) where {T<:Real}
    return GaussianObservationModel(;
        C=T[1.0 0.2; -0.3 1.0],
        d=zeros(T, PD_OBS_DIM),
        R=Matrix{T}(0.20 * I(PD_OBS_DIM)),
    )
end

function pd_lds(sm, om; fit_bool::Vector{Bool}=fill(true, 6))
    return LinearDynamicalSystem(;
        state_model=sm,
        obs_model=om,
        latent_dim=PD_LATENT_DIM,
        obs_dim=PD_OBS_DIM,
        fit_bool=fit_bool,
    )
end

pd_fresh_lds() = pd_lds(pd_state_model(), pd_obs_model())

# ============================================================================
# Validation
# ============================================================================

function test_depends_on_validation()
    om = pd_obs_model()

    om.depends_on = (Z=[:a, :b],)
    @test_throws ArgumentError SSD._resolve_dependence(om)

    # `:d` is an alias of `:C` — they must carry the same labels.
    om.depends_on = (C=[:a, :b], d=[:a, :a])
    @test_throws ArgumentError SSD._resolve_dependence(om)

    om.depends_on = (C=[:a, :b], R=[:a, :b, :b])
    @test_throws SSD.DimensionMismatchError SSD._resolve_dependence(om)

    om.depends_on = (C=Symbol[],)
    @test_throws ArgumentError SSD._resolve_dependence(om)

    # A Poisson emission has no `R`.
    pom = PoissonObservationModel(zeros(2, 2), zeros(2))
    pom.depends_on = (R=[:a, :b],)
    @test_throws ArgumentError SSD._resolve_dependence(pom)

    # A state model has no `C`.
    sm = pd_state_model()
    sm.depends_on = (C=[:a, :b],)
    @test_throws ArgumentError SSD._resolve_dependence(sm)

    return nothing
end

"""
A malformed `depends_on` is caught when the model is built, not at the first
`fit!` — `validate_LDS` resolves both sub-models' declarations.
"""
function test_depends_on_validated_at_construction()
    bad_obs = pd_obs_model()
    bad_obs.depends_on = (nope=[:a, :b],)
    @test_throws ArgumentError pd_lds(pd_state_model(), bad_obs)

    bad_state = pd_state_model()
    bad_state.depends_on = (A=[:a, :b], b=[:b, :a])   # aliases disagreeing
    @test_throws ArgumentError pd_lds(bad_state, pd_obs_model())

    ok_obs = pd_obs_model()
    ok_obs.depends_on = (C=[:a, :b],)
    @test pd_lds(pd_state_model(), ok_obs) isa LinearDynamicalSystem

    return nothing
end

# ============================================================================
# Accessors and storage layout
# ============================================================================

function test_group_accessors_and_aliasing()
    labels = [:s1, :s1, :s2, :s2]
    om = pd_obs_model()
    om.depends_on = (C=labels,)
    lds = pd_lds(pd_state_model(), om)

    @test group_labels(om, :C) == [:s1, :s2]
    @test group_labels(om, :d) == [:s1, :s2]     # alias of the same group
    @test group_labels(om, :D) == [:s1, :s2]
    @test isempty(group_labels(om, :R))          # R does not vary

    @test_throws ArgumentError group_parameter(om, :R, :s1)
    @test_throws ArgumentError group_parameter(om, :C, :nope)

    grp = SSD.parameter_grouping(lds, length(labels))
    @test grp !== nothing
    @test grp.ncells == 2
    @test grp.trial_cell == [1, 1, 2, 2]
    @test grp.cell_trials == [[1, 2], [3, 4]]

    variants = om.variants
    @test variants !== nothing
    @test length(variants) == 2
    # Slot 1 aliases the base model, so `om.C` keeps its original meaning.
    @test variants[1].C === om.C
    @test group_parameter(om, :C, :s1) === om.C
    # A varying parameter gets one array per group ...
    @test variants[1].C !== variants[2].C
    @test variants[1].d !== variants[2].d
    # ... a fixed one is shared by reference, so a single M-step write covers all.
    @test variants[1].R === variants[2].R
    @test variants[1].R === om.R

    return nothing
end

"""
A trial's cell is its position in the common refinement of every label vector,
so parameters may be grouped along different ancillary variables at once.
"""
function test_grouping_is_the_join_of_label_vectors()
    session = [:s1, :s1, :s2, :s2]
    condition = [:c1, :c2, :c1, :c2]
    sm = pd_state_model()
    sm.depends_on = (Q=condition,)
    om = pd_obs_model()
    om.depends_on = (C=session,)
    lds = pd_lds(sm, om)

    grp = SSD.parameter_grouping(lds, 4)
    # Q varies by condition and C by session, so a cell is one (session,
    # condition) pair: four singleton cells.
    @test grp.ncells == 4
    @test sort(grp.trial_cell) == [1, 2, 3, 4]
    @test grp.nslots[SSD._G_Q] == 2
    @test grp.nslots[SSD._G_CD] == 2
    @test grp.nslots[SSD._G_AB] == 1

    # The trial count must match the label vectors the model was built with.
    @test_throws SSD.DimensionMismatchError SSD.parameter_grouping(lds, 3)

    return nothing
end

"""
An override may re-assign trials to groups the model already declares, but it
cannot invent a parameter set that was never fitted, and it is meaningless on a
model that declares nothing.
"""
function test_override_key_validation()
    om = pd_obs_model()
    om.depends_on = (C=[:a, :a, :b],)
    lds = pd_lds(pd_state_model(), om)

    @test SSD.parameter_grouping(lds, 2; depends_on=(C=[:a, :b],)) !== nothing
    @test_throws ArgumentError SSD.parameter_grouping(lds, 3; depends_on=(C=[:a, :a, :x],))
    @test_throws ArgumentError SSD.parameter_grouping(lds, 3; depends_on=(nope=[:a],))
    # `:R` is a real parameter name, but this model does not group by it.
    @test_throws ArgumentError SSD.parameter_grouping(lds, 3; depends_on=(R=[:a, :b],))

    plain = pd_fresh_lds()
    @test_throws ArgumentError SSD.parameter_grouping(plain, 2; depends_on=(C=[:a, :b],))
    @test SSD.parameter_grouping(plain, 2) === nothing

    return nothing
end

# ============================================================================
# Interim guard (removed as each model family's grouped fit lands)
# ============================================================================

function test_depends_on_rejected_until_supported()
    rng = StableRNG(101)
    labels = [:a, :a, :b]
    y = [randn(rng, PD_OBS_DIM, 20) for _ in 1:3]

    om = pd_obs_model()
    om.depends_on = (C=labels,)
    lds = pd_lds(pd_state_model(), om)

    @test_throws ArgumentError fit!(lds, y; max_iter=1, progress=false)
    @test_throws ArgumentError smooth(lds, y)
    @test_throws ArgumentError elbo(lds, y)
    @test_throws ArgumentError loglikelihood(lds, y)
    @test_throws ArgumentError rand(rng, lds, 20)
    @test_throws ArgumentError rand(rng, lds, fill(20, 3))

    # A model that declares nothing is untouched by any of this.
    plain = pd_fresh_lds()
    @test fit!(plain, y; max_iter=1, progress=false) isa Vector

    pom = PoissonObservationModel([0.6 0.1; -0.2 0.5], [1.0, 0.8])
    pom.depends_on = (C=labels,)
    plds = LinearDynamicalSystem(;
        state_model=pd_state_model(),
        obs_model=pom,
        latent_dim=PD_LATENT_DIM,
        obs_dim=PD_OBS_DIM,
        fit_bool=fill(true, 5),
    )
    counts = [Float64.(rand(rng, 0:3, PD_OBS_DIM, 20)) for _ in 1:3]
    @test_throws ArgumentError fit!(plds, counts; max_iter=1, progress=false)
    @test_throws ArgumentError smooth(plds, counts)
    @test_throws ArgumentError elbo(plds, counts)

    slds = SLDS(;
        A=[0.9 0.1; 0.1 0.9],
        πₖ=[0.5, 0.5],
        LDSs=[pd_fresh_lds(), pd_lds(pd_state_model(), om)],
    )
    @test_throws ArgumentError fit!(slds, y; max_iter=1, progress=false)
    @test_throws ArgumentError elbo(slds, y)
    @test_throws ArgumentError rand(rng, slds, 20)

    return nothing
end

# ============================================================================
# Display
# ============================================================================

function test_grouped_show()
    labels = [:s1, :s1, :s2]
    om = pd_obs_model()
    om.depends_on = (C=labels, R=labels)
    out = sprint(show, om)
    @test occursin("Depends on:", out)
    @test occursin("C, d, D", out)
    @test occursin(":s2", out)

    plain = sprint(show, pd_obs_model())
    @test !occursin("Depends on:", plain)

    return nothing
end
