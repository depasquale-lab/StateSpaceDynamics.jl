# Ancillary parameter dependencies (`depends_on`): declaring that some
# parameters are estimated separately per group of trials.
#
# The load-bearing correctness check is
# `test_fully_grouped_matches_independent_fits`: when *every* parameter depends
# on the session label, a grouped fit over the pooled data must reproduce, to
# floating point, what independent fits of each session produce. Everything
# downstream of that (partial grouping, priors, held-out scoring) shares the
# same machinery.

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
        C=T[1.0 0.2; -0.3 1.0], d=zeros(T, PD_OBS_DIM), R=Matrix{T}(0.20 * I(PD_OBS_DIM))
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

"""
EM on the MAP objective is monotone in exact arithmetic; allow a relative slack
so an accumulated rounding error in the ELBO sum is not read as a real decrease.
"""
function pd_is_monotone(elbos::AbstractVector)
    length(elbos) < 2 && return true
    slack = 1e-8 * max(1.0, maximum(abs, elbos))
    return all(>=(-slack), diff(elbos))
end

"""
Two-session ground truth: shared latent dynamics, session-specific emissions.
Mirrors the stitching setup — one animal, two recording sessions.
"""
function pd_two_session_truth(; ntrials_per_session::Int=6)
    labels = vcat(fill(:s1, ntrials_per_session), fill(:s2, ntrials_per_session))
    sm = pd_state_model()
    om = pd_obs_model()
    om.depends_on = (C=labels, R=labels)
    lds = pd_lds(sm, om)

    # Session 2 sees different "neurons" and is much noisier.
    C2 = group_parameter(om, :C, :s2)
    C2 .= [0.4 -0.9; 1.1 0.5]
    d2 = group_parameter(om, :d, :s2)
    d2 .= [0.5, -0.5]
    R2 = group_parameter(om, :R, :s2)
    R2 .= Matrix(1.5 * I(PD_OBS_DIM))

    return lds, labels
end

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
`validate_LDS` resolves both sub-models' `depends_on`, so a malformed
declaration is reported by the validating positional constructor rather than at
the first `fit!`. (`LinearDynamicalSystem` is a `@kwdef` struct, so the keyword
constructor bypasses `validate_LDS` — for `depends_on` exactly as for every
other field.)
"""
function test_depends_on_validated_at_construction()
    bad_obs = pd_obs_model()
    bad_obs.depends_on = (nope=[:a, :b],)
    @test_throws ArgumentError LinearDynamicalSystem(pd_state_model(), bad_obs)

    bad_state = pd_state_model()
    bad_state.depends_on = (A=[:a, :b], b=[:b, :a])   # aliases disagreeing
    @test_throws ArgumentError LinearDynamicalSystem(bad_state, pd_obs_model())

    ok_obs = pd_obs_model()
    ok_obs.depends_on = (C=[:a, :b],)
    @test LinearDynamicalSystem(pd_state_model(), ok_obs) isa LinearDynamicalSystem

    # A model built through the keyword constructor is still rejected when
    # `validate_LDS` is run on it.
    @test_throws ArgumentError validate_LDS(pd_lds(pd_state_model(), bad_obs))

    return nothing
end

function test_depends_on_trial_count_and_override()
    om = pd_obs_model()
    om.depends_on = (C=[:a, :a, :b],)
    lds = pd_lds(pd_state_model(), om)

    rng = StableRNG(101)
    y3 = [randn(rng, PD_OBS_DIM, 25) for _ in 1:3]
    y2 = y3[1:2]

    # Labels are written for 3 trials; 2 trials is a mismatch.
    @test_throws SSD.DimensionMismatchError fit!(lds, y2; max_iter=1, progress=false)

    # ... unless the call supplies its own labels.
    @test fit!(lds, y2; depends_on=(C=[:a, :b],), max_iter=1, progress=false) isa Vector

    # An override may only re-assign trials to groups the model already knows.
    @test_throws ArgumentError fit!(
        lds, y3; depends_on=(C=[:a, :a, :unseen],), max_iter=1, progress=false
    )

    # An override is meaningless without a declared dependence.
    plain = pd_fresh_lds()
    @test_throws ArgumentError fit!(
        plain, y2; depends_on=(C=[:a, :b],), max_iter=1, progress=false
    )

    # A typo'd override key is rejected rather than silently ignored ...
    @test_throws ArgumentError fit!(
        lds, y3; depends_on=(C=[:a, :a, :b], nope=[:a, :a, :b]), max_iter=1, progress=false
    )
    # ... as is naming a parameter the model never declared as grouped.
    @test_throws ArgumentError fit!(
        lds, y3; depends_on=(C=[:a, :a, :b], R=[:a, :a, :b]), max_iter=1, progress=false
    )

    #=
    An override is one NamedTuple resolved against *both* sub-models, so keys
    belonging to the emission must not trip up the state model's resolution.
    =#
    sm = pd_state_model()
    sm.depends_on = (Q=[:a, :a, :b],)
    om2 = pd_obs_model()
    om2.depends_on = (C=[:a, :a, :b],)
    both = pd_lds(sm, om2)
    @test fit!(
        both, y2; depends_on=(Q=[:a, :b], C=[:a, :b]), max_iter=1, progress=false
    ) isa Vector

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
# Equivalences
# ============================================================================

"""
A `depends_on` whose labels are all identical adds no parameters, so it must
reproduce the ungrouped fit.
"""
function test_single_group_matches_ungrouped()
    rng = StableRNG(202)
    truth = pd_fresh_lds()
    _, y = rand(rng, truth, fill(40, 5))

    plain = pd_fresh_lds()
    elbos_plain = fit!(plain, y; max_iter=8, tol=0.0, progress=false)

    om = pd_obs_model()
    om.depends_on = (C=fill(:only, 5), R=fill(:only, 5))
    grouped = pd_lds(pd_state_model(), om)
    elbos_grouped = fit!(grouped, y; max_iter=8, tol=0.0, progress=false)

    @test elbos_grouped ≈ elbos_plain
    @test grouped.obs_model.C ≈ plain.obs_model.C
    @test grouped.obs_model.R ≈ plain.obs_model.R
    @test grouped.state_model.A ≈ plain.state_model.A
    @test grouped.state_model.Q ≈ plain.state_model.Q
    @test grouped.state_model.x0 ≈ plain.state_model.x0

    return nothing
end

"""
When *every* parameter depends on the session, the sessions share nothing, so a
grouped fit must equal two independent fits — same ELBO trace (summed) and same
fitted parameters. This is the sharpest check that the per-cell E-step and the
slot-wise M-step pooling agree with the ordinary path.
"""
function test_fully_grouped_matches_independent_fits()
    rng = StableRNG(303)

    truth1 = pd_fresh_lds()
    _, y1 = rand(rng, truth1, fill(35, 4))

    truth2 = pd_fresh_lds()
    truth2.obs_model.C .= [0.3 -1.0; 1.2 0.4]
    truth2.obs_model.R .= Matrix(0.9 * I(PD_OBS_DIM))
    truth2.state_model.A .= [0.7 0.3; -0.3 0.7]
    _, y2 = rand(rng, truth2, fill(35, 4))

    y = vcat(y1, y2)
    labels = vcat(fill(:s1, 4), fill(:s2, 4))

    fit_kwargs = (; max_iter=6, tol=0.0, progress=false)

    lds1 = pd_fresh_lds()
    elbos1 = fit!(lds1, y1; fit_kwargs...)
    lds2 = pd_fresh_lds()
    elbos2 = fit!(lds2, y2; fit_kwargs...)

    sm = pd_state_model()
    sm.depends_on = (x0=labels, P0=labels, A=labels, Q=labels)
    om = pd_obs_model()
    om.depends_on = (C=labels, R=labels)
    grouped = pd_lds(sm, om)
    elbos_g = fit!(grouped, y; fit_kwargs...)

    @test elbos_g ≈ elbos1 .+ elbos2

    for (label, ref) in ((:s1, lds1), (:s2, lds2))
        @test group_parameter(grouped.obs_model, :C, label) ≈ ref.obs_model.C
        @test group_parameter(grouped.obs_model, :d, label) ≈ ref.obs_model.d
        @test group_parameter(grouped.obs_model, :R, label) ≈ ref.obs_model.R
        @test group_parameter(grouped.state_model, :A, label) ≈ ref.state_model.A
        @test group_parameter(grouped.state_model, :b, label) ≈ ref.state_model.b
        @test group_parameter(grouped.state_model, :Q, label) ≈ ref.state_model.Q
        @test group_parameter(grouped.state_model, :x0, label) ≈ ref.state_model.x0
        @test group_parameter(grouped.state_model, :P0, label) ≈ ref.state_model.P0
    end

    return nothing
end

"""
Ragged trial lengths inside a group take the per-trial smoother fallback rather
than the shared-covariance fast path; the answer must not depend on which.
"""
function test_grouped_handles_ragged_trial_lengths()
    rng = StableRNG(404)
    lds, labels = pd_two_session_truth(; ntrials_per_session=3)
    _, y = rand(rng, lds, [30, 41, 27, 33, 30, 38])

    fitted = pd_lds(pd_state_model(), pd_obs_model())
    fitted.obs_model.depends_on = (C=labels, R=labels)
    elbos = fit!(fitted, y; max_iter=10, tol=0.0, progress=false)

    @test all(isfinite, elbos)
    @test pd_is_monotone(elbos)

    return nothing
end

# ============================================================================
# Fitting behaviour
# ============================================================================

function test_grouped_elbo_increases_and_recovers_noise()
    rng = StableRNG(505)
    truth, labels = pd_two_session_truth(; ntrials_per_session=8)
    _, y = rand(rng, truth, fill(80, length(labels)))

    fitted = pd_lds(pd_state_model(), pd_obs_model())
    fitted.obs_model.depends_on = (C=labels, R=labels)
    elbos = fit!(fitted, y; max_iter=60, tol=1e-8, progress=false)

    @test all(isfinite, elbos)
    @test pd_is_monotone(elbos)

    #=
    `C` is identifiable only up to a shared linear transform of the latent
    space, so compare what is identifiable: session 2 was simulated 7.5x
    noisier, and each session's noise covariance must come back that way rather
    than being averaged into one.
    =#
    R1 = group_parameter(fitted.obs_model, :R, :s1)
    R2 = group_parameter(fitted.obs_model, :R, :s2)
    @test isposdef(R1)
    @test isposdef(R2)
    @test tr(R2) > 3 * tr(R1)

    # Letting the emissions differ by session must fit at least as well as
    # forcing them to be shared.
    shared = pd_fresh_lds()
    fit!(shared, y; max_iter=60, tol=1e-8, progress=false)
    @test loglikelihood(fitted, y) > loglikelihood(shared, y)

    return nothing
end

function test_grouped_smooth_loglikelihood_and_heldout()
    rng = StableRNG(606)
    truth, labels = pd_two_session_truth(; ntrials_per_session=4)
    _, y = rand(rng, truth, fill(45, length(labels)))

    fitted = pd_lds(pd_state_model(), pd_obs_model())
    fitted.obs_model.depends_on = (C=labels, R=labels)
    fit!(fitted, y; max_iter=15, tol=1e-8, progress=false)

    xs, Ps = smooth(fitted, y)
    @test length(xs) == length(y)
    @test size(xs[1]) == (PD_LATENT_DIM, 45)
    @test size(Ps[1]) == (PD_LATENT_DIM, PD_LATENT_DIM, 45)
    @test all(x -> all(isfinite, x), xs)

    #=
    Each cell's smoothed covariance is computed once and shared within the
    cell, but the two sessions have different emissions, so their covariances
    must differ — i.e. sharing is per group, not global.
    =#
    @test Ps[1] ≈ Ps[2]
    @test !(Ps[1] ≈ Ps[end])

    @test isfinite(loglikelihood(fitted, y))
    @test isfinite(elbo(fitted, y))

    # Held-out scoring: a different trial count needs a `depends_on` override.
    y_new = y[[1, 5, 6]]
    lab_new = labels[[1, 5, 6]]
    @test_throws SSD.DimensionMismatchError loglikelihood(fitted, y_new)
    @test isfinite(loglikelihood(fitted, y_new; depends_on=(C=lab_new, R=lab_new)))
    xs_new, _ = smooth(fitted, y_new; depends_on=(C=lab_new, R=lab_new))
    @test length(xs_new) == 3

    return nothing
end

function test_grouped_integer_labels_and_priors()
    rng = StableRNG(707)
    labels = [1, 1, 2, 2, 2]          # integers, not Symbols
    truth = pd_fresh_lds()
    _, y = rand(rng, truth, fill(40, 5))

    sm = pd_state_model()
    sm.Q_prior = IWPrior(; Ψ=Matrix(0.01 * I(PD_LATENT_DIM)), ν=6.0)
    om = pd_obs_model()
    om.R_prior = IWPrior(; Ψ=Matrix(0.05 * I(PD_OBS_DIM)), ν=6.0)
    om.depends_on = (C=labels, R=labels)
    fitted = pd_lds(sm, om)

    elbos = fit!(fitted, y; max_iter=25, tol=1e-9, progress=false)
    @test all(isfinite, elbos)
    @test pd_is_monotone(elbos)

    @test group_labels(om, :C) == [1, 2]
    # Each version of `R` gets its own prior term and its own MAP update.
    @test isposdef(group_parameter(om, :R, 1))
    @test isposdef(group_parameter(om, :R, 2))

    return nothing
end

function test_grouped_rand_needs_a_label_for_one_trial()
    truth, labels = pd_two_session_truth(; ntrials_per_session=2)
    rng = StableRNG(808)

    # A single trial cannot be assigned to a group on its own.
    @test_throws ArgumentError rand(rng, truth, 20)
    x, y = rand(rng, truth, 20; depends_on=(C=[:s2], R=[:s2]))
    @test size(y) == (PD_OBS_DIM, 20)

    # Multi-trial sampling uses the model's own labels.
    _, ys = rand(rng, truth, fill(20, length(labels)))
    @test length(ys) == length(labels)

    return nothing
end

# ============================================================================
# Poisson
# ============================================================================

function test_grouped_poisson_fit()
    rng = StableRNG(909)
    labels = vcat(fill(:s1, 3), fill(:s2, 3))

    sm = pd_state_model()
    om = PoissonObservationModel([0.6 0.1; -0.2 0.5], [1.0, 0.8])
    om.depends_on = (C=labels,)
    truth = LinearDynamicalSystem(;
        state_model=sm,
        obs_model=om,
        latent_dim=PD_LATENT_DIM,
        obs_dim=PD_OBS_DIM,
        fit_bool=fill(true, 5),
    )
    C2 = group_parameter(om, :C, :s2)
    C2 .= [-0.4 0.7; 0.5 -0.3]
    d2 = group_parameter(om, :d, :s2)
    d2 .= [0.2, 1.4]

    _, y = rand(rng, truth, fill(40, length(labels)))

    om_fit = PoissonObservationModel([0.5 0.0; 0.0 0.5], [1.0, 1.0])
    om_fit.depends_on = (C=labels,)
    fitted = LinearDynamicalSystem(;
        state_model=pd_state_model(),
        obs_model=om_fit,
        latent_dim=PD_LATENT_DIM,
        obs_dim=PD_OBS_DIM,
        fit_bool=fill(true, 5),
    )

    elbos = fit!(fitted, y; max_iter=8, tol=0.0, progress=false)
    @test all(isfinite, elbos)
    @test elbos[end] > elbos[1]

    # The two sessions really did get separate emission parameters.
    @test !(group_parameter(om_fit, :C, :s1) ≈ group_parameter(om_fit, :C, :s2))
    @test isfinite(elbo(fitted, y))

    xs, _ = smooth(fitted, y)
    @test length(xs) == length(y)

    return nothing
end

# ============================================================================
# SLDS
# ============================================================================

function pd_slds(labels; K::Int=2)
    ldss = map(1:K) do k
        sm = pd_state_model()
        sm.A .= k == 1 ? [0.95 0.05; -0.05 0.95] : [0.60 0.30; -0.30 0.60]
        om = pd_obs_model()
        if labels !== nothing
            om.depends_on = (C=labels, R=labels)
        end
        return pd_lds(sm, om)
    end
    A = [0.9 0.1; 0.1 0.9]
    πₖ = [0.5, 0.5]
    return SLDS(; A=A, πₖ=πₖ, LDSs=ldss)
end

function test_grouped_slds_fit()
    rng = StableRNG(1010)
    labels = vcat(fill(:s1, 3), fill(:s2, 3))

    truth = pd_slds(labels)
    for k in 1:2
        C2 = group_parameter(truth.LDSs[k].obs_model, :C, :s2)
        C2 .= [0.2 -0.9; 1.0 0.3]
        R2 = group_parameter(truth.LDSs[k].obs_model, :R, :s2)
        R2 .= Matrix(1.0 * I(PD_OBS_DIM))
    end
    _, _, y = rand(rng, truth, fill(35, length(labels)))

    fitted = pd_slds(labels)
    elbos = fit!(fitted, y; max_iter=6, progress=false, rng=StableRNG(11))
    @test length(elbos) == 6
    @test all(isfinite, elbos)

    # Each regime keeps a separate emission per session.
    for k in 1:2
        om = fitted.LDSs[k].obs_model
        @test !(group_parameter(om, :C, :s1) ≈ group_parameter(om, :C, :s2))
        @test isposdef(group_parameter(om, :R, :s1))
        @test isposdef(group_parameter(om, :R, :s2))
    end

    # x0/P0 stay tied across regimes even when grouped.
    @test fitted.LDSs[2].state_model.x0 ≈ fitted.LDSs[1].state_model.x0
    @test fitted.LDSs[2].state_model.P0 ≈ fitted.LDSs[1].state_model.P0

    @test isfinite(elbo(fitted, y; rng=StableRNG(12)))

    return nothing
end

function test_grouped_slds_requires_matching_labels()
    labels = vcat(fill(:s1, 3), fill(:s2, 3))
    slds = pd_slds(labels)
    # Regime 2 disagrees about how trials are grouped.
    slds.LDSs[2].obs_model.depends_on = (C=vcat(fill(:s1, 4), fill(:s2, 2)),)
    @test_throws ArgumentError SSD._slds_parameter_grouping(slds, 6)

    # A regime that declares nothing at all is also a disagreement.
    slds2 = pd_slds(labels)
    slds2.LDSs[2].obs_model.depends_on = nothing
    @test_throws ArgumentError SSD._slds_parameter_grouping(slds2, 6)

    # No regime declaring anything is simply the ungrouped path.
    @test SSD._slds_parameter_grouping(pd_slds(nothing), 6) === nothing

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

#=
A parameter that does *not* vary is fit from every trial that carries it, even
when the trials fall in different cells. Grouping `:R` alone is the case that
exercises it: `[C d D]` stays pooled, so both cells share one emission slot and
their observation sufficient statistics have to be summed before the M-step
solves for it.
=#
function test_grouped_pools_obs_stats_across_cells()
    rng = StableRNG(808)
    labels = [:a, :a, :a, :b, :b, :b]
    _, y = rand(rng, pd_fresh_lds(), fill(40, length(labels)))

    om = pd_obs_model()
    om.depends_on = (R=labels,)
    fitted = pd_lds(pd_state_model(), om)
    elbos = fit!(fitted, y; max_iter=15, tol=1e-9, progress=false)

    @test all(isfinite, elbos)
    @test pd_is_monotone(elbos)
    @test length(om.variants) == 2
    # One `[C d D]`, shared by reference across both cells...
    @test om.variants[1].C === om.variants[2].C
    @test om.variants[1].d === om.variants[2].d
    # ... while each cell keeps its own noise covariance.
    @test !(om.variants[1].R ≈ om.variants[2].R)
    @test isposdef(group_parameter(om, :R, :a))
    @test isposdef(group_parameter(om, :R, :b))
    return nothing
end

#=
`smooth` keeps the shape convention of its input: a single trial handed over as
a plain matrix comes back as one `x`/`P` pair rather than one-element vectors.
Grouped models go through their own smoother, so the convention is re-asserted
here.
=#
function test_grouped_smooth_accepts_a_single_trial_matrix()
    rng = StableRNG(809)
    labels = [:a, :a, :a, :b, :b, :b]
    _, y = rand(rng, pd_fresh_lds(), fill(30, length(labels)))

    om = pd_obs_model()
    om.depends_on = (C=labels,)
    fitted = pd_lds(pd_state_model(), om)
    fit!(fitted, y; max_iter=3, progress=false)

    x1, P1 = smooth(fitted, y[1]; depends_on=(C=[:a],))
    @test x1 isa AbstractMatrix
    @test size(x1) == (PD_LATENT_DIM, size(y[1], 2))
    @test size(P1) == (PD_LATENT_DIM, PD_LATENT_DIM, size(y[1], 2))

    # The vector form of the same trial agrees, and keeps the vector shape.
    xs, Ps = smooth(fitted, [y[1]]; depends_on=(C=[:a],))
    @test xs isa AbstractVector
    @test xs[1] ≈ x1
    @test Ps[1] ≈ P1
    return nothing
end

#=
The matrix-normal priors contribute a term per *pair* of slots — `[A b B]` with
`Q`, `[C d D]` with `R`, `x0` with `P0` — because each term needs both halves.
`test_grouped_integer_labels_and_priors` covers the inverse-Wishart ones, which
are indexed by a single slot; these are the paired ones.
=#
function test_grouped_matrix_normal_priors()
    rng = StableRNG(810)
    labels = [:a, :a, :a, :b, :b, :b]
    _, y = rand(rng, pd_fresh_lds(), fill(40, length(labels)))

    sm = pd_state_model()
    sm.x0_prior = x0_mean_prior(zeros(PD_LATENT_DIM); κ₀=1.0)
    sm.AB_prior = StateSpaceDynamics.MNPrior(;
        M₀=zeros(PD_LATENT_DIM, PD_LATENT_DIM + 1), Λ=Matrix(0.1 * I(PD_LATENT_DIM + 1))
    )
    om = pd_obs_model()
    om.CD_prior = StateSpaceDynamics.MNPrior(;
        M₀=zeros(PD_OBS_DIM, PD_LATENT_DIM + 1), Λ=Matrix(0.1 * I(PD_LATENT_DIM + 1))
    )
    om.depends_on = (C=labels, R=labels)
    fitted = pd_lds(sm, om)

    elbos = fit!(fitted, y; max_iter=15, tol=1e-9, progress=false)
    @test all(isfinite, elbos)
    @test pd_is_monotone(elbos)

    # The prior terms are part of the objective, so the ELBO stays below the
    # unpenalised one on the same data.
    om_free = pd_obs_model()
    om_free.depends_on = (C=labels, R=labels)
    free = pd_lds(pd_state_model(), om_free)
    free_elbos = fit!(free, y; max_iter=15, tol=1e-9, progress=false)
    @test isfinite(free_elbos[end])
    @test elbos[end] < free_elbos[end]
    return nothing
end

#=
The EM loop returns as soon as the ELBO stops moving, and the vector it hands
back is truncated to the iterations actually run rather than padded to
`max_iter`. `progress=true` drives the meter alongside it.
=#
function test_grouped_fit_stops_early_and_reports_progress()
    rng = StableRNG(811)
    labels = [:a, :a, :a, :b, :b, :b]
    _, y = rand(rng, pd_fresh_lds(), fill(40, length(labels)))

    om = pd_obs_model()
    om.depends_on = (C=labels,)
    fitted = pd_lds(pd_state_model(), om)
    max_iter = 50
    elbos = fit!(fitted, y; max_iter=max_iter, tol=1e-1, progress=true)

    @test length(elbos) < max_iter
    @test all(isfinite, elbos)
    @test pd_is_monotone(elbos)
    # It really did stop on the tolerance, not on an accident of length.
    @test abs(elbos[end] - elbos[end - 1]) < 1e-1

    # Running to `max_iter` instead returns the full vector.
    om2 = pd_obs_model()
    om2.depends_on = (C=labels,)
    fitted2 = pd_lds(pd_state_model(), om2)
    @test length(fit!(fitted2, y; max_iter=4, tol=0.0, progress=false)) == 4
    return nothing
end

# A grouped Poisson truth: two sessions sharing dynamics, each with its own
# emission, matching `test_grouped_poisson_fit`'s setup.
function pd_grouped_poisson_data(seed::Int; ntrials_per::Int=3, tsteps::Int=40)
    labels = vcat(fill(:s1, ntrials_per), fill(:s2, ntrials_per))
    om = PoissonObservationModel([0.6 0.1; -0.2 0.5], [1.0, 0.8])
    om.depends_on = (C=labels,)
    truth = LinearDynamicalSystem(;
        state_model=pd_state_model(),
        obs_model=om,
        latent_dim=PD_LATENT_DIM,
        obs_dim=PD_OBS_DIM,
        fit_bool=fill(true, 5),
    )
    group_parameter(om, :C, :s2) .= [-0.4 0.7; 0.5 -0.3]
    group_parameter(om, :d, :s2) .= [0.2, 1.4]
    _, y = rand(StableRNG(seed), truth, fill(tsteps, length(labels)))
    return labels, y
end

function pd_grouped_poisson_lds(labels; Λ::Real=0.0)
    om = PoissonObservationModel([0.6 0.1; -0.2 0.5], [1.0, 0.8])
    if Λ > 0
        om.CD_prior = StateSpaceDynamics.MNPrior(;
            M₀=zeros(PD_OBS_DIM, PD_LATENT_DIM + 1), Λ=Matrix(Λ * I(PD_LATENT_DIM + 1))
        )
    end
    om.depends_on = (C=labels,)
    return LinearDynamicalSystem(;
        state_model=pd_state_model(),
        obs_model=om,
        latent_dim=PD_LATENT_DIM,
        obs_dim=PD_OBS_DIM,
        fit_bool=fill(true, 5),
    )
end

#=
A Poisson emission has no noise covariance, so its `[C d D]` prior cannot be the
conjugate matrix-normal update the Gaussian side uses; it enters the objective
as a bare quadratic penalty, once per version of `[C d D]`. The test drives it
from the outside: with `M₀ = 0`, tightening `Λ` has to pull every group's
emission towards zero and cost ELBO for doing so.
=#
function test_grouped_poisson_cd_prior()
    labels, y = pd_grouped_poisson_data(909)

    function emission_norm(om)
        return sum(
            norm(hcat(group_parameter(om, :C, l), group_parameter(om, :d, l))) for
            l in group_labels(om, :C)
        )
    end

    results = map((0.0, 10.0, 200.0)) do Λ
        fitted = pd_grouped_poisson_lds(labels; Λ=Λ)
        elbos = fit!(fitted, y; max_iter=25, tol=0.0, progress=false)
        @test all(isfinite, elbos)
        @test pd_is_monotone(elbos)
        (elbos[end], emission_norm(fitted.obs_model))
    end

    # Tighter prior, emission pulled harder towards `M₀ = 0`.
    @test results[1][2] > results[2][2] > results[3][2]
    # ... and the penalty it pays shows up in the objective.
    @test results[1][1] > results[2][1] > results[3][1]
    return nothing
end

#=
The Laplace-EM loop returns as soon as the ELBO stops moving, truncating the
vector to the iterations actually run rather than padding to `max_iter`.
`progress=true` drives the meter alongside it. The Gaussian driver has its own
copy of this loop, covered separately.
=#
function test_grouped_poisson_stops_early_and_reports_progress()
    labels, y = pd_grouped_poisson_data(910)
    fitted = pd_grouped_poisson_lds(labels)

    max_iter = 40
    elbos = fit!(fitted, y; max_iter=max_iter, tol=1e-1, progress=true)
    @test length(elbos) < max_iter
    @test all(isfinite, elbos)
    @test pd_is_monotone(elbos)
    @test abs(elbos[end] - elbos[end - 1]) < 1e-1

    # Running to `max_iter` instead returns the full vector.
    full = fit!(pd_grouped_poisson_lds(labels), y; max_iter=4, tol=0.0, progress=false)
    @test length(full) == 4
    return nothing
end
