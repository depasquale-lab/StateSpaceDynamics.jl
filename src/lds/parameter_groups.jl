#=============================================================================
Ancillary parameter dependencies (`depends_on`)

`AbstractStateModel` and `AbstractObservationModel` each carry a `depends_on`
field. When it is `nothing` (the default) the model behaves exactly as before:
one parameter set shared by every trial, and every entry point takes its
original ungrouped code path. When it is a `NamedTuple`, the named parameters
are estimated separately for each group of trials:

    session = [:a, :a, :b, :b, :b]          # one label per trial
    obs = GaussianObservationModel(C, R, d)
    obs.depends_on = (C = session, d = session, R = session)

Trials labelled `:a` then contribute to one `[C d D]` / `R` estimate and trials
labelled `:b` to another, while the latent dynamics `A`, `b`, `B`, `Q` and the
initial state `x0`, `P0` stay shared. That is the "stitching" setup for pooling
recording sessions that observe different neurons in the same animal: latent
dynamics common to the animal, emission parameters specific to the session.

Keys are canonicalized to the same groups `fit_bool` uses, because those
parameters are fit jointly as one regression:

    :x0             -> initial state mean      (fit_bool[1])
    :P0             -> initial state cov       (fit_bool[2])
    :A, :b, :B      -> dynamics [A b B]        (fit_bool[3])
    :Q              -> process noise           (fit_bool[4])
    :C, :d, :D      -> emission [C d D]        (fit_bool[5])
    :R              -> observation noise       (fit_bool[6], Gaussian only)

Because those parameters are fit jointly, grouping any one of them groups them
all. Naming one on its own would read as though only that one varies, so a
`depends_on` has to name **every** member of a group it touches: `(C = s, d =
s)` rather than `(C = s,)`, and `(A = s, b = s)` rather than `(A = s,)`. `B`
and `D` are zero-column when the model takes no inputs, and are then not part
of the group; give the model inputs and they have to be named too. `:x0`,
`:P0`, `:Q` and `:R` are groups of one and stand alone. Naming two members of
one group with different label vectors is an error.

Implementation shape: a *cell* is a maximal set of trials sharing every
parameter — one element of the common refinement of all the label vectors.
Trials in a cell are governed by a single `LinearDynamicalSystem`, so the E-step
runs on them completely unchanged (which is what keeps the equal-length
shared-covariance fast path alive within each group), and only the M-step and
the ELBO need to know about groups.

This file holds the declaration side of that design: validating `depends_on`,
resolving it into per-parameter versions, building the `variants` storage, and
computing the trial partition. The grouped M-step and ELBO that consume the
partition live in `grouped_em.jl`.
=============================================================================#

#=
Any model that carries a `depends_on` field. Spelling it out (rather than
leaving these helpers untyped) lets inference union-split over the three
concrete model types instead of dispatching dynamically on `Any`.
=#
const DependentModel = Union{AbstractStateModel,AbstractObservationModel}

# Parameter-group ordinals; identical to the `fit_bool` layout, and the index
# into a `ParameterGrouping`'s `nslots` / `cell_slot`.
const _G_X0 = 1
const _G_P0 = 2
const _G_AB = 3
const _G_Q = 4
const _G_CD = 5
const _G_R = 6

# Canonical group names, ordered to match `fit_bool`. State-model groups occupy
# `fit_bool` slots 1:4 and observation-model groups slots 5:6 (5:5 for Poisson).
_group_names(::GaussianStateModel) = (:x0, :P0, :A, :Q)
_group_names(::GaussianObservationModel) = (:C, :R)
_group_names(::PoissonObservationModel) = (:C,)

#=
Group resolution: `[A b B]` and `[C d D]` are each fit as a single regression,
so grouping any member groups them all. Because that is not what naming one of
them looks like, a `depends_on` must name every member the model carries —
see `_check_group_naming_complete`. These functions map a member to the group
it belongs to once that check has passed.

`_try_canonical_param` returns `nothing` for a name the model does not own,
which is what a *shared* `depends_on` needs: a call-site override is one
NamedTuple resolved against both sub-models, so each has to be able to ignore
the other's keys rather than reject them.
=#
function _try_canonical_param(::GaussianStateModel, name::Symbol)
    name in (:A, :b, :B) && return :A
    name in (:x0, :P0, :Q) && return name
    return nothing
end

function _try_canonical_param(::GaussianObservationModel, name::Symbol)
    name in (:C, :d, :D) && return :C
    name === :R && return :R
    return nothing
end

function _try_canonical_param(::PoissonObservationModel, name::Symbol)
    name in (:C, :d, :D) && return :C
    return nothing
end

_valid_param_names(::GaussianStateModel) = ":x0, :P0, :A, :b, :B, :Q"
_valid_param_names(::GaussianObservationModel) = ":C, :d, :D, :R"
function _valid_param_names(::PoissonObservationModel)
    return ":C, :d, :D (a Poisson emission has no noise covariance)"
end

"""
    _group_members(model, canonical) -> Tuple{Vararg{Symbol}}

The parameters the group `canonical` covers, restricted to the ones this model
actually carries. `B` and `D` default to zero-column matrices when the model
takes no inputs; there is nothing to group there, so they are left out.
"""
_group_members(m::GaussianStateModel, c::Symbol) =
    c === :A ? (size(m.B, 2) > 0 ? (:A, :b, :B) : (:A, :b)) : (c,)
function _group_members(m::GaussianObservationModel, c::Symbol)
    return c === :C ? (size(m.D, 2) > 0 ? (:C, :d, :D) : (:C, :d)) : (c,)
end
function _group_members(m::PoissonObservationModel, c::Symbol)
    return c === :C ? (size(m.D, 2) > 0 ? (:C, :d, :D) : (:C, :d)) : (c,)
end

"""
    _check_group_naming_complete(model, spec, what)

`[A b B]` and `[C d D]` are each fit as one regression, so grouping any member
groups them all. Naming one and leaving the others out reads as though only
that one varies, which is the opposite of what happens — so it is rejected
rather than quietly widened. Name every member the model carries.

Keys this model does not own are skipped: one `depends_on` is resolved against
both sub-models, so each has to ignore the other's.
"""
function _check_group_naming_complete(
    model::DependentModel, spec::NamedTuple, what::AbstractString
)
    named = Dict{Symbol,Vector{Symbol}}()
    for key in keys(spec)
        canonical = _try_canonical_param(model, key)
        canonical === nothing && continue
        push!(get!(named, canonical, Symbol[]), key)
    end
    for canonical in sort!(collect(keys(named)))
        used = named[canonical]
        members = _group_members(model, canonical)
        absent = [m for m in members if !(m in used)]
        isempty(absent) && continue
        given = join(("`:$k`" for k in used), ", ")
        want = join(("$m = ..." for m in members), ", ")
        throw(
            ArgumentError(
                "$what: $given names part of `[$(join(members, " "))]`, which is fit " *
                "jointly as one regression — grouping one member groups them all. " *
                "Say so explicitly by naming every member: `($want)`. " *
                "Missing $(join(("`:$m`" for m in absent), ", ")).",
            ),
        )
    end
    return nothing
end

function _canonical_param(model::DependentModel, name::Symbol)
    canonical = _try_canonical_param(model, name)
    canonical === nothing && throw(
        ArgumentError(
            "depends_on: `:$name` is not a parameter of a $(nameof(typeof(model))); " *
            "valid names are $(_valid_param_names(model))",
        ),
    )
    return canonical
end

"""
    ParameterDependence

**Internal.** Resolved, model-intrinsic form of a model's `depends_on`: for each
of the model's parameter groups, whether it varies, the ordered list of distinct
labels, and the per-trial labels as supplied.

`nslots[g]` is `length(labels[g])` for a varying group and `1` otherwise, so a
model's `variants` vector always has `prod(nslots)` entries and a variant's index
is a fixed function of its per-group slot indices — independent of any dataset,
which is what lets a `depends_on` override re-assign trials without invalidating
already-fitted parameters.
"""
struct ParameterDependence
    names::Vector{Symbol}
    varies::Vector{Bool}
    labels::Vector{Vector{Any}}
    trial_labels::Vector{Vector{Any}}
    nslots::Vector{Int}
end

_any_varies(dep::ParameterDependence) = any(dep.varies)

function _same_labels(a::Vector{Any}, b::Vector{Any})
    return length(a) == length(b) && all(isequal(a[i], b[i]) for i in eachindex(a))
end

"""
    _resolve_dependence(model) -> ParameterDependence

Validate `model.depends_on` and resolve it against the model's parameter groups.
Throws `ArgumentError` on unknown parameter names or on aliases of one group
carrying different label vectors, and `DimensionMismatchError` on label vectors
of unequal length.
"""
function _resolve_dependence(model::DependentModel)
    names = collect(Symbol, _group_names(model))
    ngroups = length(names)
    varies = fill(false, ngroups)
    labels = [Any[] for _ in 1:ngroups]
    trial_labels = [Any[] for _ in 1:ngroups]
    nslots = fill(1, ngroups)
    dep = ParameterDependence(names, varies, labels, trial_labels, nslots)

    spec = model.depends_on
    spec === nothing && return dep
    _check_group_naming_complete(model, spec, "depends_on")

    #=
    Track which user-facing key first claimed each group so a conflicting alias
    (`(C = s1, d = s2)`) can report both names rather than silently keeping one.
    =#
    claimed = Vector{Symbol}(undef, ngroups)
    for key in keys(spec)
        canonical = _canonical_param(model, key)
        g = findfirst(isequal(canonical), names)::Int
        supplied = getproperty(spec, key)
        supplied isa AbstractVector || throw(
            ArgumentError(
                "depends_on[:$key] must be a vector with one label per trial, " *
                "got a $(typeof(supplied))",
            ),
        )
        isempty(supplied) &&
            throw(ArgumentError("depends_on[:$key] is empty; expected one label per trial"))
        current = collect(Any, supplied)
        if varies[g]
            _same_labels(current, trial_labels[g]) || throw(
                ArgumentError(
                    "depends_on: `:$key` and `:$(claimed[g])` name the same parameter " *
                    "group (`:$canonical`, fit jointly as one regression) but were " *
                    "given different label vectors; supply one label vector per group",
                ),
            )
            continue
        end
        varies[g] = true
        claimed[g] = key
        trial_labels[g] = current
        labels[g] = collect(Any, unique(current))
        nslots[g] = length(labels[g])
    end

    ntrials = 0
    for g in 1:ngroups
        varies[g] || continue
        if ntrials == 0
            ntrials = length(trial_labels[g])
        elseif length(trial_labels[g]) != ntrials
            throw(
                DimensionMismatchError(
                    "depends_on[:$(names[g])] length", ntrials, length(trial_labels[g])
                ),
            )
        end
    end

    return dep
end

#=
Mixed-radix (column-major) index of a variant from its per-group slot indices,
and its inverse. Written out rather than going through `LinearIndices` so the
`nslots` vector never has to be splatted into a tuple.
=#
function _variant_index(nslots::AbstractVector{Int}, slots::AbstractVector{Int})
    idx = 0
    for g in length(nslots):-1:1
        idx = idx * nslots[g] + (slots[g] - 1)
    end
    return idx + 1
end

function _variant_slots(nslots::AbstractVector{Int}, index::Int)
    slots = Vector{Int}(undef, length(nslots))
    rest = index - 1
    for g in eachindex(nslots)
        slots[g] = rest % nslots[g] + 1
        rest = rest ÷ nslots[g]
    end
    return slots
end

"""
    _slot_of(dep, g, label) -> Int

Slot index of `label` within parameter group `g`. Throws when the label was not
present in the label vector the model was built with — an override may
re-assign trials to known groups but not introduce new parameter sets.
"""
function _slot_of(dep::ParameterDependence, g::Int, label)
    dep.varies[g] || return 1
    slot = findfirst(isequal(label), dep.labels[g])
    slot === nothing && throw(
        ArgumentError(
            "depends_on: label $(repr(label)) is not a known group of parameter " *
            "`:$(dep.names[g])` (known: $(join(map(repr, dep.labels[g]), ", "))). " *
            "An override may only re-assign trials to existing groups.",
        ),
    )
    return slot::Int
end

"""
    _trial_labels_for(dep, g, model, override) -> Vector{Any}

Per-trial labels for group `g`, taken from `override` when it names the group
(under any alias) and from the model's own `depends_on` otherwise.
"""
function _trial_labels_for(
    dep::ParameterDependence, g::Int, model::DependentModel, override
)
    if override !== nothing
        for key in keys(override)
            _try_canonical_param(model, key) === dep.names[g] || continue
            supplied = getproperty(override, key)
            supplied isa AbstractVector || throw(
                ArgumentError(
                    "depends_on override for `:$key` must be a vector with one label " *
                    "per trial, got a $(typeof(supplied))",
                ),
            )
            return collect(Any, supplied)
        end
    end
    return dep.trial_labels[g]
end

# ============================================================================
# Variant construction. Parameters that do not vary are shared *by reference*
# across every variant, so one M-step write updates all of them; parameters
# that do vary get one independent copy per slot, with slot 1 aliasing the base
# model's own array so `model.C` keeps its original meaning.
# ============================================================================

_slot_arrays(base, nslots::Int) = [i == 1 ? base : copy(base) for i in 1:nslots]

function _build_variants!(
    sm::GaussianStateModel{T,M,V}, dep::ParameterDependence
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    ncells = prod(dep.nslots)
    existing = sm.variants
    if existing !== nothing && length(existing) == ncells
        return existing
    end

    x0s = _slot_arrays(sm.x0, dep.nslots[1])
    P0s = _slot_arrays(sm.P0, dep.nslots[2])
    As = _slot_arrays(sm.A, dep.nslots[3])
    bs = _slot_arrays(sm.b, dep.nslots[3])
    Bs = _slot_arrays(sm.B, dep.nslots[3])
    Qs = _slot_arrays(sm.Q, dep.nslots[4])

    variants = Vector{GaussianStateModel{T,M,V}}(undef, ncells)
    for cell in 1:ncells
        s = _variant_slots(dep.nslots, cell)
        variants[cell] = GaussianStateModel{T,M,V}(;
            A=As[s[3]],
            Q=Qs[s[4]],
            b=bs[s[3]],
            x0=x0s[s[1]],
            P0=P0s[s[2]],
            B=Bs[s[3]],
            Q_prior=sm.Q_prior,
            P0_prior=sm.P0_prior,
            AB_prior=sm.AB_prior,
            x0_prior=sm.x0_prior,
        )
    end
    sm.variants = variants
    return variants
end

function _build_variants!(
    om::GaussianObservationModel{T,M,V}, dep::ParameterDependence
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    ncells = prod(dep.nslots)
    existing = om.variants
    if existing !== nothing && length(existing) == ncells
        return existing
    end

    Cs = _slot_arrays(om.C, dep.nslots[1])
    ds = _slot_arrays(om.d, dep.nslots[1])
    Ds = _slot_arrays(om.D, dep.nslots[1])
    Rs = _slot_arrays(om.R, dep.nslots[2])

    variants = Vector{GaussianObservationModel{T,M,V}}(undef, ncells)
    for cell in 1:ncells
        s = _variant_slots(dep.nslots, cell)
        variants[cell] = GaussianObservationModel{T,M,V}(;
            C=Cs[s[1]],
            R=Rs[s[2]],
            d=ds[s[1]],
            D=Ds[s[1]],
            R_prior=om.R_prior,
            CD_prior=om.CD_prior,
        )
    end
    om.variants = variants
    return variants
end

function _build_variants!(
    om::PoissonObservationModel{T,M,V}, dep::ParameterDependence
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    ncells = prod(dep.nslots)
    existing = om.variants
    if existing !== nothing && length(existing) == ncells
        return existing
    end

    Cs = _slot_arrays(om.C, dep.nslots[1])
    ds = _slot_arrays(om.d, dep.nslots[1])
    Ds = _slot_arrays(om.D, dep.nslots[1])

    variants = Vector{PoissonObservationModel{T,M,V}}(undef, ncells)
    for cell in 1:ncells
        s = _variant_slots(dep.nslots, cell)
        variants[cell] = PoissonObservationModel{T,M,V}(;
            C=Cs[s[1]], d=ds[s[1]], D=Ds[s[1]], CD_prior=om.CD_prior
        )
    end
    om.variants = variants
    return variants
end

# ============================================================================
# Trial partition
# ============================================================================

"""
    ParameterGrouping

**Internal.** The trial partition induced by a model's `depends_on`, together
with the per-cell parameter lookups the grouped E/M-steps need.

# Fields
- `names`: canonical parameter-group names, ordered as `fit_bool`
- `ncells`: number of occupied cells
- `trial_cell`: cell index of each trial
- `cell_trials`: trial indices of each cell
- `cell_state` / `cell_obs`: index into the state / observation model's
  `variants` vector for each cell
- `nslots[g]`: number of distinct versions of parameter group `g`
- `cell_slot[g][cell]`: which version of group `g` a cell uses
- `slot_labels[g][slot]`: the user's label for that version (`nothing` when the
  group does not vary)
"""
struct ParameterGrouping
    names::Vector{Symbol}
    ncells::Int
    trial_cell::Vector{Int}
    cell_trials::Vector{Vector{Int}}
    cell_state::Vector{Int}
    cell_obs::Vector{Int}
    nslots::Vector{Int}
    cell_slot::Vector{Vector{Int}}
    slot_labels::Vector{Vector{Any}}
end

"""
    _validate_override_keys(sm, om, dep_s, dep_o, override)

Reject a call-site `depends_on` that names something neither sub-model owns (a
typo), or a parameter the model does not declare as varying. An override
re-assigns trials to existing groups; it cannot introduce a parameter set that
was never fitted.
"""
function _validate_override_keys(
    sm::DependentModel,
    om::DependentModel,
    dep_s::ParameterDependence,
    dep_o::ParameterDependence,
    override::Union{Nothing,NamedTuple},
)
    override === nothing && return nothing
    for key in keys(override)
        cs = _try_canonical_param(sm, key)
        co = _try_canonical_param(om, key)
        cs === nothing &&
            co === nothing &&
            throw(
                ArgumentError(
                    "depends_on override: `:$key` is not a parameter of either sub-model " *
                    "(state: $(_valid_param_names(sm)); " *
                    "observation: $(_valid_param_names(om)))",
                ),
            )
        declared =
            (cs !== nothing && dep_s.varies[findfirst(isequal(cs), dep_s.names)::Int]) ||
            (co !== nothing && dep_o.varies[findfirst(isequal(co), dep_o.names)::Int])
        declared || throw(
            ArgumentError(
                "depends_on override: `:$key` is not declared as depending on an " *
                "ancillary variable by the model; an override may only re-assign " *
                "trials to groups the model already declares",
            ),
        )
    end
    #=
    An override is a `depends_on` like any other, and re-labels the whole
    regression group, so it has to name the group as completely as the
    declaration did.
    =#
    _check_group_naming_complete(sm, override, "depends_on override")
    _check_group_naming_complete(om, override, "depends_on override")
    return nothing
end

"""
    parameter_grouping(lds, ntrials; depends_on=nothing)

Build the trial partition for `lds` over `ntrials` trials, or return `nothing`
when no parameter of either sub-model depends on an ancillary variable.

`depends_on` optionally overrides the label vectors stored on the models, for
scoring a dataset whose trial count differs from the fitted one.
"""
function parameter_grouping(
    lds::LinearDynamicalSystem, ntrials::Int; depends_on::Union{Nothing,NamedTuple}=nothing
)
    sm = lds.state_model
    om = lds.obs_model
    dep_s = _resolve_dependence(sm)
    dep_o = _resolve_dependence(om)

    #=
    An override may only re-assign trials to groups the model already knows, so
    a model with no `depends_on` at all has nothing to override and stays on the
    ungrouped path.
    =#
    if !_any_varies(dep_s) && !_any_varies(dep_o)
        depends_on === nothing || throw(
            ArgumentError(
                "a `depends_on` override was supplied but neither sub-model declares " *
                "`depends_on`; set it on the model before fitting",
            ),
        )
        return nothing
    end

    _validate_override_keys(sm, om, dep_s, dep_o, depends_on)

    _build_variants!(sm, dep_s)
    _build_variants!(om, dep_o)

    ngroups_s = length(dep_s.names)
    ngroups_o = length(dep_o.names)
    ngroups = ngroups_s + ngroups_o
    names = vcat(dep_s.names, dep_o.names)
    varies = vcat(dep_s.varies, dep_o.varies)

    # Per-trial labels for every group (state groups first, matching fit_bool).
    trial_labels = Vector{Vector{Any}}(undef, ngroups)
    for g in 1:ngroups_s
        trial_labels[g] = _trial_labels_for(dep_s, g, sm, depends_on)
    end
    for g in 1:ngroups_o
        trial_labels[ngroups_s + g] = _trial_labels_for(dep_o, g, om, depends_on)
    end

    for g in 1:ngroups
        varies[g] || continue
        length(trial_labels[g]) == ntrials || throw(
            DimensionMismatchError(
                "depends_on labels for :$(names[g])", ntrials, length(trial_labels[g])
            ),
        )
    end

    # Per-trial slot indices, then the per-trial variant index of each sub-model.
    state_idx = Vector{Int}(undef, ntrials)
    obs_idx = Vector{Int}(undef, ntrials)
    s_slots = Vector{Int}(undef, ngroups_s)
    o_slots = Vector{Int}(undef, ngroups_o)
    for n in 1:ntrials
        for g in 1:ngroups_s
            s_slots[g] = dep_s.varies[g] ? _slot_of(dep_s, g, trial_labels[g][n]) : 1
        end
        for g in 1:ngroups_o
            gg = ngroups_s + g
            o_slots[g] = dep_o.varies[g] ? _slot_of(dep_o, g, trial_labels[gg][n]) : 1
        end
        state_idx[n] = _variant_index(dep_s.nslots, s_slots)
        obs_idx[n] = _variant_index(dep_o.nslots, o_slots)
    end

    # Occupied cells, in order of first appearance.
    trial_cell = Vector{Int}(undef, ntrials)
    cell_state = Int[]
    cell_obs = Int[]
    cell_trials = Vector{Int}[]
    for n in 1:ntrials
        cell = 0
        for c in eachindex(cell_state)
            if cell_state[c] == state_idx[n] && cell_obs[c] == obs_idx[n]
                cell = c
                break
            end
        end
        if cell == 0
            push!(cell_state, state_idx[n])
            push!(cell_obs, obs_idx[n])
            push!(cell_trials, Int[])
            cell = length(cell_state)
        end
        trial_cell[n] = cell
        push!(cell_trials[cell], n)
    end

    ncells = length(cell_state)
    cell_slot = [Vector{Int}(undef, ncells) for _ in 1:ngroups]
    for c in 1:ncells
        cs = _variant_slots(dep_s.nslots, cell_state[c])
        co = _variant_slots(dep_o.nslots, cell_obs[c])
        for g in 1:ngroups_s
            cell_slot[g][c] = cs[g]
        end
        for g in 1:ngroups_o
            cell_slot[ngroups_s + g][c] = co[g]
        end
    end

    slot_labels = Vector{Vector{Any}}(undef, ngroups)
    for g in 1:ngroups_s
        slot_labels[g] = dep_s.varies[g] ? dep_s.labels[g] : Any[nothing]
    end
    for g in 1:ngroups_o
        gg = ngroups_s + g
        slot_labels[gg] = dep_o.varies[g] ? dep_o.labels[g] : Any[nothing]
    end

    return ParameterGrouping(
        names,
        ncells,
        trial_cell,
        cell_trials,
        cell_state,
        cell_obs,
        vcat(dep_s.nslots, dep_o.nslots),
        cell_slot,
        slot_labels,
    )
end

"""
    _has_parameter_dependence(model) -> Bool

Whether a model (or either sub-model of a `LinearDynamicalSystem`) declares a
`depends_on` that actually varies a parameter.
"""
function _has_parameter_dependence(model::DependentModel)
    model.depends_on === nothing && return false
    return _any_varies(_resolve_dependence(model))
end

function _has_parameter_dependence(lds::LinearDynamicalSystem)
    return _has_parameter_dependence(lds.state_model) ||
           _has_parameter_dependence(lds.obs_model)
end

"""
    _reject_unsupported_dependence(model)

**Internal, temporary.** Refuse to run an entry point that does not yet honour
`depends_on`. The grouped E/M-step lands one model family at a time, so in the
interim a declared dependence must be an error rather than a silently pooled
fit. Each entry point drops this call as its grouped path lands.
"""
function _reject_unsupported_dependence(lds::LinearDynamicalSystem)
    _has_parameter_dependence(lds) || return nothing
    return throw(
        ArgumentError(
            "this model declares `depends_on`, but fitting and inference with " *
            "condition-dependent parameters are not implemented yet for this model " *
            "family — the grouped E/M-step lands in a follow-up. Unset `depends_on` " *
            "to fit the model with one parameter set shared by every trial.",
        ),
    )
end

function _reject_unsupported_dependence(slds::SLDS)
    for lds in slds.LDSs
        _reject_unsupported_dependence(lds)
    end
    return nothing
end

"""
    _single_trial_group_error(what)

`rand` with a scalar `tsteps` draws one trial, which a grouped model cannot
assign to a parameter version on its own.
"""
function _single_trial_group_error(what::AbstractString)
    return throw(
        ArgumentError(
            "rand(rng, $what, tsteps) samples a single trial, but this model's " *
            "parameters depend on an ancillary variable — say which group the trial " *
            "belongs to, e.g. `depends_on=(C=[:session_a],)`, or sample several trials " *
            "at once with `rand(rng, $what, fill(tsteps, ntrials))`.",
        ),
    )
end

"""
    _cell_lds(lds, grp, cell) -> LinearDynamicalSystem

A view of `lds` restricted to one cell: the state and observation model objects
holding that cell's parameter arrays. Construction is two immutable struct
allocations — the parameter arrays are shared, never copied — so the existing
kernels can be run per cell without any group awareness of their own.
"""
function _cell_lds(
    lds::LinearDynamicalSystem{T,S,O}, grp::ParameterGrouping, cell::Int
) where {T<:Real,S<:AbstractStateModel{T},O<:AbstractObservationModel{T}}
    sm = (lds.state_model.variants::Vector{S})[grp.cell_state[cell]]
    om = (lds.obs_model.variants::Vector{O})[grp.cell_obs[cell]]
    return LinearDynamicalSystem{T,S,O}(
        sm, om, lds.latent_dim, lds.obs_dim, lds.ux_dim, lds.uy_dim, lds.fit_bool
    )
end

"""
    _cell_ldss(lds, grp) -> Vector{LinearDynamicalSystem}

One `_cell_lds` per occupied cell.
"""
function _cell_ldss(
    lds::LinearDynamicalSystem{T,S,O}, grp::ParameterGrouping
) where {T<:Real,S<:AbstractStateModel{T},O<:AbstractObservationModel{T}}
    return [_cell_lds(lds, grp, c) for c in 1:(grp.ncells)]
end

"""
    _subset_data(data, trials) -> Data

The sub-`Data` holding only `trials`, sharing the per-trial matrices by
reference (nothing is copied).
"""
function _subset_data(data::Data, trials::AbstractVector{Int})
    return Data(data.y[trials], data.ux[trials], data.uy[trials], data.tsteps[trials])
end

# ============================================================================
# Public accessors
# ============================================================================

"""
    group_labels(model, name) -> Vector

Distinct labels of the parameter group `name` on a state or observation model,
in the order their parameter versions are stored. Returns an empty vector when
that parameter does not depend on an ancillary variable.

```julia
group_labels(obs_model, :C)   # [:session_a, :session_b]
```

Members of a jointly-fitted group are aliases, so `group_labels(m, :d)` is the
same as `group_labels(m, :C)`.
"""
function group_labels(model::DependentModel, name::Symbol)
    dep = _resolve_dependence(model)
    canonical = _canonical_param(model, name)
    g = findfirst(isequal(canonical), dep.names)::Int
    return dep.varies[g] ? copy(dep.labels[g]) : Any[]
end

"""
    group_parameter(model, name, label)

The version of parameter `name` fitted from the trials labelled `label`.

```julia
group_parameter(obs_model, :C, :session_a)   # that session's emission matrix
group_parameter(obs_model, :d, :session_a)   # its emission bias
```

Throws when the model declares no dependence for that parameter (read the field
directly in that case) or when `label` is not one of its groups.
"""
function group_parameter(model::DependentModel, name::Symbol, label)
    dep = _resolve_dependence(model)
    canonical = _canonical_param(model, name)
    g = findfirst(isequal(canonical), dep.names)::Int
    dep.varies[g] || throw(
        ArgumentError(
            "parameter `:$name` of this $(nameof(typeof(model))) does not depend on an " *
            "ancillary variable; read `model.$name` directly",
        ),
    )
    slots = fill(1, length(dep.nslots))
    slots[g] = _slot_of(dep, g, label)
    variants = _build_variants!(model, dep)
    return getproperty(variants[_variant_index(dep.nslots, slots)], name)
end
