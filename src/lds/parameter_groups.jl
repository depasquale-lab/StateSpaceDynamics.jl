#=============================================================================
Ancillary parameter dependencies (`depends_on`)

`AbstractStateModel` and `AbstractObservationModel` each carry a `depends_on`
field. When it is `nothing` (the default) the model behaves exactly as before:
one parameter set shared by every trial, and every entry point takes its
original ungrouped code path. When it is a `NamedTuple`, the named parameters
are estimated separately for each group of trials:

    session = [:a, :a, :b, :b, :b]          # one label per trial
    obs = GaussianObservationModel(C, R, d)
    obs.depends_on = (C = session, R = session)

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

So `:d` is an alias for `:C`; naming either makes the whole `[C d D]`
regression group-dependent, and naming two aliases of one group with different
label vectors is an error.

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
Alias resolution: `[A b B]` and `[C d D]` are each fit as a single regression,
so naming any member of a group makes the whole group group-dependent.

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

"""
    ObsSlotSpec{T}

**Internal.** Per-slot observation shape and data-derived seed for one
observation-model parameter group. `dims[s]` is slot `s`'s channel count;
`mean[s]` / `var[s]` are that slot's per-channel statistics, used only to seed
a slot whose shape differs from the template.
"""
struct ObsSlotSpec{T<:Real}
    dims::Vector{Int}
    mean::Vector{Vector{T}}
    var::Vector{Vector{T}}
end

_spec_dim(::Nothing, slot::Int, fallback::Int) = fallback
_spec_dim(spec::ObsSlotSpec, slot::Int, ::Int) = spec.dims[slot]

#=============================================================================
Per-session observation dimensions ("stitching")

`obs_dim` is a property of the `[C d D]` group, not of the model as a whole:
`C` is `obs_dim × latent_dim`, `d` is `obs_dim`, `D` is `obs_dim × uy_dim` and
`R` is `obs_dim × obs_dim`. So when sessions observe different numbers of
channels, every observation-model group has to be constant in `obs_dim` within
each of its slots. A shared `:R` alongside a per-session `:C` has no
well-defined size, and is rejected here rather than left to a downstream shape
error.

The dimensions come from the data: each slot's `obs_dim` is the row count of
the trials assigned to it. The latent dimension stays shared.

`variants` is indexed by the Cartesian product of every group's slots, so with
both `:C` and `:R` varying over two sessions it holds four entries while only
two describe a real cell. Under stitching the two cross terms pair one
session's `C` with another session's `R` and are therefore inconsistent — but
they are unreachable: `_cell_lds` only ever indexes `grp.cell_obs`, and both
groups take their width from the same per-trial data, so every combination an
occupied cell names has matching `C` and `R` widths.
=============================================================================#

"""
    _slot_obs_dims(dep, g, labels_g, obs_dims, template_dim) -> Vector{Int}

Observation dimension of each slot of observation-model group `g`.

`obs_dims[n]` is trial `n`'s channel count. A group that does not vary gets a
single slot, which then requires every trial to share one dimension.
"""
function _slot_obs_dims(
    dep::ParameterDependence,
    g::Int,
    labels_g::AbstractVector,
    obs_dims::AbstractVector{Int},
    template_dim::Int,
)
    name = dep.names[g]
    if !dep.varies[g]
        p = first(obs_dims)
        for q in obs_dims
            q == p || throw(
                ArgumentError(
                    "observations have $p and $q channels in the same dataset, but " *
                    "`:$name` is shared across all trials, so it has no well-defined " *
                    "size. Make `:$name` depend on the same ancillary variable as the " *
                    "emission, e.g. `depends_on = (C = session, R = session)`.",
                ),
            )
        end
        return [p]
    end

    dims = Vector{Int}(undef, dep.nslots[g])
    seen = falses(dep.nslots[g])
    for (n, label) in enumerate(labels_g)
        s = _slot_of(dep, g, label)
        p = obs_dims[n]
        if !seen[s]
            dims[s] = p
            seen[s] = true
        elseif dims[s] != p
            throw(
                ArgumentError(
                    "trials grouped under `:$name = $(repr(dep.labels[g][s]))` have " *
                    "differing channel counts ($(dims[s]) and $p). A parameter version " *
                    "is one matrix, so every trial sharing it must have the same number " *
                    "of channels.",
                ),
            )
        end
    end

    # A slot with no trials in this dataset keeps whatever the template implies.
    for s in eachindex(dims)
        seen[s] || (dims[s] = template_dim)
    end
    return dims
end

#=
Initialization of a variant whose shape differs from the template. Same-sized
slots keep copying the template, so a fit whose sessions happen to agree on
`obs_dim` is bit-identical to before this feature existed.

A differently-sized slot cannot copy anything, so it is seeded from the data
it will be fitted to: `d` at the per-channel mean and `R` at the per-channel
variance put the emission on the right scale immediately, and `C` cycles the
template's rows so that the loading matrix starts at the template's magnitude
rather than at zero (which the M-step could not move off of). One M-step
replaces all three.
=#
_obs_stats(::Nothing, p::Int, ::Type{T}) where {T} = (zeros(T, p), ones(T, p))

function _obs_stats(ys::AbstractVector, p::Int, ::Type{T}) where {T}
    mean_y = zeros(T, p)
    var_y = zeros(T, p)
    n = 0
    for y in ys
        n += size(y, 2)
        for t in axes(y, 2), i in 1:p
            mean_y[i] += y[i, t]
        end
    end
    n == 0 && return (mean_y, fill!(var_y, one(T)))
    mean_y ./= n
    for y in ys
        for t in axes(y, 2), i in 1:p
            var_y[i] += abs2(y[i, t] - mean_y[i])
        end
    end
    var_y ./= max(n - 1, 1)
    for i in 1:p
        var_y[i] > 0 || (var_y[i] = one(T))
    end
    return (mean_y, var_y)
end

"""
    set_group_seeds!(model, seeds) -> model

Give individual `depends_on` groups their own starting values:
`Dict(label => (C=..., d=..., D=..., R=...))`, any subset of those keys per
label. A label with no entry, or an entry naming only some parameters, keeps the
defaults for the rest.

This matters when the groups observe different channel *sets*. Nothing here
knows which channel is which, so an unseeded slot starts from the template's
emission — copied when the widths agree, its rows cycled when they do not. Both
pair a group's channel with whatever channel sits at the same row of the
template, which is only right when the groups share one channel list. A caller
that already knows each group's loadings, because it stitched them onto shared
factors before building the model, should hand them over here instead.

Seeds are *initial* values: they are read when the per-group variants are built
and ignored by a later `fit!` that reuses the cached ones, exactly as `model.C`
is. Shapes are checked against the data when the variants are built — a seed of
the wrong size is an error, never a silent reshape.

Labels are validated here rather than at fit time, because the cost of a
mistyped one is invisible: it would simply never be read.

    om.depends_on = (C = session,)
    set_group_seeds!(om, Dict(s => (C = loadings[s],) for s in sessions))
"""
function set_group_seeds!(model::DependentModel, seeds::Union{Nothing,AbstractDict})
    if seeds === nothing
        model.group_seeds = nothing
        return model
    end
    model.depends_on === nothing && throw(
        ArgumentError(
            "set_group_seeds! needs `depends_on` to be set first: with no groups there " *
            "are no labels to seed, and the seeds would never be read",
        ),
    )
    dep = _resolve_dependence(model)
    known = Set{Any}()
    for g in eachindex(dep.names)
        dep.varies[g] && union!(known, dep.labels[g])
    end
    for (label, seed) in seeds
        label in known || throw(
            ArgumentError(
                "set_group_seeds!: $(repr(label)) is not a label of any group this " *
                "model varies; it has $(join(sort!(String[repr(l) for l in known]), ", "))",
            ),
        )
        for name in keys(seed)
            _canonical_param(model, name)   # throws on a name the model does not own
        end
    end
    model.group_seeds = seeds
    return model
end

"""
    _group_seed(model, dep, g, slot) -> NamedTuple or nothing

The caller-supplied seed for one slot of observation group `g`, looked up by the
slot's label. `nothing` whenever no seeds were given, the group does not vary,
or that label has no entry.
"""
function _group_seed(
    model::AbstractObservationModel, dep::ParameterDependence, g::Int, slot::Int
)
    seeds = model.group_seeds
    seeds === nothing && return nothing
    dep.varies[g] || return nothing
    slot <= length(dep.labels[g]) || return nothing
    return get(seeds, dep.labels[g][slot], nothing)
end

_seed_entry(::Nothing, ::Symbol) = nothing
_seed_entry(seed, name::Symbol) = get(seed, name, nothing)

"""
    _slot_dim(spec, slot, fallback, seed) -> Int

The channel count one observation slot's parameters get. A dataset pins it
whenever there is one (`spec`), and that always wins. With no dataset in hand
the caller's seed is the only statement of how many channels that group has, so
it settles the size instead; the template's width is the fallback, as before.

This is what lets a group's emission be read back — `group_parameter(om, :C,
label)` — between `set_group_seeds!` and the first fit. Without it a seed
narrower or wider than the template would be rejected as the wrong shape for a
slot the template had sized, even though the seed is precisely the thing that
knows better.
"""
function _slot_dim(spec, slot::Int, fallback::Int, seed)
    spec === nothing || return _spec_dim(spec, slot, fallback)
    seed === nothing && return fallback
    C = _seed_entry(seed, :C)
    C === nothing || return size(C, 1)
    d = _seed_entry(seed, :d)
    d === nothing || return length(d)
    D = _seed_entry(seed, :D)
    D === nothing || return size(D, 1)
    return fallback
end

"""The channel count an `:R` slot gets: as `_slot_dim`, read off the `:R` seed."""
function _slot_dim_R(spec, slot::Int, fallback::Int, seed)
    spec === nothing || return _spec_dim(spec, slot, fallback)
    R = _seed_entry(seed, :R)
    return R === nothing ? fallback : size(R, 1)
end

"""Name a slot's label for an error message; `"(shared)"` when the group is pooled."""
function _slot_label(dep::ParameterDependence, g::Int, slot::Int)
    dep.varies[g] || return "(shared)"
    slot <= length(dep.labels[g]) || return "(shared)"
    return repr(dep.labels[g][slot])
end

"""
    _check_seeded_widths_agree(dep, dims_C, dims_R)

A Gaussian emission's `[C d D]` and its `:R` have to describe the same channels
— `C` is `p × latent_dim` where `R` is `p × p`. A dataset pins both from the
same trials, so they agree by construction; seeds can disagree, because a caller
may name `:d` for one group and leave that group's `:R` alone.

Only the slot pairings the trials actually produce are checked. `variants` is
the full cross product of the groups' slots, so it also holds combinations no
trial selects — one session's `[C d D]` against another's `:R` — and those
mismatch routinely whenever the sessions differ in width. Walking the trials
instead compares each group against its own `:R`.
"""
function _check_seeded_widths_agree(
    dep::ParameterDependence, dims_C::AbstractVector{Int}, dims_R::AbstractVector{Int}
)
    (dep.varies[1] || dep.varies[2]) || return nothing
    trial_labels = dep.varies[1] ? dep.trial_labels[1] : dep.trial_labels[2]
    for n in eachindex(trial_labels)
        i = dep.varies[1] ? _slot_of(dep, 1, dep.trial_labels[1][n]) : 1
        j = dep.varies[2] ? _slot_of(dep, 2, dep.trial_labels[2][n]) : 1
        dims_C[i] == dims_R[j] && continue
        throw(
            ArgumentError(
                "group_seeds imply a $(dims_C[i])-channel `[C d D]` for group " *
                "$(_slot_label(dep, 1, i)) but a $(dims_R[j])-channel `:R` for group " *
                "$(_slot_label(dep, 2, j)); one emission cannot have both. Seed `:R` " *
                "for that group as well, or leave the widths to the data at `fit!`.",
            ),
        )
    end
    return nothing
end

"""Copy a seed into a slot's storage, refusing one that is the wrong shape."""
function _apply_seed!(out, supplied, name::Symbol, slot::Int)
    size(out) == size(supplied) || throw(
        ArgumentError(
            "group_seeds gave a $(size(supplied)) `:$name` for a slot that needs " *
            "$(size(out)) (group slot $slot). A seed has to match the channel count " *
            "the data gives that group and the model's latent/input widths.",
        ),
    )
    copyto!(out, supplied)
    return out
end

#=
Slot 1 aliases the parent's array (as `_slot_arrays` has always done) so the
model object stays in sync with its first version; every other slot gets its
own storage. A slot whose shape matches the template is still a plain copy, so
nothing about the equal-`obs_dim` path changes. A seed is written *into* that
storage rather than replacing it, which keeps slot 1's aliasing intact.
=#
function _slot_storage(base::AbstractMatrix, i::Int, dims::Tuple{Int,Int})
    size(base) == dims || return similar(base, dims...)
    return i == 1 ? base : copy(base)
end

function _slot_storage(base::AbstractVector, i::Int, len::Int)
    length(base) == len || return similar(base, len)
    return i == 1 ? base : copy(base)
end

function _seed_slot_C(base::AbstractMatrix{T}, i::Int, p::Int, seed=nothing) where {T}
    out = _slot_storage(base, i, (p, size(base, 2)))
    supplied = _seed_entry(seed, :C)
    supplied === nothing || return _apply_seed!(out, supplied, :C, i)
    size(base, 1) == p && return out
    nrows = size(base, 1)
    for r in 1:p
        out[r, :] .= @view base[mod1(r, nrows), :]
    end
    return out
end

function _seed_slot_D(base::AbstractMatrix{T}, i::Int, p::Int, seed=nothing) where {T}
    out = _slot_storage(base, i, (p, size(base, 2)))
    supplied = _seed_entry(seed, :D)
    supplied === nothing || return _apply_seed!(out, supplied, :D, i)
    size(base, 1) == p && return out
    return fill!(out, zero(T))
end

function _seed_slot_d(
    base::AbstractVector{T},
    i::Int,
    p::Int,
    spec::Union{Nothing,ObsSlotSpec{T}},
    seed=nothing,
) where {T}
    out = _slot_storage(base, i, p)
    supplied = _seed_entry(seed, :d)
    supplied === nothing || return _apply_seed!(out, supplied, :d, i)
    length(base) == p && return out
    spec === nothing ? fill!(out, zero(T)) : copyto!(out, spec.mean[i])
    return out
end

function _seed_slot_R(
    base::AbstractMatrix{T},
    j::Int,
    p::Int,
    spec::Union{Nothing,ObsSlotSpec{T}},
    seed=nothing,
) where {T}
    out = _slot_storage(base, j, (p, p))
    supplied = _seed_entry(seed, :R)
    supplied === nothing || return _apply_seed!(out, supplied, :R, j)
    size(base, 1) == p && return out
    fill!(out, zero(T))
    for k in 1:p
        out[k, k] = spec === nothing ? one(T) : spec.var[j][k]
    end
    return out
end

#=
A rebuild is skipped only when the cached variants already have the shapes this
dataset asks for; otherwise re-fitting the same model against a dataset with
different channel counts would silently reuse the wrong-sized arrays.
=#
function _variants_match(
    variants::AbstractVector,
    dep::ParameterDependence,
    spec_C::Union{Nothing,ObsSlotSpec},
    spec_R::Union{Nothing,ObsSlotSpec},
)
    spec_C === nothing && spec_R === nothing && return true
    for cell in eachindex(variants)
        s = _variant_slots(dep.nslots, cell)
        v = variants[cell]
        spec_C === nothing || size(v.C, 1) == spec_C.dims[s[1]] || return false
        if spec_R !== nothing && hasproperty(v, :R)
            size(v.R, 1) == spec_R.dims[s[2]] || return false
        end
    end
    return true
end

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
    om::GaussianObservationModel{T,M,V},
    dep::ParameterDependence,
    spec_C::Union{Nothing,ObsSlotSpec{T}}=nothing,
    spec_R::Union{Nothing,ObsSlotSpec{T}}=nothing,
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    ncells = prod(dep.nslots)
    existing = om.variants
    if existing !== nothing &&
        length(existing) == ncells &&
        _variants_match(existing, dep, spec_C, spec_R)
        return existing
    end

    p0 = size(om.C, 1)
    seeds_C = [_group_seed(om, dep, 1, i) for i in 1:(dep.nslots[1])]
    dims_C = [_slot_dim(spec_C, i, p0, seeds_C[i]) for i in 1:(dep.nslots[1])]
    Cs = [_seed_slot_C(om.C, i, dims_C[i], seeds_C[i]) for i in 1:(dep.nslots[1])]
    ds = [_seed_slot_d(om.d, i, dims_C[i], spec_C, seeds_C[i]) for i in 1:(dep.nslots[1])]
    Ds = [_seed_slot_D(om.D, i, dims_C[i], seeds_C[i]) for i in 1:(dep.nslots[1])]
    seeds_R = [_group_seed(om, dep, 2, j) for j in 1:(dep.nslots[2])]
    dims_R = [_slot_dim_R(spec_R, j, p0, seeds_R[j]) for j in 1:(dep.nslots[2])]
    _check_seeded_widths_agree(dep, dims_C, dims_R)
    Rs = [_seed_slot_R(om.R, j, dims_R[j], spec_R, seeds_R[j]) for j in 1:(dep.nslots[2])]

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
    om::PoissonObservationModel{T,M,V},
    dep::ParameterDependence,
    spec_C::Union{Nothing,ObsSlotSpec{T}}=nothing,
    ::Union{Nothing,ObsSlotSpec{T}}=nothing,
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    ncells = prod(dep.nslots)
    existing = om.variants
    if existing !== nothing &&
        length(existing) == ncells &&
        _variants_match(existing, dep, spec_C, nothing)
        return existing
    end

    p0 = size(om.C, 1)
    seeds_C = [_group_seed(om, dep, 1, i) for i in 1:(dep.nslots[1])]
    dims_C = [_slot_dim(spec_C, i, p0, seeds_C[i]) for i in 1:(dep.nslots[1])]
    Cs = [_seed_slot_C(om.C, i, dims_C[i], seeds_C[i]) for i in 1:(dep.nslots[1])]
    ds = [_seed_slot_d(om.d, i, dims_C[i], spec_C, seeds_C[i]) for i in 1:(dep.nslots[1])]
    Ds = [_seed_slot_D(om.D, i, dims_C[i], seeds_C[i]) for i in 1:(dep.nslots[1])]

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
    lds::LinearDynamicalSystem,
    ntrials::Int;
    depends_on::Union{Nothing,NamedTuple}=nothing,
    y=nothing,
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

    #=
    Observation variants are built after the labels are known, because a
    stitching dataset fixes each slot's `obs_dim` from the trials assigned to
    it. `y === nothing`, or a dataset whose trials all have the template's
    channel count, reproduces the previous shapes exactly.
    =#
    labels_o = [trial_labels[ngroups_s + g] for g in 1:ngroups_o]
    spec_C, spec_R = _obs_slot_specs(om, dep_o, labels_o, ntrials, y)
    _build_variants!(om, dep_o, spec_C, spec_R)

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

#=
`d` is seeded on the emission's natural scale: the per-channel mean for a
Gaussian emission, its log for a Poisson one (`λ = exp(Cx + d)`).
=#
_natural_seed(::GaussianObservationModel, m::AbstractVector) = m
function _natural_seed(::PoissonObservationModel, m::AbstractVector{T}) where {T<:Real}
    return T[log(max(mi, eps(T))) for mi in m]
end

"""
    _obs_slot_specs(om, dep_o, labels_o, ntrials, y) -> (spec_C, spec_R)

Per-slot observation shapes for this dataset, or `(nothing, nothing)` when
every trial carries the template's channel count — the ordinary single-`obs_dim`
case, which must keep its existing arrays untouched.
"""
function _obs_slot_specs(
    om::AbstractObservationModel{T},
    dep_o::ParameterDependence,
    labels_o::AbstractVector,
    ntrials::Int,
    y,
) where {T<:Real}
    y === nothing && return (nothing, nothing)
    length(y) == ntrials || return (nothing, nothing)
    obs_dims = [size(yi, 1) for yi in y]
    p0 = size(om.C, 1)
    all(isequal(p0), obs_dims) && return (nothing, nothing)

    specs = Vector{ObsSlotSpec{T}}(undef, length(dep_o.names))
    for g in eachindex(dep_o.names)
        dims = _slot_obs_dims(dep_o, g, labels_o[g], obs_dims, p0)
        nslots = length(dims)
        means = Vector{Vector{T}}(undef, nslots)
        vars = Vector{Vector{T}}(undef, nslots)
        for s in 1:nslots
            trials = [
                n for n in 1:ntrials if
                (dep_o.varies[g] ? _slot_of(dep_o, g, labels_o[g][n]) : 1) == s
            ]
            m, v = _obs_stats(
                isempty(trials) ? nothing : [y[n] for n in trials], dims[s], T
            )
            means[s] = _natural_seed(om, m)
            vars[s] = v
        end
        specs[g] = ObsSlotSpec{T}(dims, means, vars)
    end
    return (specs[1], length(specs) > 1 ? specs[2] : nothing)
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
    #=
    The cell's `obs_dim` comes from its own emission, not from the parent: under
    stitching each session contributes a different number of channels, and the
    parent's `obs_dim` is only the template's.
    =#
    return LinearDynamicalSystem{T,S,O}(
        sm, om, lds.latent_dim, size(om.C, 1), lds.ux_dim, lds.uy_dim, lds.fit_bool
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
