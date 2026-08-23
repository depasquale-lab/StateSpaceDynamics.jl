#=============================================================================
Latent-free affine-Gaussian baselines for LDS comparison.

`AffineNullModel` covers four configurations:

  :intercept   y_t ~ N(d, R)
  :inputs      y_t ~ N(d + D v_{t-shift}, R)
  :var         y_1 ~ N(μ₀, R₀); y_t ~ N(F y_{t-1} + d, R)
  :var_inputs  y_1 ~ N(μ₀, R₀); y_t ~ N(F y_{t-1} + d + D v_{t-shift}, R)

Here `lag` enables the autoregressive term and `input_shift` matches the
corresponding LDS input timing: lagged for dynamics inputs and contemporaneous
for observation inputs. Pretrial inputs are zero. For VAR models, `(μ₀, R₀)`
scores the first observation analogously to the LDS initial-state layer.

Use `fit!` and `loglikelihood`; `r2`, `nullloglikelihood`, and `nobs` compare a
fitted LDS against a baseline. Optional priors are an `MNPrior` on the stacked
regression matrix and `IWPrior`s on `R` and `R₀`.
=============================================================================#

const NULL_BASELINES = (:intercept, :inputs, :var, :var_inputs)

_null_has_lag(baseline::Symbol) = baseline === :var || baseline === :var_inputs
_null_has_inputs(baseline::Symbol) = baseline === :inputs || baseline === :var_inputs

"""
    AffineNullModel{T}

Affine-Gaussian null model for a fitted [`LinearDynamicalSystem`](@ref). `lag`
controls whether the model includes the autoregressive term `F y_{t-1}`, and
`input_shift` (`0` or `1`) gives the input lag. When the model has no inputs,
`D` has zero columns, consistent with the LDS representation.

# Constructors
    AffineNullModel{T}(obs_dim; input_dim=0, lag=false, input_shift=0, priors...)
    AffineNullModel{T}(baseline::Symbol, obs_dim; input_dim=0, input_shift=0, priors...)

The second form selects one of `$(NULL_BASELINES)` and sets `lag`. It rejects a
`lag` keyword, and rejects `input_dim` / `input_shift` for the input-free
baselines `:intercept` and `:var`, so the baseline name is the only thing that
sets the model's structure. Set `input_shift=0` to score `v_t` (the LDS
observation input) or `input_shift=1` to score `v_{t-1}` (the LDS dynamics
input).

`obs_dim`, `input_dim`, `lag` and `input_shift` are `const`. They fix the
parameter shapes at construction, so a fitted model cannot be desynced from its
own structure.

# Fields
- `obs_dim::Int`: observation dimension.
- `input_dim::Int`: input dimension (`0` when the baseline takes no inputs).
- `lag::Bool`: whether the autoregressive term is present.
- `input_shift::Int`: input lag, `0` or `1`.
- `d::Vector{T}`: intercept.
- `D::Matrix{T}`: input matrix (`obs_dim × input_dim`).
- `F::Matrix{T}`: autoregressive matrix (`obs_dim × obs_dim`, empty when `!lag`).
- `R::Matrix{T}`: innovation covariance.
- `μ₀::Vector{T}`, `R₀::Matrix{T}`: initial-step mean/covariance (empty when `!lag`).
- `W_prior`, `R_prior`, `R₀_prior`: optional MN/IW priors, mirroring the LDS M-step.

See also [`r2`](@ref), [`nullloglikelihood`](@ref).
"""
mutable struct AffineNullModel{T<:Real}
    const obs_dim::Int
    const input_dim::Int
    const lag::Bool
    const input_shift::Int
    d::Vector{T}
    D::Matrix{T}
    F::Matrix{T}
    R::Matrix{T}
    μ₀::Vector{T}
    R₀::Matrix{T}
    W_prior::Union{Nothing,MNPrior{T,Matrix{T}}}
    R_prior::Union{Nothing,IWPrior{T}}
    R₀_prior::Union{Nothing,IWPrior{T}}
end

# Number of columns the stacked regression matrix `[F d D]` carries.
_null_ncoef(lag::Bool, obs_dim::Int, input_dim::Int) = (lag ? obs_dim : 0) + 1 + input_dim

function _check_iw_prior(::Nothing, ::Int, ::String) end

function _check_iw_prior(prior::IWPrior, dim::Int, name::String)
    size(prior.Ψ) == (dim, dim) ||
        throw(DimensionMismatchError("$name.Ψ rows/cols", dim, size(prior.Ψ, 1)))
    return nothing
end

function _check_mn_prior(::Nothing, ::Int, ::Int) end

function _check_mn_prior(prior::MNPrior, obs_dim::Int, ncoef::Int)
    size(prior.M₀) == (obs_dim, ncoef) ||
        throw(DimensionMismatchError("W_prior.M₀ cols", ncoef, size(prior.M₀, 2)))
    size(prior.Λ) == (ncoef, ncoef) ||
        throw(DimensionMismatchError("W_prior.Λ rows/cols", ncoef, size(prior.Λ, 1)))
    return nothing
end

function AffineNullModel{T}(
    obs_dim::Int;
    input_dim::Int=0,
    lag::Bool=false,
    input_shift::Int=0,
    W_prior::Union{Nothing,MNPrior{T,Matrix{T}}}=nothing,
    R_prior::Union{Nothing,IWPrior{T}}=nothing,
    R₀_prior::Union{Nothing,IWPrior{T}}=nothing,
) where {T<:Real}
    obs_dim > 0 || throw(ArgumentError("obs_dim must be positive; got $obs_dim"))
    input_dim >= 0 || throw(ArgumentError("input_dim must be non-negative; got $input_dim"))
    input_shift in (0, 1) ||
        throw(ArgumentError("input_shift must be 0 or 1; got $input_shift"))
    (input_dim == 0 && input_shift != 0) && throw(
        ArgumentError(
            "input_shift=$input_shift shifts an input block the model does not have; " *
            "set input_dim > 0 or leave input_shift at 0",
        ),
    )

    _check_mn_prior(W_prior, obs_dim, _null_ncoef(lag, obs_dim, input_dim))
    _check_iw_prior(R_prior, obs_dim, "R_prior")
    lag ||
        R₀_prior === nothing ||
        throw(ArgumentError("R₀_prior applies to the VAR initial step; it needs lag=true"))
    _check_iw_prior(R₀_prior, obs_dim, "R₀_prior")

    lag_dim = lag ? obs_dim : 0
    return AffineNullModel{T}(
        obs_dim,
        input_dim,
        lag,
        input_shift,
        zeros(T, obs_dim),
        zeros(T, obs_dim, input_dim),
        zeros(T, obs_dim, lag_dim),
        Matrix{T}(I, obs_dim, obs_dim),
        zeros(T, lag_dim),
        Matrix{T}(I, lag_dim, lag_dim),
        W_prior,
        R_prior,
        R₀_prior,
    )
end

function AffineNullModel{T}(
    baseline::Symbol,
    obs_dim::Int;
    input_dim::Int=0,
    input_shift::Int=0,
    lag::Union{Nothing,Bool}=nothing,
    kwargs...,
) where {T<:Real}
    baseline in NULL_BASELINES ||
        throw(ArgumentError("baseline must be one of $(NULL_BASELINES); got :$baseline"))
    lag === nothing || throw(
        ArgumentError(
            "the baseline name already sets lag, so lag=$lag cannot be passed " *
            "alongside :$baseline; name the baseline you want ($(NULL_BASELINES)), or " *
            "use the AffineNullModel{T}(obs_dim; lag=...) form",
        ),
    )
    if !_null_has_inputs(baseline)
        input_dim == 0 || throw(
            ArgumentError(
                "baseline :$baseline has no input block, so input_dim must be 0; got " *
                "$input_dim. Use :$(_null_has_lag(baseline) ? :var_inputs : :inputs) instead",
            ),
        )
        input_shift == 0 || throw(
            ArgumentError(
                "baseline :$baseline has no input block, so input_shift must be 0; got " *
                "$input_shift",
            ),
        )
    end
    return AffineNullModel{T}(
        obs_dim;
        input_dim=input_dim,
        lag=_null_has_lag(baseline),
        input_shift=input_shift,
        kwargs...,
    )
end

#=
Normalize the supported trial formats: an (obs_dim, T) matrix, an
(obs_dim, T, ntrials) array, or a vector of per-trial matrices (which may have
different lengths). The baseline cannot use Data(lds, y; ux, uy), because that
constructor validates the data against an LDS.
=#
_null_trials(y::AbstractVector{<:AbstractMatrix}) = y
_null_trials(y::AbstractMatrix) = [y]
_null_trials(y::AbstractArray{<:Any,3}) = [view(y, :, :, n) for n in axes(y, 3)]

# Normalize and check `y` against the model's own shape and element type.
function _null_obs_trials(null::AffineNullModel{T}, y) where {T<:Real}
    trials = _null_trials(y)
    isempty(trials) && throw(ArgumentError("y must hold at least one trial"))
    for (i, yn) in enumerate(trials)
        eltype(yn) === T || throw(
            ArgumentError(
                "y[$i] has eltype $(eltype(yn)), but the baseline is " *
                "AffineNullModel{$T}; convert the data or build the baseline at " *
                "the data's precision",
            ),
        )
        size(yn, 1) == null.obs_dim ||
            throw(DimensionMismatchError("y[$i] rows", null.obs_dim, size(yn, 1)))
        size(yn, 2) > 0 ||
            throw(ArgumentError("y[$i] has no timesteps; every trial needs at least one"))
    end
    return trials
end

# Normalize `y` and `inputs` together. Every public method enters here.
function _null_prepare(null::AffineNullModel{T}, y, inputs) where {T<:Real}
    trials = _null_obs_trials(null, y)
    tsteps = Int[size(yn, 2) for yn in trials]
    return trials, _null_input_trials(T, inputs, tsteps, null.input_dim)
end

function _null_input_trials(
    ::Type{T}, ::Nothing, tsteps::Vector{Int}, input_dim::Int
) where {T<:Real}
    input_dim == 0 || throw(
        ArgumentError(
            "inputs=nothing is only valid for a baseline with no input block; " *
            "got input_dim=$input_dim",
        ),
    )
    return [zeros(T, 0, Ti) for Ti in tsteps]
end

function _null_input_trials(
    ::Type{T}, v, tsteps::Vector{Int}, input_dim::Int
) where {T<:Real}
    # A zero-`input_dim` model accepts zero-row inputs (the collapse case); the
    # per-trial row check below rejects anything that actually carries a signal.
    trials = _null_trials(v)
    length(trials) == length(tsteps) ||
        throw(DimensionMismatchError("inputs ntrials", length(tsteps), length(trials)))
    for (i, vi) in enumerate(trials)
        eltype(vi) === T || throw(
            ArgumentError("inputs[$i] has eltype $(eltype(vi)); expected $T to match y")
        )
        size(vi, 1) == input_dim ||
            throw(DimensionMismatchError("inputs[$i] rows", input_dim, size(vi, 1)))
        size(vi, 2) == tsteps[i] ||
            throw(DimensionMismatchError("inputs[$i] tsteps", tsteps[i], size(vi, 2)))
    end
    return trials
end

#=
Design assembly
=#

# `v_{t-shift}` aligned with `y`'s columns; entries predating the trial are zero.
function _shifted_inputs(v::AbstractMatrix{T}, shift::Int) where {T<:Real}
    shift == 0 && return v
    input_dim, tsteps = size(v)
    shifted = zeros(T, input_dim, tsteps)
    tsteps > 1 && copyto!(view(shifted, :, 2:tsteps), view(v, :, 1:(tsteps - 1)))
    return shifted
end

# Response/design pair for the main regression, stacked over trials:
#   lag = false:  y_t ~ [1; v_{t-shift}]            for t = 1..T
#   lag = true:   y_t ~ [y_{t-1}; 1; v_{t-shift}]   for t = 2..T
# A length-1 trial informs only `(μ₀, R₀)`, so it drops out of the lagged design.
function _null_design(
    null::AffineNullModel{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    v::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real}
    isempty(y) && throw(ArgumentError("y must hold at least one trial"))
    responses = Matrix{T}[]
    designs = Matrix{T}[]

    for (yn, vn) in zip(y, v)
        tsteps = size(yn, 2)
        null.lag && tsteps < 2 && continue

        rows = null.lag ? (2:tsteps) : (1:tsteps)
        blocks = Matrix{T}[]
        null.lag && push!(blocks, yn[:, rows .- 1])
        push!(blocks, ones(T, 1, length(rows)))
        if null.input_dim > 0
            push!(blocks, _shifted_inputs(vn, null.input_shift)[:, rows])
        end

        push!(responses, yn[:, rows])
        push!(designs, reduce(vcat, blocks))
    end

    # Only reachable with `lag`: a lag-free design keeps every trial.
    isempty(responses) &&
        throw(ArgumentError("a VAR baseline needs at least one trial with tsteps ≥ 2"))
    return reduce(hcat, responses), reduce(hcat, designs)
end

# Initial-step regression `y_1 ~ N(μ₀, R₀)`: one column per trial, constant design.
function _null_init_design(
    ::Type{T}, y::AbstractVector{<:AbstractMatrix{T}}
) where {T<:Real}
    Y₀ = reduce(hcat, (yn[:, 1:1] for yn in y))
    return Y₀, ones(T, 1, size(Y₀, 2))
end

# Stacked regression matrix `[F d D]`, in the column order `_null_design` builds.
function _pack_W(null::AffineNullModel{T}) where {T<:Real}
    blocks = Matrix{T}[]
    null.lag && push!(blocks, null.F)
    push!(blocks, reshape(null.d, null.obs_dim, 1))
    null.input_dim > 0 && push!(blocks, null.D)
    return reduce(hcat, blocks)
end

function _unpack_W!(null::AffineNullModel{T}, W::AbstractMatrix{T}) where {T<:Real}
    offset = 0
    if null.lag
        null.F = W[:, 1:(null.obs_dim)]
        offset = null.obs_dim
    end
    null.d = vec(W[:, offset + 1])
    null.input_dim > 0 && (null.D = W[:, (offset + 2):end])
    return null
end

#=
Regression + scoring kernels
=#

#=
MAP fit of `Y = W X + ε`, `ε ~ N(0, R)`, with an optional `MNPrior` on `W` and
`IWPrior` on `R`. Mirrors the `(mn_map, iw_map)` M-step machinery behind
`update_R!` in `lds/gaussian_observations.jl`.
=#
function _null_fit_regression(
    Y::AbstractMatrix{T},
    X::AbstractMatrix{T},
    W_prior::Union{Nothing,MNPrior{T,Matrix{T}}},
    R_prior::Union{Nothing,IWPrior{T}},
) where {T<:Real}
    obs_dim, n = size(Y)
    size(X, 2) == n || throw(DimensionMismatchError("X cols vs Y cols", n, size(X, 2)))

    # `mn_map` returns a `Transpose` view; materialize so downstream BLAS calls
    # hit the concrete-matrix code paths.
    W = Matrix(mn_map(X * transpose(X), X * transpose(Y), W_prior))

    E = Y .- W * X
    S = E * transpose(E)
    Symmetrize!(S)

    # MN-prior contribution to the IW posterior scale (matches `update_R!`).
    if W_prior !== nothing
        Wm = W .- W_prior.M₀
        S .+= Wm * W_prior.Λ * transpose(Wm)
        Symmetrize!(S)
    end

    R = if R_prior === nothing
        S ./ T(n)
    else
        iw_map(R_prior.Ψ, R_prior.ν, S, T(n), obs_dim)
    end
    return W, R
end

# Plug-in Gaussian log-density of `Y` under `y ~ N(W X, R)`, shared by
# `loglikelihood` and the MAP objective.
function _null_plugin_ll(
    Y::AbstractMatrix{T}, X::AbstractMatrix{T}, W::AbstractMatrix{T}, R::AbstractMatrix{T}
) where {T<:Real}
    obs_dim, n = size(Y)
    chol = cholesky(Symmetric(R))

    # tr(R⁻¹ E E') via two triangular solves.
    E = Y .- W * X
    EE = E * transpose(E)
    Symmetrize!(EE)
    ldiv!(chol.U', EE)
    ldiv!(chol.U, EE)

    log_det_R = 2 * sum(log, diag(chol.U))
    return T(-0.5) * (T(n) * (T(obs_dim) * log(T(2π)) + log_det_R) + tr(EE))
end

#=
`R₀` is estimated from one initial observation per trial, so its residual scatter
has rank ≤ ntrials - 1 and is singular whenever `ntrials ≤ obs_dim`, including
every single-trial dataset. An `R₀_prior` keeps it positive definite through the
IW scale. Without one, fall back to the innovation covariance `R` so the baseline
scores instead of throwing a `PosDefException`.
=#
function _null_init_cov(
    R₀::AbstractMatrix{T},
    R::AbstractMatrix{T},
    ntrials::Int,
    obs_dim::Int,
    R₀_prior::Union{Nothing,IWPrior{T}},
) where {T<:Real}
    (R₀_prior === nothing && ntrials <= obs_dim) && return copy(R)
    return R₀
end

#=
StatsAPI methods on a baseline
=#

"""
    fit!(null::AffineNullModel, y; inputs=nothing)

Fit `null` in closed form on `y` and return it. `y` is a `(obs_dim, T)` matrix, an
`(obs_dim, T, ntrials)` array, or a vector of per-trial matrices; `inputs` follows
the same shape family and is required when the baseline has an input block.
"""
function StatsAPI.fit!(null::AffineNullModel, y; inputs=nothing)
    return _null_fit!(null, _null_prepare(null, y, inputs)...)
end

function _null_fit!(
    null::AffineNullModel{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    v::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real}
    Y, X = _null_design(null, y, v)
    W, R = _null_fit_regression(Y, X, null.W_prior, null.R_prior)
    _unpack_W!(null, W)
    null.R = R

    if null.lag
        Y₀, X₀ = _null_init_design(T, y)
        W₀, R₀ = _null_fit_regression(Y₀, X₀, nothing, null.R₀_prior)
        null.μ₀ = vec(W₀[:, 1])
        null.R₀ = _null_init_cov(R₀, R, size(Y₀, 2), null.obs_dim, null.R₀_prior)
    end
    return null
end

"""
    loglikelihood(null::AffineNullModel, y; inputs=nothing)

Plug-in log-likelihood of `y` under the fitted baseline. This excludes prior
terms and is directly comparable to the LDS marginal `loglikelihood(lds, y)`.
Pass held-out data to compute held-out log-likelihood.
"""
function StatsAPI.loglikelihood(null::AffineNullModel, y; inputs=nothing)
    return _null_loglikelihood(null, _null_prepare(null, y, inputs)...)
end

function _null_loglikelihood(
    null::AffineNullModel{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    v::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real}
    Y, X = _null_design(null, y, v)
    ll = _null_plugin_ll(Y, X, _pack_W(null), null.R)
    if null.lag
        Y₀, X₀ = _null_init_design(T, y)
        ll += _null_plugin_ll(Y₀, X₀, reshape(null.μ₀, null.obs_dim, 1), null.R₀)
    end
    return ll
end

# Number of scalar observations in `y`: `obs_dim * sum(tsteps)`.
function StatsAPI.nobs(null::AffineNullModel, y)
    return null.obs_dim * sum(size(yn, 2) for yn in _null_obs_trials(null, y))
end

#=
MAP objective at the fitted parameters: the plug-in data log-likelihood plus the
IW/MN log-prior terms `elbo!` carries at the EM fixed point. This is not a
likelihood, so it stays internal. It puts a prior-regularized baseline on the
same footing as a prior-regularized SSM ELBO.
=#
function _null_logmap(null::AffineNullModel, y; inputs=nothing)
    return _null_logmap(null, _null_prepare(null, y, inputs)...)
end

function _null_logmap(
    null::AffineNullModel{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    v::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real}
    ll = _null_loglikelihood(null, y, v)
    ll += mn_logprior_term(_pack_W(null), null.R, null.W_prior)
    null.R_prior === nothing || (ll += iw_logprior_term(null.R, null.R_prior))
    if null.lag && null.R₀_prior !== nothing
        ll += iw_logprior_term(null.R₀, null.R₀_prior)
    end
    return ll
end

#=
StatsAPI methods on a fitted LDS
=#

# Per-trial input matrices for the requested LDS channel, plus the lag that
# channel implies: dynamics inputs act on `x_{t-1}`, observation inputs on `y_t`.
function _null_channel(data::Data, channel::Symbol)
    channel in (:ux, :uy) ||
        throw(ArgumentError("null_inputs must be :ux or :uy; got :$channel"))
    return channel === :ux ? (data.ux, 1) : (data.uy, 0)
end

# Build the named baseline for `data`, fit it, and return its plug-in LL.
function _null_baseline_ll(
    data::Data{T}, baseline::Symbol, channel::Symbol, R_prior::Union{Nothing,IWPrior{T}}
) where {T<:Real}
    baseline in NULL_BASELINES ||
        throw(ArgumentError("null must be one of $(NULL_BASELINES); got :$baseline"))
    v, shift = _null_channel(data, channel)
    obs_dim = size(first(data.y), 1)

    # The input-free baselines reject an input block, so build them without one
    # and hand them matching zero-row inputs.
    null, inputs = if _null_has_inputs(baseline)
        AffineNullModel{T}(
            baseline,
            obs_dim;
            input_dim=size(first(v), 1),
            input_shift=shift,
            R_prior=R_prior,
        ),
        v
    else
        AffineNullModel{T}(baseline, obs_dim; R_prior=R_prior),
        [zeros(T, 0, Ti) for Ti in data.tsteps]
    end

    _null_fit!(null, data.y, inputs)
    return _null_loglikelihood(null, data.y, inputs)
end

"""
    nobs(lds::LinearDynamicalSystem, y; ux=nothing, uy=nothing)

Number of scalar observations in `y`: `obs_dim * sum(tsteps)` across trials.
[`r2`](@ref) uses this count in the Cox–Snell and Nagelkerke variants.
"""
function StatsAPI.nobs(
    lds::LinearDynamicalSystem{T}, y; ux=nothing, uy=nothing
) where {T<:Real}
    return nobs(Data(lds, y; ux=ux, uy=uy))
end

StatsAPI.nobs(data::Data) = size(first(data.y), 1) * sum(data.tsteps)

"""
    nullloglikelihood(lds, y; ux=nothing, uy=nothing, R_prior=lds.obs_model.R_prior)

Plug-in log-likelihood of the intercept baseline `y_t ~ N(d, R)`, fit to the
same data. This is the default reference model for [`r2`](@ref).

`R_prior` defaults to the LDS observation-noise prior.
"""
function StatsAPI.nullloglikelihood(
    lds::LinearDynamicalSystem{T,SM,OM},
    y;
    ux=nothing,
    uy=nothing,
    R_prior::Union{Nothing,IWPrior{T}}=lds.obs_model.R_prior,
) where {T<:Real,SM<:GaussianStateModel{T},OM<:GaussianObservationModel{T}}
    return _null_baseline_ll(Data(lds, y; ux=ux, uy=uy), :intercept, :ux, R_prior)
end

#=
Pseudo-R² variants, following the StatsBase definitions. `n` is `nobs`, the
scalar-observation count, which matches the scale of the log-likelihoods.
=#
function _pseudo_r2(variant::Symbol, ll::T, ll₀::T, n::Int) where {T<:Real}
    variant in (:McFadden, :CoxSnell, :Nagelkerke) || throw(
        ArgumentError("variant must be :McFadden, :CoxSnell or :Nagelkerke; got :$variant"),
    )
    variant === :McFadden && return one(T) - ll / ll₀

    # Nagelkerke is Cox–Snell rescaled by its attainable maximum.
    cox_snell = one(T) - exp(2 * (ll₀ - ll) / n)
    variant === :CoxSnell && return cox_snell
    return cox_snell / (one(T) - exp(2 * ll₀ / n))
end

"""
    r2(lds, y, variant=:CoxSnell; null=:intercept, null_inputs=:ux, ux=nothing, uy=nothing,
       R_prior=lds.obs_model.R_prior)

Pseudo-R² of a fitted Gaussian `LinearDynamicalSystem` against a latent-free
baseline, comparing the LDS marginal log-likelihood `ℓ` (latents integrated out)
to the baseline's plug-in log-likelihood `ℓ₀` on the same data:

- `:CoxSnell` (default) — ``1 - \\exp\\big(2(\\ell_0 - \\ell)/n\\big)``
- `:McFadden` — ``1 - \\ell/\\ell_0``
- `:Nagelkerke` — Cox–Snell rescaled by its attainable maximum

with `n = nobs(lds, y)`. A larger value means the LDS out-predicts the baseline.

For a held-out R², fit an [`AffineNullModel`](@ref) on the training split and
apply the formula above to `loglikelihood(lds, y_test)` and
`loglikelihood(null, y_test)`.

Restricted to the Gaussian LDS: a Poisson LDS has no tractable marginal
likelihood and an `SLDS` is a different type, so both raise a `MethodError`.

# Keyword Arguments
- `null::Symbol = :intercept`: baseline to score against, one of `$(NULL_BASELINES)`.
- `null_inputs::Symbol = :ux`: LDS input channel the input-bearing baselines
  consume. `:ux` enters lagged, `:uy` contemporaneous, matching each channel's
  role in the LDS.
- `R_prior`: IW prior on the baseline's `R`, defaulting to the LDS observation-noise
  prior so both are regularized alike.

See also [`nullloglikelihood`](@ref), [`nobs`](@ref), [`AffineNullModel`](@ref).
"""
function StatsAPI.r2(
    lds::LinearDynamicalSystem{T,SM,OM},
    y,
    variant::Symbol=:CoxSnell;
    null::Symbol=:intercept,
    null_inputs::Symbol=:ux,
    ux=nothing,
    uy=nothing,
    R_prior::Union{Nothing,IWPrior{T}}=lds.obs_model.R_prior,
) where {T<:Real,SM<:GaussianStateModel{T},OM<:GaussianObservationModel{T}}
    data = Data(lds, y; ux=ux, uy=uy)
    ll = loglikelihood(lds, data.y; ux=data.ux, uy=data.uy)
    ll₀ = _null_baseline_ll(data, null, null_inputs, R_prior)
    return _pseudo_r2(variant, ll, ll₀, nobs(data))
end
