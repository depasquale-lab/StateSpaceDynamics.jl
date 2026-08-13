#=============================================================================
Latent-free baseline ("null") models for LDS model comparison.

`AffineNullModel` is one affine-Gaussian baseline whose two structural switches
are type parameters — `LAG`, whether an autoregressive term `F y_{t-1}` is
present, and `SHIFT`, how far the input is lagged — so the four baselines worth
having are configurations of a single type rather than four implementations:

  :intercept   y_t ~ N(d, R)
  :inputs      y_t ~ N(d + D v_{t-SHIFT}, R)
  :var         y_1 ~ N(μ₀, R₀);  y_t ~ N(F y_{t-1} + d, R)                 (t ≥ 2)
  :var_inputs  y_1 ~ N(μ₀, R₀);  y_t ~ N(F y_{t-1} + d + D v_{t-SHIFT}, R) (t ≥ 2)

`SHIFT` follows whichever LDS input channel the baseline stands in for: the
dynamics input enters lagged (`x_t = A x_{t-1} + b + B ux_{t-1}`), the
observation input contemporaneously (`y_t = C x_t + d + D uy_t`). A lagged input
that predates the trial is zero, mirroring the LDS's input-free `x_1`.

The `μ₀`/`R₀` step of the VAR baselines mirrors the LDS's `(x0, P0)` layer, so
every timestep is scored and the baseline stays comparable to the LDS marginal
likelihood term for term.

Baselines are fit with `fit!` and scored with `loglikelihood`; `r2`,
`nullloglikelihood` and `nobs` build on those to score a fitted LDS against a
baseline. Priors are optional and mirror the LDS M-step: an `MNPrior` on the
stacked regression matrix, `IWPrior`s on `R` and `R₀`.
=============================================================================#

const NULL_BASELINES = (:intercept, :inputs, :var, :var_inputs)

"""
    AffineNullModel{T,LAG,SHIFT}

Affine-Gaussian baseline model used as a null reference for a fitted
[`LinearDynamicalSystem`](@ref). `LAG` (`Bool`) selects the autoregressive term
`F y_{t-1}`; `SHIFT` (`0` or `1`) is the input lag. An absent input block is a
zero-column `D`, matching how absent LDS inputs are canonicalized.

# Constructors
    AffineNullModel{T}(obs_dim; input_dim=0, lag=false, input_shift=0, priors...)
    AffineNullModel{T}(baseline::Symbol, obs_dim; input_dim=0, input_shift=0, priors...)

Prefer the second form, which names one of `$(NULL_BASELINES)` and sets `lag` /
`input_dim` accordingly — `input_dim` is forced to `0` for the input-free
baselines `:intercept` and `:var`. `input_shift` must be `0` (score `v_t`,
matching the LDS observation input) or `1` (score `v_{t-1}`, matching the LDS
dynamics input).

```julia
null = AffineNullModel{Float64}(:var_inputs, obs_dim; input_dim=2, input_shift=1)
fit!(null, y; inputs=ux)
loglikelihood(null, y; inputs=ux)
```

# Fields
- `obs_dim::Int`: observation dimension.
- `input_dim::Int`: input dimension (`0` when the baseline takes no inputs).
- `d::Vector{T}`: intercept.
- `D::Matrix{T}`: input matrix (`obs_dim × input_dim`).
- `F::Matrix{T}`: autoregressive matrix (`obs_dim × obs_dim`, empty when `!LAG`).
- `R::Matrix{T}`: innovation covariance.
- `μ₀::Vector{T}`, `R₀::Matrix{T}`: initial-step mean/covariance (empty when `!LAG`).
- `W_prior`, `R_prior`, `R₀_prior`: optional MN/IW priors, mirroring the LDS M-step.

See also [`r2`](@ref), [`nullloglikelihood`](@ref).
"""
mutable struct AffineNullModel{T<:Real,LAG,SHIFT}
    obs_dim::Int
    input_dim::Int
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

    lag_dim = lag ? obs_dim : 0
    return AffineNullModel{T,lag,input_shift}(
        obs_dim,
        input_dim,
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
    baseline::Symbol, obs_dim::Int; input_dim::Int=0, kwargs...
) where {T<:Real}
    baseline in NULL_BASELINES ||
        throw(ArgumentError("baseline must be one of $(NULL_BASELINES); got :$baseline"))
    lag = baseline === :var || baseline === :var_inputs
    takes_inputs = baseline === :inputs || baseline === :var_inputs
    return AffineNullModel{T}(
        obs_dim; input_dim=(takes_inputs ? input_dim : 0), lag=lag, kwargs...
    )
end

#=
Trial normalization. A baseline is a standalone model, so it cannot reuse the
`Data(lds, y; ux, uy)` constructor — that validates against an LDS. It accepts
the same public shape family instead and canonicalizes here: a single
`(obs_dim, T)` matrix, an `(obs_dim, T, ntrials)` array, or a vector of per-trial
matrices (ragged lengths allowed).
=#
_null_trials(y::AbstractVector{<:AbstractMatrix}) = y
_null_trials(y::AbstractMatrix) = [y]
_null_trials(y::AbstractArray{<:Any,3}) = [view(y, :, :, n) for n in axes(y, 3)]

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
    trials = _null_trials(v)
    length(trials) == length(tsteps) ||
        throw(DimensionMismatchError("inputs ntrials", length(tsteps), length(trials)))
    for (i, vi) in enumerate(trials)
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

# `v_{t-SHIFT}` aligned with `y`'s columns; entries predating the trial are zero.
_shifted_inputs(v::AbstractMatrix, ::Val{0}) = v

function _shifted_inputs(v::AbstractMatrix{T}, ::Val{1}) where {T<:Real}
    input_dim, tsteps = size(v)
    shifted = zeros(T, input_dim, tsteps)
    tsteps > 1 && copyto!(view(shifted, :, 2:tsteps), view(v, :, 1:(tsteps - 1)))
    return shifted
end

# Response/design pair for the main regression, stacked over trials:
#   LAG = false:  y_t ~ [1; v_{t-SHIFT}]            for t = 1..T
#   LAG = true:   y_t ~ [y_{t-1}; 1; v_{t-SHIFT}]   for t = 2..T
# A length-1 trial informs only `(μ₀, R₀)`, so it drops out of the LAG design.
function _null_design(
    null::AffineNullModel{T,LAG,SHIFT},
    y::AbstractVector{<:AbstractMatrix{T}},
    v::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real,LAG,SHIFT}
    responses = Matrix{T}[]
    designs = Matrix{T}[]

    for (yn, vn) in zip(y, v)
        tsteps = size(yn, 2)
        LAG && tsteps < 2 && continue

        rows = LAG ? (2:tsteps) : (1:tsteps)
        blocks = Matrix{T}[]
        LAG && push!(blocks, yn[:, rows .- 1])
        push!(blocks, ones(T, 1, length(rows)))
        if null.input_dim > 0
            push!(blocks, _shifted_inputs(vn, Val(SHIFT))[:, rows])
        end

        push!(responses, yn[:, rows])
        push!(designs, reduce(vcat, blocks))
    end

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
function _pack_W(null::AffineNullModel{T,LAG}) where {T<:Real,LAG}
    blocks = Matrix{T}[]
    LAG && push!(blocks, null.F)
    push!(blocks, reshape(null.d, null.obs_dim, 1))
    null.input_dim > 0 && push!(blocks, null.D)
    return reduce(hcat, blocks)
end

function _unpack_W!(null::AffineNullModel{T,LAG}, W::AbstractMatrix{T}) where {T<:Real,LAG}
    offset = 0
    if LAG
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

# Plug-in Gaussian log-density of `Y` under `y ~ N(W X, R)`. The single scoring
# kernel behind both `loglikelihood` and the MAP objective.
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
has rank ≤ ntrials - 1 and is singular whenever `ntrials ≤ obs_dim` — the common
single-trial case. An `R₀_prior` keeps it positive definite through the IW scale;
without one, fall back to the innovation covariance `R` so the baseline still
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
function StatsAPI.fit!(null::AffineNullModel{T}, y; inputs=nothing) where {T<:Real}
    trials = _null_trials(y)
    tsteps = Int[size(yn, 2) for yn in trials]
    return fit!(null, trials, _null_input_trials(T, inputs, tsteps, null.input_dim))
end

function StatsAPI.fit!(
    null::AffineNullModel{T,LAG},
    y::AbstractVector{<:AbstractMatrix{T}},
    v::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real,LAG}
    Y, X = _null_design(null, y, v)
    W, R = _null_fit_regression(Y, X, null.W_prior, null.R_prior)
    _unpack_W!(null, W)
    null.R = R

    if LAG
        Y₀, X₀ = _null_init_design(T, y)
        W₀, R₀ = _null_fit_regression(Y₀, X₀, nothing, null.R₀_prior)
        null.μ₀ = vec(W₀[:, 1])
        null.R₀ = _null_init_cov(R₀, R, size(Y₀, 2), null.obs_dim, null.R₀_prior)
    end
    return null
end

"""
    loglikelihood(null::AffineNullModel, y; inputs=nothing)

Plug-in log-likelihood of `y` under the fitted baseline — no prior terms, so it
is directly comparable to the LDS marginal `loglikelihood(lds, y)`. Scoring data
other than the fit data gives the held-out log-likelihood.
"""
function StatsAPI.loglikelihood(null::AffineNullModel{T}, y; inputs=nothing) where {T<:Real}
    trials = _null_trials(y)
    tsteps = Int[size(yn, 2) for yn in trials]
    v = _null_input_trials(T, inputs, tsteps, null.input_dim)
    return loglikelihood(null, trials, v)
end

function StatsAPI.loglikelihood(
    null::AffineNullModel{T,LAG},
    y::AbstractVector{<:AbstractMatrix{T}},
    v::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real,LAG}
    Y, X = _null_design(null, y, v)
    ll = _null_plugin_ll(Y, X, _pack_W(null), null.R)
    if LAG
        Y₀, X₀ = _null_init_design(T, y)
        ll += _null_plugin_ll(Y₀, X₀, reshape(null.μ₀, null.obs_dim, 1), null.R₀)
    end
    return ll
end

# Number of scalar observations in `y` — `obs_dim * sum(tsteps)`.
function StatsAPI.nobs(null::AffineNullModel, y)
    return null.obs_dim * sum(size(yn, 2) for yn in _null_trials(y))
end

#=
MAP objective at the fitted parameters: the plug-in data log-likelihood plus the
IW/MN log-prior terms `elbo!` carries at the EM fixed point. Not a likelihood, so
it stays internal — it exists so a prior-regularized baseline can be compared
against a prior-regularized SSM ELBO on the same footing.
=#
function _null_logmap(null::AffineNullModel{T}, y; inputs=nothing) where {T<:Real}
    trials = _null_trials(y)
    tsteps = Int[size(yn, 2) for yn in trials]
    v = _null_input_trials(T, inputs, tsteps, null.input_dim)
    return _null_logmap(null, trials, v)
end

function _null_logmap(
    null::AffineNullModel{T,LAG},
    y::AbstractVector{<:AbstractMatrix{T}},
    v::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real,LAG}
    ll = loglikelihood(null, y, v)
    ll += mn_logprior_term(_pack_W(null), null.R, null.W_prior)
    null.R_prior === nothing || (ll += iw_logprior_term(null.R, null.R_prior))
    if LAG && null.R₀_prior !== nothing
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
    v, shift = _null_channel(data, channel)
    obs_dim = size(first(data.y), 1)
    null = AffineNullModel{T}(
        baseline, obs_dim; input_dim=size(first(v), 1), input_shift=shift, R_prior=R_prior
    )

    # `input_dim` is forced to 0 for the input-free baselines, so hand those
    # matching zero-row inputs rather than the channel's.
    inputs = null.input_dim > 0 ? v : [zeros(T, 0, Ti) for Ti in data.tsteps]
    fit!(null, data.y, inputs)
    return loglikelihood(null, data.y, inputs)
end

"""
    nobs(lds::LinearDynamicalSystem, y; ux=nothing, uy=nothing)

Number of scalar observations in `y` — `obs_dim * sum(tsteps)`, summed over
trials. This is the denominator [`r2`](@ref) uses for the Cox–Snell and
Nagelkerke variants, so it is on the same scale as `loglikelihood(lds, y)`.
"""
function StatsAPI.nobs(
    lds::LinearDynamicalSystem{T}, y; ux=nothing, uy=nothing
) where {T<:Real}
    return nobs(Data(lds, y; ux=ux, uy=uy))
end

StatsAPI.nobs(data::Data) = size(first(data.y), 1) * sum(data.tsteps)

"""
    nullloglikelihood(lds, y; ux=nothing, uy=nothing, R_prior=lds.obs_model.R_prior)

Plug-in log-likelihood of the intercept baseline `y_t ~ N(d, R)` fit on the same
data — the reference point [`r2`](@ref) measures `lds` against by default.

`R_prior` defaults to the LDS's own observation-noise prior so the baseline's `R`
is regularized exactly as the LDS's is.
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
Pseudo-R² variants, following the StatsBase definitions so the numbers mean here
what they mean for a GLM. `n` is `nobs`, the scalar-observation count, matching
the scale of the log-likelihoods it divides.
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

Restricted to the Gaussian LDS: a Poisson LDS has no tractable marginal
likelihood and an `SLDS` is a different type, so both raise a `MethodError`.

# Keyword Arguments
- `null::Symbol = :intercept`: baseline to score against, one of `$(NULL_BASELINES)`.
- `null_inputs::Symbol = :ux`: LDS input channel the input-bearing baselines
  consume. `:ux` enters lagged, `:uy` contemporaneous, matching each channel's
  role in the LDS.
- `R_prior`: IW prior on the baseline's `R`, defaulting to the LDS's own
  observation-noise prior so both are regularized alike.

# Examples
Held-out R² comes from fitting a baseline on one split and scoring another,
which is the same two calls `r2` makes internally:

```julia
null = AffineNullModel{Float64}(:intercept, obs_dim)
fit!(null, y_train)
ll  = loglikelihood(lds, y_test)
ll₀ = loglikelihood(null, y_test)
r2_heldout = 1 - exp(2 * (ll₀ - ll) / nobs(lds, y_test))
```

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
