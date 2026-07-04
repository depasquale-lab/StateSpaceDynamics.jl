# =============================================================================
# Baseline / null-model log-likelihoods for an LDS-style data layout.
#
# `test_null` fits four latent-free baselines on a `Data{T}` struct and reports
# their training and (optionally) test log-likelihoods, computed under the same
# convention used by the SSM's `elbo!` at the EM fixed point:
#
#   training LL = data Gaussian LL + iw_logprior_term(R, R_prior)
#                                  + mn_logprior_term(W, R, W_prior)
#
# i.e. plug-in MAP parameters with the same partial-constant prior
# contributions that the SSM ELBO uses. The test LL is the plug-in Gaussian
# log-density on the test arrays (no prior terms), matching the convention of
# the Kalman-filter marginal `loglikelihood(lds, y)`.
#
# The four baselines are:
#   1. intercept   y_t ~ N(d, R)
#   2. inputs      y_t ~ N(d + D v_t, R)
#   3. var         y_1 ~ N(μ_0, R_0);  y_t ~ N(F y_{t-1} + d, R)         (t ≥ 2)
#   4. var_inputs  y_1 ~ N(μ_0, R_0);  y_t ~ N(F y_{t-1} + d + D v_t, R) (t ≥ 2)
#
# Each VAR variant carries an additional init regression that mirrors the SSM's
# (x0, P0) layer: μ_0 has no MN prior (matches `update_initial_state_mean!`'s
# unregularized mean), R_0 takes the optional `R0_prior` IW prior (matches
# `update_initial_state_covariance!`'s use of `P0_prior`).
# =============================================================================

"""
    test_null(train_data::Data{T}; kwargs...) -> NamedTuple

Fit four latent-free baseline models on `train_data` and return their
training and (optionally) test log-likelihoods. Designed as a fair baseline
to compare against an SSM fit on the same data: the training LL uses the
same plug-in MAP + IW/MN log-prior decomposition as the SSM's `elbo!` at
its EM fixed point.

# Arguments
- `train_data::Data{T}`: training data struct
  (`y` is `obs_dim × tsteps × ntrials`).

# Keyword Arguments
- `test_data::Union{Nothing,Data{T}} = nothing`: optional test set with the
  same `obs_dim`.
- `train_inputs::Union{Nothing,AbstractArray{T,3}} = nothing`: per-timestep
  inputs `v_t` for the input-bearing models (variants 2 and 4) on training.
  Defaults to `train_data.ux`; pass a zero-row array (e.g. `zeros(T, 0, T, N)`)
  to disable inputs on those two variants.
- `test_inputs::Union{Nothing,AbstractArray{T,3}} = nothing`: per-timestep
  inputs on test; defaults to `test_data.ux`.
- `intercept_W_prior, inputs_W_prior, var_W_prior, var_inputs_W_prior::Union{Nothing,MNPrior{T}} = nothing`:
  matrix-normal priors on the regression matrices for each of the four
  variants. Shapes must match the variant's regressor count
  (`1`, `1 + v_dim`, `obs_dim + 1`, `obs_dim + 1 + v_dim` respectively).
- `R_prior::Union{Nothing,IWPrior{T}} = nothing`: IW prior on the
  observation covariance `R` (applied to all four variants).
- `R0_prior::Union{Nothing,IWPrior{T}} = nothing`: IW prior on the
  initial-step covariance `R_0` (applied to the two VAR variants only).

# Returns
A `NamedTuple` keyed by model name (`intercept`, `inputs`, `var`,
`var_inputs`). Each entry is a `NamedTuple` with:
- `train_ll::T`: training log-likelihood under the prior-augmented convention
- `test_ll::Union{Nothing,T}`: plug-in test log-likelihood, or `nothing`
- `params::NamedTuple`: fitted MAP parameters (`d`, `R`, and where
  applicable `D`, `F`, `μ_0`, `R_0`)
"""
function test_null(
    train_data::Data{T};
    test_data::Union{Nothing,Data{T}}=nothing,
    train_inputs::Union{Nothing,AbstractArray{T,3}}=nothing,
    test_inputs::Union{Nothing,AbstractArray{T,3}}=nothing,
    intercept_W_prior::Union{Nothing,MNPrior{T}}=nothing,
    inputs_W_prior::Union{Nothing,MNPrior{T}}=nothing,
    var_W_prior::Union{Nothing,MNPrior{T}}=nothing,
    var_inputs_W_prior::Union{Nothing,MNPrior{T}}=nothing,
    R_prior::Union{Nothing,IWPrior{T}}=nothing,
    R0_prior::Union{Nothing,IWPrior{T}}=nothing,
) where {T<:Real}
    obs_dim, tsteps, ntrials = size(train_data.y)

    v_train = train_inputs === nothing ? train_data.ux : train_inputs
    v_test = if test_data === nothing
        nothing
    elseif test_inputs !== nothing
        test_inputs
    else
        test_data.ux
    end

    _null_check_inputs(v_train, tsteps, ntrials, "train_inputs")
    if test_data !== nothing
        size(test_data.y, 1) == obs_dim || throw(
            DimensionMismatchError("test_data.y obs_dim", obs_dim, size(test_data.y, 1))
        )
        _null_check_inputs(
            v_test, size(test_data.y, 2), size(test_data.y, 3), "test_inputs"
        )
    end

    intercept_res = _null_intercept(train_data, test_data, intercept_W_prior, R_prior)
    inputs_res = _null_inputs(
        train_data, test_data, v_train, v_test, inputs_W_prior, R_prior
    )
    var_res = _null_var(train_data, test_data, var_W_prior, R_prior, R0_prior)
    var_inputs_res = _null_var_inputs(
        train_data, test_data, v_train, v_test, var_inputs_W_prior, R_prior, R0_prior
    )

    return (
        intercept=intercept_res, inputs=inputs_res, var=var_res, var_inputs=var_inputs_res
    )
end

# -----------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------

function _null_check_inputs(
    v::AbstractArray{T,3}, tsteps::Int, ntrials::Int, name::String
) where {T<:Real}
    if size(v, 1) > 0 && (size(v, 2) != tsteps || size(v, 3) != ntrials)
        throw(
            DimensionMismatchError(
                "$name shape (input_dim, T, ntrials)",
                (size(v, 1), tsteps, ntrials),
                size(v),
            ),
        )
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Core regression + log-likelihood helpers
# -----------------------------------------------------------------------------

# Fit MAP (W, R) for the regression Y = W X + ε, ε ~ N(0, R), with optional
# MN prior on W and IW prior on R. `Y` is `(obs_dim, n)`, `X` is `(P, n)`.
# Returns the MAP `(W, R)`, the data-only residual scatter
# `S_data = YY - W·XY - XY'·W' + W·XX·W'`, and the sample count `n`. Mirrors
# the (mn_map + iw_map) M-step machinery used by `update_R!` in
# `lds/gaussian_observations.jl`.
function _null_fit_regression(
    Y::AbstractMatrix{T},
    X::AbstractMatrix{T},
    W_prior::Union{Nothing,MNPrior{T}},
    R_prior::Union{Nothing,IWPrior{T}},
) where {T<:Real}
    obs_dim = size(Y, 1)
    n = size(Y, 2)
    size(X, 2) == n || throw(DimensionMismatchError("X cols vs Y cols", n, size(X, 2)))

    XX = X * transpose(X)
    XY = X * transpose(Y)
    YY = Y * transpose(Y)

    # `mn_map` returns a `Transpose` view; materialize to a plain Matrix so
    # downstream BLAS-level ops hit the concrete-matrix code paths.
    W = Matrix(mn_map(XX, XY, W_prior))

    # Data-only residual scatter S_data = YY - W·XY - XY'·W' + W·XX·W'.
    Wxy = W * XY
    S_data = YY .- Wxy .- transpose(Wxy)
    S_data .+= W * XX * transpose(W)
    Symmetrize!(S_data)

    # MN-prior contribution to the IW posterior scale (matches update_R!).
    S_with_prior = copy(S_data)
    if W_prior !== nothing
        Wm = W .- W_prior.M₀
        S_with_prior .+= Wm * W_prior.Λ * transpose(Wm)
        Symmetrize!(S_with_prior)
    end

    R = if R_prior === nothing
        S_with_prior ./ T(n)
    else
        iw_map(R_prior.Ψ, R_prior.ν, S_with_prior, T(n), obs_dim)
    end

    return W, R, S_data, n
end

# Prior-augmented training log-likelihood at the MAP fit. Mirrors `elbo!`'s
# decomposition: data Gaussian LL + iw_logprior_term(R, R_prior)
# + mn_logprior_term(W, R, W_prior).
function _null_train_ll(
    W::AbstractMatrix{T},
    R::AbstractMatrix{T},
    S_data::AbstractMatrix{T},
    n::Int,
    W_prior::Union{Nothing,MNPrior{T}},
    R_prior::Union{Nothing,IWPrior{T}},
) where {T<:Real}
    obs_dim = size(R, 1)
    F = cholesky(Symmetric(R))
    log_det_R = 2 * sum(log, diag(F.U))

    # tr(R^{-1} · S_data) via two triangular solves.
    S_work = copy(S_data)
    ldiv!(F.U', S_work)
    ldiv!(F.U, S_work)
    tr_R_inv_S = tr(S_work)

    ll = T(-0.5) * (T(n) * (T(obs_dim) * log(T(2π)) + log_det_R) + tr_R_inv_S)
    ll += mn_logprior_term(W, R, W_prior)
    if R_prior !== nothing
        ll += iw_logprior_term(R, R_prior)
    end

    return ll
end

# Plug-in Gaussian log-likelihood on test data using trained (W, R).
function _null_test_ll(
    Y::AbstractMatrix{T}, X::AbstractMatrix{T}, W::AbstractMatrix{T}, R::AbstractMatrix{T}
) where {T<:Real}
    obs_dim = size(Y, 1)
    n = size(Y, 2)
    F = cholesky(Symmetric(R))
    log_det_R = 2 * sum(log, diag(F.U))

    # Residuals E = Y - W X, with quadratic term tr(R^{-1} E E').
    E = Y .- W * X
    EE = E * transpose(E)
    Symmetrize!(EE)
    ldiv!(F.U', EE)
    ldiv!(F.U, EE)

    return T(-0.5) * (T(n) * (T(obs_dim) * log(T(2π)) + log_det_R) + tr(EE))
end

# -----------------------------------------------------------------------------
# Data stacking helpers
# -----------------------------------------------------------------------------

# Stack y over (t, n) → (obs_dim × tsteps*ntrials).
@inline function _stack_y_all(y::AbstractArray{T,3}) where {T<:Real}
    obs_dim, tsteps, ntrials = size(y)
    return reshape(y, obs_dim, tsteps * ntrials)
end

# Stack a 3-D input array (input_dim, tsteps, ntrials) → (input_dim, tsteps*ntrials).
@inline function _stack_inputs_all(v::AbstractArray{T,3}) where {T<:Real}
    v_dim, tsteps, ntrials = size(v)
    return reshape(v, v_dim, tsteps * ntrials)
end

# Build a (1 × n) row of ones for the bias column.
@inline _bias_row(::Type{T}, n::Int) where {T<:Real} = fill(one(T), 1, n)

# Stack y_t for t = 2..T over trials → (obs_dim × (tsteps-1)*ntrials).
@inline function _stack_y_next(y::AbstractArray{T,3}) where {T<:Real}
    obs_dim, tsteps, ntrials = size(y)
    return reshape(y[:, 2:tsteps, :], obs_dim, (tsteps - 1) * ntrials)
end

# Stack y_{t-1} for t = 2..T over trials → (obs_dim × (tsteps-1)*ntrials).
@inline function _stack_y_prev(y::AbstractArray{T,3}) where {T<:Real}
    obs_dim, tsteps, ntrials = size(y)
    return reshape(y[:, 1:(tsteps - 1), :], obs_dim, (tsteps - 1) * ntrials)
end

# Stack inputs v_t for t = 2..T over trials.
@inline function _stack_inputs_next(v::AbstractArray{T,3}) where {T<:Real}
    v_dim, tsteps, ntrials = size(v)
    return reshape(v[:, 2:tsteps, :], v_dim, (tsteps - 1) * ntrials)
end

# Stack y_1 across trials → (obs_dim × ntrials).
@inline function _stack_y_init(y::AbstractArray{T,3}) where {T<:Real}
    obs_dim = size(y, 1)
    return reshape(y[:, 1, :], obs_dim, size(y, 3))
end

# -----------------------------------------------------------------------------
# Model 1: intercept only
# -----------------------------------------------------------------------------

function _null_intercept(
    train_data::Data{T},
    test_data::Union{Nothing,Data{T}},
    W_prior::Union{Nothing,MNPrior{T}},
    R_prior::Union{Nothing,IWPrior{T}},
) where {T<:Real}
    Y = _stack_y_all(train_data.y)
    X = _bias_row(T, size(Y, 2))

    W, R, S_data, n = _null_fit_regression(Y, X, W_prior, R_prior)
    train_ll = _null_train_ll(W, R, S_data, n, W_prior, R_prior)

    test_ll = if test_data === nothing
        nothing
    else
        Y_te = _stack_y_all(test_data.y)
        X_te = _bias_row(T, size(Y_te, 2))
        _null_test_ll(Y_te, X_te, W, R)
    end

    d = vec(W[:, 1])
    params = (d=d, R=R)
    return (train_ll=train_ll, test_ll=test_ll, params=params)
end

# -----------------------------------------------------------------------------
# Model 2: inputs only (no autocorrelation)
# -----------------------------------------------------------------------------

function _null_inputs(
    train_data::Data{T},
    test_data::Union{Nothing,Data{T}},
    v_train::AbstractArray{T,3},
    v_test::Union{Nothing,AbstractArray{T,3}},
    W_prior::Union{Nothing,MNPrior{T}},
    R_prior::Union{Nothing,IWPrior{T}},
) where {T<:Real}
    v_dim = size(v_train, 1)

    Y = _stack_y_all(train_data.y)
    n = size(Y, 2)
    X = vcat(_bias_row(T, n), _stack_inputs_all(v_train))

    W, R, S_data, _ = _null_fit_regression(Y, X, W_prior, R_prior)
    train_ll = _null_train_ll(W, R, S_data, n, W_prior, R_prior)

    test_ll = if test_data === nothing
        nothing
    else
        Y_te = _stack_y_all(test_data.y)
        n_te = size(Y_te, 2)
        X_te = vcat(_bias_row(T, n_te), _stack_inputs_all(v_test))
        _null_test_ll(Y_te, X_te, W, R)
    end

    d = vec(W[:, 1])
    D = v_dim > 0 ? W[:, 2:end] : Matrix{T}(undef, size(W, 1), 0)
    params = (d=d, D=D, R=R)
    return (train_ll=train_ll, test_ll=test_ll, params=params)
end

# -----------------------------------------------------------------------------
# Model 3: VAR(1) only (no inputs)
# -----------------------------------------------------------------------------

function _null_var(
    train_data::Data{T},
    test_data::Union{Nothing,Data{T}},
    W_prior::Union{Nothing,MNPrior{T}},
    R_prior::Union{Nothing,IWPrior{T}},
    R0_prior::Union{Nothing,IWPrior{T}},
) where {T<:Real}
    obs_dim, tsteps, _ = size(train_data.y)
    tsteps >= 2 ||
        throw(ArgumentError("VAR(1) null model requires tsteps ≥ 2 (got $tsteps)"))

    # Init regression: y_1[:, n] ~ N(μ_0, R_0); regress on a constant.
    Y_init = _stack_y_init(train_data.y)
    X_init = _bias_row(T, size(Y_init, 2))
    W_init, R0, S_init, n_init = _null_fit_regression(Y_init, X_init, nothing, R0_prior)

    # VAR(1) regression: y_t ~ N(F y_{t-1} + d, R) for t = 2..T.
    Y_var = _stack_y_next(train_data.y)
    n_var = size(Y_var, 2)
    X_var = vcat(_stack_y_prev(train_data.y), _bias_row(T, n_var))

    W_var, R, S_var, _ = _null_fit_regression(Y_var, X_var, W_prior, R_prior)

    train_ll =
        _null_train_ll(W_init, R0, S_init, n_init, nothing, R0_prior) +
        _null_train_ll(W_var, R, S_var, n_var, W_prior, R_prior)

    test_ll = if test_data === nothing
        nothing
    else
        Y_init_te = _stack_y_init(test_data.y)
        X_init_te = _bias_row(T, size(Y_init_te, 2))
        Y_var_te = _stack_y_next(test_data.y)
        X_var_te = vcat(_stack_y_prev(test_data.y), _bias_row(T, size(Y_var_te, 2)))
        _null_test_ll(Y_init_te, X_init_te, W_init, R0) +
        _null_test_ll(Y_var_te, X_var_te, W_var, R)
    end

    μ_0 = vec(W_init[:, 1])
    F = W_var[:, 1:obs_dim]
    d = vec(W_var[:, obs_dim + 1])
    params = (μ_0=μ_0, R_0=R0, F=F, d=d, R=R)
    return (train_ll=train_ll, test_ll=test_ll, params=params)
end

# -----------------------------------------------------------------------------
# Model 4: VAR(1) + inputs
# -----------------------------------------------------------------------------

function _null_var_inputs(
    train_data::Data{T},
    test_data::Union{Nothing,Data{T}},
    v_train::AbstractArray{T,3},
    v_test::Union{Nothing,AbstractArray{T,3}},
    W_prior::Union{Nothing,MNPrior{T}},
    R_prior::Union{Nothing,IWPrior{T}},
    R0_prior::Union{Nothing,IWPrior{T}},
) where {T<:Real}
    obs_dim, tsteps, _ = size(train_data.y)
    tsteps >= 2 ||
        throw(ArgumentError("VAR(1)+inputs null model requires tsteps ≥ 2 (got $tsteps)"))
    v_dim = size(v_train, 1)

    # Init regression: same as the VAR-only model — y_1 is exogenous to inputs.
    Y_init = _stack_y_init(train_data.y)
    X_init = _bias_row(T, size(Y_init, 2))
    W_init, R0, S_init, n_init = _null_fit_regression(Y_init, X_init, nothing, R0_prior)

    # VAR(1)+inputs regression: y_t ~ N(F y_{t-1} + d + D v_t, R) for t = 2..T.
    Y_var = _stack_y_next(train_data.y)
    n_var = size(Y_var, 2)
    X_var = vcat(
        _stack_y_prev(train_data.y), _bias_row(T, n_var), _stack_inputs_next(v_train)
    )

    W_var, R, S_var, _ = _null_fit_regression(Y_var, X_var, W_prior, R_prior)

    train_ll =
        _null_train_ll(W_init, R0, S_init, n_init, nothing, R0_prior) +
        _null_train_ll(W_var, R, S_var, n_var, W_prior, R_prior)

    test_ll = if test_data === nothing
        nothing
    else
        Y_init_te = _stack_y_init(test_data.y)
        X_init_te = _bias_row(T, size(Y_init_te, 2))
        Y_var_te = _stack_y_next(test_data.y)
        X_var_te = vcat(
            _stack_y_prev(test_data.y),
            _bias_row(T, size(Y_var_te, 2)),
            _stack_inputs_next(v_test),
        )
        _null_test_ll(Y_init_te, X_init_te, W_init, R0) +
        _null_test_ll(Y_var_te, X_var_te, W_var, R)
    end

    μ_0 = vec(W_init[:, 1])
    F = W_var[:, 1:obs_dim]
    d = vec(W_var[:, obs_dim + 1])
    D = v_dim > 0 ? W_var[:, (obs_dim + 2):end] : Matrix{T}(undef, size(W_var, 1), 0)
    params = (μ_0=μ_0, R_0=R0, F=F, d=d, D=D, R=R)
    return (train_ll=train_ll, test_ll=test_ll, params=params)
end

# =============================================================================
# Cox–Snell (likelihood-ratio) R² of an LDS versus each null baseline.
#
# `compute_R2` scores a *fitted* Gaussian LDS against the four `test_null`
# baselines using the Cox–Snell R²
#
#   R² = 1 - exp( (2/n) · (loglik_null - loglik_model) )
#
# where `n = obs_dim · tsteps · ntrials` is the total scalar-observation count.
# `loglik_model` is the LDS **marginal** (observed-data) log-likelihood with
# latent states integrated out — the same quantity as `loglikelihood(lds, y)`
# but extended here to honour the LDS's `B·ux` / `D·uy` input terms, which the
# base `loglikelihood` drops. `loglik_null` is the plug-in Gaussian data
# log-density of a null model fit on the training split — the `test_null`
# `test_ll` convention (no prior terms), which is the null-side analogue of the
# LDS marginal likelihood. Priors therefore enter only when *fitting* the null
# parameters: the LDS's observation-noise prior `obs_model.R_prior` is forwarded
# to the null models' `R_prior` so both estimate `R` under the same IW prior.
# (The LDS's remaining priors — `Q`, `P0`, `[A B]`, `[C D]` — live in latent
# space and have no null-model counterpart, so they are not transferred.)
#
# Restricted to the Gaussian LDS: a Poisson LDS (`PoissonObservationModel`) has
# no tractable marginal likelihood, and an `SLDS` is a different type entirely,
# so both fall through to a `MethodError` by design.
# =============================================================================

# Cox–Snell R² from a pair of total log-likelihoods and the observation count.
@inline function _cox_snell_r2(ll_null::T, ll_model::T, n::Int) where {T<:Real}
    return one(T) - exp((T(2) / T(n)) * (ll_null - ll_model))
end

# Fit the four null baselines on `train_data` and return their plug-in Gaussian
# data log-densities evaluated on `eval_data` (the `test_null` `test_ll`
# convention, no prior terms). `null_inputs` selects which of the LDS's input
# channels the input-bearing baselines consume: `"ux"` uses `data.ux` (the
# `test_null` default), `"uy"` uses `data.uy`.
function _null_plugin_lls(
    train_data::Data{T},
    eval_data::Data{T},
    null_inputs::AbstractString,
    R_prior::Union{Nothing,IWPrior{T}},
) where {T<:Real}
    res = if null_inputs == "ux"
        test_null(train_data; test_data=eval_data, R_prior=R_prior)
    elseif null_inputs == "uy"
        test_null(
            train_data;
            test_data=eval_data,
            train_inputs=train_data.uy,
            test_inputs=eval_data.uy,
            R_prior=R_prior,
        )
    else
        throw(ArgumentError("null_inputs must be \"ux\" or \"uy\"; got \"$null_inputs\""))
    end
    return (
        intercept=res.intercept.test_ll,
        inputs=res.inputs.test_ll,
        var=res.var.test_ll,
        var_inputs=res.var_inputs.test_ll,
    )
end

"""
    compute_R2(lds, data; null_inputs="ux", R_prior=lds.obs_model.R_prior) -> NamedTuple
    compute_R2(lds, train_data, test_data; kwargs...) -> NamedTuple

Cox–Snell (likelihood-ratio) R² of a fitted Gaussian `LinearDynamicalSystem`
against each of the four [`test_null`](@ref) baselines (`intercept`, `inputs`,
`var`, `var_inputs`):

```math
R^2 = 1 - \\exp\\!\\big( (2/n) \\, (\\ell_{null} - \\ell_{model}) \\big)
```

with `n = obs_dim · tsteps · ntrials` the total number of scalar observations.
`ℓ_model` is the LDS **marginal** log-likelihood (latents integrated out, via
the Kalman filter, honouring the `B·ux`/`D·uy` input terms). `ℓ_null` is the
plug-in Gaussian data log-density of a baseline fit on the training split (no
prior terms — the null-side analogue of the LDS marginal likelihood). A larger
R² means the LDS explains the data better than that baseline; it is positive
whenever the LDS out-predicts the null.

Uses the marginal likelihood, so this is restricted to the Gaussian LDS. A
Poisson LDS or an `SLDS` throws a `MethodError` (their marginals are
intractable / handled elsewhere).

# Arguments
- `lds::LinearDynamicalSystem`: a fitted Gaussian LDS.
- `data::Data{T}`: single dataset — returns one R² per baseline.
- `train_data, test_data::Data{T}`: fit the baselines on `train_data` and score
  both splits — returns `train_R2` and `test_R2` per baseline.

# Keyword Arguments
- `null_inputs::AbstractString = "ux"`: which LDS input channel the
  input-bearing baselines (`inputs`, `var_inputs`) consume — `"ux"` for the
  dynamics inputs `data.ux`, `"uy"` for the observation inputs `data.uy`.
- `R_prior::Union{Nothing,IWPrior{T}} = lds.obs_model.R_prior`: IW prior used
  when fitting each baseline's observation covariance `R`, defaulting to the
  LDS's own observation-noise prior so both are regularized alike.

# Returns
A `NamedTuple` keyed by baseline name. For the single-`data` form each entry is
`(R2, lds_ll, null_ll, n)`; for the `train_data`/`test_data` form each entry is
`(train_R2, test_R2, lds_train_ll, lds_test_ll, null_train_ll, null_test_ll,
n_train, n_test)`.
"""
function compute_R2(
    lds::LinearDynamicalSystem{T,SM,OM},
    data::Data{T};
    null_inputs::AbstractString="ux",
    R_prior::Union{Nothing,IWPrior{T}}=lds.obs_model.R_prior,
) where {T<:Real,SM<:GaussianStateModel{T},OM<:GaussianObservationModel{T}}
    obs_dim, tsteps, ntrials = size(data.y)
    n = obs_dim * tsteps * ntrials

    lds_ll = loglikelihood(lds, data)
    null_ll = _null_plugin_lls(data, data, null_inputs, R_prior)

    return map(null_ll) do ll
        return (R2=_cox_snell_r2(ll, lds_ll, n), lds_ll=lds_ll, null_ll=ll, n=n)
    end
end

function compute_R2(
    lds::LinearDynamicalSystem{T,SM,OM},
    train_data::Data{T},
    test_data::Data{T};
    null_inputs::AbstractString="ux",
    R_prior::Union{Nothing,IWPrior{T}}=lds.obs_model.R_prior,
) where {T<:Real,SM<:GaussianStateModel{T},OM<:GaussianObservationModel{T}}
    obs_dim, tsteps_tr, ntrials_tr = size(train_data.y)
    obs_dim_te, tsteps_te, ntrials_te = size(test_data.y)
    obs_dim == obs_dim_te ||
        throw(DimensionMismatchError("test_data obs_dim", obs_dim, obs_dim_te))
    n_train = obs_dim * tsteps_tr * ntrials_tr
    n_test = obs_dim_te * tsteps_te * ntrials_te

    lds_train_ll = loglikelihood(lds, train_data)
    lds_test_ll = loglikelihood(lds, test_data)

    null_train_ll = _null_plugin_lls(train_data, train_data, null_inputs, R_prior)
    null_test_ll = _null_plugin_lls(train_data, test_data, null_inputs, R_prior)

    names = (:intercept, :inputs, :var, :var_inputs)
    entries = map(names) do name
        ntr = getproperty(null_train_ll, name)
        nte = getproperty(null_test_ll, name)
        return (
            train_R2=_cox_snell_r2(ntr, lds_train_ll, n_train),
            test_R2=_cox_snell_r2(nte, lds_test_ll, n_test),
            lds_train_ll=lds_train_ll,
            lds_test_ll=lds_test_ll,
            null_train_ll=ntr,
            null_test_ll=nte,
            n_train=n_train,
            n_test=n_test,
        )
    end
    return NamedTuple{names}(entries)
end
