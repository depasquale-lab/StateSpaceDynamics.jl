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
            DimensionMismatchError(
                "test_data.y obs_dim", obs_dim, size(test_data.y, 1)
            ),
        )
        _null_check_inputs(
            v_test, size(test_data.y, 2), size(test_data.y, 3), "test_inputs"
        )
    end

    intercept_res = _null_intercept(
        train_data, test_data, intercept_W_prior, R_prior
    )
    inputs_res = _null_inputs(
        train_data, test_data, v_train, v_test, inputs_W_prior, R_prior
    )
    var_res = _null_var(
        train_data, test_data, var_W_prior, R_prior, R0_prior
    )
    var_inputs_res = _null_var_inputs(
        train_data,
        test_data,
        v_train,
        v_test,
        var_inputs_W_prior,
        R_prior,
        R0_prior,
    )

    return (
        intercept=intercept_res,
        inputs=inputs_res,
        var=var_res,
        var_inputs=var_inputs_res,
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
    size(X, 2) == n || throw(
        DimensionMismatchError("X cols vs Y cols", n, size(X, 2))
    )

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
    Y::AbstractMatrix{T},
    X::AbstractMatrix{T},
    W::AbstractMatrix{T},
    R::AbstractMatrix{T},
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
    tsteps >= 2 || throw(
        ArgumentError("VAR(1) null model requires tsteps ≥ 2 (got $tsteps)")
    )

    # Init regression: y_1[:, n] ~ N(μ_0, R_0); regress on a constant.
    Y_init = _stack_y_init(train_data.y)
    X_init = _bias_row(T, size(Y_init, 2))
    W_init, R0, S_init, n_init = _null_fit_regression(
        Y_init, X_init, nothing, R0_prior
    )

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
        X_var_te = vcat(
            _stack_y_prev(test_data.y), _bias_row(T, size(Y_var_te, 2))
        )
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
    tsteps >= 2 || throw(
        ArgumentError(
            "VAR(1)+inputs null model requires tsteps ≥ 2 (got $tsteps)"
        ),
    )
    v_dim = size(v_train, 1)

    # Init regression: same as the VAR-only model — y_1 is exogenous to inputs.
    Y_init = _stack_y_init(train_data.y)
    X_init = _bias_row(T, size(Y_init, 2))
    W_init, R0, S_init, n_init = _null_fit_regression(
        Y_init, X_init, nothing, R0_prior
    )

    # VAR(1)+inputs regression: y_t ~ N(F y_{t-1} + d + D v_t, R) for t = 2..T.
    Y_var = _stack_y_next(train_data.y)
    n_var = size(Y_var, 2)
    X_var = vcat(
        _stack_y_prev(train_data.y),
        _bias_row(T, n_var),
        _stack_inputs_next(v_train),
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
    D =
        v_dim > 0 ? W_var[:, (obs_dim + 2):end] :
        Matrix{T}(undef, size(W_var, 1), 0)
    params = (μ_0=μ_0, R_0=R0, F=F, d=d, D=D, R=R)
    return (train_ll=train_ll, test_ll=test_ll, params=params)
end
