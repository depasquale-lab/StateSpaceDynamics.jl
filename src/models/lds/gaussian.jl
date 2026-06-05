"""
Linear Dynamical System (LDS) implementation with Gaussian state and observation models.
This module defines the `GaussianStateModel` and `GaussianObservationModel` types,
as well as functions for sampling, computing log-likelihoods, gradients, Hessians,
and smoothing. The code is optimized for performance, with careful attention to
memory allocation and multi-threading for multi-trial sampling.
"""

function _extract_state_params(state_model::GaussianStateModel{T}) where {T}
    return (
        A=state_model.A,
        B=state_model.B,
        Q=state_model.Q,
        b=state_model.b,
        x0=state_model.x0,
        P0=state_model.P0,
    )
end

"""
    initialize_FilterSmooth(model, tsteps::Int)

Initialize a per-trial `FilterSmooth` buffer sized for `tsteps` timesteps.
"""
function initialize_FilterSmooth(
    model::LinearDynamicalSystem{T,S,O}, tsteps::Int; cov_alias::Bool=false
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    D = model.latent_dim
    if cov_alias
        p_smooth = zeros(T, 0, 0, 0)
        p_smooth_tt1 = zeros(T, 0, 0, 0)
        E_zz = zeros(T, 0, 0, 0)
        E_zz_prev = zeros(T, 0, 0, 0)
    else
        p_smooth = zeros(T, D, D, tsteps)
        p_smooth_tt1 = zeros(T, D, D, tsteps)
        E_zz = zeros(T, D, D, tsteps)
        E_zz_prev = zeros(T, D, D, tsteps)
    end
    return FilterSmooth{T}(
        zeros(T, D, tsteps),       # x_smooth
        p_smooth,
        p_smooth_tt1,
        zeros(T, D, tsteps),       # E_z
        E_zz,
        E_zz_prev,
        zero(T),                   # entropy
    )
end

"""
    initialize_FilterSmooth(model, tsteps_per_trial::AbstractVector{<:Integer};
                            cov_alias=false)

Initialize a `TrialFilterSmooth` with one `FilterSmooth` per trial. Trial lengths
may differ (but don't have to).

Set `cov_alias=true` only when the caller knows the cov-cache fast path will
run (equal-length multi-trial Gaussian via `_fit_tridiag!`) — in that case
every per-trial `p_smooth` / `p_smooth_tt1` is allocated as a `(0, 0, 0)` stub
because `smooth!` aliases them to `sws.p_smooth_shared` on every E-step. The
SLDS / Poisson / ragged paths invoke the per-trial smoother directly and
write into `fs.p_smooth`, so they must keep the default `cov_alias=false`.
"""
function initialize_FilterSmooth(
    model::LinearDynamicalSystem{T,S,O},
    tsteps_per_trial::AbstractVector{<:Integer};
    cov_alias::Bool=false,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    # if tsteps_per_trial has varying lengths, we can't alias the cov caches to a shared zero-array
    if cov_alias && length(unique(tsteps_per_trial)) != 1
        throw(
            ArgumentError(
                "cov_alias=true is only valid when all trials have the same number of timesteps; got tsteps_per_trial=$(tsteps_per_trial)",
            ),
        )
    end
    filter_smooths = [
        initialize_FilterSmooth(model, Int(t); cov_alias=cov_alias) for
        t in tsteps_per_trial
    ]
    return TrialFilterSmooth(filter_smooths)
end

function _extract_obs_params(obs_model::GaussianObservationModel{T}) where {T}
    return (C=obs_model.C, R=obs_model.R, d=obs_model.d, D=obs_model.D)
end

function _extract_obs_params(obs_model::PoissonObservationModel{T}) where {T}
    return (C=obs_model.C, d=obs_model.d)
end

function _get_all_params_vec(
    lds::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:AbstractStateModel{T},O<:AbstractObservationModel{T}}
    state_params = _extract_state_params(lds.state_model)
    obs_params = _extract_obs_params(lds.obs_model)

    # Convert named tuples to vectors and concatenate
    state_vec = vcat(
        vec(state_params.A),
        vec(state_params.B),
        vec(state_params.Q),
        vec(state_params.b),
        vec(state_params.x0),
        vec(state_params.P0),
    )

    if lds.obs_model isa GaussianObservationModel
        obs_vec = vcat(
            vec(obs_params.C), vec(obs_params.R), vec(obs_params.d), vec(obs_params.D)
        )
    else # PoissonObservationModel
        obs_vec = vcat(vec(obs_params.C), vec(obs_params.d))
    end

    return vcat(state_vec, obs_vec)
end

function _sample_trial!(
    rng,
    x_trial,
    y_trial,
    state_params,
    obs_params,
    obs_model::GaussianObservationModel,
    u_trial::AbstractMatrix,
    v_trial::AbstractMatrix,
)
    tsteps = size(x_trial, 2)

    # Initial state. The observation at t=1 includes the obs-input term D·v_1
    # when v_trial has nonzero rows; zero-row matmul is a no-op.
    x_trial[:, 1] = rand(rng, MvNormal(state_params.x0, state_params.P0))
    y_trial[:, 1] = rand(
        rng,
        MvNormal(
            obs_params.C * x_trial[:, 1] + obs_params.d + obs_params.D * v_trial[:, 1],
            obs_params.R,
        ),
    )

    # Subsequent states. The dynamics input B·u_{t-1} kicks the state forward;
    # again, zero-row u_trial degenerates to no input.
    for t in 2:tsteps
        x_trial[:, t] = rand(
            rng,
            MvNormal(
                state_params.A * x_trial[:, t - 1] +
                state_params.b +
                state_params.B * u_trial[:, t - 1],
                state_params.Q,
            ),
        )
        y_trial[:, t] = rand(
            rng,
            MvNormal(
                obs_params.C * x_trial[:, t] + obs_params.d + obs_params.D * v_trial[:, t],
                obs_params.R,
            ),
        )
    end
end

function _sample_trial!(
    rng,
    x_trial,
    y_trial,
    state_params,
    obs_params,
    obs_model::PoissonObservationModel,
    u_trial::AbstractMatrix,
    v_trial::AbstractMatrix,
)
    tsteps = size(x_trial, 2)
    # Poisson obs model has no D matrix; v_trial is accepted for signature
    # parity with the Gaussian path but must be empty (validated by callers).
    @assert size(v_trial, 1) == 0 "Poisson observation model does not support obs inputs"

    # Initial state
    x_trial[:, 1] = rand(rng, MvNormal(state_params.x0, state_params.P0))
    y_trial[:, 1] = rand.(rng, Poisson.(exp.(obs_params.C * x_trial[:, 1] + obs_params.d)))

    # Subsequent states
    for t in 2:tsteps
        x_trial[:, t] = rand(
            rng,
            MvNormal(
                state_params.A * x_trial[:, t - 1] +
                state_params.b +
                state_params.B * u_trial[:, t - 1],
                state_params.Q,
            ),
        )
        y_trial[:, t] = rand.(
            rng, Poisson.(exp.(obs_params.C * x_trial[:, t] + obs_params.d))
        )
    end
end

"""
    Random.rand([rng,] lds, tsteps::Integer; control_seq=nothing, obs_control_seq=nothing)
    Random.rand([rng,] lds, tsteps_per_trial::AbstractVector{<:Integer};
                control_seq=nothing, obs_control_seq=nothing)

Sample from a Linear Dynamical System.

- With a scalar `tsteps`, returns one trial as `(x::Matrix, y::Matrix)` of sizes
  `(latent_dim, tsteps)` and `(obs_dim, tsteps)` respectively.
- With a vector of per-trial lengths, returns
  `(x::Vector{Matrix}, y::Vector{Matrix})`. Lengths may differ across trials.

Optional control sequences:
- `control_seq`: dynamics-input sequence consumed by `B`. Single-trial form
  is an `(u_dim, tsteps)` matrix; multi-trial is a `Vector{<:AbstractMatrix}`
  of per-trial matrices. Required when `size(state_model.B, 2) > 0`.
- `obs_control_seq`: same shape for the observation input `D`. Required when
  `size(obs_model.D, 2) > 0`. Gaussian observation model only.
"""
function Random.rand(
    rng::AbstractRNG,
    lds::LinearDynamicalSystem{T,S,O},
    tsteps::Integer;
    control_seq::Union{Nothing,AbstractMatrix{T}}=nothing,
    obs_control_seq::Union{Nothing,AbstractMatrix{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    state_params = _extract_state_params(lds.state_model)
    obs_params = _extract_obs_params(lds.obs_model)
    Ti = Int(tsteps)

    u_trial = _check_control(control_seq, lds.state_input_dim, Ti, "control_seq")
    v_trial = _check_obs_control(obs_control_seq, lds.obs_input_dim, Ti, lds.obs_model)

    x = Matrix{T}(undef, lds.latent_dim, Ti)
    y = Matrix{T}(undef, lds.obs_dim, Ti)
    _sample_trial!(rng, x, y, state_params, obs_params, lds.obs_model, u_trial, v_trial)
    return x, y
end

function Random.rand(
    rng::AbstractRNG,
    lds::LinearDynamicalSystem{T,S,O},
    tsteps_per_trial::AbstractVector{<:Integer};
    control_seq::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    obs_control_seq::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    state_params = _extract_state_params(lds.state_model)
    obs_params = _extract_obs_params(lds.obs_model)

    ntrials = length(tsteps_per_trial)
    x = Vector{Matrix{T}}(undef, ntrials)
    y = Vector{Matrix{T}}(undef, ntrials)
    for i in 1:ntrials
        Ti = Int(tsteps_per_trial[i])
        x[i] = Matrix{T}(undef, lds.latent_dim, Ti)
        y[i] = Matrix{T}(undef, lds.obs_dim, Ti)
    end

    u_seq = _normalize_multitrial_control(
        control_seq, lds.state_input_dim, tsteps_per_trial, T, "control_seq"
    )
    v_seq = _normalize_multitrial_obs_control(
        obs_control_seq, lds.obs_input_dim, tsteps_per_trial, T, lds.obs_model
    )

    # `MersenneTwister` (and most RNG types) is not thread-safe, so sharing
    # `rng` across `@threads` races on internal state.
    if ntrials == 1
        _sample_trial!(
            rng, x[1], y[1], state_params, obs_params, lds.obs_model, u_seq[1], v_seq[1]
        )
        return x, y
    end

    ntasks = min(ntrials, Threads.maxthreadid())
    chunksize = cld(ntrials, ntasks)
    task_rngs = [MersenneTwister(rand(rng, UInt64)) for _ in 1:ntasks]

    @sync for i in 1:ntasks
        lo = (i - 1) * chunksize + 1
        hi = min(i * chunksize, ntrials)
        lo > hi && continue
        @spawn begin
            trng = task_rngs[i]
            for trial in lo:hi
                _sample_trial!(
                    trng,
                    x[trial],
                    y[trial],
                    state_params,
                    obs_params,
                    lds.obs_model,
                    u_seq[trial],
                    v_seq[trial],
                )
            end
        end
    end

    return x, y
end

function Random.rand(lds::LinearDynamicalSystem, tsteps::Integer; kwargs...)
    return rand(Random.default_rng(), lds, tsteps; kwargs...)
end

function Random.rand(
    lds::LinearDynamicalSystem, tsteps_per_trial::AbstractVector{<:Integer}; kwargs...
)
    return rand(Random.default_rng(), lds, tsteps_per_trial; kwargs...)
end

# ============================================================================
# Control-sequence normalization helpers. The public `control_seq`/`obs_control_seq`
# kwargs accept either `nothing` (no inputs — must match a zero-column `B`/`D`)
# or per-trial matrices. Internally every sampler/smoother/M-step expects an
# `AbstractMatrix{T}` of shape `(u_dim, T_i)` (possibly `0 × T_i`), so these
# helpers canonicalize on the way in.
# ============================================================================

function _check_control(cs::Nothing, expected_dim::Int, tsteps::Int, name::AbstractString)
    expected_dim == 0 || throw(
        ArgumentError(
            "$(name)=nothing is only valid when the corresponding input matrix is " *
            "zero-column; got expected_dim=$(expected_dim). Pass a $(expected_dim)×T " *
            "matrix or shrink the input matrix.",
        ),
    )
    return zeros(eltype(Float64), 0, tsteps)
end

function _check_control(
    cs::AbstractMatrix{T}, expected_dim::Int, tsteps::Int, name::AbstractString
) where {T<:Real}
    size(cs, 1) == expected_dim || throw(
        DimensionMismatchError(
            "$(name) rows vs input-matrix cols", expected_dim, size(cs, 1)
        ),
    )
    size(cs, 2) == tsteps ||
        throw(DimensionMismatchError("$(name) tsteps", tsteps, size(cs, 2)))
    return cs
end

@inline function _check_obs_control(
    cs, expected_dim::Int, tsteps::Int, ::GaussianObservationModel
)
    return _check_control(cs, expected_dim, tsteps, "obs_control_seq")
end

@inline function _check_obs_control(
    cs::Nothing, expected_dim::Int, tsteps::Int, ::PoissonObservationModel
)
    expected_dim == 0 || error(
        "Poisson observation model does not support obs_control_seq (expected_dim must be 0)",
    )
    return zeros(Float64, 0, tsteps)
end

function _normalize_multitrial_control(
    cs::Nothing, expected_dim::Int, tsteps_per_trial, ::Type{T}, name::AbstractString
) where {T<:Real}
    expected_dim == 0 || throw(
        ArgumentError(
            "$(name)=nothing is only valid when expected_dim == 0; got $(expected_dim)"
        ),
    )
    return [zeros(T, 0, Int(Ti)) for Ti in tsteps_per_trial]
end

function _normalize_multitrial_control(
    cs::AbstractVector{<:AbstractMatrix{T}},
    expected_dim::Int,
    tsteps_per_trial,
    ::Type{T},
    name::AbstractString,
) where {T<:Real}
    length(cs) == length(tsteps_per_trial) || throw(
        DimensionMismatchError("$(name) ntrials", length(tsteps_per_trial), length(cs))
    )
    for (i, ci) in enumerate(cs)
        size(ci, 1) == expected_dim ||
            throw(DimensionMismatchError("$(name)[$i] rows", expected_dim, size(ci, 1)))
        size(ci, 2) == Int(tsteps_per_trial[i]) || throw(
            DimensionMismatchError(
                "$(name)[$i] tsteps", Int(tsteps_per_trial[i]), size(ci, 2)
            ),
        )
    end
    return cs
end

@inline function _normalize_multitrial_obs_control(
    cs, expected_dim::Int, tsteps_per_trial, ::Type{T}, ::GaussianObservationModel
) where {T<:Real}
    return _normalize_multitrial_control(
        cs, expected_dim, tsteps_per_trial, T, "obs_control_seq"
    )
end

@inline function _normalize_multitrial_obs_control(
    cs::Nothing, expected_dim::Int, tsteps_per_trial, ::Type{T}, ::PoissonObservationModel
) where {T<:Real}
    expected_dim == 0 || error(
        "Poisson observation model does not support obs_control_seq (expected_dim must be 0)",
    )
    return [zeros(T, 0, Int(Ti)) for Ti in tsteps_per_trial]
end

"""
    loglikelihood!(ws, x, lds, y)

In-place version of `loglikelihood` that uses pre-computed Cholesky factors from
`ws::SmoothWorkspace` and writes into `ws.ll_vec`. Returns the sum of log-likelihoods.
"""
function loglikelihood!(
    ws::SmoothWorkspace{T},
    x::AbstractMatrix{T},
    lds::LinearDynamicalSystem{T0,S,O},
    y::AbstractMatrix{T0},
    u::AbstractMatrix{T0},
    v::AbstractMatrix{T0},
) where {T<:Real,T0<:Real,S<:GaussianStateModel{T0},O<:GaussianObservationModel{T0}}
    tsteps = size(y, 2)

    A = lds.state_model.A
    b = lds.state_model.b
    B = lds.state_model.B
    x0 = lds.state_model.x0
    d = lds.obs_model.d
    D_obs = lds.obs_model.D

    R_U = ws.R_PD[].chol.U
    Q_U = ws.Q_PD[].chol.U
    P0_U = ws.P0_PD[].chol.U

    ll_vec = ws.ll_vec
    temp_dx = ws.temp_dx
    temp_dy = ws.temp_dy
    temp_solve_Q = ws.temp_solve_Q
    temp_solve_R = ws.temp_solve_R

    latent_dim = lds.latent_dim
    obs_dim = lds.obs_dim

    cP0 = -T(0.5) * (T(latent_dim) * log(T(2π)) + logdet(ws.P0_PD[]))
    cQ = -T(0.5) * (T(latent_dim) * log(T(2π)) + logdet(ws.Q_PD[]))
    cR = -T(0.5) * (T(obs_dim) * log(T(2π)) + logdet(ws.R_PD[]))

    for t in 1:tsteps
        ll_t = zero(T)

        # Initial state (t=1): log p(x1)
        if t == 1
            @views temp_dx .= x[:, 1] .- x0
            ldiv!(temp_solve_Q, P0_U, temp_dx)
            ll_t += cP0 - T(0.5) * sum(abs2, temp_solve_Q)
        end

        # Dynamics (t>1): log p(x_t | x_{t-1}, u_{t-1}) where mean = A x_{t-1} + b + B u_{t-1}.
        # `B*u` is a no-op when `u` has zero rows (size(B,2) == size(u,1) == 0).
        if t > 1
            @views mul!(temp_dx, A, x[:, t - 1])
            @views mul!(temp_dx, B, u[:, t - 1], one(T0), one(T0))
            @views temp_dx .= x[:, t] .- temp_dx .- b
            ldiv!(temp_solve_Q, Q_U, temp_dx)
            ll_t += cQ - T(0.5) * sum(abs2, temp_solve_Q)
        end

        # Emission: log p(y_t | x_t, v_t) where mean = C x_t + d + D v_t.
        @views mul!(temp_dy, lds.obs_model.C, x[:, t])
        @views mul!(temp_dy, D_obs, v[:, t], one(T0), one(T0))
        @views temp_dy .= y[:, t] .- temp_dy .- d
        ldiv!(temp_solve_R, R_U, temp_dy)
        ll_t += cR - T(0.5) * sum(abs2, temp_solve_R)

        ll_vec[t] = ll_t
    end

    return ll_vec
end

# Backward-compatible 4-arg overload: no inputs. Forwards to the 6-arg form
# with zero-row u/v matrices, so callers that don't use controls don't have
# to pass them.
function loglikelihood!(
    ws::SmoothWorkspace{T},
    x::AbstractMatrix{T},
    lds::LinearDynamicalSystem{T0,S,O},
    y::AbstractMatrix{T0},
) where {T<:Real,T0<:Real,S<:GaussianStateModel{T0},O<:GaussianObservationModel{T0}}
    tsteps = size(y, 2)
    u = zeros(T0, 0, tsteps)
    v = zeros(T0, 0, tsteps)
    return loglikelihood!(ws, x, lds, y, u, v)
end

"""
    loglikelihood!(ws, x, lds, y)

Compute per-timestep complete-data log-likelihood contributions for a Gaussian LDS:

- `ll[1]` includes: log p(x₁) + log p(y₁ | x₁)
- `ll[t]` for t≥2 includes: log p(x_t | x_{t-1}) + log p(y_t | x_t)

Writes into `ws.ll_vec` and returns it.

Notes:
- Normalization terms (logdet + log(2π)) are included. These are constant w.r.t. `x`,
  but **not** constant across SLDS discrete states when `Q`/`R` differ by state.
"""
function loglikelihood!(
    ll::AbstractVector{T},
    ws::SLDSSmoothWorkspace{T},
    cc::LDSConstantCache{T},
    lds::LinearDynamicalSystem{T,S,O},
    x::AbstractMatrix{T},
    y::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    @assert length(ll) == tsteps

    A = lds.state_model.A
    b = lds.state_model.b
    x0 = lds.state_model.x0
    C = lds.obs_model.C
    d = lds.obs_model.d

    Q_U = cc.Q_PD[].chol.U
    P0_U = cc.P0_PD[].chol.U
    R_U = cc.R_PD[].chol.U

    dxt = ws.dxt
    dyt = ws.dyt
    tmp = ws.tmp1  # latent_dim work vector (used as transition residual)

    for t in 1:tsteps
        ll_t = zero(T)

        # emission: cR - 0.5*||R^{-1/2}(y_t - Cx_t - d)||^2
        @views mul!(dyt, C, x[:, t])
        @views dyt .= y[:, t] .- dyt .- d
        ldiv!(dyt, R_U, dyt)
        ll_t += cc.cR - T(0.5) * sum(abs2, dyt)

        if t == 1
            # prior: cP0 - 0.5*||P0^{-1/2}(x1 - x0)||^2
            @views dxt .= x[:, 1] .- x0
            ldiv!(dxt, P0_U, dxt)
            ll_t += cc.cP0 - T(0.5) * sum(abs2, dxt)
        else
            # transition: cQ - 0.5*||Q^{-1/2}(x_t - A x_{t-1} - b)||^2
            @views mul!(tmp, A, x[:, t - 1])
            @views tmp .= x[:, t] .- tmp .- b
            ldiv!(tmp, Q_U, tmp)
            ll_t += cc.cQ - T(0.5) * sum(abs2, tmp)
        end

        ll[t] = ll_t
    end

    return ll
end

"""
    Gradient!(ws, lds, y, x)

In-place version of `Gradient` that uses pre-computed Cholesky-derived terms from
`ws::SmoothWorkspace` and writes the result into `ws.grad_buf`.
Returns `ws.grad_buf`.
"""
function Gradient!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
    u::AbstractMatrix{T},
    v::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    latent_dim, tsteps = size(x)
    A = lds.state_model.A
    b = lds.state_model.b
    B = lds.state_model.B
    x0 = lds.state_model.x0
    C = lds.obs_model.C
    d_obs = lds.obs_model.d
    D_obs = lds.obs_model.D

    C_inv_R = ws.C_inv_R
    A_inv_Q = ws.A_inv_Q
    # ws.x_t = -P0^{-1}, ws.xt_given_xt_1 = -Q^{-1}
    neg_P0_inv = ws.x_t         # = -P0^{-1}
    neg_Q_inv = ws.xt_given_xt_1  # = -Q^{-1}

    grad = ws.grad_buf
    dxt = ws.dxt
    dxt_next = ws.dxt_next
    dyt = ws.dyt
    tmp1 = ws.tmp1
    tmp2 = ws.tmp2
    tmp3 = ws.tmp3

    # Helper macros (inlined): residual `x_t - A x_{t-1} - b - B u_{t-1}` and
    # `y_t - C x_t - d - D v_t`. The `B*u` / `D*v` updates are no-ops when
    # `u` / `v` have zero rows.

    # First time step
    @views dxt .= x[:, 1] .- x0
    @views mul!(dxt_next, A, x[:, 1])
    @views mul!(dxt_next, B, u[:, 1], one(T), one(T))
    @views dxt_next .= x[:, 2] .- dxt_next .- b
    @views mul!(dyt, C, x[:, 1])
    @views mul!(dyt, D_obs, v[:, 1], one(T), one(T))
    @views dyt .= y[:, 1] .- dyt .- d_obs

    mul!(tmp1, C_inv_R, dyt)
    mul!(tmp2, A_inv_Q, dxt_next)
    mul!(tmp3, neg_P0_inv, dxt)
    grad[:, 1] .= tmp1 .+ tmp2 .+ tmp3

    # Middle steps
    @views for t in 2:(tsteps - 1)
        mul!(dxt, A, x[:, t - 1])
        mul!(dxt, B, u[:, t - 1], one(T), one(T))
        dxt .= x[:, t] .- dxt .- b

        mul!(dxt_next, A, x[:, t])
        mul!(dxt_next, B, u[:, t], one(T), one(T))
        dxt_next .= x[:, t + 1] .- dxt_next .- b

        mul!(dyt, C, x[:, t])
        mul!(dyt, D_obs, v[:, t], one(T), one(T))
        dyt .= y[:, t] .- dyt .- d_obs

        mul!(tmp1, C_inv_R, dyt)
        mul!(tmp2, A_inv_Q, dxt_next)
        mul!(tmp3, neg_Q_inv, dxt)

        grad[:, t] .= tmp1 .+ tmp3 .+ tmp2
    end

    # Last time step
    @views begin
        mul!(dxt, A, x[:, tsteps - 1])
        mul!(dxt, B, u[:, tsteps - 1], one(T), one(T))
        dxt .= x[:, tsteps] .- dxt .- b
        mul!(dyt, C, x[:, tsteps])
        mul!(dyt, D_obs, v[:, tsteps], one(T), one(T))
        dyt .= y[:, tsteps] .- dyt .- d_obs

        mul!(tmp1, C_inv_R, dyt)
        mul!(tmp3, neg_Q_inv, dxt)

        grad[:, tsteps] .= tmp1 .+ tmp3
    end

    return grad
end

# Backward-compatible no-input overload.
function Gradient!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(x, 2)
    u = zeros(T, 0, tsteps)
    v = zeros(T, 0, tsteps)
    return Gradient!(ws, lds, y, x, u, v)
end

"""
    Hessian!(sws, lds, y, x)

Fill `sws.btd.H_diag`, `H_sub`, `H_super` with the log-likelihood Hessian blocks for
the active trial (length derived from `size(y, 2)`). Returns nothing — the sparse form
is **not** built here because the Newton solver consumes blocks directly.
Workspace buffers may be sized for a longer trial; only the first `tsteps` blocks are
written, which keeps this hot path safe for ragged-length fitting.
"""
function Hessian!(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    return _fill_hessian_blocks!(sws, size(y, 2))
end

# Length-only Hessian assembly. The BT Hessian for a Gaussian LDS is
# observation-independent — its blocks depend only on `A, Q, C, R, P0`
# (already cached in `sws` by `compute_smooth_constants!`) and the trial
# length. Factored out so the equal-length multi-trial fast path can fill
# blocks without constructing a dummy `y` matrix.
function _fill_hessian_blocks!(sws::SmoothWorkspace{T}, tsteps::Int) where {T<:Real}
    btd = sws.btd

    for i in 1:(tsteps - 1)
        copyto!(btd.H_sub[i], sws.H_sub_entry)
        copyto!(btd.H_super[i], sws.H_super_entry)
    end

    btd.H_diag[1] .= sws.yt_given_xt .+ sws.xt1_given_xt .+ sws.x_t
    for i in 2:(tsteps - 1)
        btd.H_diag[i] .= sws.yt_given_xt .+ sws.xt_given_xt_1 .+ sws.xt1_given_xt
    end
    btd.H_diag[tsteps] .= sws.yt_given_xt .+ sws.xt_given_xt_1

    return nothing
end

"""
    smooth(lds, y::AbstractMatrix)

Direct smoothing for a single trial.

# Arguments
- `lds::LinearDynamicalSystem`: The model.
- `y::AbstractMatrix`: Observations (obs_dim × tsteps).

# Returns
- `x_smooth::AbstractMatrix`: Smoothed latent means (latent_dim × tsteps).
- `p_smooth::Array{T,3}`: Smoothed latent covariances (latent_dim × latent_dim × tsteps).
"""
function smooth(lds::LinearDynamicalSystem, y::AbstractMatrix{T}) where {T}
    # Type assertion narrows the union for JET; runtime no-op since dispatch on
    # `size(y, 2)::Int` already lands in the FilterSmooth-returning method.
    fs = initialize_FilterSmooth(lds, size(y, 2))::FilterSmooth{T}
    sws = SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, size(y, 2))
    smooth!(lds, fs, y, sws)
    return fs.x_smooth, fs.p_smooth
end

function smooth(
    lds::LinearDynamicalSystem, y::AbstractVector{<:AbstractMatrix{T}}
) where {T}
    tsteps_per_trial = [size(yt, 2) for yt in y]
    T_max = maximum(tsteps_per_trial)
    tfs = initialize_FilterSmooth(lds, tsteps_per_trial)::TrialFilterSmooth{T}
    sws_pool = [
        SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, T_max) for
        _ in 1:Threads.maxthreadid()
    ]
    smooth!(lds, tfs, y, sws_pool)

    N = length(y)
    xs = Vector{Matrix{T}}(undef, N)
    Ps = Vector{Array{T,3}}(undef, N)
    for n in 1:N
        fs = tfs.FilterSmooths[n]
        xs[n] = copy(fs.x_smooth)
        Ps[n] = copy(fs.p_smooth)
    end
    return xs, Ps
end

"""
    smooth!(lds, fs, y, sws::SmoothWorkspace)

Low-allocation smoothing using `SmoothWorkspace`. Uses a direct single-step
Newton solver (since the Gaussian LDS has a quadratic log-likelihood),
exploiting the block tridiagonal structure of the Hessian for efficient solving.
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    fs::FilterSmooth{T},
    y::AbstractMatrix{T},
    sws::SmoothWorkspace{T},
    u::AbstractMatrix{T},
    v::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps, D = size(y, 2), lds.latent_dim
    n_active = D * tsteps
    btd = sws.btd

    compute_smooth_constants!(sws, lds)

    # Workspace buffers may be sized for a longer trial; take active-length views.
    X0 = view(sws.X₀, 1:n_active)
    grad_vec = view(sws.grad_vec, 1:n_active)
    neg_diag_v = view(btd.neg_diag, 1:tsteps)
    neg_sub_v = view(btd.neg_sub, 1:(tsteps - 1))
    neg_super_v = view(btd.neg_super, 1:(tsteps - 1))

    # Warm-start from previous E[z].
    copyto!(X0, fs.E_z)

    x_mat = reshape(X0, D, tsteps)
    Gradient!(sws, lds, y, x_mat, u, v)
    # grad_vec = -gradient (minimize negative log-likelihood)
    for t in 1:tsteps, i in 1:D
        sws.grad_vec[(t - 1) * D + i] = -sws.grad_buf[i, t]
    end

    # Hessian is independent of `u`/`v` (linear-Gaussian model has identical
    # precision blocks regardless of input means).
    Hessian!(sws, lds, y, x_mat)
    _negate_blocks!(btd, tsteps)

    # Save x_old in fs.x_smooth before we overwrite sws.X₀ with the Newton step.
    fs.x_smooth .= x_mat

    # SPD path: smoother's negated Hessian is PSD at the MAP, and the
    # sub/super blocks are transposes of each other (Hessian is
    # symmetric). At small `latent_dim` (≤ 8) this routes to LAPACK's
    # `pbsv` which is 30-60× faster than the general block-Thomas code.
    block_tridiagonal_solve_spd!(X0, neg_sub_v, neg_diag_v, neg_super_v, grad_vec, btd)

    step_mat = reshape(X0, D, tsteps)
    fs.x_smooth .-= step_mat

    logdet_precision = block_tridiagonal_inverse_logdet!(
        fs.p_smooth, fs.p_smooth_tt1, neg_sub_v, neg_diag_v, neg_super_v, btd
    )

    fs.entropy = gaussian_entropy_from_logdet(logdet_precision, n_active)

    @views for i in 1:tsteps
        Symmetrize!(fs.p_smooth[:, :, i])
    end

    return fs
end

# Backward-compatible no-input overload (zero-row u/v).
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    fs::FilterSmooth{T},
    y::AbstractMatrix{T},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    u = zeros(T, 0, tsteps)
    v = zeros(T, 0, tsteps)
    return smooth!(lds, fs, y, sws, u, v)
end

"""
    smooth!(lds, tfs, y::AbstractVector{<:AbstractMatrix}, sws_pool)

Low-allocation multi-trial smoothing. Each task in `sws_pool` owns one workspace;
trials are partitioned across tasks via `@spawn` / `fetch` (see
https://julialang.org/blog/2023/07/PSA-dont-use-threadid/).

# Arguments
- `lds::LinearDynamicalSystem`
- `tfs::TrialFilterSmooth`: one `FilterSmooth` per trial, sized at each trial's length
- `y::AbstractVector{<:AbstractMatrix}`: one `obs_dim × T_i` matrix per trial
- `sws_pool::Vector{SmoothWorkspace{T}}`: task-local workspaces, each sized at
  `max(T_i)`
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}},
    u_seq::AbstractVector{<:AbstractMatrix{T}},
    v_seq::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    ntrials = length(y)

    if ntrials == 1
        smooth!(lds, tfs[1], y[1], sws_pool[1], u_seq[1], v_seq[1])
        return tfs
    end

    # Equal-length fast path: the BT Hessian (and its inverse) is observation-
    # independent, so the smoothed covariance is identical across trials. Run
    # the cov pass once on `sws_pool[1]`, alias each FilterSmooth's
    # `p_smooth` / `p_smooth_tt1` to the shared storage, then do gradient-and-
    # solve per trial in parallel.
    T1 = size(y[1], 2)
    all_equal = all(yt -> size(yt, 2) == T1, y)

    if all_equal
        # `_precompute_shared_cov!` populates `sws_pool[1]`'s smoothing constants
        # (`R_PD`/`Q_PD`/`P0_PD`/`C_inv_R`/`A_inv_Q`/…), the `btd.neg_*` blocks,
        # and the BT forward-sweep LU cache (`btd.LU_factors`/`LU_ipivs`/`D`).
        # Per-task back-subs and gradient evaluations read from that same
        # workspace (no mutation), so it's safe to share across `@spawn`'d tasks.
        shared_entropy = _precompute_shared_cov!(sws_pool[1], lds, T1)
        source_sws = sws_pool[1]
        for trial in 1:ntrials
            tfs[trial].p_smooth = source_sws.p_smooth_shared
            tfs[trial].p_smooth_tt1 = source_sws.p_smooth_tt1_shared
            tfs[trial].entropy = shared_entropy
        end

        # Batched mean pass: when `sws_pool[1]` was constructed with the right
        # `ntrials`, every per-trial Newton step collapses into a single
        # `(D*T) × N` matrix-RHS backsubst, doing the same total math as the
        # per-trial loop below but with BLAS-3 dispatch (matches the Kalman
        # path's batched-trial efficiency).
        if size(source_sws.batched_x_mat, 3) == ntrials && ntrials > 1
            if !source_sws.batched_data_valid[]
                _populate_batched_data!(source_sws, y, u_seq, v_seq)
            end
            _smooth_mean_only_batched!(lds, tfs, source_sws)
            return tfs
        end

        ntasks = min(ntrials, length(sws_pool))
        chunksize = cld(ntrials, ntasks)
        @sync for i in 1:ntasks
            lo = (i - 1) * chunksize + 1
            hi = min(i * chunksize, ntrials)
            lo > hi && continue
            @spawn begin
                sws = sws_pool[i]
                for trial in lo:hi
                    _smooth_mean_only!(
                        lds,
                        tfs[trial],
                        y[trial],
                        sws,
                        u_seq[trial],
                        v_seq[trial],
                        source_sws,
                    )
                end
            end
        end
        return tfs
    end

    # Variable-length fallback: per-trial smoothing (each trial gets its own
    # Hessian, cov, and mean pass on the assigned worker workspace).
    ntasks = min(ntrials, length(sws_pool))
    chunksize = cld(ntrials, ntasks)

    @sync for i in 1:ntasks
        lo = (i - 1) * chunksize + 1
        hi = min(i * chunksize, ntrials)
        lo > hi && continue
        @spawn begin
            sws = sws_pool[i]
            for trial in lo:hi
                smooth!(lds, tfs[trial], y[trial], sws, u_seq[trial], v_seq[trial])
            end
        end
    end

    return tfs
end

"""
    _precompute_shared_cov!(sws, lds, tsteps)

Fill `sws.p_smooth_shared` and `sws.p_smooth_tt1_shared` with the smoothed
covariance and lag-1 cross-covariance for a single trial of length `tsteps`
(any trial; the result is shared across all equal-length trials). Returns
the per-trial Gaussian entropy contribution `H[q(x_{1:T} | y)]`, which is
identical for every trial because it depends only on the covariances.
"""
function _precompute_shared_cov!(
    sws::SmoothWorkspace{T}, lds::LinearDynamicalSystem{T,S,O}, tsteps::Int
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    D = lds.latent_dim
    btd = sws.btd
    # Hoist `p_smooth_shared` with a concrete eltype so the `Symmetrize!`
    # call below stays in JET's typed union branch (cf. `backwards_cov!`).
    p_smooth_shared = sws.p_smooth_shared::Array{T,3}

    compute_smooth_constants!(sws, lds)
    _fill_hessian_blocks!(sws, tsteps)
    _negate_blocks!(btd, tsteps)

    neg_diag_v = view(btd.neg_diag, 1:tsteps)
    neg_sub_v = view(btd.neg_sub, 1:(tsteps - 1))
    neg_super_v = view(btd.neg_super, 1:(tsteps - 1))

    p_smooth_v = view(p_smooth_shared,:,:,(1:tsteps))
    p_smooth_tt1_v = view((sws.p_smooth_tt1_shared::Array{T,3}),:,:,(1:tsteps))

    logdet_precision = block_tridiagonal_inverse_logdet!(
        p_smooth_v, p_smooth_tt1_v, neg_sub_v, neg_diag_v, neg_super_v, btd
    )

    @views for i in 1:tsteps
        Symmetrize!(p_smooth_shared[:, :, i])
    end

    return gaussian_entropy_from_logdet(logdet_precision, D * tsteps)
end

"""
    _smooth_mean_only!(lds, fs, y, sws, u, v, source_sws)

Per-trial Newton step that **assumes**:

- `fs.p_smooth` / `fs.p_smooth_tt1` are already filled (by
  `_precompute_shared_cov!`),
- `source_sws.btd` contains the **forward-sweep LU cache**, the modified
  upper diagonals `D[i+1]`, and the negated Hessian sub-diagonal blocks
  (also produced by the same `_precompute_shared_cov!` call), and
- `source_sws` itself holds the Cholesky factors and derived gradient
  constants — `compute_smooth_constants!` is **not** called per trial.

Per-task workspaces copy the constants from `source_sws` (cheap fixed-size
`copyto!`s) instead of redoing the Cholesky factorizations. When `sws ===
source_sws` (the task running on the designated workspace), even the copy
is skipped.

Computes the gradient (per-trial), then runs `block_tridiagonal_backsubst!`
against the shared LU cache. No `lu!` and no Cholesky calls happen here —
those are amortized across all equal-length trials in a single E-step.
"""
function _smooth_mean_only!(
    lds::LinearDynamicalSystem{T,S,O},
    fs::FilterSmooth{T},
    y::AbstractMatrix{T},
    sws::SmoothWorkspace{T},
    u::AbstractMatrix{T},
    v::AbstractMatrix{T},
    source_sws::SmoothWorkspace{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps, D = size(y, 2), lds.latent_dim
    n_active = D * tsteps

    # Cholesky factors and derived gradient terms were filled on `source_sws`
    # by `_precompute_shared_cov!` already; just mirror them into the local
    # task workspace. No-op when `sws === source_sws`.
    if sws !== source_sws
        _copy_smooth_constants!(sws, source_sws)
    end

    shared_btd = source_sws.btd
    X0 = view(sws.X₀, 1:n_active)
    grad_vec = view(sws.grad_vec, 1:n_active)
    neg_sub_v = view(shared_btd.neg_sub, 1:(tsteps - 1))

    copyto!(X0, fs.E_z)

    x_mat = reshape(X0, D, tsteps)
    Gradient!(sws, lds, y, x_mat, u, v)
    for t in 1:tsteps, i in 1:D
        sws.grad_vec[(t - 1) * D + i] = -sws.grad_buf[i, t]
    end

    fs.x_smooth .= x_mat
    block_tridiagonal_backsubst!(X0, neg_sub_v, grad_vec, shared_btd, tsteps)
    step_mat = reshape(X0, D, tsteps)
    fs.x_smooth .-= step_mat

    return fs
end

"""
    Gradient_batched!(ws, lds, y_batched, x_batched, u_batched, v_batched)

Batched form of `Gradient!`: every `mul!` is promoted from BLAS-2
(`bs × bs × bs`) to BLAS-3 (`bs × bs × bs × N`) by stacking the trial axis as
the trailing matrix dimension. The shared-cov fast path only ever needs
gradient evaluation at the *current iterate* across all trials, so the work
is structurally identical to N independent per-trial gradients — but BLAS
dispatch overhead is paid once instead of N times.

Writes the result into `ws.batched_grad_buf` (shape `(D, T, N)`); the affine
bias subtractions (`-b`, `-d_obs`, `-x0`) broadcast across the trial axis.
"""
function Gradient_batched!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3},
    x::AbstractArray{T,3},
    u::AbstractArray{T,3},
    v::AbstractArray{T,3},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(x, 2)
    A = lds.state_model.A
    b = lds.state_model.b
    B = lds.state_model.B
    x0 = lds.state_model.x0
    C = lds.obs_model.C
    d_obs = lds.obs_model.d
    D_obs = lds.obs_model.D

    C_inv_R = ws.C_inv_R
    A_inv_Q = ws.A_inv_Q
    neg_P0_inv = ws.x_t            # = -P0^{-1}
    neg_Q_inv = ws.xt_given_xt_1   # = -Q^{-1}

    grad = ws.batched_grad_buf
    dxt = ws.batched_dxt
    dxt_next = ws.batched_dxt_next
    dyt = ws.batched_dyt
    tmp1 = ws.batched_tmp1
    tmp2 = ws.batched_tmp2
    tmp3 = ws.batched_tmp3

    # First time step
    @views begin
        dxt .= x[:, 1, :] .- x0
        mul!(dxt_next, A, x[:, 1, :])
        mul!(dxt_next, B, u[:, 1, :], one(T), one(T))
        dxt_next .= x[:, 2, :] .- dxt_next .- b
        mul!(dyt, C, x[:, 1, :])
        mul!(dyt, D_obs, v[:, 1, :], one(T), one(T))
        dyt .= y[:, 1, :] .- dyt .- d_obs
    end

    mul!(tmp1, C_inv_R, dyt)
    mul!(tmp2, A_inv_Q, dxt_next)
    mul!(tmp3, neg_P0_inv, dxt)
    @views grad[:, 1, :] .= tmp1 .+ tmp2 .+ tmp3

    # Middle steps
    @views for t in 2:(tsteps - 1)
        mul!(dxt, A, x[:, t - 1, :])
        mul!(dxt, B, u[:, t - 1, :], one(T), one(T))
        dxt .= x[:, t, :] .- dxt .- b

        mul!(dxt_next, A, x[:, t, :])
        mul!(dxt_next, B, u[:, t, :], one(T), one(T))
        dxt_next .= x[:, t + 1, :] .- dxt_next .- b

        mul!(dyt, C, x[:, t, :])
        mul!(dyt, D_obs, v[:, t, :], one(T), one(T))
        dyt .= y[:, t, :] .- dyt .- d_obs

        mul!(tmp1, C_inv_R, dyt)
        mul!(tmp2, A_inv_Q, dxt_next)
        mul!(tmp3, neg_Q_inv, dxt)

        grad[:, t, :] .= tmp1 .+ tmp3 .+ tmp2
    end

    # Last time step
    @views begin
        mul!(dxt, A, x[:, tsteps - 1, :])
        mul!(dxt, B, u[:, tsteps - 1, :], one(T), one(T))
        dxt .= x[:, tsteps, :] .- dxt .- b
        mul!(dyt, C, x[:, tsteps, :])
        mul!(dyt, D_obs, v[:, tsteps, :], one(T), one(T))
        dyt .= y[:, tsteps, :] .- dyt .- d_obs

        mul!(tmp1, C_inv_R, dyt)
        mul!(tmp3, neg_Q_inv, dxt)

        grad[:, tsteps, :] .= tmp1 .+ tmp3
    end

    return grad
end

"""
    _populate_batched_data!(sws, y, u, v)

Stack the per-trial `y`/`u`/`v` `Vector{Matrix}` inputs into the contiguous
`(p, T, N)` / `(u_dim, T, N)` / `(d_dim, T, N)` tensors used by the batched
mean pass. Called once per fit (data is constant across EM iterations).
"""
function _populate_batched_data!(
    sws::SmoothWorkspace{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    u::AbstractVector{<:AbstractMatrix{T}},
    v::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real}
    @views for trial in eachindex(y)
        sws.batched_y[:, :, trial] .= y[trial]
    end
    if size(sws.batched_u, 1) > 0
        @views for trial in eachindex(u)
            sws.batched_u[:, :, trial] .= u[trial]
        end
    end
    if size(sws.batched_v, 1) > 0
        @views for trial in eachindex(v)
            sws.batched_v[:, :, trial] .= v[trial]
        end
    end
    sws.batched_data_valid[] = true
    return sws
end

"""
    _smooth_mean_only_batched!(lds, tfs, sws)

Batched form of `_smooth_mean_only!`: runs one Newton step for *all trials at
once* by stacking the per-trial iterate / gradient / RHS into `(D, T, N)`
tensors and performing a single BLAS-3 backsubst.

Assumes `_precompute_shared_cov!` has already populated `sws.btd`'s Cholesky
cache and `sws.batched_data_valid[] == true` (data was stacked at fit entry).
"""
function _smooth_mean_only_batched!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    ntrials = length(tfs)
    D = lds.latent_dim
    tsteps = size(tfs[1].x_smooth, 2)

    # Stage previous-iter smoothed means into the batched iterate buffer.
    @views for trial in 1:ntrials
        sws.batched_x_mat[:, :, trial] .= tfs[trial].E_z
    end

    Gradient_batched!(
        sws, lds, sws.batched_y, sws.batched_x_mat, sws.batched_u, sws.batched_v
    )

    # Pack negated gradient into the (D*T, N) matrix RHS layout.
    n_active = D * tsteps
    grad_flat = reshape(sws.batched_grad_buf, n_active, ntrials)
    x_flat = reshape(sws.batched_x_mat, n_active, ntrials)
    @. grad_flat = -grad_flat

    neg_sub_v = view(sws.btd.neg_sub, 1:(tsteps - 1))
    block_tridiagonal_backsubst!(x_flat, neg_sub_v, grad_flat, sws.btd, tsteps)

    # x_flat now holds the Newton step. Update each tfs[trial].x_smooth.
    @views for trial in 1:ntrials
        tfs[trial].x_smooth .= tfs[trial].E_z .- sws.batched_x_mat[:, :, trial]
    end

    return tfs
end

# Backward-compatible no-input overload.
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws_pool::Vector{SmoothWorkspace{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    u_seq = [zeros(T, 0, size(yt, 2)) for yt in y]
    v_seq = [zeros(T, 0, size(yt, 2)) for yt in y]
    return smooth!(lds, tfs, y, sws_pool, u_seq, v_seq)
end

"""
    Q_state!(ws, lds, E_z, E_zz, E_zz_prev, u)

State Q-term for an LDS with affine dynamics `x_t ~ N(A x_{t-1} + b + B u_{t-1}, Q)`.
In-place version of `Q_state` that uses pre-allocated buffers from `SmoothWorkspace`.
Uses cached Cholesky factors from `compute_smooth_constants!`. `u` is the per-trial
dynamics-control matrix `(u_dim, T_i)`; pass a `0×T_i` matrix when no inputs.
"""
function Q_state!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
    u::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    tstep = size(E_z, 2)
    D = lds.latent_dim
    u_dim = size(u, 1)
    A = lds.state_model.A
    b = lds.state_model.b
    B = lds.state_model.B
    x0 = lds.state_model.x0

    # Use cached Cholesky factors (already computed by compute_smooth_constants!)
    Q_U = ws.Q_PD[].chol.U
    P0_U = ws.P0_PD[].chol.U

    log_det_Q = logdet(ws.Q_PD[])
    log_det_P0 = logdet(ws.P0_PD[])

    temp = ws.elbo_temp
    sum_E_zz = ws.elbo_sum_E_zz
    sum_E_zzm1 = ws.elbo_sum_E_zzm1
    sum_E_cross = ws.elbo_sum_E_cross
    sum_mu_t = ws.elbo_sum_mu_t
    sum_mu_tm1 = ws.elbo_sum_mu_tm1
    temp2 = ws.elbo_temp2

    # Initial-state part: temp = E_zz[:,:,1] - E_z[:,1]*x0' - x0*E_z[:,1]' + x0*x0'
    fill!(temp, zero(T))
    @views begin
        temp .+= E_zz[:, :, 1]
        mul!(temp, E_z[:, 1:1], x0', -one(T), one(T))
        mul!(temp, x0, E_z[:, 1:1]', -one(T), one(T))
        mul!(temp, x0, x0', one(T), one(T))
    end
    ldiv!(P0_U', temp)
    ldiv!(P0_U, temp)
    Q_val = T(-0.5) * (log_det_P0 + tr(temp))

    # Transition part: accumulate sums over t=2:tstep
    fill!(sum_E_zz, zero(T))
    fill!(sum_E_zzm1, zero(T))
    fill!(sum_E_cross, zero(T))
    fill!(sum_mu_t, zero(T))
    fill!(sum_mu_tm1, zero(T))

    # Input-specific accumulators (only allocated when u_dim > 0). Allocating
    # 0-element arrays here would still cost an `Array` struct each call,
    # which adds up to thousands of trivial allocations across a fit.
    has_input = u_dim > 0
    sum_u = has_input ? zeros(T, u_dim) : Vector{T}()
    sum_mu_t_u = has_input ? zeros(T, D, u_dim) : Matrix{T}(undef, 0, 0)
    sum_mu_tm1_u = has_input ? zeros(T, D, u_dim) : Matrix{T}(undef, 0, 0)
    sum_uu = has_input ? zeros(T, u_dim, u_dim) : Matrix{T}(undef, 0, 0)

    @views for t in 2:tstep
        sum_E_zz .+= E_zz[:, :, t]
        sum_E_zzm1 .+= E_zz[:, :, t - 1]
        sum_E_cross .+= E_zz_prev[:, :, t]
        sum_mu_t .+= E_z[:, t]
        sum_mu_tm1 .+= E_z[:, t - 1]

        if has_input
            u_tm1 = u[:, t - 1]
            sum_u .+= u_tm1
            BLAS.ger!(one(T), E_z[:, t], u_tm1, sum_mu_t_u)
            BLAS.ger!(one(T), E_z[:, t - 1], u_tm1, sum_mu_tm1_u)
            BLAS.ger!(one(T), u_tm1, u_tm1, sum_uu)
        end
    end

    # No-input batched terms:
    #   temp = sum_E_zz - A·sum_E_cross' - sum_E_cross·A' + A·sum_E_zzm1·A'
    copyto!(temp, sum_E_zz)
    mul!(temp, A, sum_E_cross', -one(T), one(T))
    mul!(temp, sum_E_cross, A', -one(T), one(T))
    mul!(temp2, A, sum_E_zzm1)
    mul!(temp, temp2, A', one(T), one(T))

    # Bias terms (b alone):
    mul!(temp, sum_mu_t, b', -one(T), one(T))
    mul!(temp, b, sum_mu_t', -one(T), one(T))
    mul!(ws.tmp1, A, sum_mu_tm1)
    mul!(temp, ws.tmp1, b', one(T), one(T))
    mul!(temp, b, ws.tmp1', one(T), one(T))
    mul!(temp, b, b', T(tstep - 1), one(T))

    # Input cross terms (`Bu_{t-1} := B u_{t-1}`). All terms here are
    # contributions to `Σ_t E[(x_t - A x_{t-1} - b - B u_{t-1})(...)']` that
    # involve at least one `B u_{t-1}` factor.
    if u_dim > 0
        # -= sum_mu_t_u · B'  and  -= B · sum_mu_t_u'
        mul!(temp, sum_mu_t_u, B', -one(T), one(T))
        mul!(temp, B, sum_mu_t_u', -one(T), one(T))
        # += (A · sum_mu_tm1_u) · B'  and  += B · (A · sum_mu_tm1_u)'
        # Intermediate has shape (D × u_dim); no fixed-size workspace buffer.
        A_sumXU = A * sum_mu_tm1_u
        mul!(temp, A_sumXU, B', one(T), one(T))
        mul!(temp, B, A_sumXU', one(T), one(T))
        # += b · (B · sum_u)'  and  += (B · sum_u) · b'
        B_sumu = B * sum_u  # D-vector
        mul!(temp, reshape(b, :, 1), reshape(B_sumu, 1, :), one(T), one(T))
        mul!(temp, reshape(B_sumu, :, 1), reshape(b, 1, :), one(T), one(T))
        # += B · sum_uu · B'
        B_sumuu = B * sum_uu  # D × u_dim
        mul!(temp, B_sumuu, B', one(T), one(T))
    end

    # Solve Q \ temp
    ldiv!(Q_U', temp)
    ldiv!(Q_U, temp)
    Q_val += T(-0.5) * ((tstep - 1) * log_det_Q + tr(temp))

    return Q_val
end

# Backward-compatible no-input overload.
function Q_state!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    u = zeros(T, 0, size(E_z, 2))
    return Q_state!(ws, lds, E_z, E_zz, E_zz_prev, u)
end

"""
    Q_obs!(C, d, E_z, E_zz, y)

Single time-step observation component of the Q-function for
y_t ~ 𝓝(C x_t + d, R), before applying R^{-1} and constants.
"""
function Q_obs!(
    result::AbstractMatrix{T},
    C::AbstractMatrix{T},
    d::AbstractVector{T},
    E_z::AbstractVector{T},
    E_zz::AbstractMatrix{T},
    y::AbstractVector{T},
    buffers,
) where {T<:Real}

    # Unpack buffers
    ytil, sum_yy, sum_yz, work1, work2 = buffers

    # Residualize: ytil = y - d (pre-allocated buffer)
    ytil .= y .- d

    # All operations use pre-allocated buffers
    mul!(sum_yy, ytil, ytil')

    # Efficient outer product: sum_yz = ytil * E_z'
    fill!(sum_yz, zero(T))
    BLAS.ger!(one(T), ytil, E_z, sum_yz)

    # Build result using buffers
    copyto!(result, sum_yy)
    mul!(result, C, sum_yz', -one(T), one(T))   # result -= C * sum_yz'
    mul!(work1, sum_yz, C')                      # work1 = sum_yz * C'  
    result .-= work1                             # result -= work1
    mul!(work2, E_zz, C')                        # work2 = E_zz * C'
    mul!(result, C, work2, one(T), one(T))       # result += C * work2

    return result
end

"""
    Q_obs!(ws, lds, E_z, E_zz, y, v)

Full observation Q-term for Gaussian LDS over all time steps with affine
observation `y_t ~ N(C x_t + d + D v_t, R)`. `v` is the per-trial obs-control
matrix `(d_dim, T_i)`; pass a `0×T_i` matrix when no obs inputs.
"""
function Q_obs!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    y::AbstractMatrix{T},
    v::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    obs_dim = lds.obs_dim
    tsteps = size(y, 2)
    C = lds.obs_model.C
    d = lds.obs_model.d
    D_obs = lds.obs_model.D
    d_dim = size(v, 1)

    R_U = ws.R_PD[].chol.U
    log_det_R = logdet(ws.R_PD[])
    const_term = obs_dim * log(T(2π))

    temp = ws.elbo_obs_temp
    work_matrix = ws.elbo_obs_work
    ytil = ws.elbo_ytil
    sum_yy = ws.elbo_sum_yy
    sum_yz = ws.elbo_sum_yz
    work1 = ws.elbo_obs_work1
    work2 = ws.elbo_obs_work2

    fill!(temp, zero(T))

    @views for t in axes(y, 2)
        # Residualize: ytil = y[:,t] - d - D · v[:,t]
        ytil .= y[:, t] .- d
        if d_dim > 0
            mul!(ytil, D_obs, v[:, t], -one(T), one(T))
        end

        mul!(sum_yy, ytil, ytil')

        fill!(sum_yz, zero(T))
        BLAS.ger!(one(T), ytil, E_z[:, t], sum_yz)

        # work_matrix = sum_yy - C·sum_yz' - sum_yz·C' + C·E_zz·C'
        copyto!(work_matrix, sum_yy)
        mul!(work_matrix, C, sum_yz', -one(T), one(T))
        mul!(work1, sum_yz, C')
        work_matrix .-= work1
        mul!(work2, E_zz[:, :, t], C')
        mul!(work_matrix, C, work2, one(T), one(T))

        temp .+= work_matrix
    end

    ldiv!(R_U', temp)
    ldiv!(R_U, temp)
    return T(-0.5) * (tsteps * (const_term + log_det_R) + tr(temp))
end

# Backward-compatible no-input overload.
function Q_obs!(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    y::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    v = zeros(T, 0, size(y, 2))
    return Q_obs!(ws, lds, E_z, E_zz, y, v)
end

"""
    sufficient_statistics(x_smooth, p_smooth, p_smooth_t1)

Compute sufficient statistics for the EM algorithm in a Linear Dynamical System.

# Note
- The function computes the expected values for all trials.
- For single-trial data, use inputs with ntrials = 1.
"""
function sufficient_statistics!(fs::FilterSmooth{T}) where {T<:Real}
    latent_dim, tsteps = size(fs.x_smooth)

    # `initialize_FilterSmooth` leaves these as `(0, 0, 0)` stubs — the TD
    # aggregator never reads them. Materialize on demand for legacy callers.
    if size(fs.E_zz, 1) != latent_dim || size(fs.E_zz, 3) != tsteps
        fs.E_zz = zeros(T, latent_dim, latent_dim, tsteps)
        fs.E_zz_prev = zeros(T, latent_dim, latent_dim, tsteps)
    end

    # E_z is just a copy of x_smooth
    fs.E_z .= fs.x_smooth

    # Compute E_zz and E_zz_prev in-place
    @views for t in 1:tsteps
        # E_zz[:,:,t] = p_smooth[:,:,t] + x_smooth[:,t] * x_smooth[:,t]'
        mul!(fs.E_zz[:, :, t], fs.x_smooth[:, t:t], fs.x_smooth[:, t:t]')
        fs.E_zz[:, :, t] .+= fs.p_smooth[:, :, t]

        if t > 1
            # E_zz_prev[:,:,t] = p_smooth_tt1[:,:,t] + x_smooth[:,t] * x_smooth[:,t-1]'
            mul!(
                fs.E_zz_prev[:, :, t], fs.x_smooth[:, t:t], fs.x_smooth[:, (t - 1):(t - 1)]'
            )
            fs.E_zz_prev[:, :, t] .+= fs.p_smooth_tt1[:, :, t]
        else
            fs.E_zz_prev[:, :, 1] .= 0
        end
    end
end

function sufficient_statistics!(tfs::TrialFilterSmooth{T}) where {T<:Real}
    ntrials = length(tfs.FilterSmooths)

    if ntrials == 1
        sufficient_statistics!(tfs[1])
    else
        @threads for i in 1:ntrials
            sufficient_statistics!(tfs[i])
        end
    end
end

"""
    _td_init_const_blocks!(sws, lds, tsteps_per_trial, y, u_seq, v_seq)

Fill the data-only constant blocks of the sufficient-statistics buffers
(`td_obs_yy_const`, `td_obs_xy_const`, `td_obs_xx_const`, `td_dyn_xx_const`)
once at fit entry. These are observation-independent: they depend only on
the raw inputs, not on smoother output.
"""
function _td_init_const_blocks!(
    sws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    tsteps_per_trial::AbstractVector{Int},
    y::AbstractVector{<:AbstractMatrix{T}},
    u_seq::AbstractVector{<:AbstractMatrix{T}},
    v_seq::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    D = lds.latent_dim
    p = lds.obs_dim
    u_dim = lds.state_input_dim
    d_dim = lds.obs_input_dim
    ntrials = length(y)
    dyn_reg_dim = D + 1 + u_dim
    obs_reg_dim = D + 1 + d_dim
    total_obs = sum(tsteps_per_trial)
    total_dyn = total_obs - ntrials

    # Hoist workspace fields with concrete eltype to clear JET union-split
    # false positives on the syrk!/copytri! callsites below.
    td_obs_yy_const = sws.td_obs_yy_const::Matrix{T}
    td_obs_xy_const = sws.td_obs_xy_const::Matrix{T}
    td_obs_xx_const = sws.td_obs_xx_const::Matrix{T}
    td_dyn_xx_const = sws.td_dyn_xx_const::Matrix{T}

    fill!(td_obs_yy_const, zero(T))
    fill!(td_obs_xy_const, zero(T))
    fill!(td_obs_xx_const, zero(T))
    fill!(td_dyn_xx_const, zero(T))

    # obs_yy = Σ_n Σ_t y_t y_t'
    for trial in 1:ntrials
        BLAS.syrk!('U', 'N', one(T), y[trial], one(T), td_obs_yy_const)
    end
    LinearAlgebra.copytri!(td_obs_yy_const, 'U')

    # obs_xy[D+1, :] = Σ_n Σ_t y_t   (bias row sum)
    for trial in 1:ntrials
        y_trial = y[trial]
        for t in axes(y_trial, 2), j in 1:p
            td_obs_xy_const[D + 1, j] += y_trial[j, t]
        end
    end

    # obs_xx bias-bias entry
    td_obs_xx_const[D + 1, D + 1] = T(total_obs)

    if d_dim > 0
        for trial in 1:ntrials
            v_t = v_seq[trial]
            y_t = y[trial]
            # obs_xy[D+2:end, :] += Σ_t v_t y_t'
            mul!(view(td_obs_xy_const, (D + 2):obs_reg_dim, :), v_t, y_t', one(T), one(T))
            # obs_xx[D+2:end, D+2:end] += Σ_t v_t v_t'  (upper tri)
            BLAS.syrk!(
                'U',
                'N',
                one(T),
                v_t,
                one(T),
                view(td_obs_xx_const, (D + 2):obs_reg_dim, (D + 2):obs_reg_dim),
            )
        end
        LinearAlgebra.copytri!(
            view(td_obs_xx_const, (D + 2):obs_reg_dim, (D + 2):obs_reg_dim), 'U'
        )
        # obs_xx[D+1, D+2:end] / [D+2:end, D+1] = Σ_t v_t   (bias × v cross)
        for trial in 1:ntrials
            v_trial = v_seq[trial]
            for t in axes(v_trial, 2), k in 1:d_dim
                td_obs_xx_const[D + 1, D + 1 + k] += v_trial[k, t]
            end
        end
        @views td_obs_xx_const[(D + 2):obs_reg_dim, D + 1] .= td_obs_xx_const[
            D + 1, (D + 2):obs_reg_dim
        ]
    end

    td_dyn_xx_const[D + 1, D + 1] = T(total_dyn)

    if u_dim > 0
        for trial in 1:ntrials
            u_trial = u_seq[trial]
            T_n = size(u_trial, 2)
            # Convention (matches existing update_A_b!): we use u[:, 1:T_n-1]
            # as `u_{t-1}` for t = 2:T_n.
            u_used = view(u_trial, :, 1:(T_n - 1))
            BLAS.syrk!(
                'U',
                'N',
                one(T),
                u_used,
                one(T),
                view(td_dyn_xx_const, (D + 2):dyn_reg_dim, (D + 2):dyn_reg_dim),
            )
        end
        LinearAlgebra.copytri!(
            view(td_dyn_xx_const, (D + 2):dyn_reg_dim, (D + 2):dyn_reg_dim), 'U'
        )
        # bias × u cross
        for trial in 1:ntrials
            u_trial = u_seq[trial]
            T_n = size(u_trial, 2)
            for t in 1:(T_n - 1), k in 1:u_dim
                td_dyn_xx_const[D + 1, D + 1 + k] += u_trial[k, t]
            end
        end
        @views td_dyn_xx_const[(D + 2):dyn_reg_dim, D + 1] .= td_dyn_xx_const[
            D + 1, (D + 2):dyn_reg_dim
        ]
    end

    return nothing
end

"""
    _initialize_td_sufficient_statistics(T, lds, tsteps_per_trial)

Allocate a `SufficientStatistics{T}` with the right shapes for the TD path.
The PDMat refs are wrapped around identity placeholders; the aggregator
overwrites them each E-step.
"""
function _initialize_td_sufficient_statistics(
    ::Type{T}, lds::LinearDynamicalSystem{T,S,O}, tsteps_per_trial::AbstractVector{Int}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    D = lds.latent_dim
    p = lds.obs_dim
    u_dim = lds.state_input_dim
    d_dim = lds.obs_input_dim
    ntrials = length(tsteps_per_trial)
    dyn_reg_dim = D + 1 + u_dim
    obs_reg_dim = D + 1 + d_dim
    total_obs = sum(tsteps_per_trial)
    total_dyn = total_obs - ntrials

    PD_init(d) = PDMat(Matrix{T}(I, d, d))

    return SufficientStatistics{T}(
        T(ntrials),
        Ref(PDMat(fill(T(ntrials), 1, 1))),       # init_xx (1×1 = N)
        zeros(T, 1, D),                            # init_xy
        Ref(PD_init(D)),                           # init_yy
        T(total_dyn),
        Ref(PD_init(dyn_reg_dim)),                 # dyn_xx
        zeros(T, dyn_reg_dim, D),                  # dyn_xy
        Ref(PD_init(D)),                           # dyn_yy
        T(total_obs),
        Ref(PD_init(obs_reg_dim)),                 # obs_xx
        zeros(T, obs_reg_dim, p),                  # obs_xy
        Ref(PD_init(p)),                           # obs_yy
    )
end

"""
    _aggregate_td_suff_stats!(suf, tfs, lds, u_seq, v_seq, sws)

Aggregate per-trial smoother output (`x_smooth`, `p_smooth`, `p_smooth_tt1`)
into `suf` using per-trial GEMM/SYRK. Replaces the per-timestep, per-trial
loops formerly done inside `Q_state!`, `Q_obs!`, and the `update_*!`
functions.

Uses the cov-cache fast-path shortcut when all trials' `p_smooth` arrays
are aliased to the same shared storage (equal-length multi-trial fit).
"""
function _aggregate_td_suff_stats!(
    suf::SufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    lds::LinearDynamicalSystem{T,S,O},
    u_seq::AbstractVector{<:AbstractMatrix{T}},
    v_seq::AbstractVector{<:AbstractMatrix{T}},
    y::AbstractVector{<:AbstractMatrix{T}},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    D = lds.latent_dim
    p = lds.obs_dim
    u_dim = lds.state_input_dim
    d_dim = lds.obs_input_dim
    ntrials = length(tfs)
    dyn_reg_dim = D + 1 + u_dim
    obs_reg_dim = D + 1 + d_dim

    # Hoist workspace fields with concrete eltype for JET (cf. backwards_cov!).
    Szz_Ab = sws.Szz_Ab::Matrix{T}
    Szz_Cd = sws.Szz_Cd::Matrix{T}
    Q_sum = sws.Q_sum::Matrix{T}
    R_sum = sws.R_sum::Matrix{T}
    S0_sum = sws.S0_sum::Matrix{T}
    td_init_xy = sws.td_init_xy::Matrix{T}
    td_dyn_xy = sws.td_dyn_xy::Matrix{T}
    td_obs_xy = sws.td_obs_xy::Matrix{T}
    td_obs_xy_const = sws.td_obs_xy_const::Matrix{T}
    td_obs_xx_const = sws.td_obs_xx_const::Matrix{T}
    td_dyn_xx_const = sws.td_dyn_xx_const::Matrix{T}
    td_obs_yy_const = sws.td_obs_yy_const::Matrix{T}
    sum_cov_prev = sws.td_sum_smooth_cov_prev::Matrix{T}
    sum_cov_next = sws.td_sum_smooth_cov_next::Matrix{T}
    sum_cov_all = sws.td_sum_smooth_cov_all::Matrix{T}
    sum_xcov = sws.td_sum_smooth_xcov::Matrix{T}

    # Detect cov-cache fast path (equal-length trials share p_smooth storage).
    cov_cache = ntrials > 1 && tfs[1].p_smooth === tfs[2].p_smooth

    fill!(td_init_xy, zero(T))
    fill!(S0_sum, zero(T))                  # init_yy
    fill!(td_dyn_xy, zero(T))               # dyn_xy
    fill!(Q_sum, zero(T))                   # dyn_yy
    fill!(td_obs_xy, zero(T))               # obs_xy (will copy const next)
    fill!(sum_cov_prev, zero(T))
    fill!(sum_cov_next, zero(T))
    fill!(sum_cov_all, zero(T))
    fill!(sum_xcov, zero(T))

    # Seed xx/yy/xy buffers with the precomputed data-only constants.
    copyto!(Szz_Ab, td_dyn_xx_const)
    copyto!(Szz_Cd, td_obs_xx_const)
    copyto!(R_sum, td_obs_yy_const)
    copyto!(td_obs_xy, td_obs_xy_const)

    if cov_cache
        fs1 = tfs[1]
        T_shared = size(fs1.x_smooth, 2)
        p_smooth1 = fs1.p_smooth::Array{T,3}
        p_smooth_tt11 = fs1.p_smooth_tt1::Array{T,3}
        @views for t in 1:T_shared
            sum_cov_all .+= p_smooth1[:, :, t]
            if t < T_shared
                sum_cov_prev .+= p_smooth1[:, :, t]
            end
            if t > 1
                sum_cov_next .+= p_smooth1[:, :, t]
                sum_xcov .+= p_smooth_tt11[:, :, t]
            end
        end
        # Scale to total across N trials.
        N_T = T(ntrials)
        sum_cov_all .*= N_T
        sum_cov_prev .*= N_T
        sum_cov_next .*= N_T
        sum_xcov .*= N_T
    end

    for trial in 1:ntrials
        fs = tfs[trial]
        x = fs.x_smooth::Matrix{T}
        p_smooth = fs.p_smooth::Array{T,3}
        p_smooth_tt1 = fs.p_smooth_tt1::Array{T,3}
        T_n = size(x, 2)

        # Per-trial cov sums when not on the cov-cache fast path.
        if !cov_cache
            @views for t in 1:T_n
                sum_cov_all .+= p_smooth[:, :, t]
                if t < T_n
                    sum_cov_prev .+= p_smooth[:, :, t]
                end
                if t > 1
                    sum_cov_next .+= p_smooth[:, :, t]
                    sum_xcov .+= p_smooth_tt1[:, :, t]
                end
            end
        end

        # init_xy[1, :] += x[:, 1];   init_yy += x[:, 1] x[:, 1]'
        for j in 1:D
            td_init_xy[1, j] += x[j, 1]
        end
        @views BLAS.ger!(one(T), x[:, 1], x[:, 1], S0_sum)

        x_prev = view(x, :, 1:(T_n - 1))
        x_next = view(x, :, 2:T_n)

        # dyn_xx[1:D, 1:D] += x_prev x_prev'   (upper triangle via syrk)
        BLAS.syrk!('U', 'N', one(T), x_prev, one(T), view(Szz_Ab, 1:D, 1:D))
        # obs_xx[1:D, 1:D] += x x'             (upper triangle via syrk)
        BLAS.syrk!('U', 'N', one(T), x, one(T), view(Szz_Cd, 1:D, 1:D))

        # dyn_xx[1:D, D+1] += Σ x_prev   (column-sum into upper-only bias col)
        for t in 1:(T_n - 1), i in 1:D
            Szz_Ab[i, D + 1] += x_prev[i, t]
        end
        # obs_xx[1:D, D+1] += Σ x
        for t in 1:T_n, i in 1:D
            Szz_Cd[i, D + 1] += x[i, t]
        end

        # dyn_xy[1:D, :] += x_prev x_next'
        mul!(view(td_dyn_xy, 1:D, :), x_prev, x_next', one(T), one(T))
        # dyn_xy[D+1, :] += Σ x_next
        for t in 1:(T_n - 1), j in 1:D
            td_dyn_xy[D + 1, j] += x_next[j, t]
        end

        # dyn_yy += x_next x_next'  (upper tri)
        BLAS.syrk!('U', 'N', one(T), x_next, one(T), Q_sum)

        # obs_xy[1:D, :] += x y'
        mul!(view(td_obs_xy, 1:D, :), x, y[trial]', one(T), one(T))

        # Input-side cross blocks (x × u, u × x).
        if u_dim > 0
            u_trial = u_seq[trial]
            u_prev = view(u_trial, :, 1:(T_n - 1))
            mul!(view(Szz_Ab, 1:D, (D + 2):dyn_reg_dim), x_prev, u_prev', one(T), one(T))
            mul!(view(td_dyn_xy, (D + 2):dyn_reg_dim, :), u_prev, x_next', one(T), one(T))
        end
        if d_dim > 0
            v_trial = v_seq[trial]
            mul!(view(Szz_Cd, 1:D, (D + 2):obs_reg_dim), x, v_trial', one(T), one(T))
        end
    end

    # init_yy: need Σ_n P_smooth[n,:,:,1].
    if cov_cache
        @views S0_sum .+= T(ntrials) .* (tfs[1].p_smooth::Array{T,3})[:, :, 1]
    else
        @views for trial in 1:ntrials
            S0_sum .+= (tfs[trial].p_smooth::Array{T,3})[:, :, 1]
        end
    end
    @views Szz_Ab[1:D, 1:D] .+= sum_cov_prev
    @views Szz_Cd[1:D, 1:D] .+= sum_cov_all
    Q_sum .+= sum_cov_next
    # dyn_xy[1:D, :] += (Σ p_smooth_tt1)'   — adjoint because cov(x_{t-1}, x_t) = p_smooth_tt1'.
    @views td_dyn_xy[1:D, :] .+= adjoint(sum_xcov)

    LinearAlgebra.copytri!(Szz_Ab, 'U')
    LinearAlgebra.copytri!(Szz_Cd, 'U')
    LinearAlgebra.copytri!(Q_sum, 'U')
    Symmetrize!(S0_sum)

    # backing storage; each E-step rewraps so the cached Cholesky reflects
    # the latest aggregate.
    suf.init_n = T(ntrials)
    suf.dyn_n = T(sum(size(tfs[trial].x_smooth, 2) for trial in 1:ntrials) - ntrials)
    suf.obs_n = T(sum(size(tfs[trial].x_smooth, 2) for trial in 1:ntrials))

    copyto!(suf.init_xy, td_init_xy)
    copyto!(suf.dyn_xy, td_dyn_xy)
    copyto!(suf.obs_xy, td_obs_xy)

    suf.init_xx[] = PDMat(fill(T(ntrials), 1, 1))
    suf.init_yy[] = PDMat(copy(S0_sum))
    suf.dyn_xx[] = PDMat(copy(Szz_Ab))
    suf.dyn_yy[] = PDMat(copy(Q_sum))
    suf.obs_xx[] = PDMat(copy(Szz_Cd))
    suf.obs_yy[] = PDMat(copy(R_sum))

    return suf
end

"""
    _aggregate_td_suff_stats_weighted!(suf, tfs, lds, u, v, y, weights, sws)

Weighted variant of `_aggregate_td_suff_stats!`. Each per-timestep
accumulation is scaled by `weights[trial][t]`, which carries the
responsibility γₖ,ₜ in the SLDS context (`q(zₜ = k)`).

The weighted form cannot reuse the precomputed `td_*_const` blocks —
weights change every E-step, so the data-side sums must be rebuilt fresh.
Likewise, the cov-cache fast path is skipped (responsibilities vary across
trials so `P_smooth[t]` is not shared in any useful way).

Conventions, mirroring the legacy weighted M-step:
- init terms use `weights[trial][1]` (responsibility at t=1)
- dynamics factor at time t uses `weights[trial][t]` (couples xₜ₋₁ and xₜ)
- emission at time t uses `weights[trial][t]`
"""
function _aggregate_td_suff_stats_weighted!(
    suf::SufficientStatistics{T},
    tfs::TrialFilterSmooth{T},
    lds::LinearDynamicalSystem{T,S,O},
    u_seq::AbstractVector{<:AbstractMatrix{T}},
    v_seq::AbstractVector{<:AbstractMatrix{T}},
    y::AbstractVector{<:AbstractMatrix{T}},
    weights::AbstractVector{<:AbstractVector{T}},
    sws::SmoothWorkspace{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    D = lds.latent_dim
    p = lds.obs_dim
    u_dim = lds.state_input_dim
    d_dim = lds.obs_input_dim
    ntrials = length(tfs)
    dyn_reg_dim = D + 1 + u_dim
    obs_reg_dim = D + 1 + d_dim

    # Clear the accumulators we'll write into. Each field is hoisted with a
    # concrete `Matrix{T}` annotation so the BLAS.ger!/syrk! callsites below
    # stay in JET's typed union branch (cf. `backwards_cov!`).
    init_xy = sws.td_init_xy::Matrix{T};
    fill!(init_xy, zero(T))
    init_yy = sws.S0_sum::Matrix{T};
    fill!(init_yy, zero(T))
    dyn_xx = sws.Szz_Ab::Matrix{T};
    fill!(dyn_xx, zero(T))
    dyn_xy = sws.td_dyn_xy::Matrix{T};
    fill!(dyn_xy, zero(T))
    dyn_yy = sws.Q_sum::Matrix{T};
    fill!(dyn_yy, zero(T))
    obs_xx = sws.Szz_Cd::Matrix{T};
    fill!(obs_xx, zero(T))
    obs_xy = sws.td_obs_xy::Matrix{T};
    fill!(obs_xy, zero(T))
    obs_yy = sws.R_sum::Matrix{T};
    fill!(obs_yy, zero(T))

    init_n_acc = zero(T)
    dyn_n_acc = zero(T)
    obs_n_acc = zero(T)

    for trial in 1:ntrials
        fs = tfs[trial]
        x_smooth = fs.x_smooth::Matrix{T}
        P_smooth = fs.p_smooth::Array{T,3}
        P_smooth_tt1 = fs.p_smooth_tt1::Array{T,3}
        y_trial = y[trial]
        T_n = size(x_smooth, 2)
        w = weights[trial]

        # Initial term — weighted by w[1].
        w1 = w[1]
        @views begin
            x1 = x_smooth[:, 1]
            for i in 1:D
                init_xy[1, i] += w1 * x1[i]
            end
            # init_yy += w1 * (x1 x1' + P_smooth[:, :, 1])
            BLAS.ger!(w1, x1, x1, init_yy)
            init_yy .+= w1 .* P_smooth[:, :, 1]
        end
        init_n_acc += w1

        # Dynamics factors at t = 2..T_n.
        @views for t in 2:T_n
            wt = w[t]
            x_prev = x_smooth[:, t - 1]
            x_next = x_smooth[:, t]

            # dyn_xx[1:D, 1:D] += wt * (x_prev x_prev' + P_smooth[t-1])
            BLAS.ger!(wt, x_prev, x_prev, view(dyn_xx, 1:D, 1:D))
            view(dyn_xx, 1:D, 1:D) .+= wt .* P_smooth[:, :, t - 1]
            # dyn_xx bias col / row
            for i in 1:D
                dyn_xx[i, D + 1] += wt * x_prev[i]
                dyn_xx[D + 1, i] += wt * x_prev[i]
            end
            dyn_xx[D + 1, D + 1] += wt

            # dyn_xy[1:D, :] += wt * (x_prev x_next' + P_smooth_tt1[t]')
            # cov(xₜ₋₁, xₜ) = P_smooth_tt1[t]'  (cf. unweighted aggregator).
            BLAS.ger!(wt, x_prev, x_next, view(dyn_xy, 1:D, :))
            view(dyn_xy, 1:D, :) .+= wt .* transpose(P_smooth_tt1[:, :, t])
            for j in 1:D
                dyn_xy[D + 1, j] += wt * x_next[j]
            end

            # dyn_yy += wt * (x_next x_next' + P_smooth[t])
            BLAS.ger!(wt, x_next, x_next, dyn_yy)
            dyn_yy .+= wt .* P_smooth[:, :, t]

            # User-input cross blocks (only when u_dim > 0). The lower-tri
            # mirror of the off-diagonal x_prev·u_prev' block is filled
            # once at the end of the function via `copytri!(dyn_xx, 'U')`.
            if u_dim > 0
                u_trial = u_seq[trial]
                u_prev = u_trial[:, t - 1]
                # dyn_xx[1:D, D+2:end] += wt * x_prev u_prev'
                BLAS.ger!(wt, x_prev, u_prev, view(dyn_xx, 1:D, (D + 2):dyn_reg_dim))
                # dyn_xx[D+1, D+2:end] += wt * u_prev   (bias × u cross; mirrored later)
                for k in 1:u_dim
                    dyn_xx[D + 1, D + 1 + k] += wt * u_prev[k]
                end
                # dyn_xx[D+2:end, D+2:end] += wt * u_prev u_prev'
                BLAS.ger!(
                    wt,
                    u_prev,
                    u_prev,
                    view(dyn_xx, (D + 2):dyn_reg_dim, (D + 2):dyn_reg_dim),
                )
                # dyn_xy[D+2:end, :] += wt * u_prev x_next'
                BLAS.ger!(wt, u_prev, x_next, view(dyn_xy, (D + 2):dyn_reg_dim, :))
            end

            dyn_n_acc += wt
        end

        # Emissions at t = 1..T_n.
        @views for t in 1:T_n
            wt = w[t]
            x_t = x_smooth[:, t]
            y_t = y_trial[:, t]

            # obs_xx[1:D, 1:D] += wt * (x_t x_t' + P_smooth[t])
            BLAS.ger!(wt, x_t, x_t, view(obs_xx, 1:D, 1:D))
            view(obs_xx, 1:D, 1:D) .+= wt .* P_smooth[:, :, t]
            for i in 1:D
                obs_xx[i, D + 1] += wt * x_t[i]
                obs_xx[D + 1, i] += wt * x_t[i]
            end
            obs_xx[D + 1, D + 1] += wt

            # obs_xy[1:D, :] += wt * x_t y_t'
            BLAS.ger!(wt, x_t, y_t, view(obs_xy, 1:D, :))
            for j in 1:p
                obs_xy[D + 1, j] += wt * y_t[j]
            end

            # obs_yy += wt * y_t y_t'
            BLAS.ger!(wt, y_t, y_t, obs_yy)

            # Obs-input cross blocks.
            if d_dim > 0
                v_trial = v_seq[trial]
                v_t = v_trial[:, t]
                # obs_xx[1:D, D+2:end] += wt * x_t v_t'
                BLAS.ger!(wt, x_t, v_t, view(obs_xx, 1:D, (D + 2):obs_reg_dim))
                # obs_xx[D+1, D+2:end] / [D+2:end, D+1] += wt * v_t
                for k in 1:d_dim
                    obs_xx[D + 1, D + 1 + k] += wt * v_t[k]
                    obs_xx[D + 1 + k, D + 1] += wt * v_t[k]
                end
                # obs_xx[D+2:end, D+2:end] += wt * v_t v_t'
                BLAS.ger!(
                    wt, v_t, v_t, view(obs_xx, (D + 2):obs_reg_dim, (D + 2):obs_reg_dim)
                )
                # obs_xy[D+2:end, :] += wt * v_t y_t'
                BLAS.ger!(wt, v_t, y_t, view(obs_xy, (D + 2):obs_reg_dim, :))
            end

            obs_n_acc += wt
        end
    end

    if u_dim > 0
        @views dyn_xx[(D + 2):dyn_reg_dim, 1:D] .= transpose(
            dyn_xx[1:D, (D + 2):dyn_reg_dim]
        )
    end
    if d_dim > 0
        @views obs_xx[(D + 2):obs_reg_dim, 1:D] .= transpose(
            obs_xx[1:D, (D + 2):obs_reg_dim]
        )
    end

    # Symmetrize PD blocks (BLAS.ger! is not symmetric and we touched the
    # bias row/col by hand; round-trip via Symmetrize! keeps PDMat happy).
    Symmetrize!(init_yy)
    LinearAlgebra.copytri!(dyn_xx, 'U')   # use upper to mirror everything to lower
    LinearAlgebra.copytri!(dyn_yy, 'U')
    LinearAlgebra.copytri!(obs_xx, 'U')
    LinearAlgebra.copytri!(obs_yy, 'U')

    # init_xx is the (1×1) effective sample count for x_init.
    suf.init_n = init_n_acc
    suf.dyn_n = dyn_n_acc
    suf.obs_n = obs_n_acc

    copyto!(suf.init_xy, init_xy)
    copyto!(suf.dyn_xy, dyn_xy)
    copyto!(suf.obs_xy, obs_xy)

    suf.init_xx[] = PDMat(fill(init_n_acc, 1, 1))
    suf.init_yy[] = PDMat(copy(init_yy))
    suf.dyn_xx[] = PDMat(copy(dyn_xx))
    suf.dyn_yy[] = PDMat(copy(dyn_yy))
    suf.obs_xx[] = PDMat(copy(obs_xx))
    suf.obs_yy[] = PDMat(copy(obs_yy))

    return suf
end

"""
    Q_state!(sws, lds, suf)

Total log-likelihood Q-state term across all trials, computed from the
aggregated sufficient statistics in `suf`. Replaces the per-trial,
per-timestep loops of the legacy `Q_state!(sws, lds, E_z, E_zz, E_zz_prev, u)`.

Identical value (up to floating-point) to summing the legacy form across
trials.
"""
function Q_state!(
    sws::SmoothWorkspace{T}, lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    D = lds.latent_dim
    A = lds.state_model.A
    b = lds.state_model.b
    B = lds.state_model.B
    x0 = lds.state_model.x0
    u_dim = lds.state_input_dim
    dyn_reg_dim = D + 1 + u_dim

    Q_U = sws.Q_PD[].chol.U
    P0_U = sws.P0_PD[].chol.U

    log_det_Q = logdet(sws.Q_PD[])
    log_det_P0 = logdet(sws.P0_PD[])

    N = suf.init_n
    dyn_n = suf.dyn_n

    log2π = log(T(2π))
    const_init = T(N) * D * log2π
    const_trans = T(dyn_n) * D * log2π

    # S_init = init_yy - μ x0' - x0 μ' + N x0 x0'    (μ = Σ x_init)
    S_init = sws.elbo_temp
    copyto!(S_init, suf.init_yy[].mat)
    μ_sum = vec(suf.init_xy)
    BLAS.ger!(-one(T), μ_sum, x0, S_init)
    BLAS.ger!(-one(T), x0, μ_sum, S_init)
    BLAS.ger!(T(N), x0, x0, S_init)

    ldiv!(P0_U', S_init)
    ldiv!(P0_U, S_init)
    Q_val = T(-0.5) * (const_init + T(N) * log_det_P0 + tr(S_init))

    # W = [A b B] (D × dyn_reg_dim)
    W = view(sws.AB, :, 1:dyn_reg_dim)
    copyto!(view(W, :, 1:D), A)
    copyto!(view(W, :, D + 1), b)
    if u_dim > 0
        copyto!(view(W, :, (D + 2):dyn_reg_dim), B)
    end

    S_trans = sws.elbo_temp                 # reuse (S_init no longer needed)
    copyto!(S_trans, suf.dyn_yy[].mat)
    mul!(S_trans, W, suf.dyn_xy, -one(T), one(T))
    mul!(S_trans, transpose(suf.dyn_xy), transpose(W), -one(T), one(T))
    # S_trans += W · dyn_xx · W'
    W_XX = view(sws.Sxz, :, 1:dyn_reg_dim)
    mul!(W_XX, W, suf.dyn_xx[].mat)
    mul!(S_trans, W_XX, transpose(W), one(T), one(T))

    ldiv!(Q_U', S_trans)
    ldiv!(Q_U, S_trans)
    Q_val += T(-0.5) * (const_trans + T(dyn_n) * log_det_Q + tr(S_trans))

    return Q_val
end

"""
    Q_obs!(sws, lds, suf)

Total log-likelihood Q-obs term across all trials and time, computed from
the aggregated sufficient statistics in `suf`. Replaces the per-trial,
per-timestep loop of the legacy `Q_obs!(sws, lds, E_z, E_zz, y, v)`.
"""
function Q_obs!(
    sws::SmoothWorkspace{T}, lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    D = lds.latent_dim
    p = lds.obs_dim
    d_dim = lds.obs_input_dim
    obs_reg_dim = D + 1 + d_dim
    C = lds.obs_model.C
    d = lds.obs_model.d
    D_obs = lds.obs_model.D

    R_U = sws.R_PD[].chol.U
    log_det_R = logdet(sws.R_PD[])
    const_term = p * log(T(2π))

    obs_n = suf.obs_n

    # V = [C d D_obs] (p × obs_reg_dim)
    V = view(sws.CD, :, 1:obs_reg_dim)
    copyto!(view(V, :, 1:D), C)
    copyto!(view(V, :, D + 1), d)
    if d_dim > 0
        copyto!(view(V, :, (D + 2):obs_reg_dim), D_obs)
    end

    # S_obs = obs_yy - V·obs_xy - obs_xy'·V' + V·obs_xx·V'
    S_obs = sws.elbo_obs_temp
    copyto!(S_obs, suf.obs_yy[].mat)
    mul!(S_obs, V, suf.obs_xy, -one(T), one(T))
    mul!(S_obs, transpose(suf.obs_xy), transpose(V), -one(T), one(T))
    V_XX = view(sws.Syz, :, 1:obs_reg_dim)
    mul!(V_XX, V, suf.obs_xx[].mat)
    mul!(S_obs, V_XX, transpose(V), one(T), one(T))

    ldiv!(R_U', S_obs)
    ldiv!(R_U, S_obs)
    return T(-0.5) * (T(obs_n) * (const_term + log_det_R) + tr(S_obs))
end

"""
    calculate_elbo(lds, suf, sws)

Total ELBO from aggregated sufficient statistics. Computes the same quantity
as the legacy `calculate_elbo(lds, tfs, y, sws_pool, ...)` but in
O(D³ + p²·D) instead of O(N·T·p²·D). The Gaussian-posterior entropy comes
from each trial's `fs.entropy` (filled by the smoother) and is summed by
the caller before this function is invoked.
"""
function calculate_elbo(
    lds::LinearDynamicalSystem{T,S,O},
    suf::SufficientStatistics{T},
    sws::SmoothWorkspace{T},
    total_entropy::T,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    Q_total = Q_state!(sws, lds, suf) + Q_obs!(sws, lds, suf)

    prior_term = zero(T)
    if lds.state_model.Q_prior !== nothing
        prior_term += iw_logprior_term(lds.state_model.Q, lds.state_model.Q_prior)
    end
    if lds.state_model.P0_prior !== nothing
        prior_term += iw_logprior_term(lds.state_model.P0, lds.state_model.P0_prior)
    end
    if lds.obs_model.R_prior !== nothing
        prior_term += iw_logprior_term(lds.obs_model.R, lds.obs_model.R_prior)
    end

    # MN-prior log-prior contributions for [A b B] (dynamics) and [C d D] (obs).
    # Required for ELBO monotonicity under MN priors — the M-step's `mn_map`
    # update + the IW posterior scale modification together maximize the
    # MAP objective, but without this term the displayed ELBO drops the
    # MN-quadratic piece and can appear non-monotone.
    if lds.state_model.AB_prior !== nothing
        D = lds.latent_dim
        u_dim = lds.state_input_dim
        W_ab = view(sws.AB, :, 1:(D + 1 + u_dim))
        copyto!(view(W_ab, :, 1:D), lds.state_model.A)
        copyto!(view(W_ab, :, D + 1), lds.state_model.b)
        if u_dim > 0
            copyto!(view(W_ab, :, (D + 2):(D + 1 + u_dim)), lds.state_model.B)
        end
        prior_term += mn_logprior_term(W_ab, lds.state_model.Q, lds.state_model.AB_prior)
    end
    if lds.obs_model.CD_prior !== nothing
        D = lds.latent_dim
        d_dim = lds.obs_input_dim
        W_cd = view(sws.CD, :, 1:(D + 1 + d_dim))
        copyto!(view(W_cd, :, 1:D), lds.obs_model.C)
        copyto!(view(W_cd, :, D + 1), lds.obs_model.d)
        if d_dim > 0
            copyto!(view(W_cd, :, (D + 2):(D + 1 + d_dim)), lds.obs_model.D)
        end
        prior_term += mn_logprior_term(W_cd, lds.obs_model.R, lds.obs_model.CD_prior)
    end

    return Q_total + prior_term + total_entropy
end

function update_initial_state_mean!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[1] || return nothing
    inv_n = inv(T(suf.init_n))
    x0 = lds.state_model.x0
    for j in eachindex(x0)
        x0[j] = suf.init_xy[1, j] * inv_n
    end
    return nothing
end

function update_initial_state_covariance!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[2] || return nothing
    D = lds.latent_dim
    x0 = lds.state_model.x0
    N = suf.init_n

    S0 = sws.S0_sum                                  # D × D scratch
    copyto!(S0, suf.init_yy[].mat)

    # Rank-1 updates inline (BLAS.ger! would need a contiguous μ vector and
    # `view(init_xy, 1, :)` allocates a SubArray header — small but nonzero).
    for j in 1:D
        μ_j = suf.init_xy[1, j]
        x0_j = x0[j]
        for i in 1:D
            μ_i = suf.init_xy[1, i]
            x0_i = x0[i]
            S0[i, j] += T(N) * x0_i * x0_j - x0_i * μ_j - μ_i * x0_j
        end
    end
    Symmetrize!(S0)

    if lds.state_model.P0_prior === nothing
        S0 ./= T(N)
    else
        Ψ, ν = lds.state_model.P0_prior.Ψ, lds.state_model.P0_prior.ν
        # iw_map inlined: (Ψ + S0) / (ν + N + D + 1)
        denom = ν + T(N) + T(D + 1)
        for i in eachindex(S0)
            S0[i] = (Ψ[i] + S0[i]) / denom
        end
    end
    copyto!(lds.state_model.P0, S0)
    return nothing
end

function update_A_b!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[3] || return nothing
    D = lds.latent_dim
    u_dim = lds.state_input_dim
    AB_prior = lds.state_model.AB_prior

    if AB_prior === nothing
        # Zero-alloc OLS fast path. `sws.Sxz` is exactly (D × dyn_reg_dim);
        # its transpose is the (dyn_reg_dim × D) view we ldiv! into. After
        # the in-place solve, `sws.Sxz` itself holds the transposed solution
        # `transpose(dyn_xx \ dyn_xy)` = the W = [A b B] regression matrix.
        Sxz_T = transpose(sws.Sxz)
        copyto!(Sxz_T, suf.dyn_xy)
        ldiv!(suf.dyn_xx[], Sxz_T)
        W = sws.Sxz
    else
        # MN-prior MAP path — keep `mn_map` (allocates) for now.
        W = mn_map(suf.dyn_xx[], suf.dyn_xy, AB_prior)
    end

    copyto!(lds.state_model.A, view(W, :, 1:D))
    copyto!(lds.state_model.b, view(W, :, D + 1))
    if u_dim > 0
        copyto!(lds.state_model.B, view(W, :, (D + 2):(D + 1 + u_dim)))
    end
    return nothing
end

function update_Q!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[4] || return nothing
    D = lds.latent_dim
    u_dim = lds.state_input_dim

    # sws.AB is exactly (D × dyn_reg_dim); no view needed.
    W = sws.AB
    copyto!(view(W, :, 1:D), lds.state_model.A)
    copyto!(view(W, :, D + 1), lds.state_model.b)
    if u_dim > 0
        copyto!(view(W, :, (D + 2):(D + 1 + u_dim)), lds.state_model.B)
    end

    # Residual scatter S = dyn_yy - W·dyn_xy - dyn_xy'·W' + W·dyn_xx·W'
    Wxy = sws.elbo_temp                        # D × D scratch (free post-Q_state!)
    mul!(Wxy, W, suf.dyn_xy)

    S_res = sws.Q_sum                          # D × D scratch
    copyto!(S_res, suf.dyn_yy[].mat)
    S_res .-= Wxy
    S_res .-= Wxy'
    # In-place X_A_Xt = W · dyn_xx · W'. Mimic PDMats' X_A_Xt: compute
    # `WL = W · L` (where dyn_xx = L·L' via the cached Cholesky) and add
    # `WL · WL'` to the upper triangle of S_res via a symmetric rank-k
    # BLAS call, then reflect upper → lower so the matrix is EXACTLY
    # symmetric and positive-semidefinite by construction. (`mul!(S_res,
    # WL, WL', 1, 1)` followed by `Symmetrize!` is *not* equivalent —
    # BLAS gemm can produce 1-ULP-asymmetric output, and averaging then
    # halves the off-diagonal X_A_Xt contribution.)
    WL = sws.Sxz                               # (D × dyn_reg_dim) scratch
    # WL = W · L where L is the lower-triangular Cholesky factor of
    # dyn_xx. PDMats stores the *upper* factor U in `.chol.factors`
    # (uplo='U'); L = U', so the equivalent BLAS call is
    # `trmm!(…, 'U', 'T', …)` on the raw factor matrix. This avoids
    # the per-call `LowerTriangular(...)` wrapper that
    # `mul!(WL, W, chol.L)` would allocate.
    copyto!(WL, W)
    BLAS.trmm!('R', 'U', 'T', 'N', one(T), suf.dyn_xx[].chol.factors, WL)
    mul!(S_res, WL, transpose(WL), one(T), one(T))

    # MN-prior contribution to the IW posterior scale.
    AB_prior = lds.state_model.AB_prior
    if AB_prior !== nothing
        Wm = W .- AB_prior.M₀
        S_res .+= Wm * AB_prior.Λ * Wm'
    end
    # Reflect upper → lower so the matrix is exactly symmetric. (`mul!`
    # of `WL · WL'` above can give 1-ULP-asymmetric output; mirroring
    # the upper triangle wins back exact symmetry and preserves the
    # mathematically-PSD upper values.)
    for j in 2:D, i in 1:(j - 1)
        S_res[j, i] = S_res[i, j]
    end

    Q_prior = lds.state_model.Q_prior
    if Q_prior === nothing
        S_res ./= T(suf.dyn_n)
    else
        # iw_map(Ψ, ν, S, N, d) = (Ψ + S) / (ν + N + d + 1), inlined to
        # avoid a fresh `(Ψ .+ S)` matrix. `Ψ` is `AbstractMatrix` at the
        # type level (IWPrior{T,M<:AbstractMatrix} doesn't pin M on the
        # `state_model.Q_prior` field), so we assert the concrete type
        # locally to keep the loop type-stable.
        denom = Q_prior.ν + T(suf.dyn_n) + T(D + 1)
        Ψ = Q_prior.Ψ::Matrix{T}
        for i in eachindex(S_res)
            S_res[i] = (Ψ[i] + S_res[i]) / denom
        end
    end
    copyto!(lds.state_model.Q, S_res)
    return nothing
end

function update_C_d!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    lds.fit_bool[5] || return nothing
    D = lds.latent_dim
    d_dim = lds.obs_input_dim
    CD_prior = lds.obs_model.CD_prior

    if CD_prior === nothing
        # Zero-alloc OLS fast path. `sws.Syz` is exactly (p × obs_reg_dim);
        # its transpose is the (obs_reg_dim × p) view we ldiv! into. After
        # the in-place solve, `sws.Syz` itself holds V = [C d D].
        Syz_T = transpose(sws.Syz)
        copyto!(Syz_T, suf.obs_xy)
        ldiv!(suf.obs_xx[], Syz_T)
        V = sws.Syz
    else
        V = mn_map(suf.obs_xx[], suf.obs_xy, CD_prior)
    end

    copyto!(lds.obs_model.C, view(V, :, 1:D))
    copyto!(lds.obs_model.d, view(V, :, D + 1))
    if d_dim > 0
        copyto!(lds.obs_model.D, view(V, :, (D + 2):(D + 1 + d_dim)))
    end
    return nothing
end

function update_R!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    lds.fit_bool[6] || return nothing
    p = lds.obs_dim
    D = lds.latent_dim
    d_dim = lds.obs_input_dim

    # sws.CD is exactly (p × obs_reg_dim); no view needed.
    V = sws.CD
    copyto!(view(V, :, 1:D), lds.obs_model.C)
    copyto!(view(V, :, D + 1), lds.obs_model.d)
    if d_dim > 0
        copyto!(view(V, :, (D + 2):(D + 1 + d_dim)), lds.obs_model.D)
    end

    # Residual scatter S = obs_yy - V·obs_xy - obs_xy'·V' + V·obs_xx·V'
    Vxy = sws.elbo_obs_temp                    # p × p scratch (free post-Q_obs!)
    mul!(Vxy, V, suf.obs_xy)

    S_res = sws.elbo_obs_work                  # p × p scratch
    copyto!(S_res, suf.obs_yy[].mat)
    S_res .-= Vxy
    S_res .-= Vxy'
    # In-place X_A_Xt = V · obs_xx · V'. Mirror PDMats' X_A_Xt: compute
    # `VL = V · L` (obs_xx = L·L' via the cached Cholesky) and add
    # `VL · VL'` to the upper triangle via a symmetric rank-k BLAS call,
    # then reflect upper → lower for exact symmetry. (`mul!` + `Symmetrize!`
    # would halve the off-diagonal contribution because gemm can produce
    # 1-ULP-asymmetric output that averaging then collapses.)
    VL = sws.Syz                               # (p × obs_reg_dim) scratch
    # See `update_Q!`: `BLAS.trmm!` on the raw upper-stored
    # `chol.factors` (with transa='T' since L = U') avoids the
    # `LowerTriangular(...)` wrapper alloc that `mul!(VL, V, chol.L)`
    # would do.
    copyto!(VL, V)
    BLAS.trmm!('R', 'U', 'T', 'N', one(T), suf.obs_xx[].chol.factors, VL)
    mul!(S_res, VL, transpose(VL), one(T), one(T))

    CD_prior = lds.obs_model.CD_prior
    if CD_prior !== nothing
        Wm = V .- CD_prior.M₀
        S_res .+= Wm * CD_prior.Λ * Wm'
    end

    for j in 2:p, i in 1:(j - 1)
        S_res[j, i] = S_res[i, j]
    end

    if lds.obs_model.R_prior === nothing
        S_res ./= T(suf.obs_n)
    else
        Ψ, ν = lds.obs_model.R_prior.Ψ, lds.obs_model.R_prior.ν
        S_res .= iw_map(Ψ, ν, S_res, T(suf.obs_n), p)
    end
    copyto!(lds.obs_model.R, S_res)
    return nothing
end

"""
    mstep!(lds, suf::SufficientStatistics, sws)

Aggregated M-step: runs the six suf-based `update_*!` overloads in sequence
(`x0`, `P0`, `A&b&B`, `Q`, `C&d&D`, `R`). Each respects the corresponding
`lds.fit_bool` flag. The fit hot path in `_fit_tridiag!` calls this once per
EM iteration after `_aggregate_td_suff_stats!`.
"""
function mstep!(
    lds::LinearDynamicalSystem{T,S,O}, suf::SufficientStatistics{T}, sws::SmoothWorkspace{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    update_initial_state_mean!(lds, suf)
    update_initial_state_covariance!(lds, suf, sws)
    update_A_b!(lds, suf, sws)
    update_Q!(lds, suf, sws)
    update_C_d!(lds, suf, sws)
    update_R!(lds, suf, sws)
    return nothing
end

"""
    fit!(lds, y; max_iter=100, tol=1e-6, progress=true,
         control_seq=nothing, obs_control_seq=nothing)

Fit a Gaussian Linear Dynamical System via Expectation-Maximization.

# Arguments
- `lds::LinearDynamicalSystem{T,S,O}`: model to fit in place
- `y`: observations. Two shapes accepted:
    * `AbstractMatrix{T}` of size `(obs_dim, T)` — single trial
    * `AbstractVector{<:AbstractMatrix{T}}` — multi-trial, each `(obs_dim, T_i)`,
      trial lengths may differ

# Keywords
- `max_iter::Int=100`: maximum EM iterations
- `tol::Float64=1e-6`: convergence tolerance on ELBO change
- `progress::Bool=true`: show progress bar
- `control_seq`: optional dynamics-input sequence. `Vector{<:AbstractMatrix}`
  for multi-trial (each `(u_dim, T_i)`); required when `size(state_model.B, 2) > 0`.
- `obs_control_seq`: optional observation-input sequence (same shape) for the
  obs-side input matrix `D`. Required when `size(obs_model.D, 2) > 0`.

Returns a `Vector{T}` of ELBO values, one per iteration.
"""
function fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}};
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
    control_seq::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    obs_control_seq::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps_per_trial = [size(yt, 2) for yt in y]
    u_seq = _normalize_multitrial_control(
        control_seq, lds.state_input_dim, tsteps_per_trial, T, "control_seq"
    )
    v_seq = _normalize_multitrial_obs_control(
        obs_control_seq, lds.obs_input_dim, tsteps_per_trial, T, lds.obs_model
    )
    return _fit!(lds, y, max_iter, tol, progress, u_seq, v_seq, Val(lds.kalman_filter))
end

function _fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y_vec::AbstractVector{<:AbstractMatrix{T}},
    max_iter::Int,
    tol::Float64,
    progress::Bool,
    u_seq::AbstractVector{<:AbstractMatrix{T}},
    v_seq::AbstractVector{<:AbstractMatrix{T}},
    ::Val{true},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    y_combined = zeros(T, size(y_vec[1], 1), size(y_vec[1], 2), length(y_vec))
    try
        # combine y vector into matrix
        y_combined = cat(y_vec...; dims=3)
    catch
        throw(
            ArgumentError(
                """
                Failed to combine input vector of matrices into a single matrix.
                Ensure all matrices have the same number of rows (obs_dim) and that
                the total number of columns does not exceed memory limits.
                """
            ),
        )
    end

    # Kalman path consumes 3-D arrays. Stack per-trial controls if any.
    u_combined = isempty(u_seq) || size(u_seq[1], 1) == 0 ? nothing : cat(u_seq...; dims=3)
    v_combined = isempty(v_seq) || size(v_seq[1], 1) == 0 ? nothing : cat(v_seq...; dims=3)

    return _fit_kalman!(
        lds,
        y_combined;
        control_seq=u_combined,
        obs_control_seq=v_combined,
        max_iter=max_iter,
        tol=tol,
        progress=progress,
    )
end

function _fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}},
    max_iter::Int,
    tol::Float64,
    progress::Bool,
    u_seq::AbstractVector{<:AbstractMatrix{T}},
    v_seq::AbstractVector{<:AbstractMatrix{T}},
    ::Val{false},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    return _fit_tridiag!(
        lds,
        y;
        control_seq=u_seq,
        obs_control_seq=v_seq,
        max_iter=max_iter,
        tol=tol,
        progress=progress,
    )
end

function _fit_tridiag!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractVector{<:AbstractMatrix{T}};
    control_seq::AbstractVector{<:AbstractMatrix{T}},
    obs_control_seq::AbstractVector{<:AbstractMatrix{T}},
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress::Bool=true,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps_per_trial = [size(yt, 2) for yt in y]
    T_max = maximum(tsteps_per_trial)

    prev_elbo = -T(Inf)
    elbos = Vector{T}()
    sizehint!(elbos, max_iter)

    # Opt in to the cov-alias stub for `p_smooth` / `p_smooth_tt1` when the
    # cov-cache fast path is going to fire (equal-length multi-trial). The
    # smoother aliases them to shared storage on every E-step, so per-trial
    # allocations of `(D, D, T)` are pure waste at large `N`.
    ntrials_total = length(y)
    cov_alias = ntrials_total > 1 && all(t -> t == tsteps_per_trial[1], tsteps_per_trial)
    tfs = initialize_FilterSmooth(
        lds, tsteps_per_trial; cov_alias=cov_alias
    )::TrialFilterSmooth{T}

    u_dim = lds.state_input_dim
    d_dim = lds.obs_input_dim
    # Only `sws_pool[1]` needs the batched mean-pass buffers (used by the
    # equal-length cov-cache fast path); the other workspaces back the
    # per-trial fallback / @spawn'd tasks and stay at `ntrials = 1`.
    pool_size = Threads.maxthreadid()
    sws_pool = Vector{SmoothWorkspace{T}}(undef, pool_size)
    sws_pool[1] = SmoothWorkspace(
        T,
        lds.latent_dim,
        lds.obs_dim,
        T_max;
        u_dim=u_dim,
        d_dim=d_dim,
        ntrials=ntrials_total,
    )
    for i in 2:pool_size
        sws_pool[i] = SmoothWorkspace(
            T, lds.latent_dim, lds.obs_dim, T_max; u_dim=u_dim, d_dim=d_dim
        )
    end

    # Sufficient-statistics aggregator: allocated once, mutated each E-step.
    # Data-only constants (Σ y y', Σ y, Σ u u' …) are precomputed here once
    # and reused across iterations.
    suf = _initialize_td_sufficient_statistics(T, lds, tsteps_per_trial)
    _td_init_const_blocks!(
        sws_pool[1], lds, tsteps_per_trial, y, control_seq, obs_control_seq
    )

    prog = if progress
        Progress(max_iter; desc="Fitting LDS via EM...", barlen=50, showspeed=true)
    else
        nothing
    end

    for _ in 1:max_iter
        # E-step: smooth + aggregate sufficient stats from x_smooth / p_smooth /
        # p_smooth_tt1. We skip the legacy `sufficient_statistics!(tfs)` call —
        # the new aggregator reads smoother outputs directly, so the per-trial
        # E_zz / E_zz_prev arrays are no longer populated on the hot path.
        smooth!(lds, tfs, y, sws_pool, control_seq, obs_control_seq)
        _aggregate_td_suff_stats!(
            suf, tfs, lds, control_seq, obs_control_seq, y, sws_pool[1]
        )

        # ELBO uses the same Cholesky factors as the smoother just filled.
        total_entropy = zero(T)
        for fs in tfs.FilterSmooths
            total_entropy += fs.entropy
        end
        elbo = calculate_elbo(lds, suf, sws_pool[1], total_entropy)
        push!(elbos, elbo)

        # M-step: regression + IW MAP from the aggregated stats. No tfs needed.
        mstep!(lds, suf, sws_pool[1])

        prog !== nothing && next!(prog)

        if abs(elbo - prev_elbo) < tol
            prog !== nothing && finish!(prog)
            return elbos
        end
        prev_elbo = elbo
    end

    prog !== nothing && finish!(prog)
    return elbos
end

function fit!(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T}; kwargs...
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    # reshape y from [obs_dim, tsteps, trials] to vector of matrices if needed
    if ndims(y) == 3
        obs_dim, tsteps, ntrials = size(y)
        y_vec = [view(y,:,:,i) for i in 1:ntrials]
        return fit!(lds, y_vec; kwargs...)
    elseif ndims(y) == 2
        # single trial case, wrap in vector
        return fit!(lds, [y]; kwargs...)
    else
        throw(ArgumentError("Input array y must be 2D or 3D."))
    end
end

function Gradient(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    ws = SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, tsteps)
    compute_smooth_constants!(ws, lds)
    return copy(Gradient!(ws, lds, y, x))
end

function Gradient(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    compute_smooth_constants!(ws, lds)
    return copy(Gradient!(ws, lds, y, x))
end

function Hessian(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    ws = SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, tsteps)
    compute_smooth_constants!(ws, lds)
    Hessian!(ws, lds, y, x)
    block_tridgm!(ws.btd)
    return copy(ws.btd.H_sparse)
end

function Hessian(
    ws::SmoothWorkspace{T},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    compute_smooth_constants!(ws, lds)
    Hessian!(ws, lds, y, x)
    block_tridgm!(ws.btd)
    return copy(ws.btd.H_sparse)
end

function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractVector{<:AbstractMatrix{T}},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    T_max = maximum(size(yt, 2) for yt in y)
    npool = Threads.maxthreadid()
    sws_pool = [SmoothWorkspace(T, lds.latent_dim, lds.obs_dim, T_max) for _ in 1:npool]
    return smooth!(lds, tfs, y, sws_pool)
end

function loglikelihood(
    x::AbstractMatrix{XT}, lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{YT}
) where {T<:Real,YT<:Real,XT<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    WT = promote_type(T, YT, XT)
    ws = SmoothWorkspace(WT, lds.latent_dim, lds.obs_dim, tsteps)
    compute_smooth_constants!(ws, lds)
    return loglikelihood!(ws, x, lds, y)
end

"""
    filter_loglikelihood(lds, y)

One-step-ahead predictive log-likelihood ∑_{t,n} log p(y_t^n | y_{1:t-1}^n) via
the Kalman filter.  Valid for any fitted `LinearDynamicalSystem` with a
`GaussianObservationModel`, regardless of which E-step backend was used to train it.

Returns the **total** log-likelihood.  Divide by `obs_dim * tsteps * ntrials` for a
per-observation score that is comparable across configurations.
"""
function filter_loglikelihood(
    lds::LinearDynamicalSystem{T,SM,OM}, y::AbstractArray{T,3}
) where {T<:Real,SM<:GaussianStateModel{T},OM<:GaussianObservationModel{T}}
    A = lds.state_model.A
    Q = lds.state_model.Q
    b = lds.state_model.b
    x0 = lds.state_model.x0
    P0 = lds.state_model.P0
    C = lds.obs_model.C
    R = lds.obs_model.R
    d = lds.obs_model.d

    obs_dim, tsteps, ntrials = size(y)
    D = lds.latent_dim

    total_ll = zero(T)
    log2πp = T(obs_dim) * log(T(2π))

    # Pre-allocate buffers (reused across trials and time steps)
    x_p = Vector{T}(undef, D)
    x_f = Vector{T}(undef, D)
    P_p = Matrix{T}(undef, D, D)
    P_f = Matrix{T}(undef, D, D)
    tmp_DD = Matrix{T}(undef, D, D)
    innov = Vector{T}(undef, obs_dim)
    Si_e = Vector{T}(undef, obs_dim)
    Smat = Matrix{T}(undef, obs_dim, obs_dim)
    PCt = Matrix{T}(undef, D, obs_dim)
    SiPCt = Matrix{T}(undef, obs_dim, D)

    for n in 1:ntrials
        x_f .= x0
        P_f .= P0

        for t in 1:tsteps
            # Prediction (t=1: prediction == prior)
            if t == 1
                x_p .= x0
                P_p .= P0
            else
                mul!(x_p, A, x_f)
                x_p .+= b
                mul!(tmp_DD, A, P_f)
                mul!(P_p, tmp_DD, A')
                P_p .+= Q
                Symmetrize!(P_p)
            end

            # Innovation: e = y_t - C x_p - d
            mul!(innov, C, x_p)
            @views innov .= y[:, t, n] .- innov .- d

            # Innovation covariance: S = C P_p C' + R
            mul!(PCt, P_p, C')
            mul!(Smat, C, PCt)
            Smat .+= R
            Symmetrize!(Smat)

            # One-step predictive log-likelihood: log N(e; 0, S)
            S_ch = cholesky(Hermitian(Smat))
            Si_e .= innov
            ldiv!(S_ch, Si_e)           # Si_e ← S⁻¹ e
            total_ll -= T(0.5) * (log2πp + logdet(S_ch) + dot(innov, Si_e))

            # Update: x_f = x_p + K e  where K = PCt S⁻¹
            mul!(x_f, PCt, Si_e)        # x_f = PCt (S⁻¹ e)
            x_f .+= x_p

            # P_f = P_p - PCt S⁻¹ PCt'
            SiPCt .= PCt'
            ldiv!(S_ch, SiPCt)          # SiPCt ← S⁻¹ PCt'
            mul!(P_f, PCt, SiPCt)
            P_f .= P_p .- P_f
            Symmetrize!(P_f)
        end
    end

    return total_ll
end

function filter_loglikelihood(
    lds::LinearDynamicalSystem{T,SM,OM}, y::AbstractVector{<:AbstractMatrix{T}}
) where {T<:Real,SM<:GaussianStateModel{T},OM<:GaussianObservationModel{T}}
    y_comb = cat(y...; dims=3)
    return filter_loglikelihood(lds, y_comb)
end
