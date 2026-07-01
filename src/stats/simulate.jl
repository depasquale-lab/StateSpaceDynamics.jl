# Sampling / simulation for Linear Dynamical Systems.

function _sample_trial!(
    rng,
    x_trial,
    y_trial,
    state_params,
    obs_params,
    obs_model::GaussianObservationModel,
    ux_trial::AbstractMatrix,
    uy_trial::AbstractMatrix,
)
    tsteps = size(x_trial, 2)

    # Initial state. The observation at t=1 includes the obs-input term D·v_1
    # when uy_trial has nonzero rows; zero-row matmul is a no-op.
    x_trial[:, 1] = rand(rng, MvNormal(state_params.x0, state_params.P0))
    y_trial[:, 1] = rand(
        rng,
        MvNormal(
            obs_params.C * x_trial[:, 1] + obs_params.d + obs_params.D * uy_trial[:, 1],
            obs_params.R,
        ),
    )

    # Subsequent states. The dynamics input B·u_{t-1} kicks the state forward;
    # again, zero-row ux_trial degenerates to no input.
    for t in 2:tsteps
        x_trial[:, t] = rand(
            rng,
            MvNormal(
                state_params.A * x_trial[:, t - 1] +
                state_params.b +
                state_params.B * ux_trial[:, t - 1],
                state_params.Q,
            ),
        )
        y_trial[:, t] = rand(
            rng,
            MvNormal(
                obs_params.C * x_trial[:, t] + obs_params.d + obs_params.D * uy_trial[:, t],
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
    ux_trial::AbstractMatrix,
    uy_trial::AbstractMatrix,
)
    tsteps = size(x_trial, 2)
    # Poisson obs model has no D matrix; uy_trial is accepted for signature
    # parity with the Gaussian path but must be empty (validated by callers).
    @assert size(uy_trial, 1) == 0 "Poisson observation model does not support obs inputs"

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
                state_params.B * ux_trial[:, t - 1],
                state_params.Q,
            ),
        )
        y_trial[:, t] =
            rand.(rng, Poisson.(exp.(obs_params.C * x_trial[:, t] + obs_params.d)))
    end
end

"""
    Random.rand([rng,] lds, tsteps::Integer; latent_inputs=nothing, obs_inputs=nothing)
    Random.rand([rng,] lds, tsteps_per_trial::AbstractVector{<:Integer};
                latent_inputs=nothing, obs_inputs=nothing)

Sample from a Linear Dynamical System.

- With a scalar `tsteps`, returns one trial as `(x::Matrix, y::Matrix)` of sizes
  `(latent_dim, tsteps)` and `(obs_dim, tsteps)` respectively.
- With a vector of per-trial lengths, returns
  `(x::Vector{Matrix}, y::Vector{Matrix})`. Lengths may differ across trials.

Optional input sequences:
- `latent_inputs`: dynamics-input sequence consumed by `B`. Single-trial form
  is an `(ux_dim, tsteps)` matrix; multi-trial is a `Vector{<:AbstractMatrix}`
  of per-trial matrices. Required when `size(state_model.B, 2) > 0`.
- `obs_inputs`: same shape for the observation input `D`. Required when
  `size(obs_model.D, 2) > 0`. Gaussian observation model only.
"""
function Random.rand(
    rng::AbstractRNG,
    lds::LinearDynamicalSystem{T,S,O},
    tsteps::Integer;
    latent_inputs::Union{Nothing,AbstractMatrix{T}}=nothing,
    obs_inputs::Union{Nothing,AbstractMatrix{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    state_params = _extract_state_params(lds.state_model)
    obs_params = _extract_obs_params(lds.obs_model)
    Ti = Int(tsteps)

    ux_trial = _check_latent_inputs(
        latent_inputs, lds.state_input_dim, Ti, "latent_inputs", T
    )
    uy_trial = _check_obs_inputs(obs_inputs, lds.obs_input_dim, Ti, lds.obs_model)

    x = Matrix{T}(undef, lds.latent_dim, Ti)
    y = Matrix{T}(undef, lds.obs_dim, Ti)
    _sample_trial!(rng, x, y, state_params, obs_params, lds.obs_model, ux_trial, uy_trial)
    return x, y
end

function Random.rand(
    rng::AbstractRNG,
    lds::LinearDynamicalSystem{T,S,O},
    tsteps_per_trial::AbstractVector{<:Integer};
    latent_inputs::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
    obs_inputs::Union{Nothing,AbstractVector{<:AbstractMatrix{T}}}=nothing,
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

    ux_seq = _normalize_multitrial_latent_inputs(
        latent_inputs, lds.state_input_dim, tsteps_per_trial, T, "latent_inputs"
    )
    uy_seq = _normalize_multitrial_obs_inputs(
        obs_inputs, lds.obs_input_dim, tsteps_per_trial, T, lds.obs_model
    )

    # `MersenneTwister` (and most RNG types) is not thread-safe, so sharing
    # `rng` across `@threads` races on internal state.
    if ntrials == 1
        _sample_trial!(
            rng, x[1], y[1], state_params, obs_params, lds.obs_model, ux_seq[1], uy_seq[1]
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
                    ux_seq[trial],
                    uy_seq[trial],
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
