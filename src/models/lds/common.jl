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
