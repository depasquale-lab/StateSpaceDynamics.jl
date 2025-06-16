export SSDLDSImplem, build_model

struct SSD_LDSImplem <: Implementation end
Base.string(::SSD_LDSImplem) = "StateSpaceDynamics.jl"

function build_model(::SSD_LDSImplem, instance::LDSInstance, params::LDSParams)
    (; latent_dim, obs_dim, num_trials, seq_length) = instance
    (; A, Q, x0, P0, C, R)  = params

    # Create the model
    state_model = GaussianStateModel(
        A = A,
        Q = Q,
        x0 = x0,
        P0 = P0,
        )

    obs_model = GaussianObservationModel(
        C = C,
        R = R,
        )

    glds = LinearDynamicalSystem(;
            state_model=state_model,
            obs_model=obs_model,
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            fit_bool=fill(true, 6))

    return glds
end