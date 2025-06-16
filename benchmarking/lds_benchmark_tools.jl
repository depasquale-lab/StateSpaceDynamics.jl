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

"""
Build pykalman model.
"""

struct pykalman_LDSImplem <: Implementation end
Base.string(::pykalman_LDSImplem) = "pykalman"

function build_model(::pykalman_LDSImplem, instance::LDSInstance, params::LDSParams)
    pykalman = pyimport("pykalman")
    numpy = pyimport("numpy")

    (; latent_dim, obs_dim) = instance
    (; A, Q, x0, P0, C, R) = params

    kf = pykalman.KalmanFilter(
        n_dim_state=latent_dim,
        n_dim_obs=obs_dim,
        transition_matrices=numpy.array(A),
        transition_covariance=numpy.array(Q),
        initial_state_mean=numpy.array(x0),
        initial_state_covariance=numpy.array(P0),
        observation_matrices=numpy.array(C),
        observation_covariance=numpy.array(R)
    )

    return kf
end

"""
Build Dynamax model.
"""

struct Dynamax_LDSImplem <: Implementation end
Base.string(::Dynamax_LDSImplem) = "Dynamax"

function build_model(::Dynamax_LDSImplem, instance::LDSInstance, params::LDSParams)
    dynamax = pyimport("dynamax")
    jr = pyimport("jax.random")

    (; latent_dim, obs_dim) = instance
    (; A, Q, x0, P0, C, R) = params

    # Create the Dynamax model
    lds = dynamax.LinearGaussianSSM(latent_dim, obs_dim)
    key = jr.PRNGKey(0)
    params, props = lds.initialize(
        key=key,
        dynamics_weights=A,
        dynamics_covariance=Q,
        initial_mean=x0,
        initial_covariance=P0,
        emission_weights=C,
        emission_covariance=R
    )

    return (params, props)
end
