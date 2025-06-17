export SSD_LDSImplem, pykalman_LDSImplem, Dynamax_LDSImplem, build_model

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

    return (params, props, lds)
end

function run_benchmark(::SSD_LDSImplem, model::LinearDynamicalSystem, Y::AbstractArray) where T
    # Run 1 EM iteration to compile
    fit!(deepcopy(model), Y; max_iters=1)

    # run Benchmark
    bench = @benchmark begin
        fit!(model, Y; max_iters=100, tol=1e-50)
    end samples=5
    return bench
end

function run_benchmark(::pykalman_LDSImplem, model::Any, Y::AbstractArray) where T
    # No need to run an initial iteration, no JIT

    # run benchmark
    np = pyimport("numpy")
    Y_np = np.array(Y)
    bench = @benchmark begin
        model.em(Y_np, n_iter=100, tol=1e-50)
    end samples=5
    return bench
end

function run_benchmark(::Dynamax_LDSImplem, model::Tuple, Y::AbstractArray) where T
    # Run an initial iteration to compile
    dynamax = pyimport("dynamax")

    (params, props, lds) = model

    model.fit_em(params,
        props,
        Y,
        n_iter=1,
        tol=1e-50,)
end