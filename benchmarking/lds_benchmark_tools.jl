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
        observation_covariance=numpy.array(R),
        em_vars=["transition_matrices", "transition_covariance", "initial_state_mean", "initial_state_covariance", "observation_matrices", "observation_covariance"],
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
    dlds = pyimport("dynamax.linear_gaussian_ssm")
    np = pyimport("numpy")

    (; latent_dim, obs_dim) = instance
    (; A, Q, x0, P0, C, R) = params

    # Convert everything to NumPy arrays
    A_np = np.array(A)
    Q_np = np.array(Q)
    x0_np = np.array(x0)
    P0_np = np.array(P0)
    C_np = np.array(C)
    R_np = np.array(R)

    # Create the Dynamax model
    lds = dlds.LinearGaussianSSM(latent_dim, obs_dim)
    key = jr.PRNGKey(0)
    dyn_params, props = lds.initialize(
        key=key,
        dynamics_weights=A_np,
        dynamics_covariance=Q_np,
        initial_mean=x0_np,
        initial_covariance=P0_np,
        emission_weights=C_np,
        emission_covariance=R_np,
    )

    return (dyn_params, props, lds)
end

function run_benchmark(::SSD_LDSImplem, model::LinearDynamicalSystem, Y::AbstractArray)
    # Run 1 EM iteration to compile
    StateSpaceDynamics.fit!(deepcopy(model), Y, max_iter=1, tol=1e-50)

    # run Benchmark
    bench = @benchmark begin
        StateSpaceDynamics.fit!($model, $Y; max_iter=100, tol=1e-50)
    end samples=5
    return (time=median(bench).time, memory=bench.memory, allocs=bench.allocs, success=true)
end

function run_benchmark(::pykalman_LDSImplem, model::Any, Y::AbstractArray)
    # No need to run an initial iteration, no JIT
    Y = Y[:, :, 1]

    # run benchmark
    np = pyimport("numpy")
    Y_np = np.array(Y).transpose()
    bench = @benchmark begin
        $model.em($Y_np, n_iter=100)
    end samples=5
    return (time=median(bench).time, memory=bench.memory, allocs=bench.allocs, success=true)
end

function run_benchmark(::Dynamax_LDSImplem, model::Tuple, Y::AbstractArray)
    # Run an initial iteration to compile
    dynamax = pyimport("dynamax")
    np = pyimport("numpy")
    jax = pyimport("jax")

    Y = Y[:, :, 1]
    Y_np = np.array(Y).transpose()

    (params, props, lds) = model

    fit_em_ = jax.jit(lds.fit_em, static_argnames=("num_iters",))

    bench = @benchmark begin
        $fit_em_($params,
            $props,
            $Y_np,
            num_iters=100)
    end samples=5
    return (time=median(bench).time, memory=bench.memory, allocs=bench.allocs, success=true)
end