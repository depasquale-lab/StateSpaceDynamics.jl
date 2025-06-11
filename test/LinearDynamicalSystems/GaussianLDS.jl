# Define the parameters of a penndulum
g = 9.81 # gravity
l = 1.0 # length of pendulum
dt = 0.01 # time step

# Discrete-time dynamics
A = [1.0 dt; -g / l*dt 1.0]
Q = Matrix{Float64}(0.00001 * I(2))  # Process noise covariance

# Initial state/ covariance
x0 = [0.0; 1.0]
P0 = Matrix{Float64}(0.1 * I(2))  # Initial state covariance

# Observation params 
C = Matrix{Float64}(I(2))  # Observation matrix (assuming direct observation)
observation_noise_std = 0.5
R = Matrix{Float64}((observation_noise_std^2) * I(2))  # Observation noise covariance

function toy_lds(
    ntrials::Int=1, fit_bool::Vector{Bool}=[true, true, true, true, true, true]
)
    lds = GaussianLDS(;
        A=A, C=C, Q=Q, R=R, x0=x0, P0=P0, obs_dim=2, latent_dim=2, fit_bool=fit_bool
    )

    # sample data
    T = 100
    x, y = StateSpaceDynamics.rand(lds; tsteps=T, ntrials=ntrials) # 100 timepoints, 1 trials

    return lds, x, y
end

function test_lds_properties(lds)
    # check state and observation model are of correct type
    @test isa(lds.state_model, StateSpaceDynamics.GaussianStateModel)
    @test isa(lds.obs_model, StateSpaceDynamics.GaussianObservationModel)
    @test isa(lds, StateSpaceDynamics.LinearDynamicalSystem)

    # check model param dimensions
    @test size(lds.state_model.A) == (lds.latent_dim, lds.latent_dim)
    @test size(lds.obs_model.C) == (lds.obs_dim, lds.latent_dim)
    @test size(lds.state_model.Q) == (lds.latent_dim, lds.latent_dim)
    @test size(lds.obs_model.R) == (lds.obs_dim, lds.obs_dim)
    @test size(lds.state_model.x0) == (lds.latent_dim,)
    @test size(lds.state_model.P0) == (lds.latent_dim, lds.latent_dim)
end

function test_gaussian_obs_constructor_type_preservation()
    # Int
    A_int = [1 2; 3 4]
    Q_int = [1 0; 0 1]
    x0_int = [5; 6]
    P0_int = [2 0; 0 2]

    gsm_int = GaussianStateModel(
        A = A_int,
        Q = Q_int,
        x0 = x0_int,
        P0 = P0_int,
        latent_dim = 2
    )

    @test eltype(gsm_int.A) === Int
    @test eltype(gsm_int.Q) === Int
    @test eltype(gsm_int.x0) === Int
    @test eltype(gsm_int.P0) === Int


    # Float32
    A_f32 = Float32[1 2; 3 4]
    Q_f32 = Float32[1 0; 0 1]
    x0_f32 = Float32[0.5; 0.6]
    P0_f32 = Float32[0.2 0; 0 0.2]

    gsm_f32 = GaussianStateModel(
        A = A_f32,
        Q = Q_f32,
        x0 = x0_f32,
        P0 = P0_f32,
        latent_dim = 2
    )

    @test eltype(gsm_f32.A)   === Float32
    @test eltype(gsm_f32.Q)   === Float32
    @test eltype(gsm_f32.x0)  === Float32
    @test eltype(gsm_f32.P0)  === Float32


    # BigFloat
    A_bf = BigFloat[1 2; 3 4]
    Q_bf = BigFloat[1 0; 0 1]
    x0_bf = BigFloat[0.1; 0.2]
    P0_bf = BigFloat[0.3 0; 0 0.3]

    gsm_bf = GaussianStateModel(
        A = A_bf,
        Q = Q_bf,
        x0 = x0_bf,
        P0 = P0_bf,
        latent_dim = 2
    )

    @test eltype(gsm_bf.A) === BigFloat
    @test eltype(gsm_bf.Q) === BigFloat
    @test eltype(gsm_bf.x0) === BigFloat
    @test eltype(gsm_bf.P0) === BigFloat
end

function test_gaussian_lds_constructor_type_preservation()
    # Int
    A_int = [1 2; 3 4]
    C_int = [1 0; 0 1]
    Q_int = [2 0; 0 2]
    R_int = [3 0; 0 3]
    x0_int = [5, 6]
    P0_int = [4 0; 0 4]

    gls_int = GaussianLDS(Int;
        A = A_int, C = C_int, Q = Q_int, R = R_int,
        x0 = x0_int, P0 = P0_int,
        obs_dim = 2, latent_dim = 2
    )

    @test eltype(gls_int.state_model.A) === Int
    @test eltype(gls_int.state_model.Q) === Int
    @test eltype(gls_int.state_model.x0) === Int
    @test eltype(gls_int.state_model.P0) === Int
    @test eltype(gls_int.obs_model.C) === Int
    @test eltype(gls_int.obs_model.R) === Int
    @test gls_int.latent_dim == 2
    @test gls_int.obs_dim == 2

    # Float32
    A_f32 = Float32[1 2; 3 4]
    C_f32 = Float32[1 0; 0 1]
    Q_f32 = Float32[2 0; 0 2]
    R_f32 = Float32[3 0; 0 3]
    x0_f32 = Float32[0.5, 1.5]
    P0_f32 = Float32[4 0; 0 4]

    gls_f32 = GaussianLDS(Float32;
        A = A_f32, C = C_f32, Q = Q_f32, R = R_f32,
        x0 = x0_f32, P0 = P0_f32,
        obs_dim = 2, latent_dim = 2
    )

    @test eltype(gls_f32.state_model.A) === Float32
    @test eltype(gls_f32.state_model.Q) === Float32
    @test eltype(gls_f32.state_model.x0) === Float32
    @test eltype(gls_f32.state_model.P0) === Float32
    @test eltype(gls_f32.obs_model.C) === Float32
    @test eltype(gls_f32.obs_model.R) === Float32

    # BigFloat
    A_bf = BigFloat[1 2; 3 4]
    C_bf = BigFloat[1 0; 0 1]
    Q_bf = BigFloat[2 0; 0 2]
    R_bf = BigFloat[3 0; 0 3]
    x0_bf = BigFloat[0.1, 0.2]
    P0_bf = BigFloat[4 0; 0 4]

    gls_bf = GaussianLDS(BigFloat;
        A = A_bf, C = C_bf, Q = Q_bf, R = R_bf,
        x0 = x0_bf, P0 = P0_bf,
        obs_dim = 2, latent_dim = 2
    )

    @test eltype(gls_bf.state_model.A) === BigFloat
    @test eltype(gls_bf.state_model.Q) === BigFloat
    @test eltype(gls_bf.state_model.x0) === BigFloat
    @test eltype(gls_bf.state_model.P0) === BigFloat
    @test eltype(gls_bf.obs_model.C) === BigFloat
    @test eltype(gls_bf.obs_model.R) === BigFloat
end

function test_gaussian_sample_type_preservation()
    
    # Float32
    A_f32 = Matrix{Float32}(I, 2, 2)
    C_f32 = Matrix{Float32}(I, 2, 2)
    Q_f32 = Matrix{Float32}(I, 2, 2)
    R_f32 = Matrix{Float32}(I, 2, 2)
    x0_f32 = fill(one(Float32), 2)
    P0_f32 = Matrix{Float32}(I, 2, 2)

    gls_f32 = GaussianLDS(Float32;
        A     = A_f32,
        C     = C_f32,
        Q     = Q_f32,
        R     = R_f32,
        x0    = x0_f32,
        P0    = P0_f32,
        obs_dim    = 2,
        latent_dim = 2
    )

    x_f32, y_f32 = rand(gls_f32; tsteps=50, ntrials=3)

    @test eltype(x_f32) === Float32
    @test eltype(y_f32) === Float32
    @test size(x_f32) == (2, 50, 3)
    @test size(y_f32) == (2, 50, 3)

    # BigFloat
    A_bf = Matrix{BigFloat}(I, 2, 2)
    C_bf = Matrix{BigFloat}(I, 2, 2)
    Q_bf = Matrix{BigFloat}(I, 2, 2)
    R_bf = Matrix{BigFloat}(I, 2, 2)
    x0_bf = fill(one(BigFloat), 2)
    P0_bf = Matrix{BigFloat}(I, 2, 2)

    gls_bf = GaussianLDS(BigFloat;
        A     = A_bf,
        C     = C_bf,
        Q     = Q_bf,
        R     = R_bf,
        x0    = x0_bf,
        P0    = P0_bf,
        obs_dim    = 2,
        latent_dim = 2
    )
    x_bf, y_bf = rand(gls_bf; tsteps=50, ntrials=3)

    @test eltype(x_bf) === BigFloat
    @test eltype(y_bf) === BigFloat
    @test size(x_bf) == (2, 50, 3)
    @test size(y_bf) == (2, 50, 3)
end

function test_gaussian_fit_type_preservation()
    for T in (Float64,) # Float32 and BigFLoat do not work because of lack of coverage in SparseArrays/LinearAlgebra. Need to fix upstream.
                        # See GitHub issue: https://github.com/JuliaSparse/SparseArrays.jl/issues/634            
        A  = Matrix{T}(I, 2, 2)
        C  = Matrix{T}(I, 2, 2)
        Q  = Matrix{T}(I, 2, 2)
        R  = Matrix{T}(I, 2, 2)
        x0 = fill(one(T), 2)
        P0 = Matrix{T}(I, 2, 2)

        lds = GaussianLDS(T;
            A         = A,
            C         = C,
            Q         = Q,
            R         = R,
            x0        = x0,
            P0        = P0,
            obs_dim    = 2,
            latent_dim = 2
        )
        
        x, y = rand(lds; tsteps=50, ntrials=3)

        mls, param_diff = fit!(lds, y; max_iter = 10, tol = 1e-6)

        @test eltype(mls) === T
        @test eltype(param_diff) === T
    end 
end

function test_gaussian_loglikelihood_type_preservation()
    for T in (Float32, BigFloat)
        
        A  = Matrix{T}(I, 2, 2)
        C  = Matrix{T}(I, 2, 2)
        Q  = Matrix{T}(I, 2, 2)
        R  = Matrix{T}(I, 2, 2)
        x0 = fill(one(T), 2)
        P0 = Matrix{T}(I, 2, 2)

        lds = GaussianLDS(T;
            A         = A,
            C         = C,
            Q         = Q,
            R         = R,
            x0        = x0,
            P0        = P0,
            obs_dim    = 2,
            latent_dim = 2
        )

        x, y = rand(lds; tsteps=50, ntrials=3)
        x_mat = x[:, :, 1] 
        y_mat = y[:, :, 1]  

        # compute log‐likelihood and check types 
        ll = StateSpaceDynamics.loglikelihood(x_mat, lds, y_mat)

        if ll isa Number
            @test typeof(ll) === T
        else
            @test eltype(ll) === T
        end
    end 
end

function test_lds_with_params()
    lds, _, _ = toy_lds()
    test_lds_properties(lds)

    @test lds.state_model.A == A
    @test lds.state_model.Q == Q
    @test lds.obs_model.C == C
    @test lds.obs_model.R == R
    @test lds.state_model.x0 == x0
    @test lds.state_model.P0 == P0
    @test lds.obs_dim == 2
    @test lds.latent_dim == 2
    @test lds.fit_bool == [true, true, true, true, true, true]
end

function test_lds_without_params()
    lds = GaussianLDS(; obs_dim=2, latent_dim=2)
    test_lds_properties(lds)

    @test !isempty(lds.state_model.A)
    @test !isempty(lds.state_model.Q)
    @test !isempty(lds.obs_model.C)
    @test !isempty(lds.obs_model.R)
    @test !isempty(lds.state_model.x0)
    @test !isempty(lds.state_model.P0)

    # test error is thrown if nothing is passed
    @test_throws ArgumentError GaussianLDS()
end

function test_Gradient()
    lds, x, y = toy_lds()

    # for each trial check the gradient
    for i in axes(y, 3)

        # numerically calculate the gradient
        f = latents -> StateSpaceDynamics.loglikelihood(latents, lds, y[:, :, i])
        grad_numerical = ForwardDiff.gradient(f, x[:, :, i])

        # analytical gradient
        grad_analytical = StateSpaceDynamics.Gradient(lds, y[:, :, i], x[:, :, i])
        @test norm(grad_numerical - grad_analytical) < 1e-8
    end
end

function test_Hessian()
    lds, x, y = toy_lds()

    function log_likelihood(x::AbstractArray, lds, y::AbstractArray)
        return StateSpaceDynamics.loglikelihood(x, lds, y)
    end

    # for each trial check the Hessian
    for i in axes(y, 3)
        hess, main, super, sub = StateSpaceDynamics.Hessian(lds, y[:, 1:3, i], x[:, 1:3, i])
        @test size(hess) == (3 * lds.latent_dim, 3 * lds.latent_dim)
        @test size(main) == (3,)
        @test size(super) == (2,)
        @test size(sub) == (2,)

        # calculate the Hessian using autodiff
        obj = latents -> log_likelihood(latents, lds, y[:, 1:3, i])
        hess_numerical = ForwardDiff.hessian(obj, x[:, 1:3, i])

        @test norm(hess_numerical - hess) < 1e-8
    end
end

function test_smooth()
    lds, x, y = toy_lds()
    x_smooth, p_smooth, inverseoffdiag = smooth(lds, y)

    n_trials = size(y, 3)
    n_tsteps = size(y, 2)

    @test size(x_smooth) == size(x)
    @test size(p_smooth) == (lds.latent_dim, lds.latent_dim, n_tsteps, n_trials)
    @test size(inverseoffdiag) == (lds.latent_dim, lds.latent_dim, n_tsteps, n_trials)

    # test gradient is zero
    for i in axes(y, 3)
        # may as well test the gradient here too 
        f = latents -> StateSpaceDynamics.loglikelihood(latents, lds, y[:, :, i])
        grad_numerical = ForwardDiff.gradient(f, x_smooth[:, :, i])
        grad_analytical = StateSpaceDynamics.Gradient(lds, y[:, :, i], x_smooth[:, :, i])

        @test norm(grad_numerical - grad_analytical) < 1e-8
        @test maximum(abs.(grad_analytical)) < 1e-8
        @test norm(grad_analytical) < 1e-8
    end
end

function test_estep()
    lds, x, y = toy_lds()

    # run the E_Step
    E_z, E_zz, E_zz_prev, x_smooth, p_smooth, ml_total = StateSpaceDynamics.estep(lds, y)

    n_trials = size(y, 3)
    n_tsteps = size(y, 2)

    @test size(E_z) == (lds.latent_dim, n_tsteps, n_trials)
    @test size(E_zz) == (lds.latent_dim, lds.latent_dim, n_tsteps, n_trials)
    @test size(E_zz_prev) == (lds.latent_dim, lds.latent_dim, n_tsteps, n_trials)
    @test size(x_smooth) == size(x)
    @test size(p_smooth) == (lds.latent_dim, lds.latent_dim, n_tsteps, n_trials)
    @test isa(ml_total, Float64)
end

function test_initial_observation_parameter_updates(ntrials::Int=1)
    lds, x, y = toy_lds(ntrials, [true, true, false, false, false, false])

    # run the E_Step
    E_z, E_zz, E_zz_prev, x_smooth, p_smooth, ml_total = StateSpaceDynamics.estep(lds, y)

    # optimize the x0 and p0 entries using autograd
    function obj(x0::AbstractVector, P0_sqrt::AbstractMatrix, lds)
        A, Q = lds.state_model.A, lds.state_model.Q
        P0 = P0_sqrt * P0_sqrt'
        Q_val = 0.0
        for i in axes(E_z, 3)
            Q_val += StateSpaceDynamics.Q_state(
                A, Q, P0, x0, E_z[:, :, i], E_zz[:, :, :, i], E_zz_prev[:, :, :, i]
            )
        end
        return -Q_val
    end

    P0_sqrt = Matrix(cholesky(lds.state_model.P0).U)

    x0_opt =
        optimize(
            x0 -> obj(x0, P0_sqrt, lds),
            lds.state_model.x0,
            LBFGS(),
            Optim.Options(; g_abstol=1e-12),
        ).minimizer
    P0_opt = optimize(P0_ -> obj(x0_opt, P0_, lds), P0_sqrt, LBFGS()).minimizer

    # update the initial state and covariance
    StateSpaceDynamics.mstep!(lds, E_z, E_zz, E_zz_prev, p_smooth, y)

    @test isapprox(lds.state_model.x0, x0_opt, atol=1e-6)
    @test isapprox(lds.state_model.P0, P0_opt * P0_opt', atol=1e-6)
end

function test_state_model_parameter_updates(ntrials::Int=1)
    lds, x, y = toy_lds(ntrials, [false, false, true, true, false, false])

    # run the E_Step
    E_z, E_zz, E_zz_prev, x_smooth, p_smooth, ml_total = StateSpaceDynamics.estep(lds, y)

    # optimize the A and Q entries using autograd
    function obj(A::AbstractMatrix, Q_sqrt::AbstractMatrix, lds)
        Q = Q_sqrt * Q_sqrt'
        Q_val = 0.0
        for i in axes(E_z, 3)
            Q_val += StateSpaceDynamics.Q_state(
                A,
                Q,
                lds.state_model.P0,
                lds.state_model.x0,
                E_z[:, :, i],
                E_zz[:, :, :, i],
                E_zz_prev[:, :, :, i],
            )
        end
        return -Q_val
    end

    Q_sqrt = Matrix(cholesky(lds.state_model.Q).U)

    A_opt =
        optimize(
            A -> obj(A, Q_sqrt, lds),
            lds.state_model.A,
            LBFGS(),
            Optim.Options(; g_abstol=1e-12),
        ).minimizer
    Q_opt =
        optimize(
            Q_sqrt -> obj(A_opt, Q_sqrt, lds),
            Q_sqrt,
            LBFGS(),
            Optim.Options(; g_abstol=1e-12),
        ).minimizer

    # update the state model
    StateSpaceDynamics.mstep!(lds, E_z, E_zz, E_zz_prev, p_smooth, y)

    @test isapprox(lds.state_model.A, A_opt, atol=1e-6)
    @test isapprox(lds.state_model.Q, Q_opt * Q_opt', atol=1e-6)
end

function test_obs_model_params_updates(ntrials::Int=1)
    lds, x, y = toy_lds(ntrials, [false, false, false, false, true, true])

    # run the E_Step
    E_z, E_zz, E_zz_prev, x_smooth, p_smooth, ml_total = StateSpaceDynamics.estep(lds, y)

    # optimize the C and R entries using autograd
    function obj(C::AbstractMatrix, R_sqrt::AbstractMatrix, lds)
        R = R_sqrt * R_sqrt'
        Q_val = 0.0
        for i in axes(E_z, 3)
            Q_val += StateSpaceDynamics.Q_obs(
                C, R, E_z[:, :, i], E_zz[:, :, :, i], y[:, :, i]
            )
        end
        return -Q_val
    end

    R_sqrt = Matrix(cholesky(lds.obs_model.R).U)

    C_opt =
        optimize(
            C -> obj(C, R_sqrt, lds),
            lds.obs_model.C,
            LBFGS(),
            Optim.Options(; g_abstol=1e-12),
        ).minimizer
    R_opt =
        optimize(
            R_sqrt -> obj(C_opt, R_sqrt, lds),
            R_sqrt,
            LBFGS(),
            Optim.Options(; g_abstol=1e-12),
        ).minimizer

    # update the observation model
    StateSpaceDynamics.mstep!(lds, E_z, E_zz, E_zz_prev, p_smooth, y)

    @test isapprox(lds.obs_model.C, C_opt, atol=1e-6)
    @test isapprox(lds.obs_model.R, R_opt * R_opt', atol=1e-6)
end

function test_EM(n_trials::Int=1)
    # create a toy LDS
    lds, x, y = toy_lds(n_trials)

    # create a randomly initialized LDS
    lds_new = GaussianLDS(; obs_dim=2, latent_dim=2)

    # run the EM algorithm for many iterations
    ml_total, norm_diff = fit!(lds_new, y; max_iter=100)

    # test that the ml is increasing
    @test all(diff(ml_total) .>= 0)
end
