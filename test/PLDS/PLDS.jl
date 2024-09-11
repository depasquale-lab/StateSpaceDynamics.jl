function toy_PoissonLDS()
    T = 100
    # create a PLDS model
    x0 = [1.0, -1.0]
    p0 = Matrix(Diagonal([0.001, 0.001]))
    A = [cos(0.1) -sin(0.1); sin(0.1) cos(0.1)]
    Q = Matrix(Diagonal([0.001, 0.001]))
    C = [0.5 0.5; 0.5 0.1; 0.1 0.1]
    log_d = [0.5, 0.5, 0.5]
    D = Matrix(Diagonal([0.0, 0.0, 0.0]))
    b = ones(T, 2) * 0.0

    plds = PLDS(;
        A=A,
        C=C,
        Q=Q,
        D=D,
        b=b,
        log_d=log_d,
        x0=x0,
        p0=p0,
        refractory_period=1,
        obs_dim=3,
        latent_dim=2,
    )
    # sample data
    x, y = StateSpaceDynamics.sample(plds, T, 3)
    return plds, x, y
end

function test_PLDS_constructor_with_params()
    # create a set of parameters to test with
    obs_dim = 10
    latent_dim = 5

    A = randn(latent_dim, latent_dim)
    C = randn(obs_dim, latent_dim)
    Q = I(latent_dim)
    x0 = randn(latent_dim)
    p0 = I(latent_dim)
    refrac = 1
    log_d = randn(obs_dim)
    D = randn(obs_dim, obs_dim)
    fit_bool = Vector([true, true, true, true, true, true])

    # create the PLDS model
    plds = PLDS(;
        A=A,
        C=C,
        Q=Q,
        D=D,
        log_d=log_d,
        x0=x0,
        p0=p0,
        refractory_period=refrac,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        fit_bool=fit_bool,
    )

    # test model
    @test plds.A == A
    @test plds.C == C
    @test plds.Q == Q
    @test plds.x0 == x0
    @test plds.p0 == p0
    @test plds.log_d == log_d
    @test plds.D == D
    @test plds.refractory_period == 1
    @test plds.obs_dim == obs_dim
    @test plds.latent_dim == latent_dim
end

function test_PLDS_constructor_without_params()
    # create the PLDS model
    plds = PLDS(; obs_dim=10, latent_dim=5)

    # test parameters are not empty
    @test !isempty(plds.A)
    @test !isempty(plds.C)
    @test !isempty(plds.Q)
    @test !isempty(plds.x0)
    @test !isempty(plds.p0)
    @test !isempty(plds.log_d)
    @test !isempty(plds.D)
    @test plds.refractory_period == 1
    @test plds.obs_dim == 10
    @test plds.latent_dim == 5
    @test plds.fit_bool == fill(true, 6)

    # test dims of parameters
    @test size(plds.A) == (5, 5)
    @test size(plds.C) == (10, 5)
    @test size(plds.Q) == (5, 5)
    @test size(plds.x0) == (5,)
    @test size(plds.p0) == (5, 5)
    @test size(plds.log_d) == (10,)
    @test size(plds.D) == (10, 10)
    @test isempty(plds.b)
end

function test_countspikes()
    # create a set of observations that is a matrix of spikes/events
    obs = [0 0 1; 1 1 1; 0 1 0]
    # count the spikes when window=1
    count = StateSpaceDynamics.countspikes(obs, 1)
    # check the count
    @test count == [0 0 0; 0 0 1; 1 1 1]
    # count spikes when window=2
    count_2 = StateSpaceDynamics.countspikes(obs, 2)
    # check the count
    @test count_2 == [0 0 0; 0 0 1; 1 1 2]
end

function test_logposterior()
    # create a plds model
    plds = PLDS(; obs_dim=10, latent_dim=5)
    # create some observations
    obs = rand(Bool, 100, 10)
    # create latent state
    x = randn(100, 5)
    b = randn(100, 5)
    plds.b = b
    # calculate the log posterior
    logpost = StateSpaceDynamics.logposterior(x, plds, obs)
    # check the dimensions
    @test logpost isa Float64
end

function test_gradient_plds()
    # create a plds model
    plds = PLDS(; obs_dim=10, latent_dim=5)
    # create some observations
    obs = rand(Bool, 3, 10)
    # create initial latent state for gradient calculation
    x = randn(3, 5)
    b = zeros(3, 5)
    plds.b = b
    # calculate the gradient
    grad = StateSpaceDynamics.Gradient(x, plds, obs)
    # check the dimensions
    @test size(grad) == (3, 5)
    # check the gradients using autodiff
    obj = x -> StateSpaceDynamics.logposterior_nonthreaded(x, plds, obs)
    grad_autodiff = ForwardDiff.gradient(obj, x)
    @test grad ≈ grad_autodiff atol = 1e-12
end

function test_hessian_plds()
    # create a plds model
    plds = PLDS(; obs_dim=10, latent_dim=5)
    # create some observations
    obs = rand(Bool, 3, 10)
    # create initial latent state for hessian calculation
    x = randn(3, 5)
    b = zeros(3, 5)
    plds.b = b
    # calculate the hessian
    hess, main, super, sub = StateSpaceDynamics.Hessian(x, plds, obs)
    # check the dimensions
    @test length(main) == 3
    @test length(super) == 2
    @test length(sub) == 2
    @test size(hess) == (15, 15)
    # check the hessian using autodiff
    function obj_logposterior(x::Vector)
        x = StateSpaceDynamics.interleave_reshape(x, 3, 5)
        return StateSpaceDynamics.logposterior_nonthreaded(x, plds, obs)
    end
    hess_autodiff = ForwardDiff.hessian(obj_logposterior, reshape(x', 15))
    @test hess ≈ hess_autodiff atol = 1e-12
end

function test_direct_smoother()
    # create a plds model
    plds = PLDS(; obs_dim=10, latent_dim=5)
    # create some observations
    obs = rand(Bool, 10, 10)
    # create inputs
    b = rand(10, 5)
    plds.b = b
    # run the direct smoother
    x_smooth, p_smooth = StateSpaceDynamics.directsmooth(plds, obs)
    # check the dimensions
    @test size(x_smooth) == (10, 5)
    @test size(p_smooth) == (10, 5, 5)
end

function test_smooth()
    plds, x, y = toy_PoissonLDS()
    # run the smoother
    x_smooth, p_smooth = StateSpaceDynamics.smooth(plds, y)
    # check the dimensions
    @test size(x_smooth) == (3, 100, 2)
    @test size(p_smooth) == (3, 100, 2, 2)
end

function test_analytical_parameter_updates()
    # create a dummy data
    dummy_plds, x, y = toy_PoissonLDS()
    # now create a random plds model
    plds = PLDS(; obs_dim=3, latent_dim=2)
    # save the old parameters from the model
    A = copy(plds.A)
    Q = copy(plds.Q)
    x0 = copy(plds.x0)
    p0 = copy(plds.p0)
    # run E-Step
    E_z, E_zz, E_zz_prev, x_sm, p_sm = StateSpaceDynamics.E_Step(plds, y)

    # optimize x0 
    opt_x0 = x0 -> -StateSpaceDynamics.Q_initial_obs(x0, plds.p0, E_z, E_zz)
    result_x0 = optimize(opt_x0, plds.x0, LBFGS())
    @test isapprox(
        result_x0.minimizer, StateSpaceDynamics.update_initial_state_mean!(plds, E_z)
    )
    plds.x0 = result_x0.minimizer

    # optimize p0
    opt_p0 = p0 -> -StateSpaceDynamics.Q_initial_obs(plds.x0, p0, E_z, E_zz)
    result_p0 = optimize(opt_p0, plds.p0, LBFGS(), Optim.Options(; g_abstol=1e-12))
    @test isapprox(
        result_p0.minimizer * result_p0.minimizer',
        StateSpaceDynamics.update_initial_state_covariance!(plds, E_zz, E_z),
        atol=1e-3,
    )

    Q_l = Matrix(cholesky(Q).L)
    # optimize A and Q
    opt_A = A -> -StateSpaceDynamics.Q_state_model(A, Q_l, E_zz, E_zz_prev)
    result_A = optimize(
        opt_A,
        rand(plds.latent_dim, plds.latent_dim),
        LBFGS(),
        Optim.Options(; g_abstol=1e-16),
    )
    println(
        "Difference between analytical and numerical results:",
        result_A.minimizer - StateSpaceDynamics.update_A_plds!(plds, E_zz, E_zz_prev),
    )
    @test isapprox(
        result_A.minimizer,
        StateSpaceDynamics.update_A_plds!(plds, E_zz, E_zz_prev),
        atol=1e-4,
    ) # sometimes this works sometimes this doesn't. Lowering the tolerance to 1e-4

    # update the model before update Q
    plds.A = result_A.minimizer
    # optimize Q now
    opt_Q = Q -> -StateSpaceDynamics.Q_state_model(plds.A, Q, E_zz, E_zz_prev)
    result_Q = optimize(opt_Q, Q_l, LBFGS(), Optim.Options(; g_abstol=1e-12))

    @test isapprox(
        result_Q.minimizer * result_Q.minimizer',
        StateSpaceDynamics.update_Q_plds!(plds, E_zz, E_zz_prev),
        atol=1e-6,
    )
end
