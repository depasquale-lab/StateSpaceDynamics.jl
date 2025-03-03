# define parameters for a PoissonLDS
x0 = [1.0, -1.0]
p0 = Matrix(Diagonal([0.1, 0.1]))
A = [cos(0.1) -sin(0.1); sin(0.1) cos(0.1)]
Q = Matrix(Diagonal([0.1, 0.1]))
C = [0.6 0.6; 0.6 0.6; 0.6 0.6] .* 2
log_d = log.([0.1, 0.1, 0.1])

function toy_PoissonLDS(
    n_trials::Int=1, fit_bool::Vector{Bool}=[true, true, true, true, true, true]
)
    poisson_lds = PoissonLDS(;
        A=A, C=C, Q=Q, x0=x0, P0=p0, log_d=log_d, obs_dim=3, latent_dim=2, fit_bool=fit_bool
    )

    # sample data
    T = 100
    x, y = StateSpaceDynamics.sample(poisson_lds, T, n_trials) # 100 timepoints, ntrials

    return poisson_lds, x, y
end

function test_plds_properties(poisson_lds)
    @test isa(poisson_lds.state_model, StateSpaceDynamics.GaussianStateModel)
    @test isa(poisson_lds.obs_model, StateSpaceDynamics.PoissonObservationModel)
    @test isa(poisson_lds, StateSpaceDynamics.LinearDynamicalSystem)

    @test size(poisson_lds.state_model.A) ==
        (poisson_lds.latent_dim, poisson_lds.latent_dim)
    @test size(poisson_lds.obs_model.C) == (poisson_lds.obs_dim, poisson_lds.latent_dim)
    @test size(poisson_lds.state_model.Q) ==
        (poisson_lds.latent_dim, poisson_lds.latent_dim)
    @test size(poisson_lds.state_model.x0) == (poisson_lds.latent_dim,)
    @test size(poisson_lds.state_model.P0) ==
        (poisson_lds.latent_dim, poisson_lds.latent_dim)
    @test size(poisson_lds.obs_model.log_d) == (poisson_lds.obs_dim,)
end

function test_PoissonLDS_with_params()
    poisson_lds, _, _ = toy_PoissonLDS()
    test_plds_properties(poisson_lds)

    @test poisson_lds.state_model.A == A
    @test poisson_lds.state_model.Q == Q
    @test poisson_lds.obs_model.C == C
    @test poisson_lds.state_model.x0 == x0
    @test poisson_lds.state_model.P0 == p0
    @test poisson_lds.obs_dim == 3
    @test poisson_lds.latent_dim == 2
    @test poisson_lds.fit_bool == [true, true, true, true, true, true]
end

function test_poisson_lds_without_params()
    poisson_lds = PoissonLDS(; obs_dim=3, latent_dim=2)
    test_plds_properties(poisson_lds)

    @test !isempty(poisson_lds.state_model.A)
    @test !isempty(poisson_lds.state_model.Q)
    @test !isempty(poisson_lds.obs_model.C)
    @test !isempty(poisson_lds.state_model.x0)
    @test !isempty(poisson_lds.state_model.P0)
    @test !isempty(poisson_lds.obs_model.log_d)

    # check errors are thrown when nothing is passed
    # @test_throws PoissonLDS()
end

function test_Gradient()
    plds, x, y = toy_PoissonLDS()

    # for each trial check the gradient
    for i in axes(y, 3)
        # numerically calculate the gradient
        f = latents -> StateSpaceDynamics.loglikelihood(latents, plds, y[:, :, i])
        grad_numerical = ForwardDiff.gradient(f, x[:, :, i])

        # calculate the gradient
        grad = StateSpaceDynamics.Gradient(plds, y[:, :, i], x[:, :, i])

        @test norm(grad - grad_numerical) < 1e-8
    end
end

function test_Hessian()
    plds, x, y = toy_PoissonLDS()

    # create function that allows that we cna pass the latents to the loglikelihood transposed
    function log_likelihood(x::AbstractArray, plds, y::AbstractArray)
        return StateSpaceDynamics.loglikelihood(x, plds, y)
    end

    # check hessian for each trial
    for i in axes(y, 3)
        hess, main, super, sub = StateSpaceDynamics.Hessian(
            plds, y[:, 1:3, i], x[:, 1:3, i]
        )
        @test size(hess) == (plds.latent_dim * 3, plds.latent_dim * 3)
        @test size(main) == (3,)
        @test size(super) == (2,)
        @test size(sub) == (2,)

        # calcualte hess using autodiff now
        obj = latents -> log_likelihood(latents, plds, y[:, 1:3, i])
        hess_numerical = ForwardDiff.hessian(obj, x[:, 1:3, i])
        @test norm(hess_numerical - hess) < 1e-8
    end
end

function test_smooth()
    plds, x, y = toy_PoissonLDS()

    # smooth data
    x_smooth, p_smooth, inverseoffdiag = StateSpaceDynamics.smooth(plds, y)

    nTrials = size(y, 3)
    nTsteps = size(y, 2)

    @test size(x_smooth) == size(x)
    @test size(p_smooth) == (plds.latent_dim, plds.latent_dim, nTsteps, nTrials)
    @test size(inverseoffdiag) == (plds.latent_dim, plds.latent_dim, nTsteps, nTrials)

    # test gradient is zero
    for i in axes(y, 3)
        # may as well test the gradient here too 
        f = latents -> StateSpaceDynamics.loglikelihood(latents, plds, y[:, :, i])
        grad_numerical = ForwardDiff.gradient(f, x_smooth[:, :, i])
        grad_analytical = StateSpaceDynamics.Gradient(plds, y[:, :, i], x_smooth[:, :, i])

        @test norm(grad_numerical - grad_analytical) < 1e-7
    end
end

function test_parameter_gradient()
    plds, x, y = toy_PoissonLDS()

    # run estep
    E_z, E_zz, E_zz_prev, x_smooth, P_smooth, elbo = StateSpaceDynamics.estep(plds, y)

    # params
    C, log_d = plds.obs_model.C, plds.obs_model.log_d
    params = vcat(vec(C), log_d)

    # get analytical gradient
    grad_analytical = StateSpaceDynamics.gradient_observation_model!(
        zeros(length(params)), C, log_d, E_z, P_smooth, y
    )

    # get numerical gradient
    function f(params::AbstractVector{<:Real})
        C_size = plds.obs_dim * plds.latent_dim
        log_d = params[(end - plds.obs_dim + 1):end]
        C = reshape(params[1:C_size], plds.obs_dim, plds.latent_dim)
        return -StateSpaceDynamics.Q_observation_model(C, log_d, E_z, P_smooth, y)
    end

    grad = ForwardDiff.gradient(f, params)

    @test isapprox(grad, grad_analytical, rtol=1e-5, atol=1e-5)
end

function test_initial_observation_parameter_updates(ntrials::Int=1)
    plds, x, y = toy_PoissonLDS(ntrials, [true, true, false, false, false, false])

    # run the E_Step
    E_z, E_zz, E_zz_prev, x_smooth, p_smooth, ml_total = StateSpaceDynamics.estep(plds, y)

    # optimize the x0 and p0 entries using autograd
    function obj(x0::AbstractVector, P0_sqrt::AbstractMatrix, plds)
        A, Q = plds.state_model.A, plds.state_model.Q
        P0 = P0_sqrt * P0_sqrt'
        Q_val = 0.0
        for i in axes(E_z, 3)
            Q_val += StateSpaceDynamics.Q_state(
                A, Q, P0, x0, E_z[:, :, i], E_zz[:, :, :, i], E_zz_prev[:, :, :, i]
            )
        end
        return -Q_val
    end

    P0_sqrt = Matrix(cholesky(plds.state_model.P0).U)

    x0_opt =
        optimize(
            x0 -> obj(x0, P0_sqrt, plds),
            plds.state_model.x0,
            LBFGS(),
            Optim.Options(; g_abstol=1e-12),
        ).minimizer
    P0_opt = optimize(P0_ -> obj(x0_opt, P0_, plds), P0_sqrt, LBFGS()).minimizer

    # update the initial state and covariance
    StateSpaceDynamics.mstep!(plds, E_z, E_zz, E_zz_prev, p_smooth, y)

    @test isapprox(plds.state_model.x0, x0_opt, atol=1e-6)
    @test isapprox(plds.state_model.P0, P0_opt * P0_opt', atol=1e-6)
end

function test_state_model_parameter_updates(ntrials::Int=1)
    plds, x, y = toy_PoissonLDS(ntrials, [false, false, true, true, false, false])

    # run the E_Step
    E_z, E_zz, E_zz_prev, x_smooth, p_smooth, ml_total = StateSpaceDynamics.estep(plds, y)

    # optimize the A and Q entries using autograd
    function obj(A::AbstractMatrix, Q_sqrt::AbstractMatrix, plds)
        Q = Q_sqrt * Q_sqrt'
        Q_val = StateSpaceDynamics.Q_state(
            A, Q, plds.state_model.P0, plds.state_model.x0, E_z, E_zz, E_zz_prev
        )
        return -Q_val
    end

    Q_sqrt = Matrix(cholesky(plds.state_model.Q).U)

    A_opt =
        optimize(
            A -> obj(A, Q_sqrt, plds),
            plds.state_model.A,
            LBFGS(),
            Optim.Options(; g_abstol=1e-12),
        ).minimizer
    Q_opt =
        optimize(
            Q_sqrt -> obj(A_opt, Q_sqrt, plds),
            Q_sqrt,
            LBFGS(),
            Optim.Options(; g_abstol=1e-12),
        ).minimizer

    # update the state model
    StateSpaceDynamics.mstep!(plds, E_z, E_zz, E_zz_prev, p_smooth, y)

    @test isapprox(plds.state_model.A, A_opt, atol=1e-6)
    @test isapprox(plds.state_model.Q, Q_opt * Q_opt', atol=1e-6)
end

function test_EM(n_trials::Int=1)
    # generate fake data
    plds, x, y = toy_PoissonLDS(n_trials)

    # create a new plds model with random parameters
    plds_new = PoissonLDS(; obs_dim=3, latent_dim=2)
    elbo, norm_grad = fit!(plds_new, y; max_iter=100)

    # check that the ELBO increases over the whole algorithm, we cannot use monotonicity as a check as we are using Laplace EM.
    @test elbo[end] > elbo[1]
end

function test_EM_matlab()
    # read data used to smooth the results
    data_1 = Matrix(CSV.read("test_data/trial1.csv", DataFrame))
    data_2 = Matrix(CSV.read("test_data/trial2.csv", DataFrame))
    data_3 = Matrix(CSV.read("test_data/trial3.csv", DataFrame))
    y = cat(data_1, data_2, data_3; dims=3)
    y = permutedims(y, [2, 1, 3])
    # read the matlab objects to compare results
    seq = matread("test_data/seq_matlab_3_trials_plds.mat")
    params = matread("test_data/params_matlab_3_trials_plds.mat")
    # create a new plds model, run a single iteration of EM and compare the results
    plds = PoissonLDS(;
        A=[cos(0.1) -sin(0.1); sin(0.1) cos(0.1)],
        C=[1.2 1.2; 1.2 1.2; 1.2 1.2],
        log_d=log.([0.1, 0.1, 0.1]),
        Q=0.00001 * Matrix{Float64}(I(2)),
        P0=0.00001 * Matrix{Float64}(I(2)),
        x0=[1.0, -1.0],
        obs_dim=3,
        latent_dim=2,
        fit_bool=fill(true, 6),
    )
    # first smooth results
    E_z, E_zz, E_zz_prev, x_smooth, p_smooth, ml_total = StateSpaceDynamics.estep(plds, y)
    # check each E_z, E_zz, E_zz_prev are the sample
    for i in 1:3
        posterior_x = seq["seq"]["posterior"][i]["xsm"]
        posterior_cov = seq["seq"]["posterior"][i]["Vsm"]
        posterior_lagged_cov = seq["seq"]["posterior"][i]["VVsm"]

        @test isapprox(E_z[:, :, i], posterior_x, atol=1e-6)

        # TODO: Restructure matlab objects s.t. we can comparse as below
        # @test isapprox(E_zz[:, :, :, i], posterior_cov, atol=1e-6)
        # @test isapprox(E_zz_prev[:, :, :, i], posterior_lagged_cov, atol=1e-6)
    end
    # now test the params
    fit!(plds, y; max_iter=1)
    params_obj = params["params"]["model"]
    @test isapprox(plds.state_model.A, params_obj["A"], atol=1e-5)
    @test isapprox(plds.state_model.Q, params_obj["Q"], atol=1e-5)
    @test isapprox(plds.obs_model.C, params_obj["C"], atol=1e-5)
    @test isapprox(plds.state_model.x0, params_obj["x0"], atol=1e-5)
    @test isapprox(plds.state_model.P0, params_obj["Q0"], atol=1e-5)
    @test isapprox(exp.(plds.obs_model.log_d), params_obj["d"], atol=1e-5)
end
