# define parameters for a PoissonLDS
x0 = [1.0, -1.0]
p0 = Matrix(Diagonal([0.1, 0.1]))
A = [cos(0.1) -sin(0.1); sin(0.1) cos(0.1)]
Q = Matrix(Diagonal([0.1, 0.1]))
C = [0.6 0.6; 0.6 0.6; 0.6 0.6] .* 2
log_d = log.([0.1, 0.1, 0.1])

function toy_PoissonLDS(n_trials::Int=1, fit_bool::Vector{Bool}=[true, true, true, true, true, true])
    poisson_lds = PoissonLDS(;A=A, C=C, Q=Q, x0=x0, P0=p0, log_d=log_d, obs_dim=3, latent_dim=2, fit_bool=fit_bool)

    # sample data
    T = 100
    x, y = StateSpaceDynamics.sample(poisson_lds, T, n_trials) # 100 timepoints, ntrials

    return poisson_lds, x, y
end

function test_plds_properties(poisson_lds)
    @test isa(poisson_lds.state_model, StateSpaceDynamics.GaussianStateModel)
    @test isa(poisson_lds.obs_model, StateSpaceDynamics.PoissonObservationModel)
    @test isa(poisson_lds, StateSpaceDynamics.LinearDynamicalSystem)

    @test size(poisson_lds.state_model.A) == (poisson_lds.latent_dim,  poisson_lds.latent_dim)
    @test size(poisson_lds.obs_model.C) == (poisson_lds.obs_dim, poisson_lds.latent_dim)
    @test size(poisson_lds.state_model.Q) == (poisson_lds.latent_dim, poisson_lds.latent_dim)
    @test size(poisson_lds.state_model.x0) == (poisson_lds.latent_dim,)
    @test size(poisson_lds.state_model.P0) == (poisson_lds.latent_dim, poisson_lds.latent_dim)
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
    poisson_lds = PoissonLDS(obs_dim=3, latent_dim=2)
    test_plds_properties(poisson_lds)

    @test !isempty(poisson_lds.state_model.A)
    @test !isempty(poisson_lds.state_model.Q)
    @test !isempty(poisson_lds.obs_model.C)
    @test !isempty(poisson_lds.state_model.x0)
    @test !isempty(poisson_lds.state_model.P0)
    @test !isempty(poisson_lds.obs_model.log_d)
end

function test_Gradient()
    plds, x, y = toy_PoissonLDS()


    # for each trial check the gradient
    for i in axes(y, 1)
        # numerically calculate the gradient
        f = latents -> StateSpaceDynamics.loglikelihood(latents, plds, y[i, :, :])
        grad_numerical = ForwardDiff.gradient(f, x[i, :, :])

        # calculate the gradient
        grad = StateSpaceDynamics.Gradient(plds, y[i, :, :], x[i, :, :])

        @test norm(grad - grad_numerical) < 1e-9
    end
end

function test_Hessian()
    plds, x, y = toy_PoissonLDS()

    # create function that allows that we cna pass the latents to the loglikelihood transposed
    function log_likelihood(x::AbstractArray, plds, y::AbstractArray)
        x = permutedims(x)
        return StateSpaceDynamics.loglikelihood(x, plds, y)
    end
    
        # check hessian for each trial
        for i in axes(y, 1)
            hess, main, super, sub = StateSpaceDynamics.Hessian(plds, y[i, 1:3, :], x[i, 1:3, :])
            @test size(hess) == (plds.latent_dim * 3, plds.latent_dim * 3)
            @test size(main) == (3,)
            @test size(super) == (2,)
            @test size(sub) == (2,)

            # calcualte hess using autodiff now
            obj = latents -> log_likelihood(latents, plds, y[i, 1:3, :])
            hess_numerical = ForwardDiff.hessian(obj, permutedims(x[i, 1:3, :]))
            @test norm(hess_numerical - hess) < 1e-9
        end
    end

function test_smooth()
    plds, x, y = toy_PoissonLDS()

    # smooth data
    x_smooth, p_smooth, inverseoffdiag = StateSpaceDynamics.smooth(plds, y)

    nTrials = size(y, 1)
    nTsteps = size(y, 2)

    @test size(x_smooth) == size(x)
    @test size(p_smooth) == (nTrials, nTsteps, plds.latent_dim, plds.latent_dim)
    @test size(inverseoffdiag) == (nTrials, nTsteps, plds.latent_dim, plds.latent_dim)

    # test gradient is zero
    for i in axes(y, 1)
        # may as well test the gradient here too 
        f = latents -> StateSpaceDynamics.loglikelihood(latents, plds, y[i, :, :])
        grad_numerical = ForwardDiff.gradient(f, x_smooth[i, :, :])
        grad_analytical = StateSpaceDynamics.Gradient(plds, y[i, :, :], x_smooth[i, :, :])

        @test norm(grad_numerical - grad_analytical) < 1e-10
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
    grad_analytical = StateSpaceDynamics.gradient_observation_model!(zeros(length(params)), C, log_d, E_z, P_smooth, y)

    # get numerical gradient
    function f(params::AbstractVector{<:Real})
        C_size = plds.obs_dim * plds.latent_dim
        log_d = params[end-plds.obs_dim+1:end]
        C = reshape(params[1:C_size], plds.obs_dim, plds.latent_dim)
        return -StateSpaceDynamics.Q_observation_model(C, log_d, E_z, P_smooth, y)
    end

    
    grad = ForwardDiff.gradient(f, params)



    @test isapprox(grad, grad_analytical, rtol=1e-5, atol=1e-5)

end