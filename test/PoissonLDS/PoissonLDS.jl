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
    x, y = SSM.sample(poisson_lds, T, n_trials) # 100 timepoints, ntrials

    return poisson_lds, x, y
end

function test_plds_properties(poisson_lds)
    @test isa(poisson_lds.state_model, SSM.GaussianStateModel)
    @test isa(poisson_lds.obs_model, SSM.PoissonObservationModel)
    @test isa(poisson_lds, SSM.LinearDynamicalSystem)

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
        f = latents -> SSM.loglikelihood(latents, plds, y[i, :, :])
        grad_numerical = ForwardDiff.gradient(f, x[i, :, :])

        # calculate the gradient
        grad = SSM.Gradient(plds, y[i, :, :], x[i, :, :])

        @test norm(grad - grad_numerical) < 1e-10
    end
end