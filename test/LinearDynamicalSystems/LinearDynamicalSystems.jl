# Define the parameters of a penndulum
g = 9.81 # gravity
l = 1.0 # length of pendulum
dt = 0.01 # time step

# Discrete-time dynamics
A = [1.0 dt; -g/l*dt 1.0]
Q = Matrix{Float64}(0.00001 * I(2))  # Process noise covariance

# Initial state/ covariance
x0 = [0.0; 1.0]
P0 = Matrix{Float64}(0.1*I(2))  # Initial state covariance

# Observation params 
C = Matrix{Float64}(I(2))  # Observation matrix (assuming direct observation)
observation_noise_std = 0.5
R = Matrix{Float64}((observation_noise_std^2) * I(2))  # Observation noise covariance

function toy_lds()
    lds = GaussianLDS(;A=A, C=C, Q=Q, R=R, x0=x0, P0=P0, obs_dim=2, latent_dim=2)

    # sample data
    T = 100
    x, y = SSM.sample(lds, T, 1) # 100 timepoints, 1 trials

    return lds, x, y
end

function test_lds_properties(lds)
    # check state and observation model are of correct type
    @test isa(lds.state_model, GaussianStateModel)
    @test isa(lds.obs_model, GaussianObservationModel)
    @test isa(lds, LinearDynamicalSystem)

    # check model param dimensions
    @test size(lds.state_model.A) == (lds.latent_dim,  lds.latent_dim)
    @test size(lds.obs_model.C) == (lds.obs_dim, lds.latent_dim)
    @test size(lds.state_model.Q) == (lds.latent_dim, lds.latent_dim)
    @test size(lds.obs_model.R) == (lds.obs_dim, lds.obs_dim)
    @test size(lds.state_model.x0) == (lds.latent_dim,)
    @test size(lds.state_model.P0) == (lds.latent_dim, lds.latent_dim)

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
    lds = GaussianLDS(obs_dim=2, latent_dim=2)
    test_lds_properties(lds)

    @test !isempty(lds.state_model.A)
    @test !isempty(lds.state_model.Q)
    @test !isempty(lds.obs_model.C)
    @test !isempty(lds.obs_model.R)
    @test !isempty(lds.state_model.x0)
    @test !isempty(lds.state_model.P0)
end

function test_Gradient()
    lds, x, y = toy_lds()
    # for each trial check the gradient
    for i in 1:size(y:1)
        grad_numerical = ForwardDiff.gradient(x -> SSM.loglikelihood(lds, y[i, :, :], x), x[i, :, :])
        grad_analytical = SSM.Gradient(lds, y[i, :, :], x[i, :, :])
        @test norm(grad_numerical - grad_analytical) < 1e-12
    end
end

function test_smooth()
    lds, x, y = toy_lds()
    x_smooth, p_smooth, inverseoffdiag = smooth(lds, y)

    @test size(x_smooth) == size(x)
    @test size(p_smooth) == (lds.latent_dim, lds.latent_dim, size(x, 1))
    @test size(inverseoffdiag) == (lds.latent_dim, lds.latent_dim, size(x, 1))

    # test gradient is zero
    @test norm(SSM.Gradient(lds, y, x_smooth)) < 1e-12

end






