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
    x, y = StateSpaceDynamics.sample(lds, T, ntrials) # 100 timepoints, 1 trials

    return lds, x, y
end

#all(isapprox.(ForwardDiff.hessian(x-> StateSpaceDynamics.loglikelihood(x, lds, y, w), x),
#   Matrix(StateSpaceDynamics.Hessian(lds, y, x, w)[1])))

#all(isapprox.(ForwardDiff.gradient(x-> StateSpaceDynamics.loglikelihood(x, lds, y, w), x),
#    StateSpaceDynamics.Gradient(lds, y, x, w)))

function test_Gradient()
    lds, x, y = toy_lds()

    T = size(y, 2)
    w = rand(T)

    # for each trial check the gradient
    for i in axes(y, 3)

        # numerically calculate the gradient
        f = latents -> StateSpaceDynamics.loglikelihood(latents, lds, y[:, :, i], w)
        grad_numerical = ForwardDiff.gradient(f, x[:, :, i])

        # analytical gradient
        grad_analytical = StateSpaceDynamics.Gradient(lds, y[:, :, i], x[:, :, i], w)
        @test norm(grad_numerical - grad_analytical) < 1e-8
    end
end

function test_Hessian()
    lds, x, y = toy_lds()

    T = size(y, 2)
    w = rand(T)

    function log_likelihood(x::AbstractArray, lds, y::AbstractArray, w)
        return StateSpaceDynamics.loglikelihood(x, lds, y, w)
    end

    # for each trial check the Hessian
    for i in axes(y, 3)
        hess, main, super, sub = StateSpaceDynamics.Hessian(lds, y[:, 1:3, i], x[:, 1:3, i], w)
        @test size(hess) == (3 * lds.latent_dim, 3 * lds.latent_dim)
        @test size(main) == (3,)
        @test size(super) == (2,)
        @test size(sub) == (2,)

        # calculate the Hessian using autodiff
        obj = latents -> log_likelihood(latents, lds, y[:, 1:3, i], w)
        hess_numerical = ForwardDiff.hessian(obj, x[:, 1:3, i], w)

        @test norm(hess_numerical - hess) < 1e-8
    end
end

