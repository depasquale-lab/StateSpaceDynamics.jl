# Create a toy example for all LDS tests. This example represents a pendulum in a frictionless environment.
g = 9.81 # gravity
l = 1.0 # length of pendulum
dt = 0.01 # time step
T = 10.0 # total time
# Discrete-time dynamics
A = [1.0 dt; -g/l*dt 1.0]
# Initial state
x0 = [0.0; 1.0]
# Time vector
t = 0:dt:T
# Define the LDS model parameters
H = I(2)  # Observation matrix (assuming direct observation)
Q = 0.00001 * I(2)  # Process noise covariance
observation_noise_std = 0.5
R = (observation_noise_std^2) * I(2)  # Observation noise covariance
p0 = 0.1*I(2)  # Initial state covariance
x0 = [0.0; 1.0]  # Initial state mean
# Generate true data
x = zeros(2, length(t))
x[:,1] = x0
for i = 2:length(t)
    x[:,i] = A*x[:,i-1]
end
# Generate noisy data
x_noisy = zeros(2, length(t))
x_noisy[:, 1] = x0

noise = rand(Normal(0, observation_noise_std), (2, length(t)))

for i in 2:length(t)
    x_noisy[:, i] = A * x[:, i-1] + noise[:, i]
end

function test_LDS_with_params()
    # Create the Kalman filter parameter vector
    kf = LDS(;A=A, H=H, Q=Q, R=R, x0=x0, p0=p0, obs_dim=2, latent_dim=2, fit_bool=Vector([false, false, false, false, false, false, false, false]))
    # confirm parameters are set correctly
    @test kf.A == A
    @test kf.H == H
    @test isapprox(kf.B, zeros(kf.latent_dim, size(kf.inputs, 2)), atol=1e-6)
    @test kf.Q == Q
    @test kf.R == R
    @test kf.x0 == x0
    @test kf.p0 == p0
    @test isapprox(kf.inputs, zeros(1, 1), atol=1e-6)
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == Vector([false, false, false, false, false, false, false, false])
    # run the filter
    x_filt, p_filt, x_pred, p_pred, v, F, K, ml = KalmanFilter(kf, x_noisy')
    # check dimensions
    @test size(x_filt) == (length(t), 2)
    @test size(p_filt) == (length(t), 2, 2)
    @test size(x_pred) == (length(t), 2)
    @test size(p_pred) == (length(t), 2, 2)
    @test size(v) == (length(t), 2)
    @test size(F) == (length(t), 2, 2)
    @test size(K) == (length(t), 2, 2)
    # run the smoother
    x_smooth, p_smooth = KalmanSmoother(kf, x_noisy')
    # check dimensions
    @test size(x_smooth) == (length(t), 2)
    @test size(p_smooth) == (length(t), 2, 2)
end

function test_LDS_without_params()
    # Create the Kalman filter without any params
    kf = LDS(; obs_dim=2, latent_dim=2, fit_bool=Vector([true, true, true, true, true, true, true]))
    # confirm parameters are set correctly
    @test !isempty(kf.A)
    @test !isempty(kf.H)
    @test !isempty(kf.B)
    @test !isempty(kf.Q)
    @test !isempty(kf.R)
    @test !isempty(kf.x0)
    @test !isempty(kf.p0)
    @test !isempty(kf.inputs)
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == fill(true, 7)
end

function test_LDS_E_Step()
    # Create the Kalman filter parameter vector
    kf = LDS(;A=A, H=H, Q=Q, R=R, x0=x0, p0=p0, obs_dim=2, latent_dim=2, fit_bool=Vector([true, true, true, true, true, true, true]))
    # run the E_Step
    x_smooth, p_smooth, E_z, E_zz, E_zz_prev, ml = SSM.E_Step(kf, x_noisy')
    # check dimensions
    @test size(x_smooth) == (length(t), 2)
    @test size(p_smooth) == (length(t), 2, 2)
    @test size(E_z) == (length(t), 2)
    @test size(E_zz) == (length(t), 2, 2)
    @test size(E_zz_prev) == (length(t), 2, 2)
    @test size(ml) == ()
end

function test_LDS_M_Step!()
    # Create the Kalman filter parameter vector
    kf = LDS(;A=A, H=H, Q=Q, R=R, x0=x0, p0=p0, obs_dim=2, latent_dim=2, fit_bool=Vector([true, true, true, true, true, true, true]))
    # run the E_Step
    x_smooth, p_smooth, E_z, E_zz, E_zz_prev, ml = SSM.E_Step(kf, x_noisy')
    # run the M_Step
    SSM.M_Step!(kf, E_z, E_zz, E_zz_prev, x_noisy')
    # check if the parameters are updated
    @test kf.A !== A
    @test kf.H !== H
    @test isapprox(kf.B, zeros(kf.latent_dim, 1))
    @test kf.Q !== Q
    @test kf.R !== R
    @test kf.x0 !== x0
    @test kf.p0 !== p0
    @test isapprox(kf.inputs, zeros(1, 1))
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == Vector([true, true, true, true, true, true, true])
end

function test_LDS_EM()
    kf = LDS(;A=A, H=H, Q=Q, R=R, x0=x0, p0=p0, obs_dim=2, latent_dim=2, fit_bool=Vector([true, true, true, true, true, true, true]))
    # run the EM
    for i in 1:10
        ml_prev = -Inf
        l, ml = SSM.KalmanFilterEM!(kf, x_noisy', 1)
        @test ml > ml_prev
        ml_prev = ml
    end
    # check if the parameters are updated
    @test kf.A !== A
    @test kf.H !== H
    @test isapprox(kf.B, zeros(kf.latent_dim, 1))
    @test kf.Q !== Q
    @test kf.R !== R
    @test kf.x0 !== x0
    @test kf.p0 !== p0
    @test isapprox(kf.inputs, zeros(1, 1))
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == Vector([true, true, true, true, true, true, true]) 
end

function test_direct_smoother()
    # create kalman filter object
    kf = LDS(;A=A, H=H, Q=Q, R=R, x0=x0, p0=p0, obs_dim=2, latent_dim=2, fit_bool=Vector([true, true, true, true, true, true, true]))
    # run the RTS-Smoother
    x_smooth, p_smooth = KalmanSmoother(kf, x_noisy')
    # run the Direct Smoothing algorithm
    x_smooth_direct, p_smooth_direct = KalmanSmoother(kf, permutedims(x_noisy), "Direct")
    @test size(x_smooth) == size(x_smooth_direct)
    @test size(p_smooth) == size(p_smooth_direct)
    # check if the results are the same
    @test isapprox(x_smooth, x_smooth_direct, atol=1e-6)
    @test isapprox(p_smooth, p_smooth_direct, atol=1e-6)
end

function test_LDS_gradient()
    # create kalman filter object
    kf = LDS(;A=A, H=H, Q=Q, R=R, x0=x0, p0=p0, obs_dim=2, latent_dim=2, fit_bool=Vector([true, true, true, true, true, true, true]))
    # calcualte the gradient
    grad = SSM.Gradient(kf, x_noisy', zeros(size(x_noisy')))
    # check dimensions
    @test size(grad) == (length(t), kf.obs_dim)
    # calculate the gradient using autodiff
    obj(x) = x -> SSM.loglikelihood(x, kf, x_noisy')
    grad_auto = ForwardDiff.gradient(obj(x), zeros(size(x_noisy')))
    # check if the gradients are the same
    @test isapprox(grad, grad_auto, atol=1e-6)
end

function test_LDS_Hessian()
    # create kalman filter object
    kf = LDS(;A=A, H=H, Q=Q, R=R, x0=x0, p0=p0, obs_dim=2, latent_dim=2, fit_bool=Vector([true, true, true, true, true, true, true]))
    # calcualte the Hessian
    hess, main, super, sub = SSM.Hessian(kf, x_noisy[:, 1:3]') # only look at first three observations as hessian is expensive to calculate using autodiff

    # check lengths of main, super, and sub diagonals
    @test typeof(main) == Vector{Matrix{Float64}}
    @test typeof(super) == Vector{Matrix{Float64}}
    @test typeof(sub) == Vector{Matrix{Float64}}
    @test length(main) == 3
    @test length(super) == 2
    @test length(sub) == 2

    # check dimensions
    @test size(hess) == (3*kf.obs_dim, 3*kf.obs_dim)

    # calculate the Hessian using autodiff
    function log_likelihood(x::AbstractArray, l::LDS, y::AbstractArray)
        # this wrapper function just makes it so we can pass a D x T array and not a T x D array. Otherwise the Hessian is out of order.
        x = x'
        ll = SSM.loglikelihood(x, l, y)
        return ll  # Negate the log-likelihood
    end
    obj(x) = x -> log_likelihood(x, kf, zeros(size(x_noisy[:, 1:3]')))
    hess_auto = ForwardDiff.hessian(obj(x), zeros(size(x_noisy[:, 1:3])))
    # check if the Hessian are the same
    @test isapprox(Matrix(hess), hess_auto, atol=1e-6)
end