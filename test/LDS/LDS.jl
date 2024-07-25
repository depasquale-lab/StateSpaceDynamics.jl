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
    kf = LDS(A,
             H,
             nothing,
             Q, 
             R, 
             x0, 
             p0, 
             nothing, 
             2, 
             2, 
             Vector([false, false, false, false, false, false, false, false]))
    # confirm parameters are set correctly
    @test kf.A == A
    @test kf.H == H
    @test kf.B === nothing
    @test kf.Q == Q
    @test kf.R == R
    @test kf.x0 == x0
    @test kf.p0 == p0
    @test kf.inputs === nothing
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
    kf = LDS()
    # confirm parameters are set correctly
    @test kf.A !== nothing
    @test kf.H !== nothing
    @test kf.B === nothing
    @test kf.Q !== nothing
    @test kf.R !== nothing
    @test kf.x0 !== nothing
    @test kf.p0 !== nothing
    @test kf.inputs === nothing
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == fill(true, 7)
end

function test_LDS_EStep()
    # Create the Kalman filter parameter vector
    kf = LDS(A,
             H,
             nothing,
             Q, 
             R, 
             x0, 
             p0, 
             nothing, 
             2, 
             2, 
             Vector([true, true, true, true, true, true, true]))
    # run the EStep
    x_smooth, p_smooth, E_z, E_zz, E_zz_prev, ml = SSM.EStep(kf, x_noisy')
    # check dimensions
    @test size(x_smooth) == (length(t), 2)
    @test size(p_smooth) == (length(t), 2, 2)
    @test size(E_z) == (length(t), 2)
    @test size(E_zz) == (length(t), 2, 2)
    @test size(E_zz_prev) == (length(t), 2, 2)
    @test size(ml) == ()
end

function test_LDS_MStep!()
    # Create the Kalman filter parameter vector
    kf = LDS(A,
             H,
             nothing,
             Q, 
             R, 
             x0, 
             p0, 
             nothing, 
             2, 
             2, 
             Vector([true, true, true, true, true, true, true]))
    # run the EStep
    x_smooth, p_smooth, E_z, E_zz, E_zz_prev, ml = SSM.EStep(kf, x_noisy')
    # run the MStep
    SSM.MStep!(kf, E_z, E_zz, E_zz_prev, x_noisy')
    # check if the parameters are updated
    @test kf.A !== A
    @test kf.H !== H
    @test kf.B === nothing
    @test kf.Q !== Q
    @test kf.R !== R
    @test kf.x0 !== x0
    @test kf.p0 !== p0
    @test kf.inputs === nothing
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == Vector([true, true, true, true, true, true, true])
end

function test_LDS_EM()
    kf = LDS(A,
             H,
             nothing,
             Q, 
             R, 
             x0, 
             p0, 
             nothing, 
             2, 
             2, 
             Vector([true, true, true, true, true, true, true]))
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
    @test kf.B === nothing
    @test kf.Q !== Q
    @test kf.R !== R
    @test kf.x0 !== x0
    @test kf.p0 !== p0
    @test kf.inputs === nothing
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == Vector([true, true, true, true, true, true, true]) 
end