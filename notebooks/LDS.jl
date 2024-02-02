### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 53c96240-c044-11ee-2f83-e93206e06ac6
begin
	using Pkg
	Pkg.activate("/Users/ryansenne/Documents/Github/SSM")
	include("/Users/ryansenne/Documents/Github/SSM//src/SSM.jl")
	using .SSM
	using LinearAlgebra
	using Distributions
	using Random
	using Plots
end

# ╔═╡ 67829ab6-3cc9-4d57-b3a4-6ff2df6ac473
begin
	g = 9.81 # gravity
	l = 1.0 # length of pendulum
	dt = 0.001 # time step
	T = 10.0 # total time
	# Discrete-time dynamics
	A = [1.0 dt; -g/l*dt 1.0]
	# Initial state
	x0 = [0.0; 1.0]
	# Time vector
	t = 0:dt:T
	# Generate data
	x = zeros(2, length(t))
	x[:,1] = x0
	for i = 2:length(t)
    	x[:,i] = A*x[:,i-1]
	end
	# Plot data 
	plot(t, x[1,:], label="Position")
	plot!(t, x[2,:], label="Velocity")
end

# ╔═╡ fc4bab31-4866-46a6-bdec-7596db2e40ed
begin
	# Now lets add noise to the system and see if we can recover the dynamics
	# Add driving Gaussian noise to simulate observations
	x_noisy = zeros(2, length(t))
	x_noisy[:, 1] = x0
	observation_noise_std = 0.5

	noise = rand(Normal(0, observation_noise_std), (2, length(t)))

	for i in 2:length(t)
    	x_noisy[:, i] = A * x[:, i-1] + noise[:, i]
	end

	# Define the LDS model parameters
	H = I(2)  # Observation matrix (assuming direct observation)
	Q = 1e-8* I(2)  # Process noise covariance
	R = 0.25 * I(2)  # Observation noise covariance
	P0 = 1e-2 * I(2)  # Initial state covariance
	plot(t, x_noisy[1, :], label="Noisy Position")
	plot!(t, x_noisy[2, :], label="Noisy Velocity")
end

# ╔═╡ 444b013b-0f64-46d1-ab84-28b44ff998a3
begin
	# Create the Kalman filter parameter vector
	kf = SSM.LDS(A, 
			Matrix{Float64}(H), 
			nothing, 
			Matrix{Float64}(Q), 
			Matrix{Float64}(R), 
			x0, 
			Matrix{Float64}(P0), 
			nothing, 
			2, 
			2, 
			"Gaussian", 
			Vector([true, true, false, true, true, true, true, false]))

	# Run the Kalman filter
	x_filt, p_filt, x_pred, p_pred, v, F, K, ll = SSM.KalmanFilter(kf, x_noisy')
	
	# Initialize the standard deviations array.
	num_states = size(p_filt, 2)  # Number of states.
	num_steps = size(p_filt, 1)   # Number of time steps.
	std_devs = zeros(num_states, num_steps)

	# Extract standard deviations from each covariance matrix 'page'
	for i = 1:num_steps
    	for j = 1:num_states
        	std_devs[j, i] = sqrt(abs(p_filt[i, j, j]))  # Assuming the diagonal elements represent variances for each state.
    	end
	end

	# Calculate the 95% confidence interval bounds.
	z_score = 1.96  # for 95% confidence
	upper_bound = x_filt .+ (z_score .* std_devs)'
	lower_bound = x_filt .- (z_score .* std_devs)'

	# Time vector for plotting (assuming sequential time steps starting from 1)
	tt = 1:num_steps

	# Plot the state estimates.
	p = plot(tt, x_filt[:, 1], ribbon = (upper_bound[:, 1] - x_filt[:, 1], x_filt[:, 1] - lower_bound[:, 1]), fillalpha=0.1, label="State 1 with CI")
	plot!(p, tt, x_filt[:, 2], ribbon = (upper_bound[:, 2] - x_filt[:, 2], x_filt[:, 2] - lower_bound[:, 2]), fillalpha=0.1, label="State 2 with CI")

	# Set labels
	xlabel!(p, "Time")
	ylabel!(p, "State Estimate")
end

# ╔═╡ Cell order:
# ╠═53c96240-c044-11ee-2f83-e93206e06ac6
# ╠═67829ab6-3cc9-4d57-b3a4-6ff2df6ac473
# ╠═fc4bab31-4866-46a6-bdec-7596db2e40ed
# ╠═444b013b-0f64-46d1-ab84-28b44ff998a3
