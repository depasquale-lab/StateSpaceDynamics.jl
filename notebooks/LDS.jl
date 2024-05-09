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

# ╔═╡ c3f10cbe-2505-47d3-a3de-340a8b7d63de
md"""# Example: Linear Gaussian Dynamical System""" 

# ╔═╡ 69d63be2-a958-43e6-8bbf-5bdd5da60a8b
begin
	# simulate a linear gaussian dynamical system that we can filter and smooth using the KF and RTS Smoother. For this example we are simulating an observed 2-D oscillator, with a 4-D oscillating state space.

	# set up true system parameters
	oscillating_dynamics = 0.95 .* [cos(0.5) -sin(0.5) 0 0; 
							sin(0.5) cos(0.5) 0 0; 
							0 0 cos(0.25) -sin(0.25);
	                        0 0 sin(0.25) cos(0.25)]
	transition_dynamics = [1 1 0 0; 0.7 0 0.7 0]
	process_noise = 0.1 * I(4)
	measurement_noise = 75 * I(2)

	# pre-allocate arrays 
	x_init = [1, 1, 1, 1] # initial state
	P_init = I(4) # initial variance

	state_dynamics = zeros(4, 1000) # simulate 1000 t-steps
	state_dynamics[:, 1] = x_init

	process_noises = rand(MvNormal([0, 0, 0, 0], sqrt.(process_noise)), 1000)
	for i in 2:1000
		state_dynamics[:, i] = oscillating_dynamics * state_dynamics[:, i-1] + process_noises[:, i]
	end

	# now simulate a set of observations
	observation_dynamics = zeros(2, 1000)
	measurement_noises = rand(MvNormal([0, 0], sqrt.(measurement_noise)), 1000)

	for i in 1:1000
		observation_dynamics[:, i] = transition_dynamics * state_dynamics[:, i]  + measurement_noises[:, i]
	end
	
	# State Dynamics for X_1 and X_2
	state_dyns_12 = plot(state_dynamics[1, :], label="X_1", legend=:topright, xlabel="t", ylabel="state value")
	plot!(state_dyns_12, state_dynamics[2, :], label="X_2")
	
	# State Dynamics for X_3 and X_4
	state_dyns_34 = plot(state_dynamics[3, :], label="X_3", legend=:topright, xlabel="t", ylabel="state value")
	plot!(state_dyns_34, state_dynamics[4, :], label="X_4")
	
	# Phase Plot for X_1/X_2 and X_3/X_4
	phase_plots = plot(state_dynamics[1, :], state_dynamics[2, :], label="Phase 1/2", legend=:topright, xlabel="X_1", ylabel="X_2")
	plot!(phase_plots, state_dynamics[3, :], state_dynamics[4, :], label="Phase 3/4")
	
	# Observations Plot for Y_1 and Y_2
	observations_plot = plot(observation_dynamics[1, :], label="Y_1", legend=:topright, xlabel="t", ylabel="observation value")
	plot!(observations_plot, observation_dynamics[2, :], label="Y_2")
	
	# Combine all plots into a 2x2 layout
	combined_plot = plot(state_dyns_12, state_dyns_34, phase_plots, observations_plot, layout=(2, 2), size=(800, 600))
end

	

# ╔═╡ aba3f47d-4839-48e7-9f97-cb85747753c9
begin
	# lets now see how we can recover the dynamics using the kalman filter.
	lds = SSM.LDS(oscillating_dynamics, transition_dynamics, nothing, process_noise, measurement_noise, x_init, P_init, nothing, 2, 4, ones(7))

	# now lets filter
	x_filt, p_filt, x_pred, p_pred, v, F, K, ll = SSM.KalmanFilter(lds, observation_dynamics')
	
	# now smooth
	x_smooth, p_smooth = SSM.KalmanSmoother(lds, observation_dynamics')

	# # plot the results
	plot(state_dynamics[1, :])
	plot!(x_filt[:, 1])
	plot!(x_smooth[:, 1])
	# # plot!(x_filt[:, 3])
	# # plot!(x_filt[:, 4])

	
end

# ╔═╡ Cell order:
# ╠═53c96240-c044-11ee-2f83-e93206e06ac6
# ╟─c3f10cbe-2505-47d3-a3de-340a8b7d63de
# ╠═69d63be2-a958-43e6-8bbf-5bdd5da60a8b
# ╠═aba3f47d-4839-48e7-9f97-cb85747753c9
