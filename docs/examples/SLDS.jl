# ## Simulating and Fitting a Switching Linear Dynamical System

# ## Load Packages
using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using LaTeXStrings
using Statistics
using StableRNGs

rng = StableRNG(123);

# ## Simulate Data from an SLDS model
state_dim = 2
obs_dim = 10
K = 2 ## two states

## Create the HMM parameters
A_hmm = [0.92 0.08; 0.06 0.94]
π₀ = [1.0, 0.0]

## Create the state models
A₁ = 0.95 * [cos(0.05) -sin(0.05); sin(0.05) cos(0.05)] ## slower oscillator
A₂ = 0.95 * [cos(0.55) -sin(0.55); sin(0.55) cos(0.55)] ## faster oscillator

Q₁ = [0.001 0.0; 0.0 0.001]
Q₂ = [0.1 0.0; 0.0 0.1]

## Assume same initial distribution for ease
x0 = [0.0, 0.0]
P0 = [0.1 0.0; 0.0 0.1]

## create the observation models
C₁ = randn(rng, obs_dim, state_dim)
C₂ = randn(rng, obs_dim, state_dim)

R = Matrix(0.1 * I(obs_dim)) ## Assume same noise covariance for both states

## Put it all together for an SLDS model
model = SwitchingLinearDynamicalSystem(
    A_hmm,
    [LinearDynamicalSystem(GaussianStateModel(A₁, Q₁, x0, P0), GaussianObservationModel(C₁, R), state_dim, obs_dim, fill(true, 6)), 
     LinearDynamicalSystem(GaussianStateModel(A₂, Q₂, x0, P0), GaussianObservationModel(C₂, R), state_dim, obs_dim, fill(true, 6))],
     π₀,
    K)

## Simulate data
T = 1000
x, y, z = rand(rng, model, 1000)

# ## Plot the true dynamics
p1 = plot(1:T, x[1, :], label="x₁",  linewidth=1.5)
plot!(1:T, x[2, :], label="x₂", linewidth=1.5)

## Create a background shading based on the state (z)
## Find state transition points
transition_points = [1; findall(diff(z) .!= 0) .+ 1; T + 1]

for i in 1:(length(transition_points) - 1)
    start_idx = transition_points[i]
    end_idx = transition_points[i + 1] - 1
    state_value = z[start_idx]
    
    ## Choose color based on state value
    bg_color = state_value == 1 ? :lightblue : :lightyellow
    
    ## Add a background shading for this state region
    vspan!([start_idx, end_idx], fillalpha=0.5, color=bg_color, 
           label=(i == 1 ? "State $state_value" : ""))
end

## Adjust the plot appearance
title!("Latent Dynamics with State")
xlabel!("Time")
ylabel!("State Value")
ylims!(-3, 3)

## Adjust the plot appearance
title!("Latent Dynamics with State")
xlabel!("Time")
ylabel!("State Value")
ylims!(-3, 3)

p1

# ## Create a new SLDS model with different parameters and fit to the data

## Create a model to start with for EM, using reasonable guesses
A = [0.9 0.1; 0.1 0.9]
A ./= sum(A, dims=2) ## Normalize rows to sum to 1

πₖ = rand(K)
πₖ ./= sum(πₖ) ## Normalize to sum to 1

Q = Matrix(0.001 * I(state_dim))

x0 = [0.0; 0.0]
P0 = Matrix(0.001 * I(state_dim))

## set up the observation parameters
C = randn(obs_dim, state_dim)
R = Matrix(0.1 * I(obs_dim))

B = [StateSpaceDynamics.LinearDynamicalSystem(
    StateSpaceDynamics.GaussianStateModel(0.95 * [cos(f) -sin(f); sin(f) cos(f)], Q, x0, P0),
    StateSpaceDynamics.GaussianObservationModel(C, R),
    state_dim, obs_dim, fill(true, 6)) for (i,f) in zip(1:K, [0.7, 0.1])]

learned_model = SwitchingLinearDynamicalSystem(A, B, πₖ, model.K)

# ## Fit the model to the data

mls, param_diff, FB, FS = fit!(learned_model, y; max_iter=25) # use 25 iterations of EM

# ## Plot the ELBO over iterations
plot(mls, label="ELBO", linewidth=1.5)
xlabel!("Iteration")
ylabel!("ELBO")

# ## Compare the true and learned model

## Plot the latent states as a weighted function of the responsibilities for each state

latents = zeros(state_dim, T)  # Initialize with state dimension, not K
resp = exp.(FB.γ)  # Responsibilities (probabilities) for each state at each time

## For each time point, compute the weighted average of the smoothed states
for t in 1:T
    for k in 1:K
        latents[:, t] += FS[k].x_smooth[:, t] .* resp[k, t]
    end
end


## Plot the learned latent states on top of the original with improved styling
plt = plot(size=(800, 500), background_color=:white, margin=5Plots.mm)

## Plot true values
plot!(x[1, :] .+ 2, label="x₁ (True)", linewidth=2, color=:black, alpha=0.8)
plot!(x[2, :] .- 2, label="x₂ (True)", linewidth=2, color=:black, alpha=0.8)

## Plot learned values
plot!(latents[1, :] .+ 2, label="x₁ (Learned)", linewidth=1.5, color=:firebrick)
plot!(latents[2, :] .- 2, label="x₂ (Learned)", linewidth=1.5, color=:royalblue)

## Improve styling
title!("SLDS: True vs Learned Latent States")
xlabel!("Time")
ylabel!("")  # Remove the default y label

## Custom y-ticks with state labels at the correct positions
yticks!([-2, 2], ["x₂", "x₁"])  # Set custom tick positions and labels

## Add horizontal lines to emphasize the state positions
hline!([2], color=:gray, alpha=0.3, linestyle=:dash, label="")
hline!([-2], color=:gray, alpha=0.3, linestyle=:dash, label="")

xlims!(0, T)