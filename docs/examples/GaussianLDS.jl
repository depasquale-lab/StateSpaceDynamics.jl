# ## Simulating and Fitting a Linear Dynamical System

# This tutorial demonstrates how to use `StateSpaceDynamics.jl` to simulate a latent
# linear dynamical system and fit it using the EM algorithm. We'll walk through the
# complete workflow: defining a true model, generating synthetic data, initializing
# a naive model, and then learning the parameters through iterative optimization.

# ## Load Required Packages

# We begin by loading all the necessary packages for our analysis. StateSpaceDynamics.jl
# provides the core functionality, while the other packages handle linear algebra,
# random number generation, plotting, and mathematical notation.

using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using LaTeXStrings
using StableRNGs

# Set a stable random number generator for reproducible results
rng = StableRNG(123);

# ## Create a State-Space Model

# We start by defining the dimensions of our system. A linear dynamical system (LDS)
# models how a low-dimensional latent state evolves over time and generates high-dimensional
# observations. Here we use a 2D latent space (which we can visualize easily) that
# generates 10-dimensional observations.

obs_dim = 10      # Number of observed variables at each time step
latent_dim = 2    # Number of latent state variables

# Define the state transition matrix A. This matrix governs how the latent state
# evolves from one time step to the next: x_{t+1} = A * x_t + noise.
# We create a rotation matrix scaled by 0.95, which creates a stable spiral
# dynamic that slowly contracts toward the origin.
A = 0.95 * [cos(0.25) -sin(0.25); sin(0.25) cos(0.25)]

# Process noise covariance Q controls how much random variation we add to the
# latent state transitions. A smaller Q means more predictable dynamics.
Q = Matrix(0.1 * I(2))

# Initial state parameters: where the latent trajectory starts and how uncertain
# we are about this initial position.
x0 = [0.0; 0.0]           # Mean of initial state
P0 = Matrix(0.1 * I(2))   # Covariance of initial state

# Observation parameters: how the latent states map to observed data.
# C is the observation matrix (latent-to-observed mapping), and R is the
# observation noise covariance.
C = randn(rng, obs_dim, latent_dim)  # Random linear mapping from 2D latent to 10D observed
R = Matrix(0.5 * I(obs_dim))         # Independent noise on each observation dimension

# Construct the state and observation model components
true_gaussian_sm = GaussianStateModel(;A=A, Q=Q, x0=x0, P0=P0)
true_gaussian_om = GaussianObservationModel(;C=C, R=R)

# Combine them into a complete Linear Dynamical System
# The fit_bool parameter indicates which parameters should be learned during fitting
true_lds = LinearDynamicalSystem(;
    state_model=true_gaussian_sm,
    obs_model=true_gaussian_om,
    latent_dim=latent_dim,
    obs_dim=obs_dim,
    fit_bool=fill(true, 6)  # Fit all 6 parameter matrices: A, Q, C, R, x0, P0
)

# ## Simulate Latent and Observed Data

# Now we generate synthetic data from our true model. This gives us both the
# latent states (which we'll later try to recover) and the observations (which
# is all a real algorithm would see).

tSteps = 500  # Number of time points to simulate

# The rand function generates both latent trajectories and corresponding observations
latents, observations = rand(rng, true_lds; tsteps=tSteps, ntrials=1)

# ## Plot Vector Field of Latent Dynamics

# To better understand the dynamics encoded by our transition matrix A, we'll
# create a vector field plot. This shows how the latent state would evolve
# from any starting point in the 2D latent space.

# Create a grid of starting points
x = y = -3:0.5:3
X = repeat(x', length(y), 1)
Y = repeat(y, 1, length(x))

# Calculate the flow field: at each point (x,y), compute where it would move
# in one time step under the dynamics x_{t+1} = A * x_t
U = zeros(size(X))  # x-component of flow
V = zeros(size(Y))  # y-component of flow

for i in 1:size(X, 1)
    for j in 1:size(X, 2)
        # Apply dynamics and compute the displacement
        v = A * [X[i,j], Y[i,j]]
        U[i,j] = v[1] - X[i,j]  # Change in x
        V[i,j] = v[2] - Y[i,j]  # Change in y
    end
end

# Normalize arrows for cleaner visualization
magnitude = @. sqrt(U^2 + V^2)
U_norm = U ./ magnitude
V_norm = V ./ magnitude

# Create the vector field plot with the actual trajectory overlaid
p = quiver(X, Y, quiver=(U_norm, V_norm), color=:blue, alpha=0.3,
           linewidth=1, arrow=arrow(:closed, :head, 0.1, 0.1))
plot!(latents[1, :, 1], latents[2, :, 1], xlabel="x₁", ylabel="x₂",
      color=:black, linewidth=1.5, title="Latent Dynamics", legend=false)

# ## Plot Latent States and Observations

# Let's visualize both the latent states (which evolve smoothly according to our
# dynamics) and the observations (which are noisy linear combinations of the latents).

states = latents[:, :, 1]      # Extract the latent trajectory
emissions = observations[:, :, 1]  # Extract the observed data

# Create a two-panel plot: latent states on top, observations below
plot(size=(800, 600), layout=@layout[a{0.3h}; b])

# Plot latent states (offset vertically for clarity)
lim_states = maximum(abs.(states))
for d in 1:latent_dim
    plot!(1:tSteps, states[d, :] .+ lim_states * (d-1), color=:black,
          linewidth=2, label="", subplot=1)
end

plot!(subplot=1, yticks=(lim_states .* (0:latent_dim-1), [L"x_%$d" for d in 1:latent_dim]),
      xticks=[], xlims=(0, tSteps), title="Simulated Latent States",
      yformatter=y->"", tickfontsize=12)

# Plot observations (also offset vertically since there are many dimensions)
lim_emissions = maximum(abs.(emissions))
for n in 1:obs_dim
    plot!(1:tSteps, emissions[n, :] .- lim_emissions * (n-1), color=:black,
          linewidth=2, label="", subplot=2)
end

plot!(subplot=2, yticks=(-lim_emissions .* (obs_dim-1:-1:0), [L"y_{%$n}" for n in 1:obs_dim]),
      xlabel="time", xlims=(0, tSteps), title="Simulated Emissions",
      yformatter=y->"", tickfontsize=12)

plot!(link=:x, size=(800, 600), left_margin=10Plots.mm)

# ## Initialize a Model and Perform Smoothing

# In a real scenario, we would only observe the emissions, not the latent states.
# Our goal is to learn the parameters A, Q, C, R from the observations alone.
# We start by creating a "naive" model with random initial parameters.

# Initialize with random parameters (this simulates not knowing the true system)
A_init = random_rotation_matrix(2, rng)    # Random rotation matrix for dynamics
Q_init = Matrix(0.1 * I(2))                # Same process noise variance (could be random too)
C_init = randn(rng, obs_dim, latent_dim)   # Random observation mapping
R_init = Matrix(0.5 * I(obs_dim))          # Same observation noise (could vary)
x0_init = zeros(latent_dim)                # Start from origin
P0_init = Matrix(0.1 * I(latent_dim))      # Same initial uncertainty

# Create the naive model components
gaussian_sm_init = GaussianStateModel(;A=A_init, Q=Q_init, x0=x0_init, P0=P0_init)
gaussian_om_init = GaussianObservationModel(;C=C_init, R=R_init)

# Assemble the complete naive system
naive_ssm = LinearDynamicalSystem(;
    state_model=gaussian_sm_init,
    obs_model=gaussian_om_init,
    latent_dim=latent_dim,
    obs_dim=obs_dim,
    fit_bool=fill(true, 6)  # We'll learn all parameters
)

# Before fitting, let's see how well our randomly initialized model can
# infer the latent states. We use the "smoothing" algorithm, which estimates
# the latent states given all observations (past, present, and future).
x_smooth, _, _ = StateSpaceDynamics.smooth(naive_ssm, observations)

# Plot the true latent states vs. our initial (poor) estimates
plot()
for d in 1:latent_dim
    plot!(1:tSteps, states[d, :] .+ lim_states * (d-1), color=:black, linewidth=2, label="", subplot=1)
    plot!(1:tSteps, x_smooth[d, :, 1] .+ lim_states * (d-1), color=:firebrick, linewidth=2, label="", subplot=1)
end
plot!(subplot=1, yticks=(lim_states .* (0:latent_dim-1), [L"x_%$d" for d in 1:latent_dim]),
      xticks=[], xlims=(0, tSteps), yformatter=y->"", tickfontsize=12,
      title="True vs. Predicted Latent States (Pre-EM)")

# ## Fit Model Using EM Algorithm

# Now comes the crucial step: parameter learning via the Expectation-Maximization (EM)
# algorithm. EM alternates between two steps:
# 1. E-step: Estimate latent states given current parameters
# 2. M-step: Update parameters given current state estimates
# This process iteratively improves both the parameter estimates and state inferences.

println("Starting EM algorithm to learn parameters...")
elbo, _ = fit!(naive_ssm, observations; max_iter=100, tol=1e-6)

# After EM has converged, let's see how much better our latent state estimates are
x_smooth, _, _ = StateSpaceDynamics.smooth(naive_ssm, observations)

# Plot the results: true states vs. post-EM estimates
plot()
for d in 1:latent_dim
    plot!(1:tSteps, states[d, :] .+ lim_states * (d-1), color=:black, linewidth=2, label="", subplot=1)
    plot!(1:tSteps, x_smooth[d, :, 1] .+ lim_states * (d-1), color=:firebrick, linewidth=2, label="", subplot=1)
end
plot!(subplot=1, yticks=(lim_states .* (0:latent_dim-1), [L"x_%$d" for d in 1:latent_dim]),
      xticks=[], xlims=(0, tSteps), yformatter=y->"", tickfontsize=12,
      title="True vs. Predicted Latent States (Post-EM)")

# ## Confirm Model Convergence

# The Evidence Lower BOund (ELBO) is a measure of how well our model explains
# the data. In EM, this should increase monotonically and plateau when the
# algorithm has converged to a local optimum.

plot(elbo, xlabel="iteration", ylabel="ELBO", title="ELBO (Marginal Loglikelihood)", legend=false)

println("EM converged after $(length(elbo)) iterations")
println("Final ELBO: $(elbo[end])")

# ## Summary
# 
# This tutorial demonstrated the complete workflow for fitting a Linear Dynamical System:
# 1. We defined a true LDS with known parameters and generated synthetic data
# 2. We initialized a naive model with random parameters  
# 3. We used EM to iteratively improve our parameter estimates
# 4. We visualized how the latent state inference improved after learning
# 
# The EM algorithm successfully recovered the underlying dynamics from observations alone,
# as evidenced by the improved match between true and estimated latent states and the
# convergence of the ELBO objective function.