# # Simulating and Fitting a Linear Dynamical System

# This tutorial demonstrates how to use `StateSpaceDynamics.jl` to simulate a latent
# linear dynamical system and fit it using the EM algorithm. We'll walk through the
# complete workflow: defining a true model, generating synthetic data, initializing
# a naive model, and then learning the parameters through iterative optimization.

# ## Load Required Packages

using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using LaTeXStrings
using StableRNGs

# Set a stable random number generator for reproducible results
rng = StableRNG(123);

# ## Create a State-Space Model
# ## Mathematical Foundation of Linear Dynamical Systems
#
# A Linear Dynamical System describes how a hidden state evolves over time and 
# generates observations through two key equations:
#
# **State Evolution**: $x_{t+1} = \mathbf{A}  x_t + ε_t$,  where $ε_t \sim N(0, \mathbf{Q})$
# **Observation**: $y_t = \mathbf{C}  x_t + η_t$,  where $η_t \sim N(0, \mathbf{R})$
#
# The beauty of this formulation is that it separates the underlying dynamics 
# (governed by A) from what we can actually measure (governed by C). The noise
# terms ε and η represent our uncertainty about the process and measurements.

obs_dim = 10      # Number of observed variables at each time step
latent_dim = 2;   # Number of latent state variables

# Define the state transition matrix $\mathbf{A}$. This matrix governs how the latent state
# evolves from one time step to the next: $\mathbf{x}_{t+1} = \mathbf{A} \mathbf{x}_t + \boldsymbol{\epsilon}$.
# The rotation angle of 0.25 radians (≈14.3°) creates a gentle spiral, while
# the 0.95 scaling ensures the system is stable (eigenvalues < 1). Without the
# scaling factor, trajectories would spiral outward indefinitely. This particular
# combination creates visually appealing dynamics that are easy to interpret.

A = 0.95 * [cos(0.25) -sin(0.25); sin(0.25) cos(0.25)];

# Process noise covariance $\mathbf{Q}$ controls how much random variation we add to the
# latent state transitions. A smaller $\mathbf{Q}$ means more predictable dynamics.

Q = Matrix(0.1 * I(2));

# Initial state parameters: where the latent trajectory starts and how uncertain
# we are about this initial position.

x0 = [0.0; 0.0]           # Mean of initial state
P0 = Matrix(0.1 * I(2));  # Covariance of initial state

# Observation parameters: how the latent states map to observed data.
# $\mathbf{C}$ is the observation matrix (latent-to-observed mapping), and $\mathbf{R}$ is the
# observation noise covariance.

C = randn(rng, obs_dim, latent_dim)  # Random linear mapping from 2D latent to 10D observed
R = Matrix(0.5 * I(obs_dim));         # Independent noise on each observation dimension

# Construct the state and observation model components

true_gaussian_sm = GaussianStateModel(;A=A, Q=Q, x0=x0, P0=P0)
true_gaussian_om = GaussianObservationModel(;C=C, R=R);

# Combine them into a complete Linear Dynamical System
# The fit_bool parameter indicates which parameters should be learned during fitting

true_lds = LinearDynamicalSystem(;
    state_model=true_gaussian_sm,
    obs_model=true_gaussian_om,
    latent_dim=latent_dim,
    obs_dim=obs_dim,
    fit_bool=fill(true, 6)  # Fit all 6 parameter matrices: A, Q, C, R, x0, P0
);

# ## Simulate Latent and Observed Data

# Now we generate synthetic data from our true model. This gives us both the
# latent states (which we'll later try to recover) and the observations (which
# is all a real algorithm would see).

tSteps = 500;  # Number of time points to simulate

# The rand function generates both latent trajectories and corresponding observations
latents, observations = rand(rng, true_lds; tsteps=tSteps, ntrials=1);

# ## Plot Vector Field of Latent Dynamics

# To better understand the dynamics encoded by our transition matrix $\mathbf{A}$, we'll
# create a vector field plot. This shows how the latent state would evolve
# from any starting point in the 2D latent space.

# Create a grid of starting points and calculate the flow field
x = y = -3:0.5:3
X = repeat(x', length(y), 1)
Y = repeat(y, 1, length(x))

U = zeros(size(X))  # x-component of flow
V = zeros(size(Y))  # y-component of flow

for i in 1:size(X, 1), j in 1:size(X, 2)
    v = A * [X[i,j], Y[i,j]]
    U[i,j] = v[1] - X[i,j]  # Change in x
    V[i,j] = v[2] - Y[i,j]  # Change in y
end

# Normalize arrows for cleaner visualization
magnitude = @. sqrt(U^2 + V^2)
U_norm = U ./ magnitude
V_norm = V ./ magnitude;

# Create the vector field plot with the actual trajectory overlaid
p1 = quiver(X, Y, quiver=(U_norm, V_norm), color=:blue, alpha=0.3,
           linewidth=1, arrow=arrow(:closed, :head, 0.1, 0.1))
plot!(latents[1, :, 1], latents[2, :, 1], xlabel=L"x_1", ylabel=L"x_2",
      color=:black, linewidth=1.5, title="Latent Dynamics", legend=false)

# ## Plot Latent States and Observations

# Let's visualize both the latent states (which evolve smoothly according to our
# dynamics) and the observations (which are noisy linear combinations of the latents).

states = latents[:, :, 1]      # Extract the latent trajectory
emissions = observations[:, :, 1];  # Extract the observed data

# Create a two-panel plot: latent states on top, observations below
lim_states = maximum(abs.(states))
lim_emissions = maximum(abs.(emissions))

p2 = plot(size=(800, 600), layout=@layout[a{0.3h}; b])

for d in 1:latent_dim
    plot!(1:tSteps, states[d, :] .+ lim_states * (d-1), color=:black,
          linewidth=2, label="", subplot=1)
end

plot!(subplot=1, yticks=(lim_states .* (0:latent_dim-1), [L"x_%$d" for d in 1:latent_dim]),
      xticks=[], xlims=(0, tSteps), title="Simulated Latent States",
      yformatter=y->"", tickfontsize=12);

for n in 1:obs_dim
    plot!(1:tSteps, emissions[n, :] .- lim_emissions * (n-1), color=:black,
          linewidth=2, label="", subplot=2); # Plot observations (offset vertically since there are many dimensions)

end

plot!(subplot=2, yticks=(-lim_emissions .* (obs_dim-1:-1:0), [L"y_{%$n}" for n in 1:obs_dim]),
      xlabel="time", xlims=(0, tSteps), title="Simulated Emissions",
      yformatter=y->"", tickfontsize=12, left_margin=10Plots.mm)

# ## The Learning Problem
#
# In real applications, we only observe $y_t$ (the emissions) - the latent states $x_t$
# are hidden from us. Our challenge is to recover both:
# 1. The system parameters (A, Q, C, R) that generated the data
# 2. The most likely latent state sequence given our observations
#
# This is a classic "chicken and egg" problem: if we knew the parameters, we could
# infer the states; if we knew the states, we could estimate the parameters.
# The EM algorithm elegantly solves this by alternating between these two problems.

# Initialize with random parameters (this simulates not knowing the true system)
A_init = random_rotation_matrix(2, rng)    # Random rotation matrix for dynamics
Q_init = Matrix(0.1 * I(2))                # Same process noise variance
C_init = randn(rng, obs_dim, latent_dim)   # Random observation mapping
R_init = Matrix(0.5 * I(obs_dim))          # Same observation noise
x0_init = zeros(latent_dim)                # Start from origin
P0_init = Matrix(0.1 * I(latent_dim));      # Same initial uncertainty

# Our "naive" initialization uses random parameters, simulating a real scenario
# where we don't know the true system. The quality of initialization can affect
# convergence speed and which local optimum we find, but EM is generally robust
# to reasonable starting points.
gaussian_sm_init = GaussianStateModel(;A=A_init, Q=Q_init, x0=x0_init, P0=P0_init)
gaussian_om_init = GaussianObservationModel(;C=C_init, R=R_init)

# Assemble the complete naive system
naive_ssm = LinearDynamicalSystem(;
    state_model=gaussian_sm_init,
    obs_model=gaussian_om_init,
    latent_dim=latent_dim,
    obs_dim=obs_dim,
    fit_bool=fill(true, 6)  # We'll learn all parameters
);

# Before fitting, let's see how well our randomly initialized model can
# infer the latent states using the smoothing algorithm.

x_smooth, p_smooth = StateSpaceDynamics.smooth(naive_ssm, observations);

# Plot the true latent states vs. our initial (poor) estimates
p3 = plot()
for d in 1:latent_dim
    plot!(1:tSteps, states[d, :] .+ lim_states * (d-1), color=:black,
          linewidth=2, label=(d==1 ? "True" : ""), alpha=0.8)
    plot!(1:tSteps, x_smooth[d, :, 1] .+ lim_states * (d-1), color=:firebrick,
          linewidth=2, label=(d==1 ? "Predicted" : ""), alpha=0.8)
end
plot!(yticks=(lim_states .* (0:latent_dim-1), [L"x_%$d" for d in 1:latent_dim]),
      xlabel="time", xlims=(0, tSteps), yformatter=y->"", tickfontsize=12,
      title="True vs. Predicted Latent States (Pre-EM)",
      legend=:topright)

# ## Understanding the EM Algorithm
#
# EM alternates between two steps until convergence:
#
# **E-step (Expectation)**: Given current parameter estimates, compute the posterior
# distribution over latent states using the Kalman smoother. This gives us
# `p(x_{1:T} | y_1:T, θ_current)`.
#
# **M-step (Maximization)**: Given the state estimates from the E-step, update
# the parameters to maximize the expected log-likelihood. This involves solving
# closed-form equations for A, Q, C, and R.
#
# The Evidence Lower BOund (ELBO) measures how well our model explains the data.
# It's guaranteed to increase (or stay constant) at each iteration, ensuring
# convergence to at least a local optimum.

println("Starting EM algorithm to learn parameters...")

# Suppress output and capture ELBO values
elbo, _ = fit!(naive_ssm, observations; max_iter=100, tol=1e-6);

println("EM converged after $(length(elbo)) iterations")

# After EM has converged, let's see how much better our latent state estimates are
x_smooth_post, p_smooth_post = StateSpaceDynamics.smooth(naive_ssm, observations);

# Plot the results: true states vs. post-EM estimates
p4 = plot()
for d in 1:latent_dim
    plot!(1:tSteps, states[d, :] .+ lim_states * (d-1), color=:black,
          linewidth=2, label=(d==1 ? "True" : ""), alpha=0.8)
    plot!(1:tSteps, x_smooth_post[d, :, 1] .+ lim_states * (d-1), color=:firebrick,
          linewidth=2, label=(d==1 ? "Predicted" : ""), alpha=0.8)
end
plot!(yticks=(lim_states .* (0:latent_dim-1), [L"x_%$d" for d in 1:latent_dim]),
      xlabel="time", xlims=(0, tSteps), yformatter=y->"", tickfontsize=12,
      title="True vs. Predicted Latent States (Post-EM)",
      legend=:topright)

# ## Model Convergence Analysis

# The Evidence Lower Bound (ELBO) measures how well our model explains the data.
# In EM, this should increase monotonically and plateau when the algorithm
# has converged to a local optimum.

p5 = plot(elbo, xlabel="Iteration", ylabel="ELBO", 
          title="Model Convergence (ELBO)", legend=false,
          linewidth=2, color=:darkblue)

# ## Interpreting the Results
#
# The dramatic improvement in state estimation shows that EM successfully
# recovered the underlying dynamics. However, keep in mind:
# - We may have found a local optimum, not the global one
# - The recovered parameters might differ from the true ones due to identifiability
#   issues (multiple parameter sets can generate similar observations)
# - In practice, you'd validate the model on held-out data to ensure generalization

# ## Summary
# 
# This tutorial demonstrated the complete workflow for fitting a Linear Dynamical System:
# 1. We defined a true LDS with spiral dynamics and generated synthetic data
# 2. We initialized a naive model with random parameters  
# 3. We used EM to iteratively improve our parameter estimates
# 4. We visualized the dramatic improvement in latent state inference
# 
# The EM algorithm successfully recovered the underlying dynamics from observations alone,
# as evidenced by the improved match between true and estimated latent states and the
# monotonic convergence of the ELBO objective function.
