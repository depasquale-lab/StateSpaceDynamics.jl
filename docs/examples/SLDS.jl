# # Simulating and Fitting a Switching Linear Dynamical System (SLDS)

# This tutorial demonstrates building, simulating, and fitting a **Switching Linear Dynamical System (SLDS)**
# with `StateSpaceDynamics.jl`. SLDS combines a discrete Hidden Markov Model over modes with
# linear-Gaussian state-space models, capturing time series that switch among distinct linear dynamics
# (e.g., alternating between slow and fast oscillatory behaviors).

# ## Model Overview

# The SLDS combines discrete and continuous latent structure:
# - **Discrete mode** $z_t \in \{1,\ldots,K\}$ with Markov dynamics: $P(z_t | z_{t-1}) = A[z_{t-1}, z_t]$
# - **Continuous state** $\mathbf{x}_t \in \mathbb{R}^{d_x}$ evolving as: $\mathbf{x}_t = \mathbf{A}_{z_t} \mathbf{x}_{t-1} + \mathbf{b}_{z_t} + \boldsymbol{\varepsilon}_t$
# - **Observations** $\mathbf{y}_t \in \mathbb{R}^{d_y}$ via: $\mathbf{y}_t = \mathbf{C}_{z_t} \mathbf{x}_t + \mathbf{d}_{z_t} + \boldsymbol{\eta}_t$
# - **Process noise** $\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_{z_t})$
# - **Observation noise** $\boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_{z_t})$

# **Inference:** Exact EM is intractable due to exponential mode sequences. The `fit!` function
# uses variational Laplace EM (vLEM) combining HMM forward-backward with Laplace approximation
# for the continuous states.

# ## Load Required Packages

using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using LaTeXStrings
using Statistics
using StableRNGs

rng = StableRNG(1234);

# ## Create and Simulate SLDS

# ## Understanding SLDS Components
#
# An SLDS has several key components we need to specify:
# 
# **Discrete State Dynamics:** 
# - `A`: Transition matrix between discrete states (how likely to switch)
# - `πₖ`: Initial distribution over discrete states
#
# **Continuous State Dynamics (for each discrete state k):**
# - `Aₖ`: State transition matrix (how the continuous state evolves)
# - `bₖ`: State bias term
# - `Qₖ`: Process noise covariance (uncertainty in state evolution)
# - `Cₖ`: Observation matrix (how continuous states map to observations)
# - `dₖ`: Observation bias term
# - `Rₖ`: Observation noise covariance

# For our specific test case, we will create two distinct modes:
# - **Mode 1:** A slower oscillator with low process noise
# - **Mode 2:** A faster oscillator with higher process noise
# We will multiply the dynamics matrices by a 0.95 scaling factor to provide stability (eigenvalues < 1). 
# Mode 2 oscillates ~11x faster than Mode 1. The observation matrices C₁ and C₂ are different random projections.
# This means each discrete state not only has different dynamics, but also
# different ways of manifesting in the observed data.

state_dim = 2   # Latent state dimensionality
obs_dim = 10    # Observation dimensionality  
K = 2           # Number of discrete modes

# HMM parameters for mode switching
A_hmm = [0.92 0.08;    # Mode transitions: sticky dynamics
         0.06 0.94]    # High probability of staying in current mode
πₖ = [1.0, 0.0]        # Start in mode 1

# Mode-specific continuous dynamics (two oscillators with different frequencies)
A₁ = 0.95 * [cos(0.05) -sin(0.05); sin(0.05) cos(0.05)]  # Slow oscillator
A₂ = 0.95 * [cos(0.55) -sin(0.55); sin(0.55) cos(0.55)]  # Fast oscillator

# Mode-specific process noise (different noise levels)
Q₁ = [0.001 0.0; 0.0 0.001]  # Low noise for mode 1
Q₂ = [0.1   0.0; 0.0 0.1]    # Higher noise for mode 2

# Shared initial state distribution
x0 = [0.0, 0.0]
P0 = [0.1 0.0; 0.0 0.1]

# State bias terms (zero for this example)
b = zeros(state_dim)

# Mode-specific observation models
C₁ = randn(rng, obs_dim, state_dim)  # Random observation mapping for mode 1
C₂ = randn(rng, obs_dim, state_dim)  # Different mapping for mode 2
R = Matrix(0.1 * I(obs_dim))         # Shared observation noise
d = zeros(obs_dim)                    # Observation bias (zero for this example)

# Construct individual Linear Dynamical Systems for each mode
lds1 = LinearDynamicalSystem(
    GaussianStateModel(A₁, Q₁, b, x0, P0),
    GaussianObservationModel(C₁, R, d)
)

lds2 = LinearDynamicalSystem(
    GaussianStateModel(A₂, Q₂, b, x0, P0),
    GaussianObservationModel(C₂, R, d)
)

# Construct the complete SLDS
model = SLDS(
    A=A_hmm,
    πₖ=πₖ,
    LDSs=[lds1, lds2]
);

# ## Simulate data

T = 1000
z, x, y = rand(rng, model; tsteps=T, ntrials=1);

# The simulation returns:
# - z: discrete state sequence (T × 1 matrix)
# - x: continuous latent states (state_dim × T × 1 array)
# - y: observations (obs_dim × T × 1 array)
#
# Notice how the discrete states z create "regimes" in the continuous dynamics x.

# Extract single trial data for visualization
z_vec = vec(z[:, 1])
x_trial = x[:, :, 1]
y_trial = y[:, :, 1];

# ## Visualize Latent Dynamics with Mode Shading

# The plot shows the continuous latent dynamics (x₁, x₂) with background shading
# indicating the active discrete state. Notice:
# - Light blue regions: Mode 1 (slow, tight oscillations)
# - Light yellow regions: Mode 2 (fast, wide oscillations)
# - The system "remembers" where it was when switching between modes

p1 = plot(1:T, x_trial[1, :], label=L"x_1", linewidth=1.5, color=:black)
plot!(1:T, x_trial[2, :], label=L"x_2", linewidth=1.5, color=:blue)

transition_points = [1; findall(diff(z_vec) .!= 0) .+ 1; T + 1]
for i in 1:(length(transition_points) - 1)
    start_idx = transition_points[i]
    end_idx = transition_points[i + 1] - 1
    state_val = z_vec[start_idx]
    bg_color = state_val == 1 ? :lightblue : :lightyellow
    vspan!([start_idx, end_idx], fillalpha=0.3, color=bg_color, 
           label=(i == 1 && start_idx < 100 ? "Mode $state_val" : ""))
end

plot!(title="Latent Dynamics with Mode Switching",
      xlabel="Time", ylabel="State Value", 
      ylims=(-3, 3), legend=:topright)

# ## Initialize and Fit SLDS 

# Initialize SLDS with reasonable but imperfect parameters, then use variational EM to learn.

# Initialize HMM transition matrix (moderately sticky)
A_init = [0.9 0.1; 0.1 0.9]
A_init ./= sum(A_init, dims=2)  # Ensure row-stochastic

# Random initial state probabilities
πₖ_init = rand(rng, K)
πₖ_init ./= sum(πₖ_init)

# Create initial LDS models with random parameters
Random.seed!(rng, 456)  # Different seed from true model
lds_init1 = LinearDynamicalSystem(
    GaussianStateModel(
        randn(rng, state_dim, state_dim) * 0.5,  # Random A
        Matrix(0.1 * I(state_dim)),               # Q
        zeros(state_dim),                          # b
        zeros(state_dim),                          # x0
        Matrix(0.1 * I(state_dim))                # P0
    ),
    GaussianObservationModel(
        randn(rng, obs_dim, state_dim),           # Random C
        Matrix(0.1 * I(obs_dim)),                 # R
        zeros(obs_dim)                             # d
    ),
)

lds_init2 = LinearDynamicalSystem(
    GaussianStateModel(
        randn(rng, state_dim, state_dim) * 0.5,  # Random A
        Matrix(0.1 * I(state_dim)),               # Q
        zeros(state_dim),                          # b
        zeros(state_dim),                          # x0
        Matrix(0.1 * I(state_dim))                # P0
    ),
    GaussianObservationModel(
        randn(rng, obs_dim, state_dim),           # Random C
        Matrix(0.1 * I(obs_dim)),                 # R
        zeros(obs_dim)                             # d
    ),
)

learned_model = SLDS(
    A=A_init,
    πₖ=πₖ_init,
    LDSs=[lds_init1, lds_init2]
);

# Fit using variational Laplace EM
elbos = fit!(learned_model, y; max_iter=25, progress=true)

# ## Monitor ELBO Convergence

# Plot the Evidence Lower Bound (ELBO) to verify monotonic improvement.
# For SLDS, this tracks the variational approximation quality.

p2 = plot(elbos, xlabel="Iteration", ylabel="ELBO", 
          title="Variational EM Convergence", 
          marker=:circle, markersize=3, lw=2, 
          legend=false, color=:darkgreen)

annotate!(p2, length(elbos)*0.7, elbos[end]-abs(elbos[end])*0.05, 
    text("Final ELBO: $(round(elbos[end], digits=1))", 10))

# ## Compare True vs Learned Latent States

# After fitting, we can extract the smoothed latent states.
# The learned model stores the most recent smoothed states from the last EM iteration.

# For visualization, we need to run one more smoothing pass to get the final
tfs = StateSpaceDynamics.initialize_FilterSmooth(learned_model.LDSs[1], T, 1)
fbs = [StateSpaceDynamics.initialize_forward_backward(learned_model, T, Float64)]

# Initialize with uniform weights and smooth
w_uniform = ones(Float64, K, T) ./ K
StateSpaceDynamics.smooth!(learned_model, tfs[1], y_trial, w_uniform)

# Sample and run one E-step to get final discrete state posteriors
x_samples, _ = StateSpaceDynamics.sample_posterior(tfs, 1)
StateSpaceDynamics.estep!(learned_model, tfs, fbs, y, x_samples)

# Get the final smoothed continuous states
latents_learned = tfs[1].x_smooth;

# Plot comparison with offset for clarity
p3 = plot(size=(900, 400))
offset = 2.5

plot!(1:T, x_trial[1, :] .+ offset, label=L"x_1 \text{ (true)}", 
      linewidth=2, color=:black, alpha=0.8)
plot!(1:T, x_trial[2, :] .- offset, label=L"x_2 \text{ (true)}", 
      linewidth=2, color=:black, alpha=0.8)

plot!(1:T, latents_learned[1, :] .+ offset, label=L"x_1 \text{ (learned)}", 
      linewidth=1.5, color=:red, alpha=0.8)
plot!(1:T, latents_learned[2, :] .- offset, label=L"x_2 \text{ (learned)}", 
      linewidth=1.5, color=:blue, alpha=0.8)

hline!([offset, -offset], color=:gray, alpha=0.3, linestyle=:dash, label="")

plot!(title="True vs. Learned Latent States",
      xlabel="Time", ylabel="", 
      yticks=([-offset, offset], [L"x_2", L"x_1"]),
      xlims=(1, T), legend=:topright)

# ## Mode Decoding and Accuracy Assessment

# Decode discrete modes using posterior responsibilities and assess accuracy.

responsibilities = exp.(fbs[1].γ)  # Convert from log-space (K × T)
z_decoded = [argmax(responsibilities[:, t]) for t in 1:T];

# Handle label permutation by trying both assignments
function align_labels_2way(z_true::AbstractVector, z_pred::AbstractVector)
    acc_direct = mean(z_true .== z_pred)
    z_flipped = 3 .- z_pred  # Flip 1↔2
    acc_flipped = mean(z_true .== z_flipped)
    
    if acc_flipped > acc_direct
        return z_flipped, acc_flipped
    else
        return z_pred, acc_direct
    end
end

z_aligned, accuracy = align_labels_2way(z_vec, z_decoded)
print("Mode decoding accuracy: $(round(accuracy*100, digits=1))%\n");

# Visualize mode assignments over time (first 200 time steps for clarity)
t_subset = 1:200
true_modes = reshape(z_vec[t_subset], 1, :)
decoded_modes = reshape(z_aligned[t_subset], 1, :);

p4 = plot(
    heatmap(true_modes, colormap=:roma, title="True Mode Sequence", 
           xticks=false, yticks=false, colorbar=false),
    heatmap(decoded_modes, colormap=:roma, title="Decoded Mode Sequence",
           xlabel="Time Steps (1-200)", yticks=false, colorbar=false),
    layout=(2, 1), size=(800, 300)
)

# ## Summary
#
# This tutorial demonstrated the complete Switching Linear Dynamical System workflow:
#
# **Key Concepts:**
# - **Hybrid dynamics**: Discrete mode switching combined with continuous state evolution
# - **Variational Laplace EM**: Structured approximation handling intractable exact inference
# - **Mode-specific parameters**: Each discrete state has its own linear dynamics
# - **Laplace approximation**: Gaussian approximation to the continuous state posterior
#
# **Technical Insights:**
# - ELBO monitoring ensures proper variational approximation convergence
# - Label permutation requires careful accuracy assessment
# - The vLEM algorithm alternates between discrete (forward-backward) and continuous (Laplace) inference
# - Single sample Monte Carlo estimation of the ELBO
#
# **Modeling Power:**
# - Captures complex time series with multiple dynamic regimes
# - Enables automatic segmentation and regime detection
# - Provides probabilistic framework for switching systems
# - Extends both HMMs and LDS to richer model class
#
# **Applications:**
# - Neuroscience: population dynamics across behavioral states
# - Finance: regime-switching in market dynamics  
# - Engineering: fault detection in dynamical systems
# - Climate: seasonal transitions and regime changes
#
# SLDS provides a powerful framework for modeling complex temporal data with
# multiple underlying dynamics, bridging discrete regime detection with
# continuous state-space modeling in a principled probabilistic framework.