```@meta
EditURL = "../../examples/SLDS.jl"
```

# Simulating and Fitting a Switching Linear Dynamical System (SLDS)

This tutorial demonstrates building, simulating, and fitting a **Switching Linear Dynamical System (SLDS)**
with `StateSpaceDynamics.jl`. SLDS combines a discrete Hidden Markov Model over modes with
linear-Gaussian state-space models, capturing time series that switch among distinct linear dynamics
(e.g., alternating between slow and fast oscillatory behaviors).

## Model Overview

The SLDS combines discrete and continuous latent structure:
- **Discrete mode** $s_t \in \{1,\ldots,K\}$ with Markov dynamics: $P(s_t | s_{t-1}) = A_{\text{hmm}}[s_{t-1}, s_t]$
- **Continuous state** $\mathbf{x}_t \in \mathbb{R}^{d_x}$ evolving as: $\mathbf{x}_t = \mathbf{A}_{s_t} \mathbf{x}_{t-1} + \boldsymbol{\varepsilon}_t$
- **Observations** $\mathbf{y}_t \in \mathbb{R}^{d_y}$ via: $\mathbf{y}_t = \mathbf{C}_{s_t} \mathbf{x}_t + \boldsymbol{\eta}_t$
- **Process noise** $\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_{s_t})$
- **Observation noise** $\boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_{s_t})$

**Inference:** Exact EM is intractable due to exponential mode sequences. The `fit!` function
uses structured variational EM combining HMM forward-backward with mode-specific Kalman smoothing.

## Load Required Packages

````@example switching_linear_dynamical_system_example
using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using LaTeXStrings
using Statistics
using StableRNGs

rng = StableRNG(1234);
nothing #hide
````

## Create and Simulate SLDS

## Understanding SLDS Components

An SLDS has several key components we need to specify:

**Discrete State Dynamics:**
- `A_hmm`: Transition matrix between discrete states (how likely to switch)
- `π₀`: Initial distribution over discrete states

**Continuous State Dynamics (for each discrete state k):**
- `Aₖ`: State transition matrix (how the continuous state evolves)
- `Qₖ`: Process noise covariance (uncertainty in state evolution)
- `Cₖ`: Observation matrix (how continuous states map to observations)
- `R`: Observation noise covariance

For our specific test case, we will create two distinct states:
- **State 1:** A slower oscillator with low process noise
- **State 2:** A faster oscillator with higher process noise
We will multiply the dynamics matrices by a 0.95 scaling factor to provide stability (eigenvalues < 1),
State 2 oscillates ~11x faster than State 1. The observation matrices C₁ and C₂ are different random projections.
This means each discrete state not only has different dynamics, but also
different ways of manifesting in the observed data.

````@example switching_linear_dynamical_system_example
# Create the HMM parameters
A_hmm = [0.92 0.08; 0.06 0.94]
π₀ = [1.0, 0.0]

state_dim = 2   # Latent state dimensionality
obs_dim = 10    # Observation dimensionality
K = 2           # Number of discrete modes
````

HMM parameters for mode switching

````@example switching_linear_dynamical_system_example
A_hmm = [0.92 0.08;    # Mode transitions: sticky dynamics
         0.06 0.94]    # High probability of staying in current mode
π₀ = [1.0, 0.0]        # Start in mode 1
````

Mode-specific continuous dynamics (two oscillators with different frequencies)

````@example switching_linear_dynamical_system_example
A₁ = 0.95 * [cos(0.05) -sin(0.05); sin(0.05) cos(0.05)]  # Slow oscillator
A₂ = 0.95 * [cos(0.55) -sin(0.55); sin(0.55) cos(0.55)]  # Fast oscillator
````

Mode-specific process noise (different noise levels)

````@example switching_linear_dynamical_system_example
Q₁ = [0.001 0.0; 0.0 0.001]  # Low noise for mode 1
Q₂ = [0.1   0.0; 0.0 0.1]    # Higher noise for mode 2
````

Shared initial state distribution

````@example switching_linear_dynamical_system_example
x0 = [0.0, 0.0]
P0 = [0.1 0.0; 0.0 0.1]
````

Mode-specific observation models

````@example switching_linear_dynamical_system_example
C₁ = randn(rng, obs_dim, state_dim)  # Random observation mapping for mode 1
C₂ = randn(rng, obs_dim, state_dim)  # Different mapping for mode 2
R = Matrix(0.1 * I(obs_dim));        # Shared observation noise
nothing #hide
````

Construct the complete SLDS

````@example switching_linear_dynamical_system_example
model = SwitchingLinearDynamicalSystem(
    A_hmm,
    [
        LinearDynamicalSystem(GaussianStateModel(A₁, Q₁, x0, P0), GaussianObservationModel(C₁, R), state_dim, obs_dim, fill(true, 6)),
        LinearDynamicalSystem(GaussianStateModel(A₂, Q₂, x0, P0), GaussianObservationModel(C₂, R), state_dim, obs_dim, fill(true, 6)),
    ],
    π₀,
    K,
);

# Simulate data
T = 1000
x, y, z = rand(rng, model, 1000)
````

The simulation returns:
- x: continuous latent states (2 × T matrix)
- y: observations (10 × T matrix)
- z: discrete state sequence (length T vector)

Notice how the discrete states z create "regimes" in the continuous dynamics x.

## Visualize Latent Dynamics with Mode Shading

The plot shows the continuous latent dynamics (x₁, x₂) with background shading
indicating the active discrete state. Notice:
- Light blue regions: State 1 (slow, tight oscillations)
- Light yellow regions: State 2 (fast, wide oscillations)
- The system "remembers" where it was when switching between states

````@example switching_linear_dynamical_system_example
p1 = plot(1:T, x[1, :], label=L"x_1", linewidth=1.5, color=:black)
plot!(1:T, x[2, :], label=L"x_2", linewidth=1.5, color=:blue)

transition_points = [1; findall(diff(vec(z)) .!= 0) .+ 1; T + 1]
for i in 1:(length(transition_points) - 1)
    start_idx = transition_points[i]
    end_idx = transition_points[i + 1] - 1
    state_val = z[start_idx]
    bg_color = state_val == 1 ? :lightblue : :lightyellow
    vspan!([start_idx, end_idx], fillalpha=0.3, color=bg_color,
           label=(i == 1 && start_idx < 100 ? "Mode $state_val" : ""))
end

plot!(title="Latent Dynamics with Mode Switching",
      xlabel="Time", ylabel="State Value",
      ylims=(-3, 3), legend=:topright)
````

## Initialize and Fit SLDS
Initialize SLDS with reasonable but imperfect parameters, then use variational EM to learn.
Initialize HMM transition matrix (moderately sticky)

````@example switching_linear_dynamical_system_example
A_init = [0.9 0.1; 0.1 0.9]
A_init ./= sum(A_init, dims=2)  # Ensure row-stochastic
````

Random initial state probabilities

````@example switching_linear_dynamical_system_example
πₖ_init = rand(rng, K)
πₖ_init ./= sum(πₖ_init)
````

Use the initialize_slds function for proper initialization

````@example switching_linear_dynamical_system_example
learned_model = initialize_slds(;
    K=K,
    d=state_dim,
    p=obs_dim,
    self_bias=5.0,
    seed=456  # Different seed from true model
)
````

Fit using variational EM

````@example switching_linear_dynamical_system_example
mls, param_diff, FB, FS = fit!(learned_model, y; max_iter=25)
````

## Monitor ELBO Convergence
Plot the Evidence Lower BOund (ELBO) to verify monotonic improvement.
For SLDS, this tracks the variational approximation quality.

````@example switching_linear_dynamical_system_example
p2 = plot(mls, xlabel="Iteration", ylabel="ELBO",
          title="Variational EM Convergence",
          marker=:circle, markersize=3, lw=2,
          legend=false, color=:darkgreen)

annotate!(p2, length(mls)*0.7, mls[end]*0.98,
    text("Final ELBO: $(round(mls[end], digits=1))", 10))
````

## Compare True vs Learned Latent States

Reconstruct latent states by weighting mode-specific smoothed states
with their posterior responsibilities.

````@example switching_linear_dynamical_system_example
latents_learned = zeros(state_dim, T)
responsibilities = exp.(FB.γ)  # Convert from log-space

for t in 1:T
    for k in 1:K
        latents_learned[:, t] += FS[k].x_smooth[:, t] .* responsibilities[k, t]
    end
end
````

Plot comparison with offset for clarity

````@example switching_linear_dynamical_system_example
p3 = plot(size=(900, 400))
offset = 2.5

plot!(1:T, x[1, :] .+ offset, label=L"x_1 \text{ (true)}",
      linewidth=2, color=:black, alpha=0.8)
plot!(1:T, x[2, :] .- offset, label=L"x_2 \text{ (true)}",
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
````

## Mode Decoding and Accuracy Assessment

Decode discrete modes using posterior responsibilities and assess accuracy.

````@example switching_linear_dynamical_system_example
z_decoded = [argmax(responsibilities[:, t]) for t in 1:T]
````

Handle label permutation by trying both assignments

````@example switching_linear_dynamical_system_example
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

z_aligned, accuracy = align_labels_2way(vec(z), z_decoded)
print("Mode decoding accuracy: $(round(accuracy*100, digits=1))%\n");
nothing #hide
````

Visualize mode assignments over time (first 200 time steps for clarity)

````@example switching_linear_dynamical_system_example
t_subset = 1:200
true_modes = reshape(z[t_subset], 1, :)
decoded_modes = reshape(z_aligned[t_subset], 1, :)

p4 = plot(
    heatmap(true_modes, colormap=:roma, title="True Mode Sequence",
           xticks=false, yticks=false, colorbar=false),
    heatmap(decoded_modes, colormap=:roma, title="Decoded Mode Sequence",
           xlabel="Time Steps (1-200)", yticks=false, colorbar=false),
    layout=(2, 1), size=(800, 300)
)
````

## Parameter Recovery Assessment

````@example switching_linear_dynamical_system_example
print("\n=== Parameter Recovery Assessment ===\n")
````

Compare HMM transition matrices

````@example switching_linear_dynamical_system_example
A_error = norm(A_hmm - learned_model.A) / norm(A_hmm)
print("HMM transition matrix error: $(round(A_error*100, digits=1))%\n")
````

Compare dynamics matrices for each mode

````@example switching_linear_dynamical_system_example
for k in 1:K
    A_true = k == 1 ? A₁ : A₂
    A_learned = learned_model.B[k].state_model.A
    dyn_error = norm(A_true - A_learned) / norm(A_true)
    print("Mode $k dynamics error: $(round(dyn_error*100, digits=1))%\n")
end

print("True vs. Learned HMM Transitions:\n")
print("True A_hmm:\n$(round.(A_hmm, digits=3))\n")
print("Learned A_hmm:\n$(round.(learned_model.A, digits=3))\n");
nothing #hide
````

## Summary

This tutorial demonstrated the complete Switching Linear Dynamical System workflow:

**Key Concepts:**
- **Hybrid dynamics**: Discrete mode switching combined with continuous state evolution
- **Variational EM**: Structured approximation handling intractable exact inference
- **Mode-specific parameters**: Each discrete state has its own linear dynamics
- **Responsibility weighting**: Soft assignment of observations to modes

**Technical Insights:**
- ELBO monitoring ensures proper variational approximation convergence
- Label permutation requires careful accuracy assessment
- Parameter sharing across modes can improve identifiability
- Multiple initializations help avoid poor local optima

**Modeling Power:**
- Captures complex time series with multiple dynamic regimes
- Enables automatic segmentation and regime detection
- Provides probabilistic framework for switching systems
- Extends both HMMs and LDS to richer model class

**Applications:**
- Neuroscience: population dynamics across behavioral states
- Finance: regime-switching in market dynamics
- Engineering: fault detection in dynamical systems
- Climate: seasonal transitions and regime changes

SLDS provides a powerful framework for modeling complex temporal data with
multiple underlying dynamics, bridging discrete regime detection with
continuous state-space modeling in a principled probabilistic framework.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

