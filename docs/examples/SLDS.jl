# # Simulating and Fitting a Switching Linear Dynamical System (SLDS)
# 
# This tutorial walks through building, simulating, and fitting a
# **Switching Linear Dynamical System (SLDS)** with `StateSpaceDynamics.jl`.
# SLDS combines a discrete Hidden Markov Model (HMM) over modes with a set of
# linear-Gaussian state-space models (one per mode). It captures time series that
# switch among distinct linear dynamics (e.g., slow vs. fast oscillators).
#
# ## Model overview
# The SLDS has:
# - Discrete mode \(s_t \in \{1,\dots,K\}\) with Markov dynamics
#   \(p(s_t\mid s_{t-1}) = A_{\text{hmm}}[s_{t-1}, s_t]\), initial \(\pi_0\).
# - Continuous latent state \(x_t \in \mathbb R^{d_x}\) evolving as
#   \(x_t = A_{s_t} x_{t-1} + \varepsilon_t\), with \(\varepsilon_t \sim \mathcal N(0, Q_{s_t})\).
# - Observations \(y_t \in \mathbb R^{d_y}\) via
#   \(y_t = C_{s_t} x_t + \eta_t\), with \(\eta_t \sim \mathcal N(0, R_{s_t})\).
# - Initial distribution \(x_0 \sim \mathcal N(\mu_0, P_0)\).
#
# **Inference & learning.** Exact EM is intractable because of the exponential
# number of mode sequences. `fit!` uses a structured variational EM: a forward–
# backward step for the HMM (variational E-step) coupled with Kalman smoothing in
# each mode (continuous E-step), followed by M-step updates of parameters.
# The objective reported as `mls` is an ELBO that should increase monotonically.
#
# ---

# ## Load Packages
using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using LaTeXStrings
using Statistics
using StableRNGs

rng = StableRNG(123);

# ## Simulate data from an SLDS
state_dim = 2
obs_dim   = 10
K         = 2  # two modes

# HMM (mode) parameters
A_hmm = [0.92 0.08; 0.06 0.94]
π₀    = [1.0, 0.0]

# Mode-specific state dynamics (two oscillators)
A₁ = 0.95 * [cos(0.05) -sin(0.05); sin(0.05) cos(0.05)]  # slower
A₂ = 0.95 * [cos(0.55) -sin(0.55); sin(0.55) cos(0.55)]  # faster

Q₁ = [0.001 0.0; 0.0 0.001]
Q₂ = [0.1   0.0; 0.0 0.1]

# Shared initial state distribution for simplicity
x0 = [0.0, 0.0]
P0 = [0.1 0.0; 0.0 0.1]

# Mode-specific observation models
C₁ = randn(rng, obs_dim, state_dim)
C₂ = randn(rng, obs_dim, state_dim)
R  = Matrix(0.1 * I(obs_dim))  # shared observation noise

model = SwitchingLinearDynamicalSystem(
    A_hmm,
    [
        LinearDynamicalSystem(GaussianStateModel(A₁, Q₁, x0, P0), GaussianObservationModel(C₁, R), state_dim, obs_dim, fill(true, 6)),
        LinearDynamicalSystem(GaussianStateModel(A₂, Q₂, x0, P0), GaussianObservationModel(C₂, R), state_dim, obs_dim, fill(true, 6)),
    ],
    π₀,
    K,
)

# Simulate
T = 1000
x, y, z = rand(rng, model, T)

# ## Plot latent dynamics with mode shading
p1 = plot(1:T, x[1, :], label="x₁", linewidth=1.5)
plot!(1:T, x[2, :], label="x₂", linewidth=1.5)

# Shade regions by discrete mode z
transition_points = [1; findall(diff(z) .!= 0) .+ 1; T + 1]
for i in 1:(length(transition_points) - 1)
    start_idx = transition_points[i]
    end_idx   = transition_points[i + 1] - 1
    state_val = z[start_idx]
    bg_color  = state_val == 1 ? :lightblue : :lightyellow
    vspan!([start_idx, end_idx], fillalpha=0.5, color=bg_color, label=(i == 1 ? "State $state_val" : ""))
end

title!("Latent Dynamics with Mode Shading")
xlabel!("Time")
ylabel!("State Value")
ylims!(-3, 3)

p1

# ## Initialize and fit an SLDS to the observations
# Good initialization helps. We'll set a moderately sticky HMM and rough dynamics,
# then call `fit!` for variational EM.

A = [0.9 0.1; 0.1 0.9]
A ./= sum(A, dims=2)   # row-stochastic

πₖ = rand(rng, K); πₖ ./= sum(πₖ)

Q  = Matrix(0.001 * I(state_dim))
P0 = Matrix(0.001 * I(state_dim))
C  = randn(rng, obs_dim, state_dim)
R  = Matrix(0.1 * I(obs_dim))

B = [
    LinearDynamicalSystem(
        GaussianStateModel(0.95 * [cos(f) -sin(f); sin(f) cos(f)], Q, x0, P0),
        GaussianObservationModel(C, R),
        state_dim, obs_dim, fill(true, 6),
    ) for f in (0.7, 0.1)
]

learned_model = SwitchingLinearDynamicalSystem(A, B, πₖ, K)

# Fit with variational EM
mls, param_diff, FB, FS = fit!(learned_model, y; max_iter=25)

# `mls` is the ELBO trace; `param_diff` can be used as an additional stopping metric
# if desired; `FB` holds HMM posteriors and `FS` holds per-mode Kalman smoothing.

# ## ELBO over iterations
plot(mls, label="ELBO", linewidth=1.5)
xlabel!("Iteration")
ylabel!("ELBO")

# ## Compare true vs. learned latent states
# We combine mode-specific smoothed states using HMM responsibilities as weights.

latents = zeros(state_dim, T)
resp    = exp.(FB.γ) 
for t in 1:T
    for k in 1:K
        latents[:, t] += FS[k].x_smooth[:, t] .* resp[k, t]
    end
end

plt = plot(size=(800, 500), background_color=:white, margin=5Plots.mm)
plot!(x[1, :] .+ 2, label="x₁ (True)",   linewidth=2, color=:black,    alpha=0.8)
plot!(x[2, :] .- 2, label="x₂ (True)",   linewidth=2, color=:black,    alpha=0.8)
plot!(latents[1, :] .+ 2, label="x₁ (Learned)", linewidth=1.5, color=:firebrick)
plot!(latents[2, :] .- 2, label="x₂ (Learned)", linewidth=1.5, color=:royalblue)

title!("SLDS: True vs Learned Latent States")
xlabel!("Time")
ylabel!("")
yticks!([-2, 2], ["x₂", "x₁"]) 
hline!([2],  color=:gray, alpha=0.3, linestyle=:dash, label="")
hline!([-2], color=:gray, alpha=0.3, linestyle=:dash, label="")
xlims!(0, T)

plt

# ## Decoding modes and basic accuracy metrics
# Hard-decoded modes by argmax responsibilities, and a simple confusion rate with
# the simulated ground truth (up to label permutation).

z_hat = map(t -> argmax(view(resp, :, t)), 1:T)

# Since labels are arbitrary, align them to best match truth via a 2x2 sweep.
function align_labels_2way(z_true::AbstractVector{<:Integer}, z_pred::AbstractVector{<:Integer})
    acc1 = mean(z_true .== z_pred)
    acc2 = mean(z_true .== (3 .- z_pred))  # flip 1<->2
    if acc2 > acc1
        return (3 .- z_pred), acc2
    else
        return z_pred, acc1
    end
end

z_aligned, acc = align_labels_2way(vec(z), z_hat)
@info "Mode decoding accuracy (up to permutation)" acc

# ## Practical tips & pitfalls
# - **Stickiness:** If modes switch too frequently, increase self-transition mass
#   in A_hmm or add a stickiness prior in the M-step.
# - **Scaling/identifiability:** With per-mode C, Q, R all free, degeneracies can
#   appear. Consider tying certain parameters across modes (e.g., shared R).
# - **Initialization:** Seed A_hmm near diagonal; initialize A_k with different
#   frequencies/directions to avoid identical modes; run from multiple starts.
# - **Diagnostics:** Monitor ELBO and parameter differences; visualize responsibilities
#   over time and check that each mode explains distinct dynamics.
#
# ## Exercises
# 1. Increase K to 3 and create a third oscillator; verify that modes separate.
# 2. Make R state-specific and observe the tradeoff between Q and R in explaining
#    variability.
# 3. Start from nearly identical A_k and show that without good initialization the
#    model collapses to a single effective mode; then fix with sticky A_hmm.
# 4. Plot the responsibilities `resp[k, :]` over time and compare against ground truth.
#
# ---
# End of tutorial.
