# # Simulating and Fitting a Hidden Markov Model

# This tutorial demonstrates how to use `StateSpaceDynamics.jl` to create, sample from, and fit Hidden 
# Markov Models (HMMs). Unlike Linear Dynamical Systems which have continuous latent states, HMMs have
# discrete latent states that switch between a finite number of modes. This makes them ideal for modeling
# data with distinct behavioral regimes, switching dynamics, or categorical latent structure.

# We'll focus on a Gaussian generalized linear model HMM (GLM-HMM), where each hidden state corresponds
# to a different regression relationship between inputs and outputs. This is particularly useful for
# modeling data where the input-output relationship changes over time in discrete jumps.

# ## Load Required Packages

using LinearAlgebra
using Plots
using Random
using StateSpaceDynamics
using StableRNGs
using Statistics: mean

# Set up reproducible random number generation
rng = StableRNG(1234);

# ## Create a Gaussian GLM-HMM

# In a GLM-HMM, each hidden state defines a different regression model. The system switches
# between these regression models according to Markovian dynamics. This is useful for modeling
# scenarios where the relationship between predictors and outcomes changes over time. In this example,
# we will demonstrate how to use `StateSpaceDynamics.jl` to create a GLM-HMM, generate synthetic data,
# fit the model using the EM algorithm, and perform state inference with the Viterbi algorithm.

# We will start by defining a simple GLM-HMM with two hidden states.
# State 1: Positive relationship between input and output
emission_1 = GaussianRegressionEmission(
    input_dim=3,                                    # Number of input features
    output_dim=1,                                   # Number of output dimensions  
    include_intercept=true,                         # Include intercept term
    β=reshape([3.0, 2.0, 2.0, 3.0], :, 1),        # Regression coefficients [intercept, β₁, β₂, β₃]
    Σ=[1.0;;],                                     # Observation noise variance
    λ=0.0                                          # Regularization parameter
);

# State 2: Different relationship (negative intercept, different slopes)
emission_2 = GaussianRegressionEmission(
    input_dim=3,
    output_dim=1,
    include_intercept=true,
    β=reshape([-4.0, -2.0, 3.0, 2.0], :, 1),      # Different regression coefficients
    Σ=[1.0;;],                                     # Same noise level
    λ=0.0
);

# Define the state transition matrix $\mathbf{A}$:
# $A_{ij} = P(\text{state}_t = j \mid \text{state}_{t-1} = i)$

# Diagonal elements are high (states are persistent)
A = [0.99 0.01;    # From state 1: 99% stay, 1% switch to state 2
     0.05 0.95];    # From state 2: 5% switch to state 1, 95% stay

# Initial state distribution: $\pi_k = P(\text{state}_1 = k)$
πₖ = [0.8; 0.2];    # 80% chance of starting in state 1, 20% in state 2

# Now that we have defined our emission models, we can construct the complete GLM-HMM.
true_model = HiddenMarkovModel(
    K=2,                        # Number of hidden states
    A=A,                        # Transition matrix
    πₖ=πₖ,                     # Initial state distribution
    B=[emission_1, emission_2]  # Emission models for each state
);

print("Created GLM-HMM with regression models:\n")
print("State 1: y = 3.0 + 2.0x₁ + 2.0x₂ + 3.0x₃ + ε\n")
print("State 2: y = -4.0 - 2.0x₁ + 3.0x₂ + 2.0x₃ + ε\n");

# ## Sample from the GLM-HMM

# Generate synthetic data from our true model. To do this, we must define the inputs to the model.
# Then, sampling will yield the outputs nad the true hidden state sequence.

n = 20000  # Number of time points

Φ = randn(rng, 3, n);  # Generate random input features (predictors)

true_labels, data = rand(rng, true_model, Φ, n=n);  # Sample from the HMM: returns both hidden states and observations

print("Generated $(n) samples: State 1 ($(round(mean(true_labels .== 1)*100, digits=1))%), State 2 ($(round(mean(true_labels .== 2)*100, digits=1))%)\n");

# ## Visualize the Sampled Dataset

# Here, we create a scatter plot to show how the input-output relationship differs between
# the two hidden states. We plot feature 1 vs output, with points colored by true state. State 1 (blue)
# has a positive slope, while State 2 (red) has a negative slopes. In addition to the scatter plot, we overlay
# the true regression lines for each state.

colors = [:dodgerblue, :crimson]  # Blue for state 1, red for state 2

p1 = scatter(Φ[1, :], vec(data);
    color = colors[true_labels],
    ms = 2,
    alpha = 0.4,
    label = "",
    xlabel = "Input Feature 1", 
    ylabel = "Output",
    title = "GLM-HMM Data (colored by true state)"
)

xvals = range(extrema(Φ[1, :])..., length=100) # Overlay true regression lines (holding other features at 0)

β1 = emission_1.β[:, 1]
y_pred_1 = β1[1] .+ β1[2] .* xvals  # intercept + slope*x₁
plot!(xvals, y_pred_1;
    color = :dodgerblue,
    lw = 3,
    label = "State 1 regression"
)

β2 = emission_2.β[:, 1]
y_pred_2 = β2[1] .+ β2[2] .* xvals  # intercept + slope*x₁
plot!(xvals, y_pred_2;
    color = :crimson,
    lw = 3,
    label = "State 2 regression",
    legend = :topright
)

# ## Initialize and Fit HMM with EM

# In a realistic scenario, we would not have access to the latent states; we would only observe the inputs and outputs.
# We can use the Expectation-Maximization (EM) algorithm to learn the model parameters and infer the hidden states from
# the observed data alone.

# To demonstrate this process, we start with a randomly initialized GLM-HMM with different parameters than the true model.

A_init = [0.8 0.2; 0.1 0.9]     # Different transition probabilities
πₖ_init = [0.6; 0.4]            # Different initial distribution

emission_1_init = GaussianRegressionEmission(
    input_dim=3, output_dim=1, include_intercept=true,
    β=reshape([2.0, -1.0, 1.0, 2.0], :, 1),    # Random coefficients
    Σ=[2.0;;], λ=0.0
);

emission_2_init = GaussianRegressionEmission(
    input_dim=3, output_dim=1, include_intercept=true,
    β=reshape([-2.5, -1.0, 3.5, 3.0], :, 1),   # Random coefficients
    Σ=[0.5;;], λ=0.0
);


test_model = HiddenMarkovModel(K=2, A=A_init, πₖ=πₖ_init, B=[emission_1_init, emission_2_init])

# Now that we have definedour test model, we can fit it to the data using the EM algorithm.
lls = fit!(test_model, data, Φ);

print("EM converged after $(length(lls)) iterations\n")
print("Log-likelihood improved by $(round(lls[end] - lls[1], digits=1))\n");

# Plot EM convergence
p2 = plot(lls, xlabel="EM Iteration", ylabel="Log-Likelihood", 
          title="Model Convergence", legend=false, lw=2, color=:darkblue)

# ## Visualize Learned vs True Regression Models

# Now that we have done parameter learning, we can visualize how well the learned regression models
# match the true models. We plot the data again, colored by true state, and overlay both the true and learned regression lines.
# As you can see, the learned models (dashed lines) closely match the true models (solid lines).

p3 = scatter(Φ[1, :], vec(data);
    color = colors[true_labels],
    ms = 2,
    alpha = 0.3,
    label = "",
    xlabel = "Input Feature 1",
    ylabel = "Output", 
    title = "True vs. Learned Regression Models"
)

xvals = range(extrema(Φ[1, :])..., length=100)

plot!(xvals, β1[1] .+ β1[2] .* xvals;
    color = :green, lw = 3, linestyle = :solid, label = "State 1 (true)"
)
plot!(xvals, β2[1] .+ β2[2] .* xvals;
    color = :orange, lw = 3, linestyle = :solid, label = "State 2 (true)"
)

β1_learned = test_model.B[1].β[:, 1]
β2_learned = test_model.B[2].β[:, 1]
plot!(xvals, β1_learned[1] .+ β1_learned[2] .* xvals;
    color = :yellow, lw = 3, linestyle = :dash, label = "State 1 (learned)"
)
plot!(xvals, β2_learned[1] .+ β2_learned[2] .* xvals;
    color = :purple, lw = 3, linestyle = :dash, label = "State 2 (learned)",
    legend = :topright
)

# ## Hidden State Decoding with Viterbi Algorithm

# Now we use the Viterbi algorithm to find the most likely sequence of hidden states given the
# observed data. We'll compare true vs predicted state sequences.

pred_labels = viterbi(test_model, data, Φ);

accuracy = mean(true_labels .== pred_labels)
print("Hidden state prediction accuracy: $(round(accuracy*100, digits=1))%\n");

# We can also visualize the true and predicted state sequences as heatmaps (first 1000 time points).
n_display = 1000
true_seq = reshape(true_labels[1:n_display], 1, :)
pred_seq = reshape(pred_labels[1:n_display], 1, :)

p4 = plot(
    heatmap(true_seq, colormap=:roma, title="True State Sequence", 
           xticks=false, yticks=false, colorbar=false),
    heatmap(pred_seq, colormap=:roma, title="Predicted State Sequence (Viterbi)",
           xlabel="Time Steps (1-$n_display)", xticks=0:200:n_display, 
           yticks=false, colorbar=false),
    layout=(2, 1), size=(800, 300)
)

# ## Multiple Independent Trials

# Real-world scenarios often involve multiple independent sequences. Here, we generate 
# multiple trials of synthetic data and show how to fit GLM-HMMs to this data structure
# using `StateSpaceDynamics.jl`.

num_trials = 100   # Number of independent sequences
n_trial = 1000;    # Length of each sequence

print("Generating $num_trials independent trials of length $n_trial...\n")

all_data = Vector{Matrix{Float64}}()
Φ_total = Vector{Matrix{Float64}}()
all_true_labels = Vector{Vector{Int64}}()

for i in 1:num_trials  # Generate multiple trials from our ground truth model
    Φ_trial = randn(rng, 3, n_trial)
    true_labels_trial, data_trial = rand(rng, true_model, Φ_trial, n=n_trial)
    push!(all_true_labels, true_labels_trial)
    push!(all_data, data_trial)
    push!(Φ_total, Φ_trial)
end

print("Total data points: $(num_trials * n_trial)\n");

# ## Fitting HMM to Multiple Trials

# When we have multiple independent sequences, EM accounts for each sequence 
# starting fresh from the initial distribution, providing more robust estimates.
# To fit models of this kind, simply create a test model as before and call `fit!` with
# the data and inputs as vectors of the independent sequences! We will use the same 
# random initialization as before.

test_model_multi = HiddenMarkovModel(
    K=2, A=A_init, πₖ=πₖ_init, 
    B=[deepcopy(emission_1_init), deepcopy(emission_2_init)]
)  # Initialize fresh model for multi-trial fitting

lls_multi = fit!(test_model_multi, all_data, Φ_total);

print("Multi-trial EM converged after $(length(lls_multi)) iterations\n")
print("Log-likelihood improved by $(round(lls_multi[end] - lls_multi[1], digits=1))\n");

# Plot multi-trial convergence
p5 = plot(lls_multi, xlabel="EM Iteration", ylabel="Log-Likelihood",
          title="Multi-Trial Model Convergence", legend=false, lw=2, color=:darkgreen)

# ## Multi-Trial State Decoding

# Decode hidden states for all trials and visualize as a multi-trial heatmap.

all_pred_labels = viterbi(test_model_multi, all_data, Φ_total);

# Calculate overall accuracy across all trials
all_true_matrix = hcat(all_true_labels...);
all_pred_matrix = hcat(all_pred_labels...);
total_accuracy = mean(all_true_matrix .== all_pred_matrix);

print("Overall state prediction accuracy: $(round(total_accuracy*100, digits=1))%\n");

# Visualize subset of trials (first 10 trials, first 500 time points)
n_trials_display = 10
n_time_display = 500

true_subset = hcat(all_true_labels[1:n_trials_display]...)'[:, 1:n_time_display]
pred_subset = hcat(all_pred_labels[1:n_trials_display]...)'[:, 1:n_time_display]

p6 = plot(
    heatmap(true_subset, colormap=:roma, title="True States ($n_trials_display trials)",
           xticks=false, ylabel="Trial", colorbar=false),
    heatmap(pred_subset, colormap=:roma, title="Predicted States (Viterbi)", 
           xlabel="Time Steps", ylabel="Trial", colorbar=false),
    layout=(2, 1), size=(900, 400)
)

# ## Model Assessment Summary

# Compare learned parameters with true parameters
true_A_orig = [0.99 0.01; 0.05 0.95]
A_error = norm(true_A_orig - test_model_multi.A) / norm(true_A_orig)
print("Transition matrix relative error: $(round(A_error, digits=4))\n")

true_π_orig = [0.8; 0.2]
π_error = norm(true_π_orig - test_model_multi.πₖ) / norm(true_π_orig)
print("Initial distribution relative error: $(round(π_error, digits=4))\n")

print("\nRegression Coefficient Recovery:\n")
print("State 1 - True β:    [3.0, 2.0, 2.0, 3.0]\n")
print("State 1 - Learned β: $(round.(test_model_multi.B[1].β[:, 1], digits=2))\n")
print("State 2 - True β:    [-4.0, -2.0, 3.0, 2.0]\n") 
print("State 2 - Learned β: $(round.(test_model_multi.B[2].β[:, 1], digits=2))\n");

# ## Summary
#
# This tutorial demonstrated the complete workflow for Hidden Markov Models with regression emissions:
#
# **Key Concepts:**
# - **Discrete latent states** with different regression relationships in each state
# - **Markovian dynamics** governing state transitions over time  
# - **EM algorithm** for joint parameter learning and state inference
# - **Viterbi decoding** for finding most likely state sequences
#
# **Applications:**
# - Modeling switching dynamics and regime changes
# - Context-dependent input-output relationships  
# - Multiple independent trial analysis
# - Robust parameter estimation across sequences
#
# GLM-HMMs provide a powerful framework for modeling data with discrete latent structure,
# making them valuable for neuroscience, economics, and other domains with switching behaviors.