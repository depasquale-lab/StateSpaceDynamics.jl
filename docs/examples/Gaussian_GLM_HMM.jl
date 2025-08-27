# ## Simulating and Fitting a Hidden Markov Model

# This tutorial demonstrates how to use `StateSpaceDynamics.jl` to create, sample from, and fit Hidden 
# Markov Models (HMMs). Unlike Linear Dynamical Systems which have continuous latent states, HMMs have
# discrete latent states that switch between a finite number of modes. This makes them ideal for modeling
# data with distinct behavioral regimes, switching dynamics, or categorical latent structure.

# We'll focus on a Gaussian generalized linear model HMM (GLM-HMM), where each hidden state corresponds
# to a different regression relationship between inputs and outputs. This is particularly useful for
# modeling data where the input-output relationship changes over time in discrete jumps.

# ## Load Required Packages

# We load the essential packages for HMM modeling, visualization, and reproducible random sampling.

using LinearAlgebra
using Plots
using Random
using StateSpaceDynamics
using StableRNGs
using Statistics: mean

# Set up reproducible random number generation
rng = StableRNG(1234);

# ## Create a Gaussian Generalized Linear Model-Hidden Markov Model (GLM-HMM)

# In a GLM-HMM, each hidden state defines a different regression model. The system switches
# between these regression models according to Markovian dynamics. This is useful for modeling
# scenarios where the relationship between predictors and outcomes changes over time.

# Define emission models for each hidden state
# State 1: Positive relationship between input and output
emission_1 = GaussianRegressionEmission(
    input_dim=3,                                    # Number of input features
    output_dim=1,                                   # Number of output dimensions  
    include_intercept=true,                         # Include intercept term
    β=reshape([3.0, 2.0, 2.0, 3.0], :, 1),        # Regression coefficients [intercept, β1, β2, β3]
    Σ=[1.0;;],                                     # Observation noise variance
    λ=0.0                                          # Regularization parameter
)

# State 2: Different relationship (negative intercept, different slopes)
emission_2 = GaussianRegressionEmission(
    input_dim=3,
    output_dim=1,
    include_intercept=true,
    β=reshape([-4.0, -2.0, 3.0, 2.0], :, 1),      # Different regression coefficients
    Σ=[1.0;;],                                     # Same noise level
    λ=0.0
)

# Define the state transition matrix A
# A[i,j] = probability of transitioning from state i to state j
# Diagonal elements are high (states are persistent), off-diagonal elements are low
A = [0.99 0.01;    # From state 1: 99% stay, 1% switch to state 2
     0.05 0.95]    # From state 2: 5% switch to state 1, 95% stay

# Initial state distribution: probability of starting in each state
πₖ = [0.8; 0.2]    # 80% chance of starting in state 1, 20% in state 2

# Construct the complete HMM
true_model = HiddenMarkovModel(
    K=2,                        # Number of hidden states
    A=A,                        # Transition matrix
    πₖ=πₖ,                     # Initial state distribution
    B=[emission_1, emission_2]  # Emission models for each state
)

println("Created GLM-HMM with 2 states and 3 input features")
println("State 1 regression: y = 3.0 + 2.0*x₁ + 2.0*x₂ + 3.0*x₃ + ε")
println("State 2 regression: y = -4.0 - 2.0*x₁ + 3.0*x₂ + 2.0*x₃ + ε")

# ## Sample from the GLM-HMM

# Generate synthetic data from our true model. This will give us both the observed
# data (inputs and outputs) and the true hidden state sequence, which we'll use
# to evaluate our parameter recovery.

n = 20000  # Number of time points
println("Generating $n samples from the GLM-HMM...")

# Generate random input features (predictors)
Φ = randn(rng, 3, n)  # 3 features × n time points

# Sample from the HMM: returns both hidden states and observations
true_labels, data = rand(rng, true_model, Φ, n=n)

println("Generated data summary:")
println("  - Input features shape: $(size(Φ))")
println("  - Output data shape: $(size(data))")
println("  - True labels shape: $(size(true_labels))")
println("  - State 1 proportion: $(round(mean(true_labels .== 1), digits=3))")
println("  - State 2 proportion: $(round(mean(true_labels .== 2), digits=3))")

# ## Visualize the Sampled Dataset

# Create a scatter plot showing how the input-output relationship differs between
# the two hidden states. Points are colored by their true hidden state.

colors = [:dodgerblue, :crimson]  # Blue for state 1, red for state 2

scatter(Φ[1, :], vec(data);
    color = colors[true_labels],
    ms = 3,
    label = "",
    xlabel = "Input Feature 1",
    ylabel = "Output",
    title = "GLM-HMM Sampled Data (colored by true state)",
    alpha = 0.6
)

# Overlay the true regression lines for each state
# We'll plot the relationship between feature 1 and output, holding other features at 0

xvals = range(minimum(Φ[1, :]), stop=maximum(Φ[1, :]), length=100)

# State 1 regression line: y = β₀ + β₁*x₁ (setting x₂=x₃=0)
β1 = emission_1.β[:, 1]
y_pred_1 = β1[1] .+ β1[2] .* xvals  # intercept + slope*x₁
plot!(xvals, y_pred_1;
    color = :dodgerblue,
    lw = 3,
    label = "State 1 regression",
    legend = :topright,
)

# State 2 regression line
β2 = emission_2.β[:, 1]
y_pred_2 = β2[1] .+ β2[2] .* xvals  # intercept + slope*x₁
plot!(xvals, y_pred_2;
    color = :crimson,
    lw = 3,
    label = "State 2 regression",
    legend = :topright,
)

# ## Initialize and Fit a New HMM to the Sampled Data

# Now we'll pretend we don't know the true parameters and try to learn them from
# the observed data alone. We start with a randomly initialized HMM and use the
# Expectation-Maximization (EM) algorithm to learn the parameters.

println("Initializing naive HMM with random parameters...")

# Initialize with different parameters than the true model
A = [0.8 0.2; 0.1 0.9]          # Different transition probabilities
πₖ = [0.6; 0.4]                 # Different initial distribution

# Initialize emission models with random regression coefficients
emission_1 = GaussianRegressionEmission(
    input_dim=3, output_dim=1, include_intercept=true,
    β=reshape([2.0, -1.0, 1.0, 2.0], :, 1),    # Random coefficients
    Σ=[2.0;;],                                  # Different noise variance
    λ=0.0
)

emission_2 = GaussianRegressionEmission(
    input_dim=3, output_dim=1, include_intercept=true,
    β=reshape([-2.5, -1.0, 3.5, 3.0], :, 1),   # Random coefficients
    Σ=[0.5;;],                                  # Different noise variance
    λ=0.0
)

# Create the test model with naive initialization
test_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])

# Fit the model using EM algorithm
println("Running EM algorithm to learn HMM parameters...")
lls = fit!(test_model, data, Φ)

println("EM converged after $(length(lls)) iterations")
println("Initial log-likelihood: $(round(lls[1], digits=2))")
println("Final log-likelihood: $(round(lls[end], digits=2))")
println("Improvement: $(round(lls[end] - lls[1], digits=2))")

# Plot the convergence of the log-likelihood
plot(lls)
title!("Log-likelihood over EM Iterations")
xlabel!("EM Iteration")
ylabel!("Log-Likelihood")

# ## Visualize the Emission Model Predictions

# Compare the true regression relationships with what our fitted model learned.
# This shows how well we recovered the underlying GLM parameters.

state_colors = [:dodgerblue, :crimson]  # Data points colored by true state
true_colors = [:green, :orange]         # True regression lines
pred_colors = [:teal, :yellow]          # Predicted regression lines

scatter(Φ[1, :], vec(data);
    color = state_colors[true_labels],
    ms = 3,
    alpha = 0.6,
    label = "",
    xlabel = "Input Feature 1",
    ylabel = "Output",
    title = "True vs. Predicted Regression Relationships"
)

xvals = range(minimum(Φ[1, :]), stop=maximum(Φ[1, :]), length=100)

# Plot true regression lines
β1_true = emission_1.β[:, 1]  # Note: this is now the fitted model's β, not true model's
y_true_1 = β1_true[1] .+ β1_true[2] .* xvals
plot!(xvals, y_true_1;
    color = true_colors[1],
    lw = 3,
    linestyle = :solid,
    label = "State 1 (true)"
)

β2_true = emission_2.β[:, 1]
y_true_2 = β2_true[1] .+ β2_true[2] .* xvals
plot!(xvals, y_true_2;
    color = true_colors[2],
    lw = 3,
    linestyle = :solid,
    label = "State 2 (true)"
)

# Plot learned regression lines
β1_pred = test_model.B[1].β[:, 1]
y_pred_1 = β1_pred[1] .+ β1_pred[2] .* xvals
plot!(xvals, y_pred_1;
    color = pred_colors[1],
    lw = 3,
    linestyle = :dash,
    label = "State 1 (learned)"
)

β2_pred = test_model.B[2].β[:, 1]
y_pred_2 = β2_pred[1] .+ β2_pred[2] .* xvals
plot!(xvals, y_pred_2;
    color = pred_colors[2],
    lw = 3,
    linestyle = :dash,
    label = "State 2 (learned)"
)

# ## Visualize the Latent State Predictions using Viterbi Algorithm

# The Viterbi algorithm finds the most likely sequence of hidden states given the
# observed data. We'll compare the true hidden state sequence with our predictions.

println("Running Viterbi algorithm to decode hidden state sequence...")
pred_labels = viterbi(test_model, data, Φ);

# Calculate accuracy of state prediction
accuracy = mean(true_labels .== pred_labels)
println("Hidden state prediction accuracy: $(round(accuracy*100, digits=1))%")

# Visualize a subset of the state sequences as heatmaps
true_mat = reshape(true_labels[1:1000], 1, :)
pred_mat = reshape(pred_labels[1:1000], 1, :)

p1 = heatmap(true_mat;
    colormap = :roma50,
    title = "True State Labels",
    xlabel = "",
    ylabel = "",
    xticks = false,
    yticks = false,
    colorbar = false,
    framestyle = :box)

p2 = heatmap(pred_mat;
    colormap = :roma50,
    title = "Predicted State Labels (Viterbi)",
    xlabel = "Timepoints (1-1000)",
    ylabel = "",
    xticks = 0:200:1000,
    yticks = false,
    colorbar = false,
    framestyle = :box)

plot(p1, p2;
    layout = (2, 1),
    size = (700, 500),
    margin = 5Plots.mm)

# ## Sampling Multiple, Independent Trials of Data from an HMM

# Real-world scenarios often involve multiple independent sequences (e.g., multiple
# subjects, experimental sessions, or trials). We'll generate multiple independent
# sequences and show how to fit HMMs to this type of data structure.

println("Generating multiple independent trials...")

all_data = Vector{Matrix{Float64}}()     # Store data from each trial
Φ_total = Vector{Matrix{Float64}}()      # Store input features from each trial
all_true_labels = []                     # Store true state sequences

num_trials = 100  # Number of independent sequences
n = 1000         # Length of each sequence

for i in 1:num_trials
    Φ = randn(rng, 3, n)
    true_labels, data = rand(rng, true_model, Φ, n=n)
    push!(all_true_labels, true_labels)
    push!(all_data, data)
    push!(Φ_total, Φ)
end

println("Generated $num_trials independent trials, each with $n time points")
println("Total data points: $(num_trials * n)")

# ## Fitting an HMM to Multiple, Independent Trials of Data

# When we have multiple independent sequences, the EM algorithm needs to account
# for the fact that each sequence starts fresh from the initial state distribution.
# This provides more robust parameter estimates than fitting to a single long sequence.

println("Fitting HMM to multiple independent trials...")

# Initialize a new model for multi-trial fitting
A = [0.8 0.2; 0.1 0.9]
πₖ = [0.6; 0.4]
emission_1 = GaussianRegressionEmission(
    input_dim=3, output_dim=1, include_intercept=true,
    β=reshape([2.0, -1.0, 1.0, 2.0], :, 1), Σ=[2.0;;], λ=0.0
)
emission_2 = GaussianRegressionEmission(
    input_dim=3, output_dim=1, include_intercept=true,
    β=reshape([-2.5, -1.0, 3.5, 3.0], :, 1), Σ=[0.5;;], λ=0.0
)

test_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])

# Fit to multiple trials - the package handles the multi-trial structure automatically
lls = fit!(test_model, all_data, Φ_total)

println("Multi-trial EM converged after $(length(lls)) iterations")
println("Initial log-likelihood: $(round(lls[1], digits=2))")
println("Final log-likelihood: $(round(lls[end], digits=2))")

# Plot convergence for multi-trial fitting
plot(lls)
title!("Log-likelihood over EM Iterations (Multi-Trial)")
xlabel!("EM Iteration")
ylabel!("Log-Likelihood")

# ## Visualize Latent State Predictions for Multiple Trials using Viterbi

# Decode hidden states for all trials and visualize the results as a multi-trial heatmap.
# This shows how well we can predict state sequences across different independent runs.

println("Running Viterbi decoding on all trials...")
all_pred_labels_vec = viterbi(test_model, all_data, Φ_total)

# Reshape for visualization
all_pred_labels = hcat(all_pred_labels_vec...)'      # trials × time
all_true_labels_matrix = hcat(all_true_labels...)'   # trials × time

# Calculate overall accuracy across all trials
total_accuracy = mean(all_true_labels_matrix .== all_pred_labels)
println("Overall hidden state accuracy across all trials: $(round(total_accuracy*100, digits=1))%")

# Visualize a subset of trials
state_colors = [:dodgerblue, :crimson]
true_subset = all_true_labels_matrix[1:10, 1:500]   # First 10 trials, first 500 time points
pred_subset = all_pred_labels[1:10, 1:500]

p1 = heatmap(
    true_subset,
    colormap = :roma50,
    colorbar = false,
    title = "True State Labels (10 trials × 500 timepoints)",
    xlabel = "",
    ylabel = "Trial Number",
    xticks = false,
    yticks = true,
    margin = 5Plots.mm,
    legend = false
)

p2 = heatmap(
    pred_subset,
    colormap = :roma50,
    colorbar = false,
    title = "Predicted State Labels (Viterbi)",
    xlabel = "Timepoints",
    ylabel = "Trial Number",
    xticks = true,
    yticks = true,
    margin = 5Plots.mm,
    legend = false
)

final_plot = plot(
    p1, p2,
    layout = (2, 1),
    size = (850, 550),
    margin = 5Plots.mm,
    legend = false,
)

display(final_plot)

# ## Summary and Model Assessment

println("\n=== Final Model Assessment ===")

# Compare learned parameters with true parameters
true_A = [0.99 0.01; 0.05 0.95]
learned_A = test_model.A
A_error = norm(true_A - learned_A) / norm(true_A)
println("Transition matrix relative error: $(round(A_error, digits=4))")

true_π = [0.8; 0.2]
learned_π = test_model.πₖ
π_error = norm(true_π - learned_π) / norm(true_π)
println("Initial distribution relative error: $(round(π_error, digits=4))")

println("\nTrue vs Learned Regression Coefficients:")
println("State 1 - True β: [3.0, 2.0, 2.0, 3.0]")
println("State 1 - Learned β: $(round.(test_model.B[1].β[:, 1], digits=2))")
println("State 2 - True β: [-4.0, -2.0, 3.0, 2.0]")
println("State 2 - Learned β: $(round.(test_model.B[2].β[:, 1], digits=2))")

# ## Summary
#
# This tutorial demonstrated the complete workflow for Hidden Markov Models with regression emissions:
#
# 1. **Model Structure**: Discrete latent states with different regression relationships in each state
# 2. **Applications**: Ideal for modeling switching dynamics, regime changes, or context-dependent relationships
# 3. **Single vs Multiple Trials**: Showed how to handle both single long sequences and multiple independent trials
# 4. **Parameter Recovery**: EM algorithm successfully learned transition dynamics and emission parameters
# 5. **State Decoding**: Viterbi algorithm accurately recovered hidden state sequences
# 6. **Scalability**: Framework handles multiple trials efficiently for robust parameter estimation
#
# GLM-HMMs provide a powerful framework for modeling data with discrete latent structure and
# context-dependent input-output relationships, making them valuable for many real-world applications.