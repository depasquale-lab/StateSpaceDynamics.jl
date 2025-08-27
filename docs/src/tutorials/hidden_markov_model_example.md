```@meta
EditURL = "../../examples/HMM.jl"
```

## Simulating and Fitting a Hidden Markov Model with Gaussian Emissions

This tutorial demonstrates how to use `StateSpaceDynamics.jl` to create, sample from, and fit Hidden
Markov Models (HMMs) with Gaussian emission distributions. This is the classical HMM formulation where
each hidden state generates observations from a different multivariate Gaussian distribution.

Unlike the GLM-HMM in the previous tutorial, this model doesn't have input features - each state
simply emits observations from its own characteristic Gaussian distribution. This makes it ideal
for clustering time series data, identifying behavioral regimes, or modeling switching dynamics
in systems where each state has a distinct statistical signature.

## Load Required Packages

We load the essential packages for HMM modeling, visualization, and reproducible analysis.

````@example hidden_markov_model_example
using LinearAlgebra
using Plots
using Random
using StateSpaceDynamics
using StableRNGs
using Statistics: mean, std
````

Set up reproducible random number generation

````@example hidden_markov_model_example
rng = StableRNG(1234);
nothing #hide
````

## Create a Gaussian Emission HMM

We'll create an HMM with two hidden states, each emitting 2D Gaussian observations.
This creates a simple but illustrative model where the hidden states correspond
to different regions in the observation space.

````@example hidden_markov_model_example
output_dim = 2  # Each observation is a 2D vector
````

Define the state transition dynamics
High diagonal values mean states are "sticky" (tend to persist)

````@example hidden_markov_model_example
A = [0.99 0.01;    # From state 1: 99% stay in state 1, 1% switch to state 2
     0.05 0.95];   # From state 2: 5% switch to state 1, 95% stay in state 2
nothing #hide
````

Initial state probabilities (equal probability of starting in either state)

````@example hidden_markov_model_example
πₖ = [0.5; 0.5]
````

Define emission distributions for each hidden state
State 1: Centered at (-1, -1) with small variance (tight cluster)

````@example hidden_markov_model_example
μ_1 = [-1.0, -1.0]                                          # Mean vector
Σ_1 = 0.1 * Matrix{Float64}(I, output_dim, output_dim)      # Covariance matrix (0.1 * Identity)
emission_1 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)
````

State 2: Centered at (1, 1) with larger variance (more spread out)

````@example hidden_markov_model_example
μ_2 = [1.0, 1.0]                                           # Mean vector
Σ_2 = 0.2 * Matrix{Float64}(I, output_dim, output_dim)     # Covariance matrix (0.2 * Identity)
emission_2 = GaussianEmission(output_dim=output_dim, μ=μ_2, Σ=Σ_2)
````

Construct the complete HMM

````@example hidden_markov_model_example
model = HiddenMarkovModel(
    K=2,                        # Number of hidden states
    B=[emission_1, emission_2], # Emission distributions
    A=A,                        # State transition matrix
    πₖ=πₖ                      # Initial state distribution
)

println("Created Gaussian HMM with 2 states:")
println("  State 1: μ = $μ_1, σ² = $(Σ_1[1,1]) (tight cluster in lower-left)")
println("  State 2: μ = $μ_2, σ² = $(Σ_2[1,1]) (looser cluster in upper-right)")
println("  Transition probabilities encourage state persistence")
````

## Sample from the HMM

Generate synthetic data from our true model. Unlike GLM-HMMs, we don't need
input features - each state generates observations from its own Gaussian distribution.

````@example hidden_markov_model_example
num_samples = 10000
println("Generating $num_samples samples from the Gaussian HMM...")
````

Sample both hidden state sequence and corresponding observations

````@example hidden_markov_model_example
true_labels, data = rand(rng, model, n=num_samples)

println("Generated data summary:")
println("  Data shape: $(size(data)) (dimensions × time)")
println("  Labels shape: $(size(true_labels))")
println("  State 1 proportion: $(round(mean(true_labels .== 1), digits=3))")
println("  State 2 proportion: $(round(mean(true_labels .== 2), digits=3))")
println("  Data range: [$(round(minimum(data), digits=2)), $(round(maximum(data), digits=2))]")
````

## Visualize the Sampled Dataset

Create a 2D scatter plot showing the observations colored by their true hidden state.
This illustrates how each state generates observations from a distinct region of space.

````@example hidden_markov_model_example
x_vals = data[1, 1:num_samples]  # First dimension
y_vals = data[2, 1:num_samples]  # Second dimension
labels_slice = true_labels[1:num_samples]

state_colors = [:dodgerblue, :crimson]  # Blue for state 1, red for state 2

plt = plot()
````

Plot observations for each state separately to get proper legends

````@example hidden_markov_model_example
for state in 1:2
    idx = findall(labels_slice .== state)
    scatter!(x_vals[idx], y_vals[idx];
        color=state_colors[state],
        label="State $state",
        markersize=4,
        alpha=0.6)
end
````

Add a trajectory line to show temporal evolution (faded)

````@example hidden_markov_model_example
plot!(x_vals[1:1000], y_vals[1:1000];  # Show first 1000 points for clarity
    color=:gray,
    lw=1.5,
    linealpha=0.4,
    label="Trajectory")
````

Mark start and end points

````@example hidden_markov_model_example
scatter!([x_vals[1]], [y_vals[1]];
    color=:green,
    markershape=:star5,
    markersize=10,
    label="Start")

scatter!([x_vals[end]], [y_vals[end]];
    color=:black,
    markershape=:diamond,
    markersize=8,
    label="End")

xlabel!("Output Dimension 1")
ylabel!("Output Dimension 2")
title!("HMM Emissions Colored by True Hidden State")

println("Note: The trajectory line shows the temporal sequence connecting observations")
````

## Initialize and Fit a New HMM to the Sampled Data

Now we'll simulate the realistic scenario where we observe only the data, not the
hidden states. We'll initialize an HMM with incorrect parameters and use EM to
learn the true parameters from the observations alone.

````@example hidden_markov_model_example
println("Initializing naive HMM with incorrect parameters...")
````

Initialize with biased/incorrect parameters

````@example hidden_markov_model_example
μ_1 = [-0.25, -0.25]  # Closer to center than true value
Σ_1 = 0.3 * Matrix{Float64}(I, output_dim, output_dim)  # Larger variance than true
emission_1 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)

μ_2 = [0.25, 0.25]    # Closer to center than true value
Σ_2 = 0.5 * Matrix{Float64}(I, output_dim, output_dim)  # Much larger variance than true
````

Note: There's a bug in the original code - emission_2 uses μ_1 and Σ_1, let's fix it:

````@example hidden_markov_model_example
emission_2 = GaussianEmission(output_dim=output_dim, μ=μ_2, Σ=Σ_2)
````

Different transition matrix and initial distribution

````@example hidden_markov_model_example
A = [0.8 0.2;     # Less persistent than true model
     0.05 0.95]   # Asymmetric transitions
πₖ = [0.6, 0.4]   # Biased toward state 1
````

Create the test model

````@example hidden_markov_model_example
test_model = HiddenMarkovModel(K=2, B=[emission_1, emission_2], A=A, πₖ=πₖ)

println("Initial guesses:")
println("  State 1: μ = $μ_1, σ² = $(Σ_1[1,1])")
println("  State 2: μ = $μ_2, σ² = $(Σ_2[1,1])")
````

Fit the model using the Expectation-Maximization algorithm

````@example hidden_markov_model_example
println("Running EM algorithm to learn parameters...")
lls = fit!(test_model, data)

println("EM algorithm converged after $(length(lls)) iterations")
println("Log-likelihood improvement: $(round(lls[end] - lls[1], digits=2))")
````

Plot the convergence of the log-likelihood

````@example hidden_markov_model_example
plot(lls)
title!("Log-likelihood over EM Iterations")
xlabel!("EM Iteration")
ylabel!("Log-Likelihood")
````

Display learned parameters

````@example hidden_markov_model_example
println("Learned parameters:")
println("  State 1: μ = $(round.(test_model.B[1].μ, digits=3)), σ² = $(round(test_model.B[1].Σ[1,1], digits=3))")
println("  State 2: μ = $(round.(test_model.B[2].μ, digits=3)), σ² = $(round(test_model.B[2].Σ[1,1], digits=3))")
````

## Visualize the Latent State Predictions using Viterbi Algorithm

The Viterbi algorithm finds the most likely sequence of hidden states given the
observed data and learned parameters. We'll compare this with the true sequence.

````@example hidden_markov_model_example
println("Running Viterbi algorithm to decode most likely state sequence...")
pred_labels = viterbi(test_model, data);
nothing #hide
````

Calculate the accuracy of state prediction

````@example hidden_markov_model_example
accuracy = mean(true_labels .== pred_labels)
println("State sequence prediction accuracy: $(round(accuracy*100, digits=1))%")
````

Handle potential label switching (EM can converge with states swapped)
Check if swapping labels gives better accuracy

````@example hidden_markov_model_example
swapped_pred = 3 .- pred_labels  # Convert 1→2, 2→1
swapped_accuracy = mean(true_labels .== swapped_pred)

if swapped_accuracy > accuracy
    println("Detected label switching - corrected accuracy: $(round(swapped_accuracy*100, digits=1))%")
    pred_labels = swapped_pred
    accuracy = swapped_accuracy
end
````

Visualize state sequences as heatmaps

````@example hidden_markov_model_example
true_mat = reshape(true_labels[1:1000], 1, :)
pred_mat = reshape(pred_labels[1:1000], 1, :)

p1 = heatmap(true_mat;
    colormap = :roma50,
    title = "True State Labels (first 1000 timepoints)",
    xlabel = "",
    ylabel = "",
    xticks = false,
    yticks = false,
    colorbar = false,
    framestyle = :box)

p2 = heatmap(pred_mat;
    colormap = :roma50,
    title = "Predicted State Labels (Viterbi)",
    xlabel = "Timepoints",
    ylabel = "",
    xticks = 0:200:1000,
    yticks = false,
    colorbar = false,
    framestyle = :box)

plot(p1, p2;
    layout = (2, 1),
    size = (700, 500),
    margin = 5Plots.mm)
````

## Sampling Multiple, Independent Trials of Data from an HMM

In many real applications, we have multiple independent sequences rather than
one long sequence. For example: multiple subjects in an experiment, multiple
recording sessions, or multiple independent time series. We'll demonstrate
how to handle this scenario.

````@example hidden_markov_model_example
println("Generating multiple independent trials...")

n_trials = 100    # Number of independent sequences
n_samples = 1000  # Length of each sequence
````

Pre-allocate storage for efficiency

````@example hidden_markov_model_example
all_true_labels = Vector{Vector{Int}}(undef, n_trials)
all_data = Vector{Matrix{Float64}}(undef, n_trials)
````

Generate independent sequences

````@example hidden_markov_model_example
for i in 1:n_trials
    true_labels, data = rand(rng, model, n=n_samples)
    all_true_labels[i] = true_labels
    all_data[i] = data
end

println("Generated $n_trials independent trials of length $n_samples each")
println("Total data points: $(n_trials * n_samples)")
````

Calculate statistics across trials

````@example hidden_markov_model_example
total_state1_prop = mean([mean(labels .== 1) for labels in all_true_labels])
println("Average proportion of state 1 across trials: $(round(total_state1_prop, digits=3))")
````

## Fitting an HMM to Multiple, Independent Trials of Data

When fitting to multiple independent sequences, the EM algorithm must account
for the fact that each sequence starts independently from the initial state
distribution. This typically provides more robust parameter estimates.

````@example hidden_markov_model_example
println("Fitting HMM to multiple independent trials...")
````

Initialize a fresh model for multi-trial fitting

````@example hidden_markov_model_example
μ_1 = [-0.25, -0.25]
Σ_1 = 0.3 * Matrix{Float64}(I, output_dim, output_dim)
emission_1 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)

μ_2 = [0.25, 0.25]
Σ_2 = 0.5 * Matrix{Float64}(I, output_dim, output_dim)
emission_2 = GaussianEmission(output_dim=output_dim, μ=μ_2, Σ=Σ_2)

A = [0.8 0.2; 0.05 0.95]
πₖ = [0.6, 0.4]
test_model = HiddenMarkovModel(K=2, B=[emission_1, emission_2], A=A, πₖ=πₖ)
````

Fit to all trials simultaneously
The package automatically handles the multi-trial structure

````@example hidden_markov_model_example
lls = fit!(test_model, all_data)

println("Multi-trial EM converged after $(length(lls)) iterations")
println("Final log-likelihood: $(round(lls[end], digits=2))")
````

Plot convergence for multi-trial case

````@example hidden_markov_model_example
plot(lls)
title!("Log-likelihood over EM Iterations (Multi-Trial Fitting)")
xlabel!("EM Iteration")
ylabel!("Log-Likelihood")

println("Final learned parameters (multi-trial):")
println("  State 1: μ = $(round.(test_model.B[1].μ, digits=3)), σ² = $(round(test_model.B[1].Σ[1,1], digits=3))")
println("  State 2: μ = $(round.(test_model.B[2].μ, digits=3)), σ² = $(round(test_model.B[2].Σ[1,1], digits=3))")
````

## Visualize Latent State Predictions for Multiple Trials using Viterbi

Decode the hidden state sequences for all trials and visualize the results
as a heatmap showing state assignments across multiple independent sequences.

````@example hidden_markov_model_example
println("Running Viterbi decoding on all trials...")
all_pred_labels_vec = viterbi(test_model, all_data)
````

Reshape data for easier analysis and visualization

````@example hidden_markov_model_example
all_pred_labels = hcat(all_pred_labels_vec...)'      # trials × time
all_true_labels_matrix = hcat(all_true_labels...)'   # trials × time
````

Calculate overall accuracy across all trials and timepoints

````@example hidden_markov_model_example
overall_accuracy = mean(all_true_labels_matrix .== all_pred_labels)
````

Check for label switching across the entire dataset

````@example hidden_markov_model_example
swapped_pred_all = 3 .- all_pred_labels
swapped_accuracy_all = mean(all_true_labels_matrix .== swapped_pred_all)

if swapped_accuracy_all > overall_accuracy
    println("Detected label switching in multi-trial analysis")
    all_pred_labels = swapped_pred_all
    overall_accuracy = swapped_accuracy_all
end

println("Overall state prediction accuracy across all trials: $(round(overall_accuracy*100, digits=1))%")
````

Calculate per-trial accuracies for robustness assessment

````@example hidden_markov_model_example
trial_accuracies = [mean(all_true_labels_matrix[i, :] .== all_pred_labels[i, :]) for i in 1:n_trials]
println("Mean per-trial accuracy: $(round(mean(trial_accuracies)*100, digits=1))% ± $(round(std(trial_accuracies)*100, digits=1))%")
````

Visualize a subset of trials to show consistency across independent sequences

````@example hidden_markov_model_example
state_colors = [:dodgerblue, :crimson]
true_subset = all_true_labels_matrix[1:10, 1:500]   # First 10 trials, first 500 timepoints
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
    title = "Predicted State Labels (Viterbi Decoding)",
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
````

## Parameter Recovery Analysis

````@example hidden_markov_model_example
println("\n=== Parameter Recovery Assessment ===")
````

Compare true vs learned emission parameters

````@example hidden_markov_model_example
true_μ1, true_μ2 = [-1.0, -1.0], [1.0, 1.0]
learned_μ1 = test_model.B[1].μ
learned_μ2 = test_model.B[2].μ

μ1_error = norm(true_μ1 - learned_μ1) / norm(true_μ1)
μ2_error = norm(true_μ2 - learned_μ2) / norm(true_μ2)

println("Mean vector recovery errors:")
println("  State 1: $(round(μ1_error*100, digits=1))%")
println("  State 2: $(round(μ2_error*100, digits=1))%")
````

Compare transition matrices

````@example hidden_markov_model_example
true_A = [0.99 0.01; 0.05 0.95]
learned_A = test_model.A
A_error = norm(true_A - learned_A) / norm(true_A)
println("Transition matrix recovery error: $(round(A_error*100, digits=1))%")
````

Compare covariance matrices

````@example hidden_markov_model_example
true_Σ1, true_Σ2 = 0.1, 0.2
learned_Σ1 = test_model.B[1].Σ[1,1]
learned_Σ2 = test_model.B[2].Σ[1,1]

Σ1_error = abs(true_Σ1 - learned_Σ1) / true_Σ1
Σ2_error = abs(true_Σ2 - learned_Σ2) / true_Σ2

println("Variance recovery errors:")
println("  State 1: $(round(Σ1_error*100, digits=1))%")
println("  State 2: $(round(Σ2_error*100, digits=1))%")
````

## Summary

This tutorial demonstrated the complete workflow for Gaussian emission Hidden Markov Models:

1. **Model Structure**: Discrete hidden states with Gaussian emission distributions
2. **Applications**: Time series clustering, regime detection, behavioral state analysis
3. **Parameter Learning**: EM algorithm successfully recovered emission parameters and transition dynamics
4. **State Inference**: Viterbi algorithm accurately decoded hidden state sequences
5. **Multi-Trial Analysis**: Robust parameter estimation from multiple independent sequences
6. **Label Switching**: Demonstrated detection and handling of the label switching problem
7. **Scalability**: Efficient handling of large datasets with multiple trials

Gaussian HMMs provide a fundamental framework for modeling time series with discrete latent structure,
forming the foundation for more complex state-space models and serving as a powerful tool for
exploratory data analysis in temporal datasets.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

