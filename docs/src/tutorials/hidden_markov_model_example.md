```@meta
EditURL = "../../examples/HMM.jl"
```

# Simulating and Fitting a Hidden Markov Model with Gaussian Emissions

This tutorial demonstrates how to use `StateSpaceDynamics.jl` to create, sample from, and fit Hidden
Markov Models (HMMs) with Gaussian emission distributions. This is the classical HMM formulation where
each hidden state generates observations from a different multivariate Gaussian distribution.

Unlike GLM-HMMs, this model doesn't use input features - each state simply emits observations
from its own characteristic Gaussian distribution. This makes it ideal for clustering time series data,
identifying behavioral regimes, or modeling switching dynamics where each state has a distinct
statistical signature.

## Load Required Packages

````@example hidden_markov_model_example
using LinearAlgebra
using Plots
using Random
using StateSpaceDynamics
using StableRNGs
using Statistics: mean, std
using LaTeXStrings
````

Set up reproducible random number generation

````@example hidden_markov_model_example
rng = StableRNG(1234);
nothing #hide
````

## Create a Gaussian Emission HMM

We'll create an HMM with two hidden states, each emitting 2D Gaussian observations.
This creates a simple but illustrative model where hidden states correspond
to different regions in the observation space.

````@example hidden_markov_model_example
output_dim = 2;  # Each observation is a 2D vector
nothing #hide
````

Define state transition dynamics: $A_{ij} = P(\text{state}_t = j \mid \text{state}_{t-1} = i)$  \\
High diagonal values mean states are "sticky" (tend to persist)

````@example hidden_markov_model_example
A = [0.99 0.01;    # From state 1: 99% stay, 1% switch to state 2
     0.05 0.95];   # From state 2: 5% switch to state 1, 95% stay
nothing #hide
````

Initial state probabilities: $\pi_k = P(\text{state}_1 = k)$

````@example hidden_markov_model_example
πₖ = [0.5; 0.5];
nothing #hide
````

Define emission distributions for each hidden state
State 1: Centered at (-1, -1) with small variance (tight cluster)

````@example hidden_markov_model_example
μ_1 = [-1.0, -1.0]
Σ_1 = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
emission_1 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1);
nothing #hide
````

State 2: Centered at (1, 1) with larger variance (more spread out)

````@example hidden_markov_model_example
μ_2 = [1.0, 1.0]
Σ_2 = 0.2 * Matrix{Float64}(I, output_dim, output_dim)
emission_2 = GaussianEmission(output_dim=output_dim, μ=μ_2, Σ=Σ_2);
nothing #hide
````

Construct the complete HMM

````@example hidden_markov_model_example
model = HiddenMarkovModel(
    K=2,                        # Number of hidden states
    B=[emission_1, emission_2], # Emission distributions
    A=A,                        # State transition matrix
    πₖ=πₖ                      # Initial state distribution
);

print("Created Gaussian HMM with 2 states:\n")
print("State 1: μ = $μ_1, σ² = $(Σ_1[1,1]) (tight cluster)\n")
print("State 2: μ = $μ_2, σ² = $(Σ_2[1,1]) (looser cluster)\n");
nothing #hide
````

## Sample from the HMM

Generate synthetic data from our true model. Each state generates observations
from its own Gaussian distribution without requiring input features. The rand function
samples both the hidden state sequence and the corresponding observations.

````@example hidden_markov_model_example
num_samples = 10000;
true_labels, data = rand(rng, model, n=num_samples);
nothing #hide
````

## Visualize the Sampled Dataset

Create a 2D scatter plot showing observations colored by their true hidden state.
This illustrates how each state generates observations from a distinct region of space.
We will also plot a trajectory line to show the temporal evolution for the first 1000 timepoints.

````@example hidden_markov_model_example
x_vals = data[1, 1:num_samples]
y_vals = data[2, 1:num_samples]
labels_slice = true_labels[1:num_samples]

state_colors = [:dodgerblue, :crimson]

p1 = plot()

for state in 1:2
    idx = findall(labels_slice .== state)
    scatter!(x_vals[idx], y_vals[idx];
        color=state_colors[state],
        label="State $state",
        markersize=3,
        alpha=0.6)
end

plot!(x_vals[1:1000], y_vals[1:1000];
    color=:gray, lw=1, alpha=0.3, label="Trajectory")

scatter!([x_vals[1]], [y_vals[1]]; marker=:star5, markersize=8,
         color=:green, label="Start")
scatter!([x_vals[end]], [y_vals[end]]; marker=:diamond, markersize=6,
         color=:black, label="End")

plot!(xlabel=L"x_1", ylabel=L"x_2",
      title="HMM Emissions by True Hidden State",
      legend=:topleft)
````

## Initialize and Fit HMM with EM

In reality, we only observe the data, not the hidden states. The goal of fitting
is to learn the latent state sequence and the model parameters that best explain the data.
We will initialize a new HMM with incorrect parameters and use the Expectation-Maximization (EM)
algorithm to iteratively refine the parameters and infer the hidden states.

````@example hidden_markov_model_example
μ_1_init = [-0.25, -0.25]  # Closer to center than true
Σ_1_init = 0.3 * Matrix{Float64}(I, output_dim, output_dim)  # Larger variance
emission_1_init = GaussianEmission(output_dim=output_dim, μ=μ_1_init, Σ=Σ_1_init);

μ_2_init = [0.25, 0.25]    # Closer to center than true
Σ_2_init = 0.5 * Matrix{Float64}(I, output_dim, output_dim)  # Much larger variance
emission_2_init = GaussianEmission(output_dim=output_dim, μ=μ_2_init, Σ=Σ_2_init);

A_init = [0.8 0.2; 0.05 0.95]  # Less persistent than true model
πₖ_init = [0.6, 0.4];           # Biased toward state 1

test_model = HiddenMarkovModel(K=2, B=[emission_1_init, emission_2_init],
                              A=A_init, πₖ=πₖ_init);
nothing #hide
````

Fit using Expectation-Maximization

````@example hidden_markov_model_example
lls = fit!(test_model, data);

print("EM converged in $(length(lls)) iterations\n")
print("Log-likelihood improved by $(round(lls[end] - lls[1], digits=1))\n");
nothing #hide
````

Plot EM convergence

````@example hidden_markov_model_example
p2 = plot(lls, xlabel="EM Iteration", ylabel="Log-Likelihood",
          title="EM Algorithm Convergence", legend=false,
          marker=:circle, markersize=3, lw=2, color=:darkblue)

p2
````

## Hidden State Decoding with Viterbi

Now that we have learned the model parameters from the observed data, we can decode the most likely sequence
of hidden states using the Viterbi algorithm. Then, in this toy example where we know the true latent state path,
we can assess the accuracy of our state predictions.

````@example hidden_markov_model_example
pred_labels = viterbi(test_model, data);

accuracy = mean(true_labels .== pred_labels)
````

Calling a specific set of parameters "state 1" and "state 2" is arbitrary and does not affect the
correctness of the model. The EM algorithm can converge with the states swapped from our
original convention. We check for this and correct it if necessary.

````@example hidden_markov_model_example
swapped_pred = 3 .- pred_labels  # Convert 1→2, 2→1
swapped_accuracy = mean(true_labels .== swapped_pred)

if swapped_accuracy > accuracy
    pred_labels = swapped_pred
    accuracy = swapped_accuracy
    print("Detected and corrected label switching\n")
end

print("State prediction accuracy: $(round(accuracy*100, digits=1))%\n");
nothing #hide
````

Our model looks like it is doing pretty well! Let's visualize the predicted and true
state sequences as heatmaps (first 1000 timepoints)

````@example hidden_markov_model_example
n_display = 1000
true_seq = reshape(true_labels[1:n_display], 1, :)
pred_seq = reshape(pred_labels[1:n_display], 1, :)

p3 = plot(
    heatmap(true_seq, colormap=:roma, title="True State Sequence",
           xticks=false, yticks=false, colorbar=false),
    heatmap(pred_seq, colormap=:roma, title="Predicted State Sequence (Viterbi)",
           xlabel="Time Steps (1-$n_display)", xticks=0:200:n_display,
           yticks=false, colorbar=false),
    layout=(2, 1), size=(800, 300)
)
````

## Multiple Independent Trials

Many real applications involve multiple independent sequences (e.g., multiple subjects,
sessions, or trials). In `StateSpaceDynamics.jl`, it is easy to incorporate data from multiple
trials in parameters learning. Once again, we will generate a synthetic dataset from our
ground truth model to illustrate this process.

````@example hidden_markov_model_example
n_trials = 100    # Number of independent sequences
n_samples = 1000  # Length of each sequence

all_true_labels = Vector{Vector{Int}}(undef, n_trials);
all_data = Vector{Matrix{Float64}}(undef, n_trials);

for i in 1:n_trials  # Sample each trial independently
    labels_trial, data_trial = rand(rng, model, n=n_samples)
    all_true_labels[i] = labels_trial
    all_data[i] = data_trial
end

total_state1_prop = mean([mean(labels .== 1) for labels in all_true_labels])
print("Average State 1 proportion: $(round(total_state1_prop, digits=3))\n");
nothing #hide
````

## Multi-Trial HMM Fitting

When fitting to multiple independent sequences, EM accounts for each sequence
starting independently from the initial state distribution. Here, we initialize
a new model and fit it to all trials simultaneously.

````@example hidden_markov_model_example
test_model_multi = HiddenMarkovModel(
    K=2,
    B=[deepcopy(emission_1_init), deepcopy(emission_2_init)],
    A=A_init, πₖ=πₖ_init
)

lls_multi = fit!(test_model_multi, all_data);
nothing #hide
````

Let's check on how our training went and what parameters we learned.

````@example hidden_markov_model_example
print("Multi-trial EM converged in $(length(lls_multi)) iterations\n")
print("Log-likelihood improved by $(round(lls_multi[end] - lls_multi[1], digits=1))\n");
print("Multi-trial learned parameters:\n")
print("State 1: μ = $(round.(test_model_multi.B[1].μ, digits=3)), σ² = $(round(test_model_multi.B[1].Σ[1,1], digits=3))\n")
print("State 2: μ = $(round.(test_model_multi.B[2].μ, digits=3)), σ² = $(round(test_model_multi.B[2].Σ[1,1], digits=3))\n");
nothing #hide
````

Visualize multi-trial EM convergence

````@example hidden_markov_model_example
p4 = plot(lls_multi, xlabel="EM Iteration", ylabel="Log-Likelihood",
          title="Multi-Trial EM Convergence", legend=false,
          marker=:circle, markersize=3, lw=2, color=:darkgreen)
````

## Multi-Trial State Decoding

Now that we have done parameter learning, we can use Viterbi to find the
most likely hidden state sequence for each trial with a single function call.

````@example hidden_markov_model_example
all_pred_labels_vec = viterbi(test_model_multi, all_data);

all_pred_labels = hcat(all_pred_labels_vec...)';      # trials × time
all_true_labels_matrix = hcat(all_true_labels...)';   # trials × time
````

Calculate overall accuracy across all trials accounting for label switching

````@example hidden_markov_model_example
overall_accuracy = mean(all_true_labels_matrix .== all_pred_labels);

swapped_pred_all = 3 .- all_pred_labels;
swapped_accuracy_all = mean(all_true_labels_matrix .== swapped_pred_all);

if swapped_accuracy_all > overall_accuracy
    all_pred_labels = swapped_pred_all
    overall_accuracy = swapped_accuracy_all
    print("Corrected label switching in multi-trial analysis\n")
end

print("Overall state prediction accuracy: $(round(overall_accuracy*100, digits=1))%\n");
nothing #hide
````

We can also look at per-trial accuracies to see how consistent the model is across trials.

````@example hidden_markov_model_example
trial_accuracies = [mean(all_true_labels_matrix[i, :] .== all_pred_labels[i, :]) for i in 1:n_trials]
print("Per-trial accuracy: $(round(mean(trial_accuracies)*100, digits=1))% ± $(round(std(trial_accuracies)*100, digits=1))%\n");
nothing #hide
````

Visualize subset of trials (first 10 trials, first 500 timepoints)

````@example hidden_markov_model_example
n_trials_display = 10
n_time_display = 500

true_subset = all_true_labels_matrix[1:n_trials_display, 1:n_time_display]
pred_subset = all_pred_labels[1:n_trials_display, 1:n_time_display]

p5 = plot(
    heatmap(true_subset, colormap=:roma, title="True States ($n_trials_display trials)",
           xticks=false, ylabel="Trial", colorbar=false),
    heatmap(pred_subset, colormap=:roma, title="Predicted States (Viterbi)",
           xlabel="Time Steps", ylabel="Trial", colorbar=false),
    layout=(2, 1), size=(900, 400)
)
````

## Parameter Recovery Assessment
Since we have access to the true model parameters, we can quantitatively assess
how well the multi-trial fitting procedure recovered them.

````@example hidden_markov_model_example
true_μ1_orig, true_μ2_orig = [-1.0, -1.0], [1.0, 1.0]
learned_μ1 = test_model_multi.B[1].μ
learned_μ2 = test_model_multi.B[2].μ
````

Compare emission model mean vectors

````@example hidden_markov_model_example
μ1_error = norm(true_μ1_orig - learned_μ1) / norm(true_μ1_orig)
μ2_error = norm(true_μ2_orig - learned_μ2) / norm(true_μ2_orig)

print("Mean vector recovery errors:\n")
print("State 1: $(round(μ1_error*100, digits=1))%, State 2: $(round(μ2_error*100, digits=1))%\n")
````

Compare covariance matrices

````@example hidden_markov_model_example
true_Σ1_orig, true_Σ2_orig = 0.1, 0.2
learned_Σ1 = test_model_multi.B[1].Σ[1,1]
learned_Σ2 = test_model_multi.B[2].Σ[1,1]

Σ1_error = abs(true_Σ1_orig - learned_Σ1) / true_Σ1_orig
Σ2_error = abs(true_Σ2_orig - learned_Σ2) / true_Σ2_orig

print("Variance recovery errors:\n")
print("State 1: $(round(Σ1_error*100, digits=1))%, State 2: $(round(Σ2_error*100, digits=1))%\n");
nothing #hide
````

Compare transition matrices

````@example hidden_markov_model_example
true_A_orig = [0.99 0.01; 0.05 0.95]
A_error = norm(true_A_orig - test_model_multi.A) / norm(true_A_orig)
print("Transition matrix error: $(round(A_error*100, digits=1))%\n")
````

## Summary

This tutorial demonstrated the complete workflow for Gaussian emission Hidden Markov Models.
We covered how to create, sample from, fit, and perform state inference with HMMs using
`StateSpaceDynamics.jl`.

**Key Concepts:**
- **Discrete hidden states** with Gaussian emission distributions
- **Temporal dependencies** through Markovian state transitions
- **EM algorithm** for joint parameter learning and state inference
- **Viterbi decoding** for finding most likely state sequences

**Technical Insights:**
- **Label switching** is a common identifiability issue requiring detection and correction
- **Multi-trial analysis** provides more robust parameter estimates than single sequences
- **Parameter recovery** quality depends on state separation and sequence length
- **Convergence monitoring** through log-likelihood plots ensures proper algorithm behavior

**Applications:**
- Time series clustering and regime detection
- Behavioral state analysis in sequential data
- Exploratory analysis of temporal datasets with latent structure
- Foundation for more complex state-space models

Gaussian HMMs provide a fundamental framework for modeling sequential data with discrete
latent structure, serving as both standalone models and building blocks for more
sophisticated probabilistic time series methods.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

