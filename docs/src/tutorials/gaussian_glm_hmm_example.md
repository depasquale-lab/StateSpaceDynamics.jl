```@meta
EditURL = "../../examples/Gaussian_GLM_HMM.jl"
```

# Simulating and Fitting a Hidden Markov Model

This tutorial demonstrates how to use `StateSpaceDynamics.jl` to create, sample from, and fit Hidden
Markov Models (HMMs). Unlike Linear Dynamical Systems which have continuous latent states, HMMs have
discrete latent states that switch between a finite number of modes. This makes them ideal for modeling
data with distinct behavioral regimes, switching dynamics, or categorical latent structure.

We'll focus on a Gaussian generalized linear model HMM (GLM-HMM), where each hidden state corresponds
to a different regression relationship between inputs and outputs. This is particularly useful for
modeling data where the input-output relationship changes over time in discrete jumps.

## Load Required Packages

````@example gaussian_glm_hmm_example
using LinearAlgebra
using Plots
using Random
using StateSpaceDynamics
using StableRNGs
using Statistics: mean
````

Set up reproducible random number generation

````@example gaussian_glm_hmm_example
rng = StableRNG(1234);
nothing #hide
````

## Create a Gaussian GLM-HMM

In a GLM-HMM, each hidden state defines a different regression model. The system switches
between these regression models according to Markovian dynamics. This is useful for modeling
scenarios where the relationship between predictors and outcomes changes over time.

Define emission models for each hidden state
State 1: Positive relationship between input and output

````@example gaussian_glm_hmm_example
emission_1 = GaussianRegressionEmission(
    input_dim=3,                                    # Number of input features
    output_dim=1,                                   # Number of output dimensions
    include_intercept=true,                         # Include intercept term
    β=reshape([3.0, 2.0, 2.0, 3.0], :, 1),        # Regression coefficients [intercept, β₁, β₂, β₃]
    Σ=[1.0;;],                                     # Observation noise variance
    λ=0.0                                          # Regularization parameter
);
nothing #hide
````

State 2: Different relationship (negative intercept, different slopes)

````@example gaussian_glm_hmm_example
emission_2 = GaussianRegressionEmission(
    input_dim=3,
    output_dim=1,
    include_intercept=true,
    β=reshape([-4.0, -2.0, 3.0, 2.0], :, 1),      # Different regression coefficients
    Σ=[1.0;;],                                     # Same noise level
    λ=0.0
);
nothing #hide
````

Define the state transition matrix $\mathbf{A}$:
$A_{ij} = P(\text{state}_t = j \mid \text{state}_{t-1} = i)$
Diagonal elements are high (states are persistent), off-diagonal elements are low

````@example gaussian_glm_hmm_example
A = [0.99 0.01;    # From state 1: 99% stay, 1% switch to state 2
     0.05 0.95];    # From state 2: 5% switch to state 1, 95% stay
nothing #hide
````

Initial state distribution: $\pi_k = P(\text{state}_1 = k)$

````@example gaussian_glm_hmm_example
πₖ = [0.8; 0.2];    # 80% chance of starting in state 1, 20% in state 2
nothing #hide
````

Construct the complete HMM

````@example gaussian_glm_hmm_example
true_model = HiddenMarkovModel(
    K=2,                        # Number of hidden states
    A=A,                        # Transition matrix
    πₖ=πₖ,                     # Initial state distribution
    B=[emission_1, emission_2]  # Emission models for each state
);

print("Created GLM-HMM with regression models:\n")
print("State 1: y = 3.0 + 2.0x₁ + 2.0x₂ + 3.0x₃ + ε\n")
print("State 2: y = -4.0 - 2.0x₁ + 3.0x₂ + 2.0x₃ + ε\n");
nothing #hide
````

## Sample from the GLM-HMM

Generate synthetic data from our true model. This will give us both the observed
data (inputs and outputs) and the true hidden state sequence.

````@example gaussian_glm_hmm_example
n = 20000  # Number of time points
````

Generate random input features (predictors)

````@example gaussian_glm_hmm_example
Φ = randn(rng, 3, n);  # 3 features × n time points
nothing #hide
````

Sample from the HMM: returns both hidden states and observations

````@example gaussian_glm_hmm_example
true_labels, data = rand(rng, true_model, Φ, n=n);

print("Generated $(n) samples: State 1 ($(round(mean(true_labels .== 1)*100, digits=1))%), State 2 ($(round(mean(true_labels .== 2)*100, digits=1))%)\n");
nothing #hide
````

## Visualize the Sampled Dataset

Create a scatter plot showing how the input-output relationship differs between
the two hidden states. We'll plot feature 1 vs output, with points colored by true state.

````@example gaussian_glm_hmm_example
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
````

Overlay true regression lines (holding other features at 0)

````@example gaussian_glm_hmm_example
xvals = range(extrema(Φ[1, :])..., length=100)

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
````

## Initialize and Fit HMM with EM

Now we'll learn the parameters from observed data alone using EM algorithm.
Start with a randomly initialized HMM with different parameters than the true model.

Initialize with different parameters

````@example gaussian_glm_hmm_example
A_init = [0.8 0.2; 0.1 0.9]     # Different transition probabilities
πₖ_init = [0.6; 0.4]            # Different initial distribution
````

Initialize emission models with random regression coefficients

````@example gaussian_glm_hmm_example
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
nothing #hide
````

Create and fit the test model

````@example gaussian_glm_hmm_example
test_model = HiddenMarkovModel(K=2, A=A_init, πₖ=πₖ_init, B=[emission_1_init, emission_2_init])

print("Running EM algorithm...")
lls = fit!(test_model, data, Φ);

print("EM converged after $(length(lls)) iterations\n")
print("Log-likelihood improved by $(round(lls[end] - lls[1], digits=1))\n");
nothing #hide
````

Plot EM convergence

````@example gaussian_glm_hmm_example
p2 = plot(lls, xlabel="EM Iteration", ylabel="Log-Likelihood",
          title="Model Convergence", legend=false, lw=2, color=:darkblue)
````

## Visualize Learned vs True Regression Models

Compare the true regression relationships with what our fitted model learned.

````@example gaussian_glm_hmm_example
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
````

Plot true regression lines

````@example gaussian_glm_hmm_example
plot!(xvals, β1[1] .+ β1[2] .* xvals;
    color = :green, lw = 3, linestyle = :solid, label = "State 1 (true)"
)
plot!(xvals, β2[1] .+ β2[2] .* xvals;
    color = :orange, lw = 3, linestyle = :solid, label = "State 2 (true)"
)
````

Plot learned regression lines

````@example gaussian_glm_hmm_example
β1_learned = test_model.B[1].β[:, 1]
β2_learned = test_model.B[2].β[:, 1]
plot!(xvals, β1_learned[1] .+ β1_learned[2] .* xvals;
    color = :teal, lw = 3, linestyle = :dash, label = "State 1 (learned)"
)
plot!(xvals, β2_learned[1] .+ β2_learned[2] .* xvals;
    color = :purple, lw = 3, linestyle = :dash, label = "State 2 (learned)",
    legend = :topright
)
````

## Hidden State Decoding with Viterbi Algorithm

The Viterbi algorithm finds the most likely sequence of hidden states given the
observed data. We'll compare true vs predicted state sequences.

````@example gaussian_glm_hmm_example
pred_labels = viterbi(test_model, data, Φ);
nothing #hide
````

Calculate accuracy

````@example gaussian_glm_hmm_example
accuracy = mean(true_labels .== pred_labels)
print("Hidden state prediction accuracy: $(round(accuracy*100, digits=1))%\n");
nothing #hide
````

Visualize state sequences as heatmaps (subset for clarity)

````@example gaussian_glm_hmm_example
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
````

## Multiple Independent Trials

Real-world scenarios often involve multiple independent sequences. We'll generate
multiple trials and show how to fit HMMs to this data structure.

````@example gaussian_glm_hmm_example
num_trials = 100   # Number of independent sequences
n_trial = 1000;    # Length of each sequence

print("Generating $num_trials independent trials of length $n_trial...\n")

all_data = Vector{Matrix{Float64}}()
Φ_total = Vector{Matrix{Float64}}()
all_true_labels = Vector{Vector{Int64}}()

for i in 1:num_trials
    Φ_trial = randn(rng, 3, n_trial)
    true_labels_trial, data_trial = rand(rng, true_model, Φ_trial, n=n_trial)
    push!(all_true_labels, true_labels_trial)
    push!(all_data, data_trial)
    push!(Φ_total, Φ_trial)
end

print("Total data points: $(num_trials * n_trial)\n");
nothing #hide
````

## Fitting HMM to Multiple Trials

When we have multiple independent sequences, EM accounts for each sequence
starting fresh from the initial distribution, providing more robust estimates.

Initialize fresh model for multi-trial fitting

````@example gaussian_glm_hmm_example
test_model_multi = HiddenMarkovModel(
    K=2, A=A_init, πₖ=πₖ_init,
    B=[deepcopy(emission_1_init), deepcopy(emission_2_init)]
)

print("Fitting HMM to multiple trials...")
lls_multi = fit!(test_model_multi, all_data, Φ_total);

print("Multi-trial EM converged after $(length(lls_multi)) iterations\n")
print("Log-likelihood improved by $(round(lls_multi[end] - lls_multi[1], digits=1))\n");
nothing #hide
````

Plot multi-trial convergence

````@example gaussian_glm_hmm_example
p5 = plot(lls_multi, xlabel="EM Iteration", ylabel="Log-Likelihood",
          title="Multi-Trial Model Convergence", legend=false, lw=2, color=:darkgreen)
````

## Multi-Trial State Decoding

Decode hidden states for all trials and visualize as a multi-trial heatmap.

````@example gaussian_glm_hmm_example
all_pred_labels = viterbi(test_model_multi, all_data, Φ_total)
````

Calculate overall accuracy across all trials

````@example gaussian_glm_hmm_example
all_true_matrix = hcat(all_true_labels...)
all_pred_matrix = hcat(all_pred_labels...)
total_accuracy = mean(all_true_matrix .== all_pred_matrix)

print("Overall state prediction accuracy: $(round(total_accuracy*100, digits=1))%\n");
nothing #hide
````

Visualize subset of trials (first 10 trials, first 500 time points)

````@example gaussian_glm_hmm_example
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
````

## Model Assessment Summary

````@example gaussian_glm_hmm_example
print("\n=== Final Model Assessment ===\n")
````

Compare learned parameters with true parameters

````@example gaussian_glm_hmm_example
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
nothing #hide
````

## Summary

This tutorial demonstrated the complete workflow for Hidden Markov Models with regression emissions:

**Key Concepts:**
- **Discrete latent states** with different regression relationships in each state
- **Markovian dynamics** governing state transitions over time
- **EM algorithm** for joint parameter learning and state inference
- **Viterbi decoding** for finding most likely state sequences

**Applications:**
- Modeling switching dynamics and regime changes
- Context-dependent input-output relationships
- Multiple independent trial analysis
- Robust parameter estimation across sequences

GLM-HMMs provide a powerful framework for modeling data with discrete latent structure,
making them valuable for neuroscience, economics, and other domains with switching behaviors.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

