```@meta
EditURL = "../../examples/HMM_ModelSelection.jl"
```

# Choosing "K" for a hidden Markov model.
In principle, one can fit an HMM with any number of states, but how do we choose?
One generally has no ground truth, except for the most rare cases. So it begs the question:
How do we select the number of hidden states K? In this tutorial we will demonstrate a few
typical approaches for model selection.

## Load Required Packages

````@example hmm_model_selection_example
using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using Statistics
using StableRNGs
using Printf
````

Fix RNG for reproducible simulation and k-means seeding

````@example hmm_model_selection_example
rng = StableRNG(1234);
nothing #hide
````

## Create a True HMM
For our example we will create a simple HMM with K=3 states and 2D Gaussian emissions.
We will then generate ground truth data from this model for use in the tutorial.

````@example hmm_model_selection_example
K = 3  # Number of hidden states
D = 2  # Dimensionality of observations
T = 500  # Number of time steps to simulate
````

Define the true transition matrix

````@example hmm_model_selection_example
true_A = [0.90 0.05 0.05;
          0.10 0.80 0.10;
          0.15 0.15 0.70];
nothing #hide
````

Define the true emission parameters (means and covariances)

````@example hmm_model_selection_example
true_μs = [-1.0 0.0 1.0;
            0.6 -1.0 0.0]

true_Σs =[[0.1 -0.3; -0.3 1.0],
          [0.5 0.2; 0.2 0.3],
          [0.3 0.0; 0.0 0.3]];
nothing #hide
````

Define the initial state distribution

````@example hmm_model_selection_example
true_π = [0.5, 0.3, 0.2];
nothing #hide
````

Create the true HMM

````@example hmm_model_selection_example
true_hmm = HiddenMarkovModel(true_A,
                            [GaussianEmission(2, true_μs[:, k],
                            true_Σs[k]) for k in 1:K],
                            true_π,
                            K)

states, observations = rand(rng, true_hmm; n=T)
````

Now plot the data colored by the true states

````@example hmm_model_selection_example
p1 = scatter(observations[1, :], observations[2, :],
    group=states,
    xlabel="x1", ylabel="x2",
    title="HMM Observations Colored by True State",
    legend=:topright, alpha=0.7)
````

## Model Selection Approaches

With real data, we don't know the true number of states K. We'll explore several
methods to estimate the optimal K by fitting HMMs with different numbers of states
and comparing their performance using various criteria.

Define range of K values to test

````@example hmm_model_selection_example
K_range = 1:8
````

We'll compare models using multiple criteria:
1. Log-likelihood (higher is better, but tends to overfit)
2. AIC (Akaike Information Criterion - penalizes complexity)
3. BIC (Bayesian Information Criterion - stronger complexity penalty)
4. Cross-validation likelihood

````@example hmm_model_selection_example
results = Dict(
    "K" => Int[],
    "log_likelihood" => Float64[],
    "AIC" => Float64[],
    "BIC" => Float64[],
    "n_params" => Int[]
)
````

## Helper Functions for Model Selection

````@example hmm_model_selection_example
function initialize_hmm_kmeans(obs, k, rng)
    """Initialize HMM parameters using k-means clustering"""
    if k == 1
        means = [mean(obs, dims=2)[:, 1]]
        covs = [cov(obs')]
    else
        Random.seed!(rng, 42)
        cluster_centers = obs[:, randperm(rng, size(obs, 2))[1:k]]
        means = [cluster_centers[:, i] for i in 1:k]
        covs = [Matrix(0.5 * I(size(obs, 1))) for _ in 1:k]
    end

    A_init = fill(0.1/(k-1), k, k)
    for i in 1:k
        A_init[i, i] = 0.9
    end
    A_init = A_init ./ sum(A_init, dims=2)  # Normalize rows

    π_init = fill(1/k, k)

    return HiddenMarkovModel(A_init,
                           [GaussianEmission(2, means[i], covs[i]) for i in 1:k],
                           π_init,
                           k)
end

function count_parameters(hmm)
    """Count the number of free parameters in an HMM"""
    K = hmm.K
    D = length(hmm.B[1].μ)

    transition_params = K * (K - 1)
    initial_params = K - 1
    emission_params = K * D + K * D * (D + 1) ÷ 2

    return transition_params + initial_params + emission_params
end

println("Fitting HMMs with different numbers of states...")

for k in K_range
    println("Fitting HMM with K=$k states...")

    hmm_k = initialize_hmm_kmeans(observations, k, rng)

    fit!(hmm_k, observations; max_iters=100, tol=1e-6)

    ll = loglikelihood(hmm_k, observations)
    n_params = count_parameters(hmm_k)
    aic_val = -2*ll + 2*n_params
    bic_val = -2*ll + log(T)*n_params

    push!(results["K"], k)
    push!(results["log_likelihood"], ll)
    push!(results["AIC"], aic_val)
    push!(results["BIC"], bic_val)
    push!(results["n_params"], n_params)
end
````

## Visualize Model Selection Results

Create a comprehensive plot showing all criteria

````@example hmm_model_selection_example
p2 = plot(layout=(2, 2), size=(800, 600))
````

Plot 1: Log-likelihood

````@example hmm_model_selection_example
plot!(results["K"], results["log_likelihood"],
      marker=:circle, linewidth=2, label="Log-likelihood",
      xlabel="Number of States (K)", ylabel="Log-likelihood",
      title="Model Log-likelihood", subplot=1)
vline!([K], linestyle=:dash, color=:red, label="True K=$K", subplot=1)
````

Plot 2: AIC

````@example hmm_model_selection_example
plot!(results["K"], results["AIC"],
      marker=:circle, linewidth=2, label="AIC", color=:orange,
      xlabel="Number of States (K)", ylabel="AIC",
      title="Akaike Information Criterion", subplot=2)
aic_min_idx = argmin(results["AIC"])
vline!([results["K"][aic_min_idx]], linestyle=:dash, color=:orange,
       label="AIC min (K=$(results["K"][aic_min_idx]))", subplot=2)
vline!([K], linestyle=:dash, color=:red, label="True K=$K", subplot=2)
````

Plot 3: BIC

````@example hmm_model_selection_example
plot!(results["K"], results["BIC"],
      marker=:circle, linewidth=2, label="BIC", color=:green,
      xlabel="Number of States (K)", ylabel="BIC",
      title="Bayesian Information Criterion", subplot=3)
bic_min_idx = argmin(results["BIC"])
vline!([results["K"][bic_min_idx]], linestyle=:dash, color=:green,
       label="BIC min (K=$(results["K"][bic_min_idx]))", subplot=3)
vline!([K], linestyle=:dash, color=:red, label="True K=$K", subplot=3)
````

Plot 4: Number of parameters

````@example hmm_model_selection_example
plot!(results["K"], results["n_params"],
      marker=:circle, linewidth=2, label="# Parameters", color=:purple,
      xlabel="Number of States (K)", ylabel="Number of Parameters",
      title="Model Complexity", subplot=4)

display(p2)
````

## Cross-Validation Approach

````@example hmm_model_selection_example
function cross_validate_hmm(observations, k, n_folds=5)
    """Perform k-fold cross-validation for HMM with k states"""
    T = size(observations, 2)
    fold_size = T ÷ n_folds
    cv_scores = Float64[]

    for fold in 1:n_folds

        test_start = (fold - 1) * fold_size + 1
        test_end = min(fold * fold_size, T)

        train_idx = [1:(test_start-1); (test_end+1):T]
        test_idx = test_start:test_end

        if length(train_idx) == 0 || length(test_idx) == 0
            continue
        end

        train_obs = observations[:, train_idx]
        test_obs = observations[:, test_idx]

        hmm_cv = initialize_hmm_kmeans(train_obs, k, rng)
        fit!(hmm_cv, train_obs; max_iters=50, tol=1e-4)

        test_ll = loglikelihood(hmm_cv, test_obs)
        push!(cv_scores, test_ll / length(test_idx))  # Normalize by sequence length
    end

    return mean(cv_scores)
end
````

Perform cross-validation for each K

````@example hmm_model_selection_example
println("\nPerforming cross-validation...")
cv_scores = Float64[]
for k in K_range
    println("Cross-validating K=$k...")
    cv_score = cross_validate_hmm(observations, k, 5)
    push!(cv_scores, cv_score)
end
````

Plot cross-validation results

````@example hmm_model_selection_example
p3 = plot(K_range, cv_scores,
          marker=:circle, linewidth=2, label="CV Score",
          xlabel="Number of States (K)", ylabel="CV Log-likelihood",
          title="Cross-Validation Results", color=:blue)
cv_max_idx = argmax(cv_scores)
vline!([K_range[cv_max_idx]], linestyle=:dash, color=:blue,
       label="CV max (K=$(K_range[cv_max_idx]))")
vline!([K], linestyle=:dash, color=:red, label="True K=$K")
````

## Summary of Results

````@example hmm_model_selection_example
println("\n" * "="^50)
println("MODEL SELECTION SUMMARY")
println("="^50)
println("True K: $K")
println("AIC selects: K = $(results["K"][aic_min_idx])")
println("BIC selects: K = $(results["K"][bic_min_idx])")
println("Cross-validation selects: K = $(K_range[cv_max_idx])")
println("\nDetailed Results:")
println("K\tLog-lik\t\tAIC\t\tBIC\t\tCV Score")
println("-"^60)
for i in 1:length(K_range)
    @printf("%d\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\n",
            results["K"][i], results["log_likelihood"][i],
            results["AIC"][i], results["BIC"][i], cv_scores[i])
end
````

## Compare Best Models Visually

Fit models with AIC and BIC selected K values for visual comparison

````@example hmm_model_selection_example
best_aic_k = results["K"][aic_min_idx]
best_bic_k = results["K"][bic_min_idx]

hmm_aic = initialize_hmm_kmeans(observations, best_aic_k, rng)
fit!(hmm_aic, observations; max_iters=100, tol=1e-6)

hmm_bic = initialize_hmm_kmeans(observations, best_bic_k, rng)
fit!(hmm_bic, observations; max_iters=100, tol=1e-6)
````

Get most likely state sequences

````@example hmm_model_selection_example
states_aic = viterbi(hmm_aic, observations)
states_bic = viterbi(hmm_bic, observations)
````

Create comparison plots

````@example hmm_model_selection_example
p4 = plot(layout=(1, 3), size=(1200, 400))
````

True states

````@example hmm_model_selection_example
scatter!(observations[1, :], observations[2, :], group=states,
         xlabel="x1", ylabel="x2", title="True States (K=$K)",
         legend=false, alpha=0.7, subplot=1)
````

AIC selected model

````@example hmm_model_selection_example
scatter!(observations[1, :], observations[2, :], group=states_aic,
         xlabel="x1", ylabel="x2", title="AIC Model (K=$best_aic_k)",
         legend=false, alpha=0.7, subplot=2)
````

BIC selected model

````@example hmm_model_selection_example
scatter!(observations[1, :], observations[2, :], group=states_bic,
         xlabel="x1", ylabel="x2", title="BIC Model (K=$best_bic_k)",
         legend=false, alpha=0.7, subplot=3)

display(p4)
````

## Key Takeaways

**Information Criteria**:
- **AIC** tends to favor more complex models (higher K)
- **BIC** has stronger penalty for complexity, often selects simpler models
- Both can help avoid overfitting compared to raw likelihood

**Cross-Validation**:
- More robust estimate of generalization performance
- Computationally expensive but often worth it
- Less sensitive to the specific penalty terms in AIC/BIC

**Practical Recommendations**:
1. Start with domain knowledge if available
2. Use multiple criteria - they don't always agree
3. Consider interpretability alongside statistical fit
4. Visualize results when possible (as we did here)
5. Remember that the "best" K depends on your specific goals

**Model Selection Caveats**:
- Local optima in EM can affect comparisons
- Small datasets can make selection unreliable
- True model may not be in your candidate set
- Consider ensemble approaches for robust predictions

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

