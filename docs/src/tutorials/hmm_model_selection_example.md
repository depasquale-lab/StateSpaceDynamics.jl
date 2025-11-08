```@meta
EditURL = "../../examples/HMM_ModelSelection.jl"
```

# Choosing "K" for a hidden Markov model.
In principle, one can fit an HMM with any number of states, but how do we choose?
One generally has no ground truth, except for the most rare cases. So it begs the question:
How do we select the number of hidden states K? In this tutorial we will demonstrate a few
typical approaches for model selection with enhanced cross-validation integration.

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
    "n_params" => Int[],
    "CV_score" => Float64[]
);
nothing #hide
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

for k in K_range
    println("Evaluating HMM with K=$k states...")

    hmm_k = initialize_hmm_kmeans(observations, k, rng)
    fit!(hmm_k, observations; max_iters=100, tol=1e-6)

    ll = loglikelihood(hmm_k, observations)
    n_params = count_parameters(hmm_k)
    aic_val = -2*ll + 2*n_params
    bic_val = -2*ll + log(T)*n_params

    cv_score = cross_validate_hmm(observations, k, 5)

    push!(results["K"], k)
    push!(results["log_likelihood"], ll)
    push!(results["AIC"], aic_val)
    push!(results["BIC"], bic_val)
    push!(results["n_params"], n_params)
    push!(results["CV_score"], cv_score)
end
````

## Find optimal K for each criterion

````@example hmm_model_selection_example
aic_min_idx = argmin(results["AIC"])
bic_min_idx = argmin(results["BIC"])
cv_max_idx = argmax(results["CV_score"])

best_aic_k = results["K"][aic_min_idx]
best_bic_k = results["K"][bic_min_idx]
best_cv_k = results["K"][cv_max_idx]
````

## Interpreting Information Criteria

**AIC (Akaike Information Criterion)**:
- Estimates relative model quality for prediction
- Asymptotically equivalent to leave-one-out cross-validation
- Tends to select more complex models (higher K)
- Better for prediction tasks

**BIC (Bayesian Information Criterion)**:
- Estimates probability that model is true among candidates
- Stronger complexity penalty, especially for large datasets
- Tends to select simpler models (lower K)
- Better for identifying "true" model structure

**Key insight**: Lower values are better for both AIC and BIC
(they measure "badness" - deviance plus penalty)

## Cross-Validation: The Gold Standard

Cross-validation provides the most honest estimate of generalization performance:
- Trains on subset of data, tests on held-out portion
- Directly measures what we care about: performance on unseen data
- Less dependent on specific penalty terms than AIC/BIC
- More computationally expensive but often worth it

**Challenges for HMMs**:
- Temporal data makes random splits problematic
- Sequential structure should be preserved when possible
- We use contiguous blocks to maintain temporal coherence

**Interpreting CV results**:
- Higher CV likelihood indicates better generalization
- Plateauing suggests additional complexity isn't helpful
- Large variance across folds may indicate unstable model

## Visualization of Model Selection Results
In our plots we will plot 1.) the loglikelihood 2.) negative AIC 3.) negative BIC and 4.) the loglikelihood of the test dataset
We plot the negative AIC and BIC as those metrics are defined such that a lower score is better, so we invert the statistic, so
like regular likelihood, higher indicates better model performance.

Create comprehensive plot showing all criteria including CV

````@example hmm_model_selection_example
p2 = plot(layout=(2, 2), size=(1000, 800))

plot!(results["K"], results["log_likelihood"],
      marker=:circle, linewidth=2, label="Log-likelihood",
      xlabel="Number of States (K)", ylabel="Log-likelihood",
      title="Model Log-likelihood", subplot=1)
vline!([K], linestyle=:dash, color=:red, label="True K=$K", subplot=1)

plot!(results["K"], -results["AIC"],
      marker=:circle, linewidth=2, label="AIC", color=:orange,
      xlabel="Number of States (K)", ylabel="AIC",
      title="Negative AIC", subplot=2)
vline!([best_aic_k], linestyle=:dash, color=:orange,
       label="nAIC max (K=$best_aic_k)", subplot=2)
vline!([K], linestyle=:dash, color=:red, label="True K=$K", subplot=2)

plot!(results["K"], -results["BIC"],
      marker=:circle, linewidth=2, label="BIC", color=:green,
      xlabel="Number of States (K)", ylabel="BIC",
      title="Negative BIC", subplot=3)
vline!([best_bic_k], linestyle=:dash, color=:green,
       label="nBIC max (K=$best_bic_k)", subplot=3)
vline!([K], linestyle=:dash, color=:red, label="True K=$K", subplot=3)

plot!(results["K"], results["CV_score"],
      marker=:circle, linewidth=2, label="CV Score", color=:blue,
      xlabel="Number of States (K)", ylabel="CV Log-likelihood",
      title="Cross-Validation Results", subplot=4)
vline!([best_cv_k], linestyle=:dash, color=:blue,
       label="CV max (K=$best_cv_k)", subplot=4)
vline!([K], linestyle=:dash, color=:red, label="True K=$K", subplot=4)

p2
````

## Unified Model Selection Comparison
Create a single plot showing all criteria on normalized scales for direct comparison

````@example hmm_model_selection_example
p3 = plot(size=(800, 500))

norm_aic = (-results["AIC"] .- maximum(-results["AIC"])) ./ (maximum(-results["AIC"]) - minimum(-results["AIC"])) .+ 1
norm_bic = (-results["BIC"] .- maximum(-results["BIC"])) ./ (maximum(-results["BIC"]) - minimum(-results["BIC"])) .+ 1
norm_cv = (results["CV_score"] .- maximum(results["CV_score"])) ./ (maximum(results["CV_score"]) - minimum(results["CV_score"])) .+ 1

plot!(results["K"], norm_aic, marker=:circle, linewidth=2,
      label="nAIC (normalized)", color=:orange)
plot!(results["K"], norm_bic, marker=:square, linewidth=2,
      label="nBIC (normalized)", color=:green)
plot!(results["K"], norm_cv, marker=:diamond, linewidth=2,
      label="CV (normalized)", color=:blue)

xlabel!("Number of States (K)")
ylabel!("Normalized Score (lower is better)")
title!("Unified Model Selection Comparison")
vline!([K], linestyle=:dash, color=:red, linewidth=2, label="True K=$K")

scatter!([best_aic_k], [norm_aic[aic_min_idx]], markersize=8, color=:orange, markershape=:star5, label="")
scatter!([best_bic_k], [norm_bic[bic_min_idx]], markersize=8, color=:green, markershape=:star5, label="")
scatter!([best_cv_k], [norm_cv[cv_max_idx]], markersize=8, color=:blue, markershape=:star5, label="");

p3
````

## Compare ALL Best Models Visually (including CV)

Fit models with AIC, BIC, and CV selected K values

````@example hmm_model_selection_example
hmm_aic = initialize_hmm_kmeans(observations, best_aic_k, rng)
fit!(hmm_aic, observations; max_iters=100, tol=1e-6)

hmm_bic = initialize_hmm_kmeans(observations, best_bic_k, rng)
fit!(hmm_bic, observations; max_iters=100, tol=1e-6)

hmm_cv = initialize_hmm_kmeans(observations, best_cv_k, rng)
fit!(hmm_cv, observations; max_iters=100, tol=1e-6);
nothing #hide
````

Get most likely state sequences

````@example hmm_model_selection_example
states_aic = viterbi(hmm_aic, observations)
states_bic = viterbi(hmm_bic, observations)
states_cv = viterbi(hmm_cv, observations);
nothing #hide
````

Create enhanced comparison plots (2x2 layout)

````@example hmm_model_selection_example
p4 = plot(layout=(2, 2), size=(1000, 800))

scatter!(observations[1, :], observations[2, :], group=states,
         xlabel="x1", ylabel="x2", title="True States (K=$K)",
         legend=false, alpha=0.7, subplot=1)

scatter!(observations[1, :], observations[2, :], group=states_aic,
         xlabel="x1", ylabel="x2", title="AIC Model (K=$best_aic_k)",
         legend=false, alpha=0.7, subplot=2)

scatter!(observations[1, :], observations[2, :], group=states_bic,
         xlabel="x1", ylabel="x2", title="BIC Model (K=$best_bic_k)",
         legend=false, alpha=0.7, subplot=3)

scatter!(observations[1, :], observations[2, :], group=states_cv,
         xlabel="x1", ylabel="x2", title="Cross-Validation Model (K=$best_cv_k)",
         legend=false, alpha=0.7, subplot=4)

p4
````

## Key Takeaways

**Information Criteria**:
- **AIC** tends to favor more complex models (higher K)
- **BIC** has stronger penalty for complexity, often selects simpler models
- Both can help avoid overfitting compared to raw likelihood

**Cross-Validation**:
- More robust estimate of generalization performance
- Computationally expensive but often provides the most reliable selection
- Less sensitive to the specific penalty terms in AIC/BIC
- Directly measures out-of-sample performance

**Practical Recommendations**:
1. **Use multiple criteria** - they don't always agree, and disagreement is informative
2. **Prioritize cross-validation** when computational resources allow
3. Consider the **practical significance** of differences between close K values
4. **Visualize results** to understand model behavior across different K values
5. Remember that model selection depends on your **specific goals and constraints**
6. When methods disagree, consider **ensemble approaches** or **domain knowledge**

**Model Selection Insights from This Example**:
- Raw likelihood always increases with K (overfitting tendency)
- Information criteria balance fit vs. complexity differently
- Cross-validation provides unbiased performance estimates
- Visual inspection can reveal whether selected models make practical sense

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

