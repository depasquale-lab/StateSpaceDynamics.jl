```@meta
EditURL = "../../examples/GaussianMixtureModel.jl"
```

## Simulating and Fitting a Gaussian Mixture Model

This tutorial demonstrates how to use `StateSpaceDynamics.jl` to create a Gaussian Mixture Model
(GMM) and fit it using the EM algorithm. Unlike Hidden Markov Models which model temporal sequences,
GMMs are designed for clustering and density estimation of independent observations. Each data point
is assumed to come from one of several Gaussian components, but there's no temporal dependence.

GMMs are fundamental in machine learning for tasks like:
- Unsupervised clustering of data
- Density estimation for anomaly detection
- Dimensionality reduction (when combined with factor analysis)
- As building blocks for more complex models

The key insight is that complex data distributions can often be well-approximated as mixtures
of simpler Gaussian distributions, each representing a different "mode" or cluster in the data.

## Load Required Packages

We need several packages for GMM modeling, data generation, and comprehensive visualization.

````@example gaussian_mixture_model_example
using StateSpaceDynamics  # Core GMM functionality
using LinearAlgebra       # Matrix operations
using Random             # Random number generation
using Plots              # Basic plotting
using StableRNGs         # Reproducible randomness
using Distributions      # Statistical distributions
using StatsPlots         # Enhanced statistical plotting
using Combinatorics
````

Set up reproducible random number generation

````@example gaussian_mixture_model_example
rng = StableRNG(1234);
nothing #hide
````

## Create a True Gaussian Mixture Model to Simulate From

We'll create a "ground truth" GMM with known parameters, generate data from it,
then see how well we can recover these parameters using only the observed data.

````@example gaussian_mixture_model_example
k = 3  # Number of mixture components (clusters)
D = 2  # Data dimensionality (2D for easy visualization)
````

Define the true component means
Each column represents the mean vector for one component

````@example gaussian_mixture_model_example
true_μs = [
    -1.0  1.0  0.0;   # x₁ coordinates of the 3 component centers
    -1.0 -1.5  2.0    # x₂ coordinates of the 3 component centers
]  # Shape: (D, K) = (2, 3)
````

Define covariance matrices for each component
Using isotropic (spherical) covariances for simplicity

````@example gaussian_mixture_model_example
true_Σs = [Matrix{Float64}(0.3 * I(2)) for _ in 1:k]  # All components have same shape
````

Define mixing weights (must sum to 1)
These represent the probability that a random sample comes from each component

````@example gaussian_mixture_model_example
true_πs = [0.5, 0.2, 0.3]  # Component 1 is most likely, component 2 least likely
````

Construct the complete GMM

````@example gaussian_mixture_model_example
true_gmm = GaussianMixtureModel(k, true_μs, true_Σs, true_πs)

println("Created true GMM with $k components in $D dimensions:")
for i in 1:k
    println("  Component $i: μ = $(true_μs[:, i]), π = $(true_πs[i])")
end
println("  All components have isotropic covariance with σ² = 0.3")
````

## Sample Data from the True GMM

Generate synthetic data from our true model. We'll sample both the component
assignments (for visualization) and the actual observations.

````@example gaussian_mixture_model_example
n = 500  # Number of data points to generate
println("Generating $n samples from the true GMM...")
````

First, determine which component each sample comes from

````@example gaussian_mixture_model_example
labels = rand(rng, Categorical(true_πs), n)
````

Count samples per component

````@example gaussian_mixture_model_example
component_counts = [sum(labels .== i) for i in 1:k]
println("Samples per component: $(component_counts)")
println("Empirical mixing proportions: $(round.(component_counts ./ n, digits=3))")
````

Generate the actual data points

````@example gaussian_mixture_model_example
X = Matrix{Float64}(undef, D, n)
for i in 1:n
    component = labels[i]
    X[:, i] = rand(rng, MvNormal(true_μs[:, component], true_Σs[component]))
end

println("Generated data summary:")
println("  Data shape: $(size(X)) (dimensions × samples)")
println("  Data range: x₁ ∈ [$(round(minimum(X[1,:]), digits=2)), $(round(maximum(X[1,:]), digits=2))], x₂ ∈ [$(round(minimum(X[2,:]), digits=2)), $(round(maximum(X[2,:]), digits=2))]")
````

Visualize the generated data colored by true component membership

````@example gaussian_mixture_model_example
p1 = scatter(
    X[1, :], X[2, :];
    group=labels,                    # Color by true component
    title="GMM Samples (colored by true component)",
    xlabel="x₁", ylabel="x₂",
    markersize=4,
    alpha=0.8,
    legend=:topright,
    palette=:Set1_3
)
````

Add component centers for reference

````@example gaussian_mixture_model_example
for i in 1:k
    scatter!(p1, [true_μs[1, i]], [true_μs[2, i]];
        marker=:star, markersize=10, color=i,
        markerstrokewidth=2, markerstrokecolor=:black,
        label="Center $i")
end

display(p1)
````

## Fit a New Gaussian Mixture Model to the Data

Now we simulate the realistic scenario: we observe only the data points X,
not the true component labels or parameters. Our goal is to recover the
underlying mixture structure using the EM algorithm.

````@example gaussian_mixture_model_example
println("Initializing GMM for fitting...")
println("Note: We assume we know the correct number of components k=$k")
println("      (In practice, this often requires model selection)")
````

Initialize a GMM with the correct number of components but unknown parameters

````@example gaussian_mixture_model_example
fit_gmm = GaussianMixtureModel(k, D)

println("Running EM algorithm to learn GMM parameters...")
````

Fit the model using EM algorithm
- maxiter: maximum number of EM iterations
- tol: convergence tolerance (change in log-likelihood)
- initialize_kmeans: use k-means to initialize component centers

````@example gaussian_mixture_model_example
class_probabilities, lls = fit!(fit_gmm, X;
    maxiter=100,
    tol=1e-6,
    initialize_kmeans=true  # This often helps convergence
)

println("EM algorithm completed:")
println("  Converged after $(length(lls)) iterations")
println("  Final log-likelihood: $(round(lls[end], digits=2))")
println("  Log-likelihood improvement: $(round(lls[end] - lls[1], digits=2))")
````

Display learned parameters

````@example gaussian_mixture_model_example
println("\nLearned GMM parameters:")
for i in 1:k
    println("  Component $i: μ = $(round.(fit_gmm.μₖ[:, i], digits=3)), π = $(round(fit_gmm.πₖ[i], digits=3))")
end
````

## Plot Log-Likelihoods to Visualize EM Convergence

The EM algorithm should monotonically increase the log-likelihood at each iteration.
Plotting this helps us verify convergence and understand the optimization process.

````@example gaussian_mixture_model_example
p2 = plot(
    lls;
    xlabel="EM Iteration",
    ylabel="Log-Likelihood",
    title="EM Algorithm Convergence",
    label="Log-Likelihood",
    marker=:circle,
    markersize=4,
    linewidth=2,
    grid=true
)
````

Add annotations about convergence behavior

````@example gaussian_mixture_model_example
if length(lls) > 1
    initial_rate = lls[min(5, end)] - lls[1]
    final_rate = lls[end] - lls[max(1, end-5)]

    annotate!(p2, length(lls)*0.7, lls[end]*0.95,
        text("Final LL: $(round(lls[end], digits=1))", 10))

    if length(lls) < 100  # Converged before max iterations
        annotate!(p2, length(lls)*0.7, lls[end]*0.90,
            text("Converged in $(length(lls)) iterations", 10))
    end
end

display(p2)
````

## Visualize Model Contours Over the Data

Create a comprehensive visualization showing both the data and the fitted model.
We'll plot probability density contours for each learned component.

````@example gaussian_mixture_model_example
println("Creating visualization of fitted GMM...")
````

Create a grid for plotting contours

````@example gaussian_mixture_model_example
x_range = range(minimum(X[1, :]) - 1, stop=maximum(X[1, :]) + 1, length=150)
y_range = range(minimum(X[2, :]) - 1, stop=maximum(X[2, :]) + 1, length=150)
xs = collect(x_range)
ys = collect(y_range)
````

Start with a scatter plot of the data (without true labels this time)

````@example gaussian_mixture_model_example
p3 = scatter(
    X[1, :], X[2, :];
    markersize=3,
    alpha=0.5,
    color=:gray,
    xlabel="x₁",
    ylabel="x₂",
    title="Data with Fitted GMM Components",
    legend=:topright,
    label="Data points"
)
````

Plot probability density contours for each learned component

````@example gaussian_mixture_model_example
colors = [:red, :green, :blue]
for i in 1:fit_gmm.k
    comp_dist = MvNormal(fit_gmm.μₖ[:, i], fit_gmm.Σₖ[i])
    Z_i = [fit_gmm.πₖ[i] * pdf(comp_dist, [x, y]) for y in ys, x in xs]

    contour!(
        p3, xs, ys, Z_i;
        levels=8,
        linewidth=2,
        c=colors[i],
        label="Component $i (π=$(round(fit_gmm.πₖ[i], digits=2)))"
    )

    scatter!(p3, [fit_gmm.μₖ[1, i]], [fit_gmm.μₖ[2, i]];
        marker=:star, markersize=8, color=colors[i],
        markerstrokewidth=2, markerstrokecolor=:black,
        label="")
end

display(p3)
````

## Analyze Component Assignments

Use the fitted model to assign each data point to its most likely component
and compare with the true assignments.

````@example gaussian_mixture_model_example
println("Analyzing component assignments...")
````

Get posterior probabilities for each data point
class_probabilities[i, j] = P(component i | data point j)

````@example gaussian_mixture_model_example
predicted_labels = [argmax(class_probabilities[:, j]) for j in 1:n]
````

Calculate assignment accuracy (accounting for possible label permutation)
Since EM can converge with components in different order, we need to find
the best permutation of labels

````@example gaussian_mixture_model_example
function best_permutation_accuracy(true_labels, pred_labels, k)
    best_acc = 0.0
    best_perm = collect(1:k)

    for perm in Combinatorics.permutations(1:k)
        mapped_pred = [perm[pred_labels[i]] for i in 1:length(pred_labels)]
        acc = mean(true_labels .== mapped_pred)
        if acc > best_acc
            best_acc = acc
            best_perm = perm
        end
    end

    return best_acc, best_perm
end
````

Calculate accuracy with best label permutation

````@example gaussian_mixture_model_example
accuracy, best_perm = best_permutation_accuracy(labels, predicted_labels, k)
println("Component assignment accuracy: $(round(accuracy*100, digits=1))%")
println("Best label permutation: $best_perm")
````

## Parameter Recovery Analysis

````@example gaussian_mixture_model_example
println("\n=== Parameter Recovery Assessment ===")
````

Compare true vs learned parameters (accounting for label permutation)

````@example gaussian_mixture_model_example
mapped_μs = fit_gmm.μₖ[:, best_perm]
mapped_πs = fit_gmm.πₖ[best_perm]
mapped_Σs = fit_gmm.Σₖ[best_perm]
````

Mean vector errors

````@example gaussian_mixture_model_example
μ_errors = [norm(true_μs[:, i] - mapped_μs[:, i]) for i in 1:k]
println("Mean vector recovery errors:")
for i in 1:k
    println("  Component $i: $(round(μ_errors[i], digits=3))")
end
````

Mixing weight errors

````@example gaussian_mixture_model_example
π_errors = [abs(true_πs[i] - mapped_πs[i]) for i in 1:k]
println("Mixing weight recovery errors:")
for i in 1:k
    println("  Component $i: $(round(π_errors[i], digits=3))")
end
````

Covariance matrix errors (Frobenius norm)

````@example gaussian_mixture_model_example
Σ_errors = [norm(true_Σs[i] - mapped_Σs[i]) for i in 1:k]
println("Covariance matrix recovery errors:")
for i in 1:k
    println("  Component $i: $(round(Σ_errors[i], digits=3))")
end
````

## Create Final Comparison Visualization

Create a side-by-side comparison of true vs learned GMM

````@example gaussian_mixture_model_example
p_true = scatter(X[1, :], X[2, :]; group=labels, title="True GMM",
                xlabel="x₁", ylabel="x₂", markersize=3, alpha=0.7,
                palette=:Set1_3, legend=false)

p_learned = scatter(X[1, :], X[2, :]; group=predicted_labels, title="Learned GMM",
                   xlabel="x₁", ylabel="x₂", markersize=3, alpha=0.7,
                   palette=:Set1_3, legend=false)

final_comparison = plot(p_true, p_learned, layout=(1, 2), size=(800, 400))
display(final_comparison)
````

## Model Selection Considerations

````@example gaussian_mixture_model_example
println("\n=== Model Selection Notes ===")
println("In this tutorial, we assumed the correct number of components k=$k was known.")
println("In practice, you would need to select k using techniques like:")
println("  - Information criteria (AIC, BIC)")
println("  - Cross-validation")
println("  - Gap statistic")
println("  - Elbow method on within-cluster sum of squares")
println("")
println("The EM algorithm is guaranteed to converge to a local optimum, but not")
println("necessarily the global optimum. Multiple random initializations are often")
println("used to find better solutions.")
````

## Summary

This tutorial demonstrated the complete workflow for Gaussian Mixture Models:

1. **Model Structure**: Independent observations from a mixture of Gaussian distributions
2. **Parameter Learning**: EM algorithm iteratively improves component parameters and mixing weights
3. **Initialization**: K-means initialization helps EM converge to better solutions
4. **Visualization**: Contour plots reveal the learned probability landscape
5. **Evaluation**: Component assignment accuracy and parameter recovery assessment
6. **Label Permutation**: Handling the identifiability issue where components can be reordered
7. **Convergence Monitoring**: Log-likelihood plots verify proper algorithm convergence

GMMs provide a flexible framework for clustering and density estimation, serving as building
blocks for more complex probabilistic models while remaining interpretable and efficient to fit.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

