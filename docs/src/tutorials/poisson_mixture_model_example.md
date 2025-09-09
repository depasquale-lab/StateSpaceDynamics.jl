```@meta
EditURL = "../../examples/PoissonMixtureModel.jl"
```

# Simulating and Fitting a Poisson Mixture Model

This tutorial demonstrates how to build and fit a **Poisson Mixture Model (PMM)**
with `StateSpaceDynamics.jl` using the Expectation-Maximization (EM) algorithm.
We'll cover simulation, fitting, diagnostics, interpretation, and practical considerations.

## What is a Poisson Mixture Model?

A PMM assumes each observation $x_i \in \{0,1,2,\ldots\}$ is drawn from one of
$k$ Poisson distributions with rates $\lambda_1,\ldots,\lambda_k$.
The component assignment is a latent categorical variable $z_i \in \{1,\ldots,k\}$
with mixing weights $\pi_1,\ldots,\pi_k$ where $\sum_j \pi_j = 1$.

**Generative process:**
1. Draw $z_i \sim \text{Categorical}(\boldsymbol{\pi})$
2. Given $z_i = j$, draw $x_i \sim \text{Poisson}(\lambda_j)$

PMMs are useful for **count data** from heterogeneous sub-populations
(e.g., spike counts from different neuron types, customer transaction counts
from different segments, or event frequencies across different regimes).

## EM Algorithm Overview

EM maximizes the marginal log-likelihood $\log p(\mathbf{x} | \boldsymbol{\pi}, \boldsymbol{\lambda})$ by iterating:
- **E-step:** Compute responsibilities $\gamma_{ij} = P(z_i = j | x_i, \boldsymbol{\theta})$
- **M-step:** Update parameters to maximize expected complete-data log-likelihood

For Poisson mixtures, the M-step has closed-form updates:
$$\pi_j \leftarrow \frac{1}{n} \sum_i \gamma_{ij}, \quad \lambda_j \leftarrow \frac{\sum_i \gamma_{ij} x_i}{\sum_i \gamma_{ij}}$$

## Load Required Packages

````@example poisson_mixture_model_example
using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using StableRNGs
using StatsPlots
using Distributions
````

Fix RNG for reproducible simulation and k-means seeding

````@example poisson_mixture_model_example
rng = StableRNG(1234);
nothing #hide
````

## Create True Poisson Mixture Model

We'll simulate from a mixture of $k=3$ Poisson components with distinct
rates and mixing weights. These parameters create well-separated components
that should be recoverable by EM.

````@example poisson_mixture_model_example
k = 3
true_λs = [5.0, 10.0, 25.0]   # Poisson rates per component
true_πs = [0.25, 0.45, 0.30]  # Mixing weights (sum to 1)

true_pmm = PoissonMixtureModel(k, true_λs, true_πs);

print("True model: k=$k components\n")
for i in 1:k
    print("Component $i: λ=$(true_λs[i]), π=$(true_πs[i])\n")
end
````

## Generate Synthetic Data

Draw $n$ independent samples. The `labels` indicate true component membership
for each observation (unknown in practice and must be inferred).

````@example poisson_mixture_model_example
n = 500
labels = rand(rng, Categorical(true_πs), n)
data = [rand(rng, Poisson(true_λs[labels[i]])) for i in 1:n];

print("Generated $n samples with count range [$(minimum(data)), $(maximum(data))]\n");
nothing #hide
````

Visualize samples by true component membership
Components with larger $\lambda$ shift mass toward higher counts

````@example poisson_mixture_model_example
p1 = histogram(data;
    group=labels,
    bins=0:1:maximum(data),
    bar_position=:dodge,
    xlabel="Count", ylabel="Frequency",
    title="Poisson Mixture Samples by True Component",
    alpha=0.7, legend=:topright
)
````

## Fit Poisson Mixture Model with EM

Construct model with $k$ components and fit using EM algorithm.
Key options:
- `maxiter`: Maximum EM iterations
- `tol`: Convergence tolerance (relative log-likelihood improvement)
- `initialize_kmeans=true`: Use k-means for stable initialization

````@example poisson_mixture_model_example
fit_pmm = PoissonMixtureModel(k)
_, lls = fit!(fit_pmm, data; maxiter=100, tol=1e-6, initialize_kmeans=true);

print("EM converged in $(length(lls)) iterations\n")
print("Log-likelihood improved by $(round(lls[end] - lls[1], digits=1))\n");
nothing #hide
````

Display learned parameters

````@example poisson_mixture_model_example
print("Learned parameters:\n")
for i in 1:k
    print("Component $i: λ=$(round(fit_pmm.λₖ[i], digits=2)), π=$(round(fit_pmm.πₖ[i], digits=3))\n")
end
````

## Monitor EM Convergence

EM guarantees non-decreasing log-likelihood. Monotonic ascent indicates proper convergence.

````@example poisson_mixture_model_example
p2 = plot(lls;
    xlabel="Iteration", ylabel="Log-Likelihood",
    title="EM Convergence",
    marker=:circle, markersize=3, lw=2,
    legend=false, color=:darkblue
)

annotate!(p2, length(lls)*0.7, lls[end]*0.98,
    text("Final LL: $(round(lls[end], digits=1))", 10))
````

## Visual Model Assessment

Overlay fitted component PMFs and overall mixture PMF on normalized histogram.
Components should explain major modes and tail behavior in the data.

````@example poisson_mixture_model_example
p3 = histogram(data;
    bins=0:1:maximum(data), normalize=true, alpha=0.3,
    xlabel="Count", ylabel="Probability Density",
    title="Data vs. Fitted Mixture Components",
    label="Data", color=:gray
)

x_range = collect(0:maximum(data))
colors = [:red, :green, :blue]
````

Plot individual component PMFs

````@example poisson_mixture_model_example
for i in 1:k
    λᵢ = fit_pmm.λₖ[i]
    πᵢ = fit_pmm.πₖ[i]
    pmf_i = πᵢ .* pdf.(Poisson(λᵢ), x_range)
    plot!(p3, x_range, pmf_i;
        lw=2, color=colors[i],
        label="Component $i (λ=$(round(λᵢ, digits=1)))"
    )
end
````

Plot overall mixture PMF

````@example poisson_mixture_model_example
mixture_pmf = reduce(+, (πᵢ .* pdf.(Poisson(λᵢ), x_range)
                        for (λᵢ, πᵢ) in zip(fit_pmm.λₖ, fit_pmm.πₖ)))
plot!(p3, x_range, mixture_pmf;
    lw=3, linestyle=:dash, color=:black,
    label="Mixture PMF"
)
````

## Posterior Responsibilities (Soft Clustering)

Responsibilities $\gamma_{ij} = P(z_i = j | x_i, \hat{\boldsymbol{\theta}})$ quantify
how likely each observation belongs to each component. These provide soft assignments
and uncertainty quantification.

````@example poisson_mixture_model_example
function responsibilities_pmm(λs::AbstractVector, πs::AbstractVector, x::AbstractVector)
    k, n = length(λs), length(x)
    Γ = zeros(n, k)

    for i in 1:n
        for j in 1:k
            Γ[i, j] = πs[j] * pdf(Poisson(λs[j]), x[i])
        end

        row_sum = sum(Γ[i, :]) # Normalize to get probabilities
        if row_sum > 0
            Γ[i, :] ./= row_sum
        end
    end
    return Γ
end

Γ = responsibilities_pmm(fit_pmm.λₖ, fit_pmm.πₖ, data);
nothing #hide
````

Hard assignments (if needed) are argmax over responsibilities

````@example poisson_mixture_model_example
hard_labels = [argmax(Γ[i, :]) for i in 1:n];
nothing #hide
````

Calculate assignment accuracy compared to true labels

````@example poisson_mixture_model_example
accuracy = mean(labels .== hard_labels)
print("Component assignment accuracy: $(round(accuracy*100, digits=1))%\n");
nothing #hide
````

## Information Criteria for Model Selection

When $k$ is unknown, compare models using AIC/BIC:
- AIC = $2p - 2\text{LL}$
- BIC = $p \log(n) - 2\text{LL}$
where parameter count $p = (k-1) + k = 2k-1$ (mixing weights + rates)

````@example poisson_mixture_model_example
function compute_ic(lls::AbstractVector, n::Int, k::Int)
    ll = last(lls)
    p = 2k - 1
    return (AIC = 2p - 2ll, BIC = p*log(n) - 2ll)
end

ic = compute_ic(lls, n, k)
print("Information criteria: AIC = $(round(ic.AIC, digits=1)), BIC = $(round(ic.BIC, digits=1))\n");
nothing #hide
````

## Parameter Recovery Assessment

````@example poisson_mixture_model_example
print("\n=== Parameter Recovery Assessment ===\n")
````

Compare true vs learned rates (account for potential label permutation)

````@example poisson_mixture_model_example
λ_errors = [abs(true_λs[i] - fit_pmm.λₖ[i]) / true_λs[i] for i in 1:k]
π_errors = [abs(true_πs[i] - fit_pmm.πₖ[i]) for i in 1:k]

print("Rate recovery errors (%):\n")
for i in 1:k
    print("Component $i: $(round(λ_errors[i]*100, digits=1))%\n")
end

print("Mixing weight recovery errors:\n")
for i in 1:k
    print("Component $i: $(round(π_errors[i], digits=3))\n")
end
````

## Practical Considerations and Extensions

````@example poisson_mixture_model_example
print("\n=== Practical Tips ===\n")
print("• Initialization matters: try multiple random starts or k-means seeding\n")
print("• Label switching: component indices are arbitrary, sort by λ for stability\n")
print("• Empty components: if πⱼ ≈ 0, consider reducing k\n")
print("• Model selection: use cross-validation or information criteria to choose k\n")
print("• Zero-inflation: consider zero-inflated Poisson for excess zeros\n")
print("• Overdispersion: use Negative Binomial mixtures if variance >> mean\n");
nothing #hide
````

## Model Selection Example

Demonstrate fitting multiple values of k and comparing via BIC

````@example poisson_mixture_model_example
print("\n=== Model Selection Demo ===\n")

k_range = 1:5
bic_scores = Float64[]
aic_scores = Float64[]

for k_test in k_range
    pmm_test = PoissonMixtureModel(k_test)
    _, lls_test = fit!(pmm_test, data; maxiter=50, tol=1e-6, initialize_kmeans=true)

    ic_test = compute_ic(lls_test, n, k_test)
    push!(aic_scores, ic_test.AIC)
    push!(bic_scores, ic_test.BIC)

    print("k=$k_test: AIC=$(round(ic_test.AIC, digits=1)), BIC=$(round(ic_test.BIC, digits=1))\n")
end
````

Plot information criteria vs number of components

````@example poisson_mixture_model_example
p4 = plot(k_range, [aic_scores bic_scores];
    xlabel="Number of Components (k)", ylabel="Information Criterion",
    title="Model Selection via Information Criteria",
    label=["AIC" "BIC"], marker=:circle, lw=2
)

optimal_k_aic = k_range[argmin(aic_scores)]
optimal_k_bic = k_range[argmin(bic_scores)]

print("Optimal k: AIC suggests k=$optimal_k_aic, BIC suggests k=$optimal_k_bic\n");
nothing #hide
````

## Summary

This tutorial demonstrated the complete Poisson Mixture Model workflow:

**Key Concepts:**
- **Discrete mixtures**: Model count data as mixture of Poisson distributions
- **EM algorithm**: Iterative optimization with closed-form M-step updates
- **Soft clustering**: Posterior responsibilities provide probabilistic assignments
- **Model selection**: Information criteria help choose appropriate number of components

**Applications:**
- Spike count analysis in neuroscience
- Customer transaction modeling in business analytics
- Event frequency analysis in reliability engineering
- Gene expression count clustering in bioinformatics

**Technical Insights:**
- Initialization strategy significantly affects final solution quality
- Label switching is a fundamental identifiability issue in mixture models
- Information criteria provide principled approach to model complexity selection
- Component separation quality affects parameter recovery accuracy

**Extensions:**
- Zero-inflated Poisson mixtures for excess zero counts
- Negative Binomial mixtures for overdispersed count data
- Bayesian approaches for uncertainty quantification
- Mixture regression models for count data with covariates

Poisson mixture models provide a flexible framework for modeling heterogeneous
count data, enabling both clustering and density estimation while maintaining
interpretable probabilistic structure.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

