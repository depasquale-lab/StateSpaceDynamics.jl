```@meta
EditURL = "../../examples/ProbabilisticPCA.jl"
```

# Simulating and Fitting a Probabilistic PCA (PPCA) Model

This tutorial demonstrates **Probabilistic PCA (PPCA)** in `StateSpaceDynamics.jl`:
simulating data, fitting with EM, and interpreting results. PPCA is a maximum-likelihood,
probabilistic version of PCA with an explicit latent-variable generative model and noise model.

## The PPCA Model

The generative model for observations $\mathbf{x} \in \mathbb{R}^D$ with $k$ latent factors:
$$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_k), \quad \mathbf{x} | \mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu} + \mathbf{W}\mathbf{z}, \sigma^2 \mathbf{I}_D)$$

where $\mathbf{W} \in \mathbb{R}^{D \times k}$ (factor loadings), $\boldsymbol{\mu} \in \mathbb{R}^D$ (mean),
and $\sigma^2 > 0$ (isotropic noise variance).

**Key properties:**
- Marginal covariance: $\text{Cov}(\mathbf{x}) = \mathbf{W}\mathbf{W}^T + \sigma^2 \mathbf{I}$
- As $\sigma^2 \to 0$, PPCA approaches standard PCA
- Rotational non-identifiability: $\mathbf{W}\mathbf{R}$ for orthogonal $\mathbf{R}$ spans same subspace

**Posterior over latents:** Given observation $\mathbf{x}$, the posterior is Gaussian with:
$$\mathbf{M} = \mathbf{I}_k + \frac{1}{\sigma^2}\mathbf{W}^T\mathbf{W}, \quad \mathbb{E}[\mathbf{z}|\mathbf{x}] = \mathbf{M}^{-1}\mathbf{W}^T(\mathbf{x}-\boldsymbol{\mu})/\sigma^2$$

## Load Required Packages

````@example Probabilistic_PCA_example
using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using StatsPlots
using StableRNGs
using Distributions
using LaTeXStrings
````

Set reproducible randomness for simulation and initialization

````@example Probabilistic_PCA_example
rng = StableRNG(1234);
nothing #hide
````

## Create and Simulate PPCA Model

We'll work in 2D with two latent factors for easy visualization and interpretation.

````@example Probabilistic_PCA_example
D = 2  # Observation dimensionality
k = 2  # Number of latent factors
````

True parameters for data generation

````@example Probabilistic_PCA_example
W_true = [-1.64  0.2;   # Factor loading matrix
           0.9  -2.8]
σ²_true = 0.5           # Noise variance
μ_true = [1.65, -1.3];  # Mean vector

ppca = ProbabilisticPCA(W_true, σ²_true, μ_true)
````

Generate synthetic data

````@example Probabilistic_PCA_example
num_obs = 500
X, z = rand(rng, ppca, num_obs);

print("Generated $num_obs observations in $D dimensions with $k latent factors\n")
print("Data range: X₁ ∈ [$(round(minimum(X[1,:]), digits=2)), $(round(maximum(X[1,:]), digits=2))], ")
print("X₂ ∈ [$(round(minimum(X[2,:]), digits=2)), $(round(maximum(X[2,:]), digits=2))]\n");
nothing #hide
````

## Visualize Simulated Data

Color points by dominant latent dimension for intuition (latent variables are unobserved in practice)

````@example Probabilistic_PCA_example
x1, x2 = X[1, :], X[2, :]
labels = [abs(z[1,i]) > abs(z[2,i]) ? 1 : 2 for i in 1:size(z,2)]

p1 = scatter(x1, x2;
    group=labels, xlabel=L"X_1", ylabel=L"X_2",
    title="Simulated Data (colored by dominant latent factor)",
    markersize=4, alpha=0.7,
    palette=[:dodgerblue, :crimson],
    legend=:topright
)
````

## Fit PPCA Using EM Algorithm

Start from random initialization and use EM to learn parameters.
The algorithm maximizes the marginal log-likelihood of the observed data.

Initialize with random parameters

````@example Probabilistic_PCA_example
W_init = randn(rng, D, k)
σ²_init = 0.5
μ_init = randn(rng, D)

fit_ppca = ProbabilisticPCA(W_init, σ²_init, μ_init)

print("Running EM algorithm...")
````

Fit with EM - returns log-likelihood trace for convergence monitoring

````@example Probabilistic_PCA_example
lls = fit!(fit_ppca, X);

print("EM converged in $(length(lls)) iterations\n")
print("Log-likelihood improved by $(round(lls[end] - lls[1], digits=1))\n");
nothing #hide
````

Monitor EM convergence - should show monotonic increase

````@example Probabilistic_PCA_example
p2 = plot(lls;
    xlabel="Iteration", ylabel="Log-Likelihood",
    title="EM Convergence", marker=:circle, markersize=3,
    lw=2, legend=false, color=:darkblue
)
````

## Visualize Learned Loading Directions

The columns of $\mathbf{W}$ span the principal subspace (up to rotation).
We'll plot these loading vectors from the learned mean.

````@example Probabilistic_PCA_example
μ_fit = fit_ppca.μ
W_fit = fit_ppca.W
w1, w2 = W_fit[:, 1], W_fit[:, 2]

p3 = scatter(x1, x2;
    xlabel=L"X_1", ylabel=L"X_2",
    title="Data with Learned PPCA Loading Directions",
    label="Data", alpha=0.5, markersize=3, color=:gray
)
````

Draw loading vectors in both directions for visibility

````@example Probabilistic_PCA_example
scale = 2.0  # Scale for better visualization
quiver!(p3, [μ_fit[1]], [μ_fit[2]];
    quiver=([scale*w1[1]], [scale*w1[2]]),
    arrow=:arrow, lw=3, color=:red, label="W₁")
quiver!(p3, [μ_fit[1]], [μ_fit[2]];
    quiver=([-scale*w1[1]], [-scale*w1[2]]),
    arrow=:arrow, lw=3, color=:red, label="")
quiver!(p3, [μ_fit[1]], [μ_fit[2]];
    quiver=([scale*w2[1]], [scale*w2[2]]),
    arrow=:arrow, lw=3, color=:green, label="W₂")
quiver!(p3, [μ_fit[1]], [μ_fit[2]];
    quiver=([-scale*w2[1]], [-scale*w2[2]]),
    arrow=:arrow, lw=3, color=:green, label="")
````

## Posterior Inference and Reconstruction

Compute posterior means $\mathbb{E}[\mathbf{z}|\mathbf{x}]$ and reconstructions $\hat{\mathbf{x}} = \boldsymbol{\mu} + \mathbf{W}\mathbb{E}[\mathbf{z}|\mathbf{x}]$

````@example Probabilistic_PCA_example
function ppca_posterior_means(W::AbstractMatrix, σ²::Real, μ::AbstractVector, X::AbstractMatrix)
    k = size(W, 2)
    M = I(k) + (W' * W) / σ²            # Posterior precision matrix
    B = M \ (W' / σ²)                   # Efficient computation of M⁻¹W^T/σ²
    Z_mean = B * (X .- μ)               # Posterior means
    return Z_mean
end
````

Compute posterior latent means and reconstructions

````@example Probabilistic_PCA_example
Ẑ = ppca_posterior_means(W_fit, fit_ppca.σ², μ_fit, X)
X̂ = μ_fit .+ W_fit * Ẑ
````

Calculate reconstruction error

````@example Probabilistic_PCA_example
recon_mse = mean(sum((X - X̂).^2, dims=1)) / D
print("Reconstruction MSE: $(round(recon_mse, digits=4))\n");
nothing #hide
````

## Variance Explained Analysis

Compare sample covariance eigenvalues to PPCA model structure.
PPCA should capture top-k eigenvalues via $\mathbf{W}\mathbf{W}^T$ and
approximate remainder with isotropic noise $\sigma^2$.

````@example Probabilistic_PCA_example
Σ_sample = cov(X, dims=2)  # Sample covariance matrix
λs = sort(eigvals(Σ_sample), rev=true)  # Eigenvalues in descending order
````

Proportion of variance explained by top-k eigenvalues

````@example Probabilistic_PCA_example
pve_sample = sum(λs[1:k]) / sum(λs)
````

PPCA-implied variance components

````@example Probabilistic_PCA_example
total_var_ppca = tr(W_fit * W_fit') + D * fit_ppca.σ²
explained_var_ppca = tr(W_fit * W_fit')
pve_ppca = explained_var_ppca / total_var_ppca

print("Variance Analysis:\n")
print("Sample eigenvalues: $(round.(λs, digits=3))\n")
print("PVE (sample top-$k): $(round(pve_sample*100, digits=1))%\n")
print("PVE (PPCA model): $(round(pve_ppca*100, digits=1))%\n");
nothing #hide
````

## Parameter Recovery Assessment

````@example Probabilistic_PCA_example
print("\n=== Parameter Recovery Assessment ===\n")
````

Compare true vs learned parameters

````@example Probabilistic_PCA_example
W_error = norm(W_true - W_fit) / norm(W_true)
μ_error = norm(μ_true - μ_fit) / norm(μ_true)
σ²_error = abs(σ²_true - fit_ppca.σ²) / σ²_true

print("Parameter recovery errors:\n")
print("Loading matrix W: $(round(W_error*100, digits=1))%\n")
print("Mean vector μ: $(round(μ_error*100, digits=1))%\n")
print("Noise variance σ²: $(round(σ²_error*100, digits=1))%\n")

print("True parameters:\n")
print("W = $(round.(W_true, digits=2))\n")
print("μ = $(round.(μ_true, digits=2)), σ² = $(σ²_true)\n")

print("Learned parameters:\n")
print("W = $(round.(W_fit, digits=2))\n")
print("μ = $(round.(μ_fit, digits=2)), σ² = $(round(fit_ppca.σ², digits=2))\n");
nothing #hide
````

## Model Selection Example

Demonstrate fitting models with different numbers of latent factors
and comparing via information criteria.

````@example Probabilistic_PCA_example
function compute_aic_bic(ll::Real, n_params::Int, n_obs::Int)
    aic = 2*n_params - 2*ll
    bic = n_params*log(n_obs) - 2*ll
    return (aic, bic)
end

print("\n=== Model Selection Demo ===\n")

k_range = 1:min(D, 4)
aic_scores = Float64[]
bic_scores = Float64[]
lls_final = Float64[]

for k_test in k_range
    W_test = randn(rng, D, k_test) # Initialize and fit model with k_test factors
    ppca_test = ProbabilisticPCA(W_test, 0.5, zeros(D))
    lls_test = fit!(ppca_test, X)

    n_params = D * k_test + D + 1 # Parameter count: D*k_test (W) + D (μ) + 1 (σ²)
    ll_final = lls_test[end]
    aic, bic = compute_aic_bic(ll_final, n_params, num_obs)  # Calculate information criteria

    push!(aic_scores, aic)
    push!(bic_scores, bic)
    push!(lls_final, ll_final)

    print("k=$k_test: LL=$(round(ll_final, digits=1)), AIC=$(round(aic, digits=1)), BIC=$(round(bic, digits=1))\n")
end
````

Plot information criteria

````@example Probabilistic_PCA_example
p4 = plot(k_range, [aic_scores bic_scores];
    xlabel="Number of Latent Factors (k)", ylabel="Information Criterion",
    title="Model Selection via Information Criteria",
    label=["AIC" "BIC"], marker=:circle, lw=2
)

optimal_k = k_range[argmin(bic_scores)]
print("BIC suggests optimal k = $optimal_k\n");
nothing #hide
````

## Summary

This tutorial demonstrated the complete Probabilistic PCA workflow:

**Key Concepts:**
- **Probabilistic framework**: Explicit generative model with latent factors and noise
- **EM algorithm**: Iterative maximum-likelihood parameter estimation
- **Posterior inference**: Probabilistic latent variable estimates and reconstructions
- **Model selection**: Information criteria for choosing appropriate number of factors

**Advantages over Standard PCA:**
- Principled handling of missing data and noise
- Probabilistic interpretation enables uncertainty quantification
- Natural framework for model selection and comparison
- Seamless extension to more complex latent variable models

**Applications:**
- Dimensionality reduction for high-dimensional data
- Exploratory data analysis and visualization
- Feature extraction for machine learning pipelines
- Foundation for more complex factor models and state-space models

**Technical Insights:**
- Loading matrix $\mathbf{W}$ captures principal directions of variation
- Noise parameter $\sigma^2$ quantifies unexplained variance
- Rotational non-identifiability requires care in interpretation
- EM convergence monitoring ensures reliable parameter estimates

PPCA provides a flexible, probabilistic approach to factor analysis that bridges
classical multivariate statistics with modern latent variable modeling, serving as
both a standalone technique and building block for more sophisticated models.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

