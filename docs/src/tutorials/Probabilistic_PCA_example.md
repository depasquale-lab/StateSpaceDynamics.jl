```@meta
EditURL = "../../examples/ProbabilisticPCA.jl"
```

## Simulating and Fitting a Gaussian Mixture Model

This tutorial demonstrates how to use `StateSpaceDynamics.jl` to create a
Probabilistic PCA model and fit it using the EM algorithm.

## Load Packages

````@example Probabilistic_PCA_example
using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using StatsPlots
using StableRNGs
using Distributions
````

````@example Probabilistic_PCA_example
rng = StableRNG(1234);
nothing #hide
````

## Create a State-Space Model

````@example Probabilistic_PCA_example
D = 2
k = 2
````

define true parameters for the model to sample from

````@example Probabilistic_PCA_example
W_true = [
   -1.64   0.2;
    0.9  -2.8
]

σ²_true = 0.5
μ_true = [1.65, -1.3]

ppca = ProbabilisticPCA(W_true, σ²_true, μ_true)
````

Sample data from the model

````@example Probabilistic_PCA_example
num_obs = 500
X, z = rand(rng, ppca, num_obs)
````

## Plot sampled data from the model

````@example Probabilistic_PCA_example
x1 = X[1, :]
x2 = X[2, :]
labels = map(i -> (abs(z[1,i]) > abs(z[2,i]) ? 1 : 2), 1:size(z,2))

p = plot()

scatter!(
    p, x1, x2;
    group      = labels,
    xlabel     = "X₁",
    ylabel     = "X₂",
    title      = "Samples grouped by dominant latent factor",
    label= ["Latent 1" "Latent 2"],
    legend     = :topright,
    markersize = 5,
)

display(p)
````

## Paramter recovery: Initialize a new model with default parameters and fit to the data using EM.

````@example Probabilistic_PCA_example
#define default parameters
D = 2
k = 2
W = randn(D, k)
σ² = 0.5
μ_vector = randn(2)

fit_ppca = ProbabilisticPCA(W, σ², μ_vector)
````

## Fit model using EM Algorithm

````@example Probabilistic_PCA_example
lls = fit!(fit_ppca, X)
````

## Confirm model convergence using log likelihoods

````@example Probabilistic_PCA_example
ll_plot = plot(
    lls;
    xlabel="Iteration",
    ylabel="Log-Likelihood",
    title="EM Convergence (PPCA)",
    marker=:circle,
    label="log_likelihood",
    reuse=false,
)

display(ll_plot)
````

## Plot the learned lower dimension latent space over the Data

````@example Probabilistic_PCA_example
x1, x2 = X[1, :], X[2, :]
μ1, μ2  = fit_ppca.μ
W_fit = fit_ppca.W
w1      = W_fit[:, 1]
w2      = W_fit[:, 2]

P = plot()

scatter!(
    P, x1, x2;
    xlabel     = "X₁",
    ylabel     = "X₂",
    title      = "Data with PPCA loading directions",
    label      = "Data",
    alpha      = 0.5,
    markersize = 4,
)

k = size(W_fit, 2)
bases_x = repeat([μ_fit[1]], k)
bases_y = repeat([μ_fit[2]], k)
````

Add the component arrows in both directions

````@example Probabilistic_PCA_example
quiver!(
  P, [μ1], [μ2];
  quiver      = ([ w1[1]], [ w1[2]]),
  arrow       = :arrow, lw=3,
  color       = :red,
  label       = "W₁"
)
quiver!(
  P, [μ1], [μ2];
  quiver      = ([-w1[1]], [-w1[2]]),
  arrow       = :arrow, lw=3,
  color       = :red,
  label       = ""
)

quiver!(
  P, [μ1], [μ2];
  quiver      = ([ w2[1]], [ w2[2]]),
  arrow       = :arrow, lw=3,
  color       = :green,
  label       = "W₂"
)
quiver!(
  P, [μ1], [μ2];
  quiver      = ([-w2[1]], [-w2[2]]),
  arrow       = :arrow, lw=3,
  color       = :green,
  label       = ""
)

display(P)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

