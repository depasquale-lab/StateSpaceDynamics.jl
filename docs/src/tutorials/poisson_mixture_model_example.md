```@meta
EditURL = "../../examples/PoissonMixtureModel.jl"
```

## Simulating and Fitting a Poisson Mixture Model

This tutorial demonstrates how to use `StateSpaceDynamics.jl` to
create a Gaussian Mixture Model and fit it using the EM algorithm.

## Load Packages

````@example poisson_mixture_model_example
using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using StableRNGs
using StatsPlots
using Distributions
````

````@example poisson_mixture_model_example
rng = StableRNG(1234);
nothing #hide
````

## Create a State-Space Model

````@example poisson_mixture_model_example
k = 3
data_dim = 2
````

define true parameters for the model to sample from

````@example poisson_mixture_model_example
true_λs = [5.0, 10.0, 25.0]
true_πs = [0.25, 0.45, 0.3]

true_pmm = PoissonMixtureModel(k, true_λs, true_πs)
````

Generate sample data from the model

````@example poisson_mixture_model_example
n = 500
labels = rand(rng, Categorical(true_πs), n)
data   = [rand(rng, Poisson(true_λs[labels[i]])) for i in 1:n]
````

## Plot the sample data with distinct mixtures.

````@example poisson_mixture_model_example
p = plot()

histogram!(
  p, data;
  group     = labels,
  bins      = 0:1:maximum(data),
  bar_position = :dodge,
  xlabel    = "Count",
  ylabel    = "Frequency",
  title     = "Poisson‐Mixture Samples by Component",
  alpha     = 0.7,
  legend    = :topright,
)

p
````

## Paramter recovery: Initialize a new model with default parameters and fit to the data using EM.

````@example poisson_mixture_model_example
k = 3
````

define default parameters

````@example poisson_mixture_model_example
λs = ones(k)
πs = ones(k) ./ k

fit_pmm = PoissonMixtureModel(k, λs, πs)
````

## Fit model using EM Algorithm

````@example poisson_mixture_model_example
lls = fit!(fit_pmm, data; maxiter=100, tol=1e-6, initialize_kmeans=true)
````

## Confirm model convergence using log likelihoods

````@example poisson_mixture_model_example
plot(
  lls;
  xlabel="Iteration",
  ylabel="Log-Likelihood",
  title="EM Convergence (Poisson Mixture)",
  marker=:circle,
  label="log_likelihood",
  reuse=false,
)
````

## Plot the model pmf imposed over the generated data with distinct Mixtures

````@example poisson_mixture_model_example
p = plot()

histogram!(
  p, data;
  bins      = 0:1:maximum(data),
  normalize = true,
  alpha     = 0.3,
  label     = "Data",
  xlabel    = "Count",
  ylabel    = "Density",
  title     = "Poisson Mixtures: Data and PMFs",
)

x      = 0:maximum(data)
colors = [:red, :green, :blue]
for i in 1:k
    λi    = fit_pmm.λₖ[i]
    πi    = fit_pmm.πₖ[i]
    pmf_i = πi .* pdf.(Poisson(λi), x)
    plot!(
      p, x, pmf_i;
      lw    = 2,
      c     = colors[i],
      label = "Comp $i (λ=$(round(λi, sigdigits=3)))",
    )
end

mix_pmf = sum(πi .* pdf.(Poisson(λi), x) for (λi,πi) in zip(fit_pmm.λₖ, fit_pmm.πₖ))
plot!(
  p, x, mix_pmf;
  lw    = 3, ls=:dash, c=:black,
  label = "Mixture",
)

p
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

