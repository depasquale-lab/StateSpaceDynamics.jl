# ## Simulating and Fitting a Poisson Mixture Model 

# This tutorial demonstrates how to use `StateSpaceDynamics.jl` to simulate a latent
# linear dynamical system and fit it using the EM algorithm.

# ## Load Packages

using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using StableRNGs
using Distributions

rng = StableRNG(1234);
gr()

# Create a State-Space Model

k = 3
data_dim = 2

true_λs = [5.0, 10.0, 25.0]  
true_πs = [0.25, 0.45, 0.3]

true_pmm = PoissonMixtureModel(k, true_λs, true_πs)

# Generate and Plot sampled data from the model. 
n = 500
labels = rand(rng, Categorical(true_πs), n)
data   = [rand(rng, Poisson(true_λs[labels[i]])) for i in 1:n]

poisson_mixtures = histogram(
  data;
  group     = labels,                      
  bins      = 0:1:maximum(data),          
  xlabel    = "Count",
  ylabel    = "Frequency",
  title     = "Poisson‐Mixture Samples by Component",
  alpha     = 0.7,
  legend    = :topright,
  reuse     = false,
)
display(poisson_mixtures)

# Paramter recovery: Initialize a new model and fit the model to the data using EM.  
k = 3

λs = ones(k)
πs = ones(k) ./ k

fit_pmm = PoissonMixtureModel(k, λs, πs)

lls = fit!(fit_pmm, data; maxiter=100, tol=1e-6, initialize_kmeans=true)

# confirm model convergence

lls_convergence = plot(
  lls;
  xlabel="Iteration",
  ylabel="Log-Likelihood",
  title="EM Convergence (Poisson Mixture)",
  marker=:circle,
  label="log_likelihood",
  reuse=false,
)

display(lls_convergence) 

# # Contour plot of the fitted model density with the generated data

colors = [:red, :green, :blue]

histogram(
  data;
  bins      = 0:1:maximum(data),
  normalize = true,
  alpha     = 0.3,
  label     = "Data",
  xlabel    = "Count",
  ylabel    = "Density",
  title     = "Poisson Mixtures: Data and PMFs",
)

x      = 0:maximum(data)
colors = [:blue, :red, :green]
for i in 1:k
    λi    = fit_pmm.λₖ[i]
    πi    = fit_pmm.πₖ[i]
    pmf_i = πi .* pdf.(Poisson(λi), x)
    plot!(
      x, pmf_i;
      lw    = 2,
      c     = colors[i],
      label = "Comp $i (λ=$(round(λi, sigdigits=3)))",
    )
end

mix_pmf = sum(πi .* pdf.(Poisson(λi), x) for (λi,πi) in zip(fit_pmm.λₖ, fit_pmm.πₖ))
plot!(
  x, mix_pmf;
  lw    = 3, ls=:dash, c=:black,
  label = "Mixture",
)

display(current())