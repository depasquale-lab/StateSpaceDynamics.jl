# ## Simulating and Fitting a Gaussian Mixture Model 

# This tutorial demonstrates how to use `StateSpaceDynamics.jl` to 
# create a Gaussian Mixture Model and fit it using the EM algorithm.

# ## Load Packages

using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using StableRNGs
using Distributions

#
rng = StableRNG(1234);

# ## Create a State-Space Model

k = 3
data_dim = 2

# define true parameters for the model to sample from 
true_μs = [
    -1.0  -1.0;
     1.0  -1.5;
     0.0   2.0
]
true_Σs = [0.3 * Matrix{Float64}(I, data_dim, data_dim) for _ in 1:k]
true_πs = [0.5, 0.2, 0.3]

true_gmm = GaussianMixtureModel(k, true_μs, true_Σs, true_πs)

# Generate sample data
n = 500
X = rand(true_gmm, n)

# ## Plot sampled data from the model 
labels = rand(rng, Distributions.Categorical(true_πs), n)
X2 = Array{Float64,2}(undef, n, data_dim)
for i in 1:n
    comp = labels[i]
    X2[i, :] = rand(rng, MvNormal(true_μs[comp, :], true_Σs[comp]))'
end

scatter(
  X2[:,1], X2[:,2];
  group=labels,
  title="GMM Samples Coloured by Component",
  xlabel="x₁", ylabel="x₂",
  markersize=4,
  alpha=0.8,
)

# ## Paramter recovery: Initialize a new model with default parameters and fit to the data 

k = 3
data_dim = 2

# define default parameters 
μs = zeros(Float64, k, data_dim)
Σs = [Matrix{Float64}(I, data_dim, data_dim) for _ in 1:k]
πs = ones(k) ./ k

fit_gmm = GaussianMixtureModel(k, μs, Σs, πs)

# ## Fit model using EM Algorithm 
class_probabilities, lls = fit!(fit_gmm, X; maxiter=100, tol=1e-6, initialize_kmeans=true)

# ## Confirm model convergence using log likelihoods 

plot(
  lls;
  xlabel="Iteration",
  ylabel="Log-Likelihood",
  title="EM Convergence",
  marker=:circle,
  reuse=false,
)


# ## Build a contour plot of the model imposed over the generated data 
xs = range(minimum(X[:,1]) - 1, stop=maximum(X[:,1]) + 1, length=150)
ys = range(minimum(X[:,2]) - 1, stop=maximum(X[:,2]) + 1, length=150)

scatter(
  X[:,1], X[:,2];
  markersize=3, alpha=0.5,
  xlabel="x₁", ylabel="x₂",
  title="Data & Fitted GMM Contours by Component",
  legend=:topright,              
)

colors = [:red, :green, :blue] 

for i in 1:fit_gmm.k
    comp_dist = MvNormal(fit_gmm.μₖ[i, :], fit_gmm.Σₖ[i])

    Z_i = [fit_gmm.πₖ[i] * pdf(comp_dist, [x,y]) for y in ys, x in xs]

    contour!(
      contour_plot, xs, ys, Z_i;
      levels    = 10,
      linewidth = 2,
      c         = colors[i],
      label     = "Comp $i",
    )
end
