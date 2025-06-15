# ## Simulating and Fitting a Gaussian Mixture Model

# This tutorial demonstrates how to use `StateSpaceDynamics.jl` to 
# create a Gaussian Mixture Model and fit it using the EM algorithm.

using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using StableRNGs
using Distributions
using StatsPlots

rng = StableRNG(1234);

# ## Create a true GaussianMixtureModel to simulate from

k = 3
D = 2  # data dimension

true_μs = [
    -1.0  1.0  0.0;
    -1.0 -1.5  2.0
]  # shape (D, K)

true_Σs = [Matrix{Float64}(0.3 * I(2)) for _ in 1:k]
true_πs = [0.5, 0.2, 0.3]

true_gmm = GaussianMixtureModel(k, true_μs, true_Σs, true_πs)

# ## Sample data from the true GMM
n = 500

# generate component labels (for plotting)
labels = rand(rng, Categorical(true_πs), n)
# generate samples from the GMM
X = Matrix{Float64}(undef, D, n)
for i in 1:n
    X[:, i] = rand(rng, MvNormal(true_μs[:, labels[i]], true_Σs[labels[i]]))
end

p1 = scatter(
    X[1, :], X[2, :];
    group=labels,
    title="GMM Samples",
    xlabel="x₁", ylabel="x₂",
    markersize=4,
    alpha=0.8,
    legend=false,
)
p1

# ## Fit a new GaussianMixtureModel to the data

fit_gmm = GaussianMixtureModel(k, D)

class_probabilities, lls = fit!(fit_gmm, X;
    maxiter=100, tol=1e-6, initialize_kmeans=true)

# ## Plot log-likelihoods to visualize EM convergence

p2 = plot(
    lls;
    xlabel="Iteration",
    ylabel="Log-Likelihood",
    title="EM Convergence",
    label="log_likelihood",
    marker=:circle,
)
p2

# ## Visualize model contours over the data

xs = collect(range(minimum(X[1, :]) - 1, stop=maximum(X[1, :]) + 1, length=150))
ys = collect(range(minimum(X[2, :]) - 1, stop=maximum(X[2, :]) + 1, length=150))

p3 = scatter(
    X[1, :], X[2, :];
    markersize=3, alpha=0.5,
    xlabel="x₁", ylabel="x₂",
    title="Data & Fitted GMM Contours by Component",
    legend=:topright,
)

colors = [:red, :green, :blue]

for i in 1:fit_gmm.k
    comp_dist = MvNormal(fit_gmm.μₖ[:, i], fit_gmm.Σₖ[i])
    Z_i = [fit_gmm.πₖ[i] * pdf(comp_dist, [x, y]) for y in ys, x in xs]

    contour!(
        p3, xs, ys, Z_i;
        levels=10,
        linewidth=2,
        c=colors[i],
        label="Comp $i",
    )
end

p3