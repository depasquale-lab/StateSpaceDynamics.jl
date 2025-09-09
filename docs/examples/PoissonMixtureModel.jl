# # Simulating and Fitting a Poisson Mixture Model
# 
# This tutorial shows how to build and fit a **Poisson Mixture Model (PMM)**
# with `StateSpaceDynamics.jl` using the Expectation–Maximization (EM) algorithm.
# It starts from simulation, fits the model, and then walks through diagnostics,
# interpretation, and common extensions.
#
# ## What is a Poisson Mixture Model?
# A PMM assumes each observation \(x_i\in \{0,1,2,\dots\}\) is drawn from one of
# \(k\) Poisson distributions, with means (rates) \(\lambda_1,\dots,\lambda_k\).
# The component is a latent categorical variable \(z_i\in\{1,\dots,k\}\) with
# mixing weights \(\pi_1,\dots,\pi_k\) (\(\sum_j \pi_j = 1\)). The generative process:
#
# 1. Draw \(z_i \sim \text{Categorical}(\pi)\)
# 2. Given \(z_i=j\), draw \(x_i \sim \text{Poisson}(\lambda_j)\)
#
# PMMs are handy for **count data** that come from a few heterogeneous sub-populations
# (e.g., spike counts from cells with different firing rates, click counts from multiple
# user segments, etc.).
#
# ## EM in one paragraph
# EM maximizes the marginal log-likelihood \(\log p(x\,|\,\pi,\lambda)\) by iterating:
# - **E-step:** compute responsibilities \(\gamma_{ij} = p(z_i=j\,|\,x_i,\theta)\)
# - **M-step:** update \(\pi_j\) and \(\lambda_j\) to maximize the expected complete-data log-likelihood.
# For Poisson mixtures, the closed-form M-step is:
# \[
# \pi_j \leftarrow \frac{1}{n} \sum_i \gamma_{ij}, \qquad
# \lambda_j \leftarrow \frac{\sum_i \gamma_{ij} x_i}{\sum_i \gamma_{ij}}.
# \]
# `fit!` in `StateSpaceDynamics.jl` performs these steps under the hood.
#
# ---

using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using StableRNGs
using StatsPlots
using Distributions

# We fix the RNG for reproducibility of simulated data and k-means seeding.
rng = StableRNG(1234);

# ## Create a true PoissonMixtureModel to simulate from
# We'll simulate from a mixture of k=3 Poisson components with distinct means and
# mixing weights. Feel free to change these and observe how the fitted model behaves.

k = 3
true_λs = [5.0, 10.0, 25.0]   # Poisson means (rates) per component
true_πs = [0.25, 0.45, 0.30]  # Mixing weights (must sum to 1)

true_pmm = PoissonMixtureModel(k, true_λs, true_πs);

# ## Generate data from the true model
# We'll draw n IID samples. `labels` are the latent component indices used for
# simulation; in practice these are unknown and must be inferred.

n = 500
labels = rand(rng, Categorical(true_πs), n)
data   = [rand(rng, Poisson(true_λs[labels[i]])) for i in 1:n];  # Vector{Int}

# ## Quick look: histogram of Poisson samples by component
# In real applications we don't see the true `labels`, but plotting them here helps
# build intuition: components with larger λ shift mass to larger counts.

p1 = histogram(
    data;
    group=labels,
    bins=0:1:maximum(data),
    bar_position=:dodge,
    xlabel="Count",
    ylabel="Frequency",
    title="Poisson Mixture Samples by Component",
    alpha=0.7,
    legend=:topright,
)
p1

# ## Fit a new PoissonMixtureModel to the data
# We construct an empty model with k components and call `fit!`. Options:
# - `maxiter`: EM iterations cap
# - `tol`: stop if relative improvement in log-likelihood is smaller than this
# - `initialize_kmeans=true`: seed with k-means on the 1D counts for more stable starts
#
# Note: EM is sensitive to initialization; try toggling `initialize_kmeans` or running
# from multiple random starts when you care about global optima.

fit_pmm = PoissonMixtureModel(k)
_, lls = fit!(fit_pmm, data; maxiter=100, tol=1e-6, initialize_kmeans=true);

# ## Convergence diagnostics
# EM guarantees non-decreasing log-likelihood. Monotone ascent is a good basic check.

p2 = plot(
    lls;
    xlabel="Iteration",
    ylabel="Log-Likelihood",
    title="EM Convergence (Poisson Mixture)",
    marker=:circle,
    label="log_likelihood",
)
p2

# ## Visual model check: mixture PMFs vs data
# Overlay the fitted component probability mass functions (PMFs) and the overall
# mixture PMF on the normalized histogram. Components should plausibly explain the
# major modes and tail behavior in the data.

p3 = histogram(
    data;
    bins=0:1:maximum(data),
    normalize=true,
    alpha=0.3,
    label="Data",
    xlabel="Count",
    ylabel="Density",
    title="Poisson Mixtures: Data and PMFs",
)

x = collect(0:maximum(data))
colors = [:red, :green, :blue]

for i in 1:k
    λi = fit_pmm.λₖ[i]
    πi = fit_pmm.πₖ[i]
    pmf_i = πi .* pdf.(Poisson(λi), x)
    plot!(
        p3, x, pmf_i;
        lw=2,
        c=colors[i],
        label="Comp $i (λ=$(round(λi, sigdigits=3)))",
    )
end

mix_pmf = reduce(+, (πi .* pdf.(Poisson(λi), x) for (λi, πi) in zip(fit_pmm.λₖ, fit_pmm.πₖ)))
plot!(
    p3, x, mix_pmf;
    lw=3, ls=:dash, c=:black,
    label="Mixture",
)

p3

# ## Interpreting the fitted parameters
# - `fit_pmm.λₖ` are the estimated Poisson rates per component.
# - `fit_pmm.πₖ` are the estimated mixing weights (sum to 1).
# Larger λ means the component puts more mass on larger counts. If two λs are close,
# EM may swap their order from run to run (label switching); the mixture distribution
# is unchanged.

# ## Posterior responsibilities (soft clustering)
# Responsibilities \(\gamma_{ij}=p(z_i=j\,|\,x_i,\hat\theta)\) quantify how likely
# each point belongs to each component. These are useful for soft assignments,
# uncertainty-aware summaries, and inspecting ambiguous points.

function responsibilities_pmm(λs::AbstractVector, πs::AbstractVector, x::AbstractVector)
    k = length(λs)
    n = length(x)
    Γ = zeros(n, k)
    for i in 1:n
        for j in 1:k
            Γ[i, j] = πs[j] * pdf(Poisson(λs[j]), x[i])
        end
        s = sum(Γ[i, :])
        Γ[i, :] ./= s > 0 ? s : 1.0
    end
    return Γ
end

Γ = responsibilities_pmm(fit_pmm.λₖ, fit_pmm.πₖ, data);

# Hard labels (if you need them) are argmax over responsibilities.
hard_labels = map(i -> argmax(view(Γ, i, :)), 1:n);

# ## Information criteria: picking k
# If you don't know k, you can fit several models and compare AIC/BIC:
#   AIC = 2p - 2LL,   BIC = p*log(n) - 2LL
# with parameter count p = (k-1) mixing weights + k rates = 2k-1.

function aic_bic(lls::AbstractVector, n::Int, k::Int)
    ll = last(lls)
    p = 2k - 1
    return (AIC = 2p - 2ll, BIC = p*log(n) - 2ll)
end

(ic_aic, ic_bic) = aic_bic(lls, n, k), aic_bic(lls, n, k)

# (In practice, compute these for multiple k and choose the one with the smallest
# AIC/BIC, balancing parsimony and fit.)

# ## Practical tips & pitfalls
# - **Initialization matters.** Try multiple random starts or k-means seeding.
# - **Label switching.** Component indices are arbitrary; sort components by λ for
#   stable presentation if needed.
# - **Small/empty components.** If some π_j \approx 0, consider reducing k or adding
#   a tiny ridge prior on λ to avoid degenerate updates.
# - **Zero-inflation / overdispersion.** If data have excess zeros or variance >> mean,
#   consider a Negative Binomial mixture or a zero-inflated Poisson.
# - **Train/validation split.** Use held-out likelihood or posterior predictive checks
#   for model assessment beyond AIC/BIC.

# ## Next steps (ideas for exercises)
# 1. **Unknown k:** loop over k=1:6, fit each, compare AIC/BIC.
# 2. **Stress test:** change true λs to be closer (e.g., [10, 12, 14]) and see how EM
#    behaves and how responsibilities reflect ambiguity.
# 3. **Posterior predictive check:** simulate new data from the fitted mixture and
#    compare histograms / tail behavior.
# 4. **Zero-inflation:** inject extra zeros and try a more flexible mixture class.

# ---
