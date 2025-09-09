# # Simulating and Fitting a Probabilistic PCA (PPCA) Model
# 
# This tutorial walks through **Probabilistic PCA (PPCA)** in
# `StateSpaceDynamics.jl`: simulating data, fitting with EM, and interpreting
# the results. PPCA is a maximum-likelihood, probabilistic version of PCA with a
# simple latent-variable generative model and an explicit noise model.
#
# ## The PPCA model at a glance
# The generative story for \(x\in\mathbb R^D\) with \(k\) latent factors:
# \[
# z \sim \mathcal N(0, I_k),\qquad
# x \mid z \sim \mathcal N(\mu + W z,\ \sigma^2 I_D),
# \]
# where \(W\in\mathbb R^{D\times k}\) (factor loadings), \(\mu\in\mathbb R^D\), and
# \(\sigma^2>0\) (isotropic noise). The marginal covariance is
# \(\operatorname{Cov}(x)=W W^\top + \sigma^2 I\). When \(\sigma^2\to 0\), PPCA
# approaches standard PCA. Rotations of \(W\) (\(W R\) for orthogonal \(R\)) span
# the same principal subspace—this is the usual rotational non-identifiability.
#
# **Posterior over latents.** Given an observed \(x\), the posterior is Gaussian
# with
# \[
# M = I_k + \tfrac{1}{\sigma^2} W^\top W,\qquad
# \mathbb E[z\mid x] = M^{-1} W^\top (x-\mu)/\sigma^2,\qquad
# \operatorname{Cov}(z\mid x) = M^{-1}.
# \]
# `fit!` in `StateSpaceDynamics.jl` performs EM to maximize the likelihood.
#
# ---

# ## Load Packages
using StateSpaceDynamics
using LinearAlgebra
using Random
using Plots
using StatsPlots
using StableRNGs
using Distributions

# Reproducible randomness for simulation and initialization.
rng = StableRNG(1234);

# ## Create a PPCA model and simulate
# We'll work in 2D with two latent factors for easy visualization.

D = 2
k = 2; 

# True parameters used to generate synthetic data
W_true = [
   -1.64   0.2;
    0.9   -2.8
]

σ²_true = 0.5
μ_true  = [1.65, -1.3];

ppca = ProbabilisticPCA(W_true, σ²_true, μ_true)

# Draw IID samples from the model
num_obs = 500 
X, z = rand(rng, ppca, num_obs);

# ## Visualize the simulated data
# We'll color points by the dominant latent dimension (for intuition only—latent
# variables are unobserved in real data).

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
    label      = ["Latent 1" "Latent 2"],
    legend     = :topright,
    markersize = 5,
)

p

# ## Parameter recovery: fit PPCA with EM
# We'll start from random loadings/mean and a reasonable noise variance, then
# call `fit!` to run EM until convergence.

# (Re)define defaults to emphasize the fitting path
D = 2
k = 2 
W = randn(rng, D, k)
σ² = 0.5
μ_vector = randn(rng, 2)

fit_ppca = ProbabilisticPCA(W, σ², μ_vector);

# ### Fit with EM
# `fit!` returns the log-likelihood trace. Monotone ascent is a good sanity check.
lls = fit!(fit_ppca, X)

# ### Log-likelihood diagnostic
ll_plot = plot(
    lls;
    xlabel="Iteration",
    ylabel="Log-Likelihood",
    title="EM Convergence (PPCA)",
    marker=:circle,
    label="log_likelihood",
    reuse=false,
)

ll_plot

# ## Interpreting the learned parameters
# - `fit_ppca.W` are the learned loading directions. Columns span the principal
#   subspace (up to rotation). For k=2 in 2D, they form a basis centered at μ.
# - `fit_ppca.μ` is the learned mean of the data.
# - `fit_ppca.σ²` is the isotropic residual variance.

x1, x2 = X[1, :], X[2, :]
μ1, μ2  = fit_ppca.μ
W_fit   = fit_ppca.W
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
);

# Draw loading vectors from the mean in both directions for visibility
quiver!(P, [μ1], [μ2]; quiver=([ w1[1]], [ w1[2]]), arrow=:arrow, lw=3, color=:red,   label="W₁")
quiver!(P, [μ1], [μ2]; quiver=([-w1[1]], [-w1[2]]), arrow=:arrow, lw=3, color=:red,   label="")
quiver!(P, [μ1], [μ2]; quiver=([ w2[1]], [ w2[2]]), arrow=:arrow, lw=3, color=:green, label="W₂")
quiver!(P, [μ1], [μ2]; quiver=([-w2[1]], [-w2[2]]), arrow=:arrow, lw=3, color=:green, label="")

P

# ## Posterior latents and reconstructions
# Compute \nE[z|x]\n and optional reconstructions \(\hat x = \mu + W\,\mathbb E[z|x]\).

function ppca_posterior_means(W::AbstractMatrix, σ²::Real, μ::AbstractVector, X::AbstractMatrix)
    D, N = size(X)
    k    = size(W, 2)
    M = I(k) + (W' * W) / σ²            # k×k
    B = M \ (W' / σ²)                  # k×D, equals M^{-1} W^T / σ²
    Zmean = B * (X .- μ)               # k×N
    return Zmean
end

Ẑ = ppca_posterior_means(W_fit, fit_ppca.σ², fit_ppca.μ, X)
X̂ = fit_ppca.μ .+ W_fit * Ẑ;

# Example: reconstruction error (per-dimension MSE)
recon_mse = mean(norm.(eachcol(X - X̂)).^2) / size(X,1)

# ## Variance explained and choosing k
# A quick check is to compare the sample covariance eigenvalues to the PPCA model.
# For PPCA with k factors, the top-k eigenvalues should be captured by W W^T and
# the remainder approximated by σ².

Σ̂ = cov(permutedims(X))            # D×D sample covariance
λs = sort(eigvals(Symmetric(Σ̂)); rev=true);

# Proportion of variance explained by top-k sample eigenvalues
pve_sample = sum(λs[1:k]) / sum(λs)

# PPCA-implied total variance: tr(WW^T) + D*σ²
pve_ppca = (tr(W_fit * W_fit') ) / (tr(W_fit * W_fit') + size(X,1) * 0 + length(μ1:μ2) * fit_ppca.σ²)  # placeholder formula to keep inline; see note below

# NOTE: For PPCA in D dims, total variance is tr(WW^T) + D*σ².
# If you prefer, compute: pve_ppca = tr(W_fit*W_fit') / (tr(W_fit*W_fit') + D*fit_ppca.σ²)

# ## Practical tips & pitfalls
# - **Scaling matters.** Standardize your features if units differ.
# - **Initialization:** Multiple random starts can help avoid poor local optima.
# - **Rotational ambiguity:** For presentation, you can orthonormalize W or align it
#   with PCA loadings via Procrustes.
# - **Choosing k:** Use scree plots, PVE, or information criteria (AIC/BIC) on the
#   marginal likelihood returned by `fit!`.
# - **Outliers/heavy tails:** Consider robust variants (e.g., t-PCA) if needed.

# ## Where this fits in `StateSpaceDynamics.jl`
# PPCA is an IID latent-factor model. In the ecosystem, it bridges to time-series
# models with latent structure, such as LDS/PLDS where factors evolve over time.
# The workflow mirrors those models: specify (W, σ², μ), simulate, fit with EM,
# and validate with likelihood curves and posterior diagnostics.

# ## Exercises
# 1. **Compare to PCA:** Compute the top-2 eigenvectors of the sample covariance
#    and align `fit_ppca.W` to them with an orthogonal Procrustes transform.
# 2. **Vary noise:** Increase `σ²_true` and see how loading directions and
#    convergence behave.
# 3. **Model selection:** Fit k=1..D and report AIC/BIC (parameter count is
#    p = D*k + 1 (σ²) + D (μ)).
# 4. **Held-out likelihood:** Split X into train/validation and compare models.

# ---
# End of tutorial.
