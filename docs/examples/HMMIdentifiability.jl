# # Understanding Non-Identifiability in Hidden Markov Models
#
# This hands-on literate tutorial mirrors the LDS/SSM label-indeterminacy example but for HMMs. We will:
#
# 1. **Build a reference HMM and synthesize data.**
# 2. **Show permutation invariance of the likelihood** (log-likelihood is unchanged under state relabelings).
# 3. **Fit from multiple random initializations** to surface **label switching**.
# 4. **Align states post‑hoc** to a reference ordering and outline a simple canonicalization.
# 5. **Summarize diagnostics** and clarify what **is** and **is not** identifiable.
#
# > **Takeaway.** In mixture models and HMMs, state labels are arbitrary. Without constraints or post‑hoc alignment, EM fits from different starts can represent the *same* model up to a permutation of labels.

# ## Load Required Packages
# We’ll use `StateSpaceDynamics.jl` for HMM construction and EM fitting. `StableRNGs` gives bit‑reproducibility across runs.

using StateSpaceDynamics
using Random
using LinearAlgebra
using Statistics
using StableRNGs
using Printf
using Distributions

rng = StableRNG(12345);

# ## Create a Reference ("True") Gaussian-Emission HMM
# We define a simple 3‑state, 1D Gaussian‑emission HMM with well‑separated means. The transition matrix is near‑symmetric
# with strong self‑transition probability, making states persist for stretches—this helps visualize label switching cleanly.

K = 3
D = 1
T = 400

π_true = fill(1/3, K)  # uniform prior

ρ = 0.88                # near-symmetric transitions with strong self-prob
A_true = fill((1-ρ)/(K-1), K, K)
for i in 1:K
    A_true[i,i] = ρ
end

μ_true  = [-2.5, 0.0, 2.5]
σ2_true = fill(0.1, K)
Σ_true  = [fill(σ2_true[k], D, D) for k in 1:K]

emissions = [GaussianEmission(1, [μ_true[k]], Σ_true[k]) for k in 1:K]
true_hmm = HiddenMarkovModel(A_true, emissions, π_true, K)

# Generate data
s_true, x_true = rand(rng, true_hmm; n=T)

# ## Permutation (Label) Invariance: Likelihood is Unchanged
# The HMM likelihood is **invariant** to permutation of state labels. If we apply a permutation matrix `P` to the transition
# matrix and reorder emissions accordingly, the joint distribution (and hence log-likelihood on data) is identical.
# We implement a helper `permute_states!` and verify that the log-likelihood difference is numerically ~0.

function permute_states!(m::HiddenMarkovModel, perm::Vector{Int})
    K = length(perm)
    P = zeros(eltype(m.A), K, K)
    for (i,j) in enumerate(perm)   # new i  <- old j
        P[i,j] = 1
    end
    m.A  .= P * m.A * P'
    m.πₖ .= P * m.πₖ
    m.B   = m.B[perm]
    return m
end

mtest = deepcopy(true_hmm)
ll0   = StateSpaceDynamics.loglikelihood(mtest, x_true)
perm  = [2,3,1]  # arbitrary relabeling
permute_states!(mtest, perm)
ll1   = StateSpaceDynamics.loglikelihood(mtest, x_true)

@printf("\nPermutation invariance: ΔLL = %.3e (should be ~0)\n", abs(ll0 - ll1))

# ## Multi-Start EM: Label Switching in Practice
# Because the likelihood has multiple symmetric optima (one per permutation of labels), EM can converge to any of these
# equivalent solutions depending on initialization. Running EM from several random starts reveals this “label switching”.

function random_init_model(Y::AbstractMatrix; K::Int, D::Int, rng)
    A0 = zeros(Float64, K, K)
    for i in 1:K
        v = rand(rng, Dirichlet(ones(K)))
        A0[i, :] .= v ./ sum(v)
    end
    π0 = rand(rng, Dirichlet(ones(K)))
    μ0s = rand(mean(Y) .+ (-1.0:0.1:1.0), K)  # spread out over data range
    Σ0s = [fill(0.5, D, D) for _ in 1:K]
    B0  = [GaussianEmission(D, [μ0s[k]], Σ0s[k]) for k in 1:K]

    HiddenMarkovModel(A0, B0, π0, K)
end

M = 5  # number of independent starts
fits = Vector{HiddenMarkovModel}(undef, M)
lls  = zeros(Float64, M)

for m in 1:M
    m0 = random_init_model(x_true; K=K, D=D, rng=StableRNG(10_000 + m))
    hist = fit!(m0, x_true; max_iters=100, tol=1e-6)
    fits[m] = m0
    lls[m]  = last(hist)
end

fitted_means = [[fits[m].B[k].μ[1] for k in 1:K] for m in 1:M]
print("\nFitted emission means per run (unordered):\n")
for row in fitted_means
    @printf("  %s\n", string(round.(row, digits=3)))
end

# ## Post-hoc Alignment: Best Permutation to a Reference
# To compare runs or build averaged summaries, we need a consistent label convention. A simple approach is to choose a target
# ordering (e.g., increasing emission mean) and, for each fitted model, find the permutation that best matches that ordering
# using a cost (e.g., squared error in means). Below we brute‑force all permutations (fine for small `K`) and permute models.

function all_permutations(K::Int)
    K == 1 && return [[1]]
    out = Vector{Vector{Int}}()
    for p in all_permutations(K-1)
        for i in 0:(K-1)
            q = copy(p)
            insert!(q, i+1, K)
            push!(out, q)
        end
    end
    return out
end

# Match a model to a target ordering by minimizing squared error of emission means (D=1)
function best_perm_by_means(model::HiddenMarkovModel, μ_target::Vector{Float64})
    perms = all_permutations(length(μ_target))
    best_i, best_cost = 1, Inf
    for (i, p) in enumerate(perms)
        s = 0.0
        for k in 1:length(p)
            μk = model.B[p[k]].μ[1]
            s += (μk - μ_target[k])^2
        end
        if s < best_cost
            best_cost = s; best_i = i
        end
    end
    return perms[best_i]
end

μ_target = sort(μ_true)               # choose an ordering convention (ascending mean)
aligned  = deepcopy(fits)
for m in 1:M
    p = best_perm_by_means(aligned[m], μ_target)
    permute_states!(aligned[m], p)
end

aligned_means = [[aligned[m].B[k].μ[1] for k in 1:K] for m in 1:M]
print("\nAligned emission means per run (ascending):\n")
for row in aligned_means
    @printf("  %s\n", string(round.(row, digits=3)))
end

# ## What *is* Identifiable?
# - **Transition structure up to permutation.** Eigenvalues of `A` and invariants like the stationary distribution are
#   identifiable, but the rows/columns correspond to states whose labels are arbitrary.
# - **Emission parameters up to permutation.** Means/variances (or their analogues) are identifiable modulo label permutations;
#   if components overlap heavily, practical identifiability degrades.
# - **State sequence itself is not identifiable** without constraints: we interpret distributions over sequences (e.g.,
#   smoothed posteriors), and these too are label‑ambiguous.

# ## How is this different from non-convexity?
# **Label symmetry (non‑identifiability)** and **non‑convexity** can both yield multiple optima, but for different reasons
# and with different remedies.
#
# - **Label symmetry (non‑identifiability).** The objective is *exactly invariant* under a finite group of permutations
#   (K! relabelings). All these optima are **mathematically equivalent**—they represent the same distribution after permuting
#   state labels. You can move between them by explicitly permuting parameters (A, π, emissions).
#   - **Symptoms:** Equal (up to numerical noise) log‑likelihoods; parameters across runs are permutations of each other;
#     posteriors match after relabeling.
#   - **Handling:** Choose a **canonicalization** (e.g., sort states by a statistic) or use **identifying priors/constraints**;
#     always **align models post‑hoc** before comparison.
#
# - **Non‑convexity (true multimodality).** The objective has *distinct* local maxima not related by symmetry. Fits may
#   differ in likelihood and yield different predictions even **after** optimal alignment.
#   - **Symptoms:** Unequal log‑likelihoods across runs; differences persist after trying all label permutations; predictive
#     metrics (e.g., held‑out log‑lik) differ.
#   - **Handling:** Use **multiple restarts**, better initializations, **continuation/annealing**, **regularization**, longer
#     EM runs or alternative optimizers, and **cross‑validation**.
#
# **Both can co‑exist.** You may have several *families* of solutions: each family contains K! symmetric copies (label
# switching), and there can be several **distinct** families due to non‑convexity. Our alignment step removes the symmetry
# so you can then compare **true** local optima.
#
# **Practical checklist**
# 1. Align by permutations → do likelihoods/predictions match? If yes, it was **label symmetry**.
# 2. If not, you’re seeing **non‑convexity**; compare held‑out performance and pick the best (or ensemble, if appropriate).

# ## Diagnostics & Practical Tips
# - **Check invariance.** Recompute log‑likelihood after permuting fitted models—differences should be ~0.
# - **Use multiple random starts.** If runs converge to permuted solutions with similar likelihoods, that’s expected; if you
#   see materially different likelihoods, you may have local maxima worth investigating.
# - **Align before comparing.** Always align models (or posterior state marginals) before averaging, plotting, or computing
#   distances.
# - **Stability tricks.** Increase `T`, separate emissions more, or regularize covariances to improve practical identifiability.

# ## Reproducibility
# All random draws are made with fixed seeds via `StableRNGs`, so your numbers should match the tutorial outputs. If you
# change `K`, dimensionality, or the emission family, expect small edits to the alignment cost and canonicalization rules.
