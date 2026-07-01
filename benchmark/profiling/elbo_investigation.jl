# Reproduce the symptom: BTD ELBO never early-stops while a true
# marginal log-likelihood does. Fit a small Gaussian LDS for K iters,
# record the ELBO at each iter, AND independently compute the marginal
# log-likelihood at the same iterate. Compare.

using StateSpaceDynamics
using LinearAlgebra, Random

const D, p, T_steps, N = 3, 5, 100, 4
const MAX_EM = 50

function make_lds(rng)
    A = StateSpaceDynamics.random_rotation_matrix(D, rng)
    Q = (M = randn(rng, D, D); M * M' + 1e-3 * I)
    x0 = randn(rng, D)
    P0 = (M = randn(rng, D, D); M * M' + 1e-3 * I)
    C = randn(rng, p, D)
    R = (M = randn(rng, p, p); M * M' + 1e-3 * I)
    b = randn(rng, D)
    d = randn(rng, p)
    sm = GaussianStateModel(; A=Matrix(A), Q=Matrix(Q), b=b, x0=x0, P0=Matrix(P0))
    om = GaussianObservationModel(; C=C, R=Matrix(R), d=d)
    return LinearDynamicalSystem(sm, om)
end

rng = MersenneTwister(123)
lds_true = make_lds(rng)
_, y = StateSpaceDynamics.rand(rng, lds_true, fill(T_steps, N))

# Use a perturbed starting point so we see a real EM trajectory.
lds_fit = make_lds(MersenneTwister(999))

# Manual EM loop so we can record the marginal LL after each iter too.
# The package's fit! doesn't expose this hook, so we do the components
# by hand using the same suf-based pipeline.
elbos = Float64[]
margs = Float64[]

tsteps_per_trial = fill(T_steps, N)
tfs = StateSpaceDynamics.initialize_FilterSmooth(lds_fit, tsteps_per_trial)
T_max = T_steps
sws_pool = [
    StateSpaceDynamics.SmoothWorkspace(Float64, D, p, T_max; ux_dim=0, uy_dim=0) for
    _ in 1:Threads.maxthreadid()
]
suf = StateSpaceDynamics._initialize_td_sufficient_statistics(
    Float64, lds_fit, tsteps_per_trial
)
ux_seq = [zeros(Float64, 0, T_steps) for _ in 1:N]
uy_seq = [zeros(Float64, 0, T_steps) for _ in 1:N]
StateSpaceDynamics._td_init_const_blocks!(
    sws_pool[1], lds_fit, tsteps_per_trial, y, ux_seq, uy_seq
)

for iter in 1:MAX_EM
    # E-step
    StateSpaceDynamics.smooth!(lds_fit, tfs, y, sws_pool, ux_seq, uy_seq)
    StateSpaceDynamics._aggregate_td_suff_stats!(
        suf, tfs, lds_fit, ux_seq, uy_seq, y, sws_pool[1]
    )

    # ELBO via the suf-based pipeline (same path `fit!` uses)
    total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths)
    elbo = StateSpaceDynamics.elbo!(lds_fit, suf, sws_pool[1], total_entropy)
    push!(elbos, elbo)

    # Marginal log-likelihood via Kalman filter (exact for Gaussian LDS)
    marg = StateSpaceDynamics.loglikelihood(lds_fit, y)
    push!(margs, marg)

    # M-step
    StateSpaceDynamics.mstep!(lds_fit, suf, sws_pool[1])
end

println("\nELBO and marginal log-likelihood trajectory")
println("(They should be identical for a Gaussian LDS at the exact posterior,")
println(" up to a constant. ELBO trajectory drives EM; marginal LL is the")
println(" `true` likelihood we'd report.)\n")
println(
    rpad("iter", 6),
    rpad("ELBO", 22),
    rpad("marg_LL", 22),
    rpad("ΔELBO", 22),
    rpad("Δmarg_LL", 22),
    rpad("ELBO - marg_LL", 22),
)
for k in eachindex(elbos)
    e = elbos[k]
    m = margs[k]
    de = k == 1 ? NaN : e - elbos[k - 1]
    dm = k == 1 ? NaN : m - margs[k - 1]
    println(
        rpad(k, 6),
        rpad(string(round(e; digits=8)), 22),
        rpad(string(round(m; digits=8)), 22),
        rpad(string(round(de; digits=10)), 22),
        rpad(string(round(dm; digits=10)), 22),
        rpad(string(round(e - m; digits=6)), 22),
    )
end

# Diagnostic — what's the expected offset?
obs_n = N * T_steps
n_active = N * T_steps * D
# Q_state in elbo! is missing -0.5 * (N + dyn_n) * D * log(2π).
# Entropy contributes +0.5 * n_active * log(2π).
# Q_obs contributes -0.5 * obs_n * p * log(2π) — present.
# True marg_LL has -0.5 * obs_n * p * log(2π) only (no latent 2π's since x is
# integrated out).
N_total = N
dyn_n = N * (T_steps - 1)
state_2pi_missing = -0.5 * (N_total + dyn_n) * D * log(2π)
entropy_2pi_extra = 0.5 * n_active * log(2π)
println("\nDiagnostic constants (per Gaussian-LDS ELBO assembly):")
println("  -0.5*(N+dyn_n)*D*log(2π)  (missing from Q_state) = ", state_2pi_missing)
println("  +0.5*n_active*log(2π)     (entropy normalizer)   = ", entropy_2pi_extra)
println(
    "  Sum of the two terms                              = ",
    state_2pi_missing + entropy_2pi_extra,
)
println("  Observed (ELBO - marg_LL) at last iter            = ", elbos[end] - margs[end])
