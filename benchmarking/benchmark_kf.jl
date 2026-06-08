using Pkg
Pkg.activate("benchmarking")

# Load the in-tree source rather than the installed package so that the
# `kalman_filter` branch's new Kalman/RTS E-step is exercised.
include(joinpath(@__DIR__, "..", "src", "StateSpaceDynamics.jl"))
using .StateSpaceDynamics
const SSD = StateSpaceDynamics

using BenchmarkTools
using StableRNGs
using Random
using LinearAlgebra
using Printf
using Statistics
using DataFrames
using CSV
using Plots
using Measures

BLAS.set_num_threads(1)
set_zero_subnormals(true);

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

struct BenchConfig
    latent_dims::Vector{Int}
    obs_dims::Vector{Int}
    seq_length::Int
    n_iters::Int
    n_repeats::Int
end

kf_config = BenchConfig(
    [4, 8, 16],   # latent_dims
    [4, 8],      # obs_dims
    100,         # seq_length
    10,          # n_iters (EM iterations per fit)
    100,         # n_repeats (benchmark samples)
)

const NUM_TRIALS = 100

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 60.0

# ----------------------------------------------------------------------
# Parameter + model helpers (local — not dependent on benchmarking/Params.jl,
# which `using StateSpaceDynamics` would import the installed package)
# ----------------------------------------------------------------------

struct LDSParams{T<:Real}
    A::Matrix{T}
    Q::Matrix{T}
    x0::Vector{T}
    P0::Matrix{T}
    C::Matrix{T}
    R::Matrix{T}
    b::Vector{T}
    d::Vector{T}
end

function init_params(rng::AbstractRNG, latent_dim::Int, obs_dim::Int)
    A = 0.99 * SSD.random_rotation_matrix(latent_dim, rng)

    Q = randn(rng, latent_dim, latent_dim)
    Q = Q * Q' + I

    x0 = randn(rng, latent_dim)
    P0 = randn(rng, latent_dim, latent_dim)
    P0 = P0 * P0' + I

    C = randn(rng, obs_dim, latent_dim)
    R = randn(rng, obs_dim, obs_dim)
    R = R * R' + I

    b = randn(rng, latent_dim)
    d = randn(rng, obs_dim)

    return LDSParams(A, Q, x0, P0, C, R, b, d)
end

function build_lds(p::LDSParams, kalman::Bool)
    state_model = SSD.GaussianStateModel(; A=p.A, Q=p.Q, x0=p.x0, P0=p.P0, b=p.b)
    obs_model = SSD.GaussianObservationModel(; C=p.C, R=p.R, d=p.d)
    return SSD.LinearDynamicalSystem(
        state_model, obs_model;
        kalman_filter=kalman,
    )
end


# ----------------------------------------------------------------------
# Benchmark loop
# ----------------------------------------------------------------------

eig_dims = minimum(kf_config.latent_dims)


methods = [(:tridiag, false), (:kalman, true)]

results = DataFrame(
    method          = Symbol[],
    latent_dim      = Int[],
    obs_dim         = Int[],
    seq_len         = Int[],
    num_trials      = Int[],
    n_iters         = Int[],
    time_sec        = Float64[],
    memory          = Int[],
    allocs          = Int[],
    test_ll_per_obs = Float64[],
    elbo            = Float64[],    
)

for latent_dim in kf_config.latent_dims

    for obs_dim in kf_config.obs_dims
        # obs_dim < latent_dim && continue

        println("\n→ latent_dim=$latent_dim, obs_dim=$obs_dim, seq_len=$(kf_config.seq_length), num_trials=$NUM_TRIALS")

        rng = StableRNG(1234)
        params = init_params(rng, latent_dim, obs_dim)

        # Both methods fit the same training data and are evaluated on the same
        # held-out test set drawn from the true generative model.
        ref = build_lds(params, false)

        tsteps=[kf_config.seq_length for _ in 1:NUM_TRIALS]
        _, y      = rand(rng, ref, tsteps)
        _, y_test = rand(StableRNG(5678), ref, tsteps)

        # Normalization constant for per-observation log-likelihood
        n_obs = kf_config.seq_length * NUM_TRIALS

        # random initial parameters for fitting (same for both methods)
        params0 = init_params(rng, latent_dim, obs_dim)

        # print the first few eigenvalues of the true A matrix for reference
        true_A_eig = eigen(params.A).values[1:eig_dims]
        @printf("  True A eigenvalues (first %d): %s\n", eig_dims, string(round.(true_A_eig, sigdigits=4)))

        for (name, kf) in methods
            print("  $(rpad(string(name), 8)) ")

            model = build_lds(params0, kf)

            # Warm up / precompile
            SSD.fit!(deepcopy(model), y; max_iter=10, tol=1e-6, progress=false)

            max_iter = kf_config.n_iters
            bench = @benchmark SSD.fit!(m, $y;
                                        max_iter=$max_iter,
                                        tol=1e-6,
                                        progress=false) setup=(m = deepcopy($model)) samples=kf_config.n_repeats evals=1

            med_sec = median(bench).time / 1e9

          

            # Fit once outside the benchmark to get final parameters for evaluation.
            fitted = deepcopy(model)
            elbo = SSD.fit!(fitted, y; max_iter=max_iter, tol=1e-6, progress=false)
            ll_per_obs = SSD.loglikelihood(fitted, y_test) / n_obs

            push!(results, (
                name, latent_dim, obs_dim,
                kf_config.seq_length, NUM_TRIALS, max_iter,
                med_sec, bench.memory, bench.allocs,
                ll_per_obs, elbo[end],
            ))
            @printf("median = %.6f sec || elbo = %.6f | test_ll/obs = %.6f || alloc = %d | mem = %.2f MB\n", med_sec, elbo[end], ll_per_obs, bench.allocs, bench.memory/1e6)

            # print the first min(latent_dim) eigenvalues of the fitted A matrix
            A_eig = eigen(fitted.state_model.A).values[1:eig_dims]
            @printf("  A eigenvalues (first %d): %s\n", eig_dims, string(round.(A_eig, sigdigits=4)))

        end
    end
end

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------

results_dir = joinpath(@__DIR__, "results")
isdir(results_dir) || mkpath(results_dir)

csv_path = joinpath(results_dir, "kf_benchmark_results_ntrials$(NUM_TRIALS)_seqlen$(kf_config.seq_length).csv")
CSV.write(csv_path, results)
println("\nWrote results → $csv_path")

# ----------------------------------------------------------------------
# Plot: 2-row figure
#   Row 1 — median fit time vs latent_dim  (one subplot per obs_dim)
#   Row 2 — test log-likelihood/obs vs latent_dim
# ----------------------------------------------------------------------

method_colors = Dict(:tridiag => "#E69F00", :kalman => "#0072B2")
method_markers = Dict(:tridiag => :diamond, :kalman => :circle)
method_labels  = Dict(:tridiag => "Tridiagonal", :kalman => "Kalman/RTS")

gb_time = groupby(results, [:method, :latent_dim, :obs_dim])
agg     = combine(gb_time,
                  :time_sec        => median => :median_time,
                  :test_ll_per_obs => median => :median_ll)

ovals = sort(unique(agg.obs_dim))
lvals = sort(unique(agg.latent_dim))
mvals = [:tridiag, :kalman]

t_min, t_max = extrema(agg.median_time)
ll_min, ll_max = extrema(agg.median_ll)

subplots = Plots.Plot[]

# ---- Row 1: timing ----
for (j, o) in enumerate(ovals)
    panel = @view agg[agg.obs_dim .== o, :]
    p = plot(
        title     = "obs_dim = $o",
        xlabel    = "",
        ylabel    = (j == 1) ? "Fit time (s)" : "",
        framestyle = :box,
        grid      = :y,
        minorgrid = true,
        legend    = false,
        xticks    = lvals,
        ylims     = (t_min * 0.9, t_max * 1.1),
    )
    for m in mvals
        sub = sort(@view(panel[panel.method .== m, :]), :latent_dim)
        isempty(sub) && continue
        plot!(p, sub.latent_dim, sub.median_time;
              label     = method_labels[m],
              color     = method_colors[m],
              marker    = method_markers[m],
              linewidth = 2, markersize = 6)
    end
    push!(subplots, p)
end

# ---- Row 2: test log-likelihood / obs ----
for (j, o) in enumerate(ovals)
    panel = @view agg[agg.obs_dim .== o, :]
    p = plot(
        xlabel    = "latent_dim",
        ylabel    = (j == 1) ? "Test ll / obs" : "",
        framestyle = :box,
        grid      = :y,
        minorgrid = true,
        legend    = false,
        xticks    = lvals,
        ylims     = (ll_min * 1.05, ll_max * 0.95),
    )
    for m in mvals
        sub = sort(@view(panel[panel.method .== m, :]), :latent_dim)
        isempty(sub) && continue
        plot!(p, sub.latent_dim, sub.median_ll;
              label     = method_labels[m],
              color     = method_colors[m],
              marker    = method_markers[m],
              linewidth = 2, markersize = 6)
    end
    push!(subplots, p)
end

# Assemble: 2 rows × length(ovals) columns, then legend panel on right
grid_plot = plot(subplots...;
                 layout       = (2, length(ovals)),
                 size         = (350 * length(ovals), 560),
                 left_margin  = 12mm,
                 bottom_margin = 8mm)

legend_plot = plot(framestyle=:none, showaxis=false, size=(160, 560))
for m in mvals
    plot!(legend_plot, [NaN], [NaN];
          label        = method_labels[m],
          linewidth    = 2,
          marker       = method_markers[m],
          seriescolor  = method_colors[m])
end

final_plt = plot(grid_plot, legend_plot;
                 layout      = @layout([a{0.85w} b{0.15w}]),
                 plot_title  = "Tridiagonal vs Kalman/RTS  (seq_len=$(kf_config.seq_length), trials=$NUM_TRIALS, iters=$(kf_config.n_iters))")
plot!(final_plt; tickfontsize=10, guidefontsize=12, legendfontsize=11)

png_path = joinpath(results_dir, "kf_benchmark_ntrials$(NUM_TRIALS)_seqlen$(kf_config.seq_length).png")
savefig(final_plt, png_path)
println("Wrote plot → $png_path")
