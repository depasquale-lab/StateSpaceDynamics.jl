using Pkg
Pkg.activate("benchmarking")

include("SSD_Benchmark.jl")
using .SSD_Benchmark
using BenchmarkTools
using StableRNGs
using Printf
using CSV
using DataFrames

# Benchmark configuration
struct BenchConfig
    latent_dims::Vector{Int}
    obs_dims::Vector{Int}
    seq_lengths::Vector{Int}
    n_iters::Int
    n_repeats::Int
end

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0


instance = HMMInstance(; num_states=2, emission_dim=1, num_trials=1, seq_length=100)
rng = StableRNG(1234)
params = init_params(rng, instance)

ssd_model = build_model(SSD_HMMImplem(), instance, params)
y = build_data(rng, ssd_model, instance)

println("building model")
model = build_model(Dynamax_HMMImplem(), instance, params)

println("running benchmark")
result = run_benchmark(Dynamax_HMMImplem(), model, y[1])