include("SSD_Benchmark.jl")
using .SSD_Benchmark
using BenchmarkTools
using StableRNGs
using Printf
using CSV

struct BenchConfig
    latent_dims::Vector{Int}
    obs_dims::Vector{Int} 
    seq_lengths::Vector{Int}
    n_iters::Int
    n_repeats::Int
end

# Optional: reduce GC effects in timing
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

# Define benchmarking configurations
config = BenchConfig(
    [2, 4, 6, 8],
    [2, 4, 6, 8],
    [100, 500, 1000],
    100,
    5
)

# Define which implementations to run
implementations = [
    SSD_LDSImplem(),
    pykalman_LDSImplem(),
    Dynamax_LDSImplem()
]

# Results will be stored here
all_results = []

# Loop over configurations
for latent_dim in config.latent_dims
    for obs_dim in config.obs_dims
        obs_dim < latent_dim && continue
        for seq_len in config.seq_lengths
            println("\n→ Benchmarking LDS with latent_dim=$latent_dim, obs_dim=$obs_dim, seq_len=$seq_len")

            # Construct instance & params
            instance = LDSInstance(; latent_dim=latent_dim, obs_dim=obs_dim, num_trials=1, seq_length=seq_len)
            rng = StableRNG(1234)
            params = init_params(rng, instance)

            results_row = Dict("config" => (latent_dim=latent_dim, obs_dim=obs_dim, seq_len=seq_len))

            for impl in implementations
                print("  Running $(string(impl))... ")

                try
                    model = build_model(impl, instance, params)
                    _, y = build_data(rng, model, instance)

                    result = run_benchmark(impl, model, nothing, y)
                    results_row[string(impl)] = result

                    if result.success
                        @printf("✓ time = %.3f sec\n", result.time / 1e9)
                    else
                        println("✗ failed")
                    end
                catch e
                    results_row[string(impl)] = (time=NaN, memory=0, allocs=0, success=false)
                    println("✗ exception: ", e)
                end
            end

            push!(all_results, results_row)
            println("-"^50)
        end
    end
end

# write to CSV or show sumary
using DataFrames
df = DataFrame()
for row in all_results
    config = row["config"]
    for (name, result) in row
        name == "config" && continue
        push!(df, (
            implementation = name,
            latent_dim = config.latent_dim,
            obs_dim = config.obs_dim,
            seq_len = config.seq_len,
            time_sec = result.time / 1e9,
            memory = result.memory,
            allocs = result.allocs,
            success = result.success
        ))
    end
end


CSV.write("lds_benchmark_results.csv", df)

