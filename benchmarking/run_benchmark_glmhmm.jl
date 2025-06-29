# Activate the benchmarking environment
using Pkg
cd("benchmarking")
Pkg.activate(".")

# Import the necessary packages
include("SSD_Benchmark.jl")
using .SSD_Benchmark
using StableRNGs
using StateSpaceDynamics
import HiddenMarkovModels as HMMs
using StableRNGs
using Printf
using DataFrames
using CSV

# Benchmark configuration
latent_dims = [2, 4, 6, 8]
input_dims = [2, 4, 6, 8]
obs_dims = [1]
seq_lengths = [100, 500, 1000]
num_trials = 10  # can increase if you want

# Implementations to benchmark
implementations = [
    SSD_GLMHMM_Implem(),
    HMM_GLMHMM_Implem(),
    DYNAMAX_GLMHMM_Implem()
]

all_results = []

for latent_dim in latent_dims
    for obs_dim in obs_dims
        for input_dim in input_dims
            for seq_len in seq_lengths
                println("\n→ Benchmarking GLMHMM with latent_dim=$latent_dim, intput_dim=$input_dim, obs_dim=$obs_dim, seq_len=$seq_len")

                # Build instance and RNG
                rng = StableRNG(1234)
                gen_instance = HMMInstance(
                    num_states=latent_dim,
                    num_trials=num_trials,
                    seq_length=seq_len,
                    input_dim=input_dim,
                    output_dim=obs_dim
                )

                gen_params = init_params(rng, gen_instance)
                gen_model = build_model(SSD_GLMHMM_Implem(), gen_instance, gen_params)
                labels, X, Y, obs_seq, control_seq, seq_ends = build_data(rng, gen_model, gen_instance)

                # Prepare results row
                results_row = Dict{String, Any}()
                results_row["config"] = (latent_dim=latent_dim, input_dim=input_dim, obs_dim=obs_dim, seq_len=seq_len)

                # Generate benchmarking init params
                instance_bench = HMMInstance(
                    num_states=latent_dim,
                    num_trials=num_trials,
                    seq_length=seq_len,
                    input_dim=input_dim,
                    output_dim=obs_dim
                )
                params_bench = init_params(rng, instance_bench)

                # Loop over implementations and run benchmarks
                for impl in implementations
                    print("  Running $(string(impl))... ")
                    try
                        if impl isa DYNAMAX_GLMHMM_Implem
                            model, dparams, dprops = build_model(impl, instance_bench, params_bench)
                            result = run_benchmark(impl, model, dparams, dprops, X, Y)
                        else
                            model = build_model(impl, instance_bench, params_bench)
                            result = run_benchmark(impl, model, X, Y)
                        end
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
end

# write to CSV or show sumary
df = DataFrame()
for row in all_results
    config = row["config"]
    for (name, result) in row
        if name == "config"
            continue
        end
        push!(df, (
            implementation = name,
            latent_dim = config.latent_dim,
            input_dim = config.input_dim,
            obs_dim = config.obs_dim,
            seq_len = config.seq_len,
            time_sec = result[:time] / 1e9,
            memory = result[:memory],
            allocs = result[:allocs],
            success = result[:success]
        ))
    end
end


CSV.write("glmhmm_benchmark_results.csv", df)