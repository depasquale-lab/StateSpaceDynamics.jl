using Pkg
Pkg.activate("benchmarking")

include("SSD_Benchmark.jl")
using .SSD_Benchmark
using BenchmarkTools
using StableRNGs
using Printf
using CSV
using DataFrames
using PythonCall


dynamax = pyimport("dynamax")
np = pyimport("numpy")
jax = pyimport("jax")
jnp = pyimport("jax.numpy")
dynamax_hmm = pyimport("dynamax.hidden_markov_model")

"""
Minimal Dynamax working example
"""

num_states = 2
# variables in julia
A = [0.9 0.1; 0.2 0.8]
p = [0.5, 0.5]
u = [[0.0], [1.0]]
Σ = Vector{Matrix{Float64}}(undef, num_states)

for i in 1:num_states
    Σ[i] = randn(1, 1)
    Σ[i] = Σ[i] * Σ[i]' .+ 1e-3  # Ensure positive definiteness
end

# convert variables to python
A_jax = jnp.array(Py(A).to_numpy())
p_jax = jnp.array(Py(p).to_numpy())
u_jax = jnp.stack([jnp.array(Py(ui).to_numpy()) for ui in u])
sig_jax = jnp.stack([jnp.array(Σi) for Σi in Σ])


# Create and initialize the model
hmm = dynamax_hmm.GaussianHMM(num_states=2, emission_dim=1)

params, props = hmm.initialize(
    initial_probs=p_jax,
    transition_matrix=A_jax,
    emission_means=u_jax,
    emission_covariances=sig_jax
)

data = jnp.array(np.random.rand(100,1))

em_params, log_probs = hmm.fit_em(params,
                                  props,
                                  data,
                                  num_iters=100)

"""
Reproduction of SSD benchmarking example
"""
instance = SSD_Benchmark.HMMInstance(; num_states=2, emission_dim=1, num_trials=1, seq_length=100)
rng = StableRNG(1234)
params = SSD_Benchmark.init_params(rng, instance)

ssd_model = SSD_Benchmark.build_model(SSD_Benchmark.SSD_HMMImplem(), instance, params)
y = SSD_Benchmark.build_data(rng, ssd_model, instance)

model, dparams, dprops = SSD_Benchmark.build_model(SSD_Benchmark.Dynamax_HMMImplem(), instance, params)


# em_params, log_probs = model.fit_em(dparams,
#                                   dprops,
#                                   dat_jax,
#                                   num_iters=100)


println("running outside")
dat_jax = jnp.array(np.expand_dims(Py(y[1]).to_numpy(), axis=1))
params, lps = model.fit_em(dparams, dprops, dat_jax, num_iters=100)





result = SSD_Benchmark.run_benchmark(SSD_Benchmark.Dynamax_HMMImplem(), model, dprops, dparams, y[1])