import HiddenMarkovModels as HMMs
using PythonCall

export SSD_GLMHMM_Implem, HMM_GLMHMM_Implem, DYNAMAX_GLMHMM_Implem, build_model, run_benchmark

"""
StateSpaceDynamics.jl implementation functions
"""

struct SSD_GLMHMM_Implem <: Implementation end

function build_model(::SSD_GLMHMM_Implem, instance::HMMInstance, params::HMMParams)
    # Read in instance and params
    (; num_states, num_trials, seq_length, input_dim, output_dim) = instance
    (; πₖ, A, β) = params

    # Build the Bernoulli regression emission models
    B = Vector{StateSpaceDynamics.BernoulliRegressionEmission}(undef, num_states)

    for i=1:size(B,1)
        B[i] = BernoulliRegressionEmission(; input_dim=input_dim, output_dim=output_dim, include_intercept=false, β=β[i], λ=0.0)
    end

    # Build the HMM
    return StateSpaceDynamics.HiddenMarkovModel(K=num_states, A=A, πₖ=πₖ, B=B)
end

function run_benchmark(::SSD_GLMHMM_Implem, model::StateSpaceDynamics.HiddenMarkovModel, X::AbstractVector{<:AbstractMatrix}, Y::AbstractVector{<:AbstractMatrix})

    # Run 1 EM iteration to compile
    ll = StateSpaceDynamics.fit!(model, Y, X; max_iters=1)

    # Run the benchmarking
    bench = @benchmark begin
        model_bench = deepcopy($model)
        StateSpaceDynamics.fit!(model_bench, $Y, $X; max_iters=100, tol=1e-100)
    end samples=5
        
    return (time=median(bench).time, memory=bench.memory, allocs=bench.allocs, success=true)
end

"""
HiddenMarkovModels.jl implementation functions
"""
struct HMM_GLMHMM_Implem <: Implementation end

function build_model(::HMM_GLMHMM_Implem, instance::HMMInstance, params::HMMParams)
    (; num_states, input_dim) = instance
    (; πₖ, A, β) = params

    dist_coeffs = [vec(β[i]) for i in 1:num_states]  # ensure flat vectors

    return ControlledBernoulliHMM(πₖ, A, dist_coeffs)
end

function run_benchmark(::HMM_GLMHMM_Implem, model::ControlledBernoulliHMM, X::AbstractVector, Y::AbstractVector)

    # Convert data format from SSD.jl compatible to HMM.jl compatible.
    # Synthetic data is generated from SSD.jl
    obs_seq, control_seq, seq_ends = format_glmhmm_data(X, Y)

    # Compile step
    _, _ = HMMs.baum_welch(model, obs_seq, control_seq;seq_ends=seq_ends, max_iterations=1)

    # Run the benchmarking
    bench = @benchmark begin
        model_bench = deepcopy($model)
        HMMs.baum_welch($model, $obs_seq, $control_seq;seq_ends=$seq_ends, max_iterations=100, atol=1e-30)
    end samples=5
        
    return (time=median(bench).time, memory=bench.memory, allocs=bench.allocs, success=true)

end

"""
Dynamax implementation functions
"""

const np = pyimport("numpy")
const dynamax = pyimport("dynamax.hidden_markov_model")
const jr = pyimport("jax.random")
const jnp = pyimport("jax.numpy")

struct DYNAMAX_GLMHMM_Implem <: Implementation end

function build_model(::DYNAMAX_GLMHMM_Implem, instance::HMMInstance, params::HMMParams)
    (; num_states, input_dim) = instance
    (; πₖ, A, β) = params

    # Convert Julia parameters to NumPy arrays
    pi_jax = jnp.array(πₖ)
    A_jax = jnp.array(A)

    # Single output dimension models only
    β_mat = hcat(β...)'
    B_jax = jnp.array(β_mat)

    # Create the Dynamax model
    dynamax_model = dynamax.LogisticRegressionHMM(
    num_states=num_states,
    input_dim=input_dim
    )

    # Initialize the Dynamax model
    dparams, dprops = dynamax_model.initialize(
        method="prior",
        initial_probs=pi_jax,
        transition_matrix=A_jax,
        emission_weights=B_jax)


    return dynamax_model, dparams, dprops
end


function run_benchmark(::DYNAMAX_GLMHMM_Implem, model::PythonCall.Core.Py, dparams::PythonCall.Core.Py, dprops::PythonCall.Core.Py, X::AbstractVector, Y::AbstractVector)
    
    X_jax = jnp.stack([jnp.array(permutedims(x, (2,1))) for x in X])
    Y_jax = jnp.squeeze(jnp.stack([jnp.array(permutedims(y, (2,1))) for y in Y]))

    # Compile evrything
    model.fit_em(dparams, dprops, Y_jax, inputs=X_jax, num_iters=1, verbose=false)

    # Run the benchmarking
    bench = @benchmark begin
        model_bench = deepcopy($model)
        model_bench.fit_em($dparams, $dprops, $Y_jax, inputs=$X_jax, num_iters=100, verbose=false)
    end samples=5

    return (time=median(bench).time, memory=bench.memory, allocs=bench.allocs, success=true)
end
