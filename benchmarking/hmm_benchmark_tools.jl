export SSD_HMMImplem, HiddenMarkovModels_Implem, Dynamax_HMMImplem, run_benchmark

struct SSD_HMMImplem <: Implementation end
Base.string(::SSD_HMMImplem) = "ssd_hmm"

function build_model(::SSD_HMMImplem, instance::HMMInstance, params::HMMParams)
    (; num_states, num_trials, seq_length, emission_dim) = instance
    (; πₖ, A, μ, Σ) = params

    # Create the model
    hmm = HiddenMarkovModel(
        A,
        [GaussianEmission(emission_dim, μ[i], Σ[i]) for i in 1:num_states],
        πₖ,
        num_states
    )

    return hmm
end

struct HiddenMarkovModels_Implem <: Implementation end
Base.string(::HiddenMarkovModels_Implem) = "HiddenMarkovModels.jl"

function build_model(::HiddenMarkovModels_Implem, instance::HMMInstance, params::HMMParams)
    (; num_states, num_trials, seq_length, emission_dim) = instance
    (; πₖ, A, μ, Σ) = params

    initial_dists = [MvNormal(μ[i], Σ[i]) for i in 1:num_states]

    # Create the model
    hmm = HiddenMarkovModels.HMM(
        πₖ,
        A,
        initial_dists
    )

    return hmm
end

struct Dynamax_HMMImplem <: Implementation end
Base.string(::Dynamax_HMMImplem) = "Dynamax.jl"

function build_model(::Dynamax_HMMImplem, instance::HMMInstance, params::HMMParams)
    (; num_states, num_trials, seq_length, emission_dim) = instance
    (; πₖ, A, μ, Σ) = params

    dynamax_hmm = pyimport("dynamax.hidden_markov_model")
    jnp = pyimport("jax.numpy")
    np = pyimport("numpy")

    # convert to Dynamax arrays
    π_jax = jnp.array(πₖ)
    A_jax = jnp.array(A)

    μ_jax = jnp.stack([np.array(μi) for μi in μ])
    Σ_jax = jnp.stack([np.array(Σi) for Σi in Σ])

    # Create the model
    hmm = dynamax_hmm.GaussianHMM(
        num_states,
        emission_dim
    )

    params, props = hmm.initialize(
        initial_probs=π_jax,
        transition_matrix=A_jax,
        emission_means=μ_jax,
        emission_covariances=Σ_jax
    )

    return (hmm, params, props)
end

function run_benchmark(::SSD_HMMImplem, model::HiddenMarkovModel, data::Vector{Float64})
    data_mat = reshape(data, 1, :)

    # Run 1 EM iteration to compile
    StateSpaceDynamics.fit!(deepcopy(model), data_mat; max_iters=1, tol=1e-50);

    bench = @benchmark begin
        StateSpaceDynamics.fit!(deepcopy($model), $data_mat; max_iters=100, tol=1e-50);
    end samples=5

    return (time=median(bench).time, memory=bench.memory, allocs=bench.allocs, success=true)
end

function run_benchmark(::HiddenMarkovModels_Implem, model::HiddenMarkovModels.HMM, data::Vector{Float64})
    data_vec_vec = [[data[i]] for i in 1:length(data)]

    # Run initial iteration to compile
    _, _ = baum_welch(deepcopy(model), data_vec_vec; max_iterations=1, atol=1e-50);

    bench = @benchmark begin
        hmm_est, lls = baum_welch(deepcopy($model), $data_vec_vec; max_iterations=100, atol=1e-50);
    end samples=5

    return (time=median(bench).time, memory=bench.memory, allocs=bench.allocs, success=true)
end

function run_benchmark(::Dynamax_HMMImplem, model::Tuple{Any, Any, Any}, data::Vector{Float64})
    dynamax = pyimport("dynamax")
    np = pyimport("numpy")
    jax = pyimport("jax")

    data = np.array(data)
    
    hmm = model[1]
    params = model[2]
    props = model[3]

    # JIT compile the function call
    fit_fn = jax.jit(hmm.fit_em)

    bench = @benchmark begin
        _, _ = $fit_fn(
            $params,
            $props,
            $data,
            num_iters=100
        )
    end samples=5
    return (time=median(bench).time, memory=bench.memory, allocs=bench.allocs, success=true)
end

