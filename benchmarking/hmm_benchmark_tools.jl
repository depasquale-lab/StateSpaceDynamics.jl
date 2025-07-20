export SSD_HMMImplem, HiddenMarkovModels_Implem

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

    initial_dists = [Normal(μ[i], Σ[i]) for i in 1:num_states]

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

