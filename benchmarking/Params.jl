using StateSpaceDynamics
export HMMParams, init_glmhmm_params, build_glmhmm_data

@kwdef struct HMMParams{T<:Real, V<:AbstractVector{<:AbstractMatrix{T}}, M<:AbstractMatrix{T}}
    πₖ::Vector{T}
    A::M
    β::V
end

function init_glmhmm_params(rng::AbstractRNG, instance::HMMInstance)
    (; num_states, input_dim, output_dim) = instance

    # Initialize state distribution and transition matrix
    πₖ = StateSpaceDynamics.initialize_state_distribution(num_states)
    A = StateSpaceDynamics.initialize_transition_matrix(num_states)

    # Initialize regression weights (β) and covariances (Σ)
    β = [randn(rng, input_dim, output_dim) for _ in 1:num_states]

    return HMMParams(πₖ=πₖ, A=A, β=β)
end

function build_glmhmm_data(rng::AbstractRNG, model::HiddenMarkovModel, instance::HMMInstance)
    (; num_states, num_trials, seq_length, input_dim, output_dim) = instance

    # Sample from the model
    all_data = Vector{Matrix{Float64}}()  # Store each data matrix
    Φ_total = Vector{Matrix{Float64}}()
    all_true_labels = []

    for i in 1:num_trials
        Φ = randn(input_dim, seq_length)
        true_labels, data = rand(rng, model, Φ, n=seq_length)
        push!(all_true_labels, true_labels)
        push!(all_data, data)
        push!(Φ_total, Φ)
    end

    return all_true_labels, Φ_total, all_data
end
