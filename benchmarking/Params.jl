using StateSpaceDynamics
export Params, init_params, build_data

@kwdef struct Params{T<:Real, V<:AbstractVector{<:AbstractMatrix{T}}, M<:AbstractMatrix{T}}
    πₖ::Vector{T}
    A::M
    β::V
end

function init_params(rng::AbstractRNG, instance::Instance)
    (; num_states, input_dim, output_dim) = instance

    # Initialize state distribution and transition matrix
    πₖ = StateSpaceDynamics.initialize_state_distribution(num_states)
    A = StateSpaceDynamics.initialize_transition_matrix(num_states)

    # Initialize regression weights (β) and covariances (Σ)
    β = [randn(rng, output_dim, input_dim) for _ in 1:num_states]

    return Params(πₖ=πₖ, A=A, β=β)
end

function build_data(rng::AbstractRNG, model::HiddenMarkovModel, instance::Instance)
    (; num_states, num_trials, seq_length, input_dim, output_dim) = instance

    # Create lists to hold data and labels for each trial
    Φ_trials = [randn(input_dim, seq_length) for _ in 1:num_trials]
    true_labels_trials = Vector{Vector{Int}}(undef, num_trials)
    data_trials = Vector{Matrix{Float64}}(undef, num_trials)

    # Sample data for each trial
    for i in 1:num_trials
        true_labels_trials[i], data_trials[i] = rand(rng, model, Φ_trials[i]; n=seq_length)
    end

    return true_labels_trials, data_trials
end
