using StateSpaceDynamics
export HMMParams, init_params, build_data

@kwdef struct HMMParams{T<:Real, V<:AbstractVector{<:AbstractMatrix{T}}, M<:AbstractMatrix{T}}
    πₖ::Vector{T}
    A::M
    β::V
end

@kwdef struct LDSParams{T<:Real, V<:AbstractVector{<:AbstractMatrix{T}}, M<:AbstractMatrix{T}}
    A::M
    Q::M
    x0::V
    P0::M
    C::M
    R::M
end

function init_params(rng::AbstractRNG, instance::HMMInstance)
    (; num_states, input_dim, output_dim) = instance

    # Initialize state distribution and transition matrix
    πₖ = StateSpaceDynamics.initialize_state_distribution(num_states)
    A = StateSpaceDynamics.initialize_transition_matrix(num_states)

    # Initialize regression weights (β) and covariances (Σ)
    β = [randn(rng, input_dim, output_dim) for _ in 1:num_states]

    return HMMParams(πₖ=πₖ, A=A, β=β)
end

function build_data(rng::AbstractRNG, model::HiddenMarkovModel, instance::HMMInstance)
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

    obs_seq, control_seq, seq_ends = format_glmhmm_data(Φ_total, all_data)

    return all_true_labels, Φ_total, all_data, obs_seq, control_seq, seq_ends
end

function init_params(rng::AbstractRNG, instance::LDSInstance)
    (; latent_dim, obs_dim, num_trials, seq_length) = instance
    
    # Initialize state transition matrix (A), process noise covariance (Q)
    A = random_rotation_matrix(latent_dim, rng)

    Q = randn(rng, latent_dim, latent_dim)
    Q = Q * Q'  # Ensure positive semi-definite

    x0 = randn(rng, latent_dim)
    P0 = randn(rng, latent_dim, latent_dim)
    P0 = P0 * P0'  # Ensure positive semi-definite

    C = randn(rng, obs_dim, latent_dim)
    R = randn(rng, obs_dim, obs_dim)

    return LDSParams(;A=A, Q=Q, x0=x0, P0=P0, C=C, R=R)
end

function build_data(rng::AbstractRNG, model::LinearDynamicalSystem, instance::LDSInstance)
    (; latent_dim, obs_dim, num_trials, seq_length) = instance

    latents, observations = rand(rng, model; num_trials=num_trials, tsteps=seq_length)
    return latents, observations
end