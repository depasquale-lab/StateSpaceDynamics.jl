using StateSpaceDynamics
export HMMParams, init_params, build_data, LDSParams

@kwdef struct HMMParams{T<:Real, M<:AbstractMatrix{T}}
    πₖ::Vector{T}
    A::M
    μ::Vector{Vector{T}}
    Σ::Vector{Matrix{T}}
end

@kwdef struct LDSParams{T<:Real, V<:AbstractVector{T}, M<:AbstractMatrix{T}}
    A::M
    Q::M
    x0::V
    P0::M
    C::M
    R::M
end

function init_params(rng::AbstractRNG, instance::HMMInstance)
    (; num_states, emission_dim) = instance

    # Initialize state distribution and transition matrix
    πₖ = StateSpaceDynamics.initialize_state_distribution(num_states)
    A = StateSpaceDynamics.initialize_transition_matrix(num_states)

    # Initialize Gaussian params (μ) and covariances (Σ)
    μ = [randn(rng, emission_dim) for _ in 1:num_states]
    Σ = Vector{Matrix{Float64}}(undef, num_states)
    for i in 1:num_states
        Σ[i] = randn(rng, emission_dim, emission_dim)
        Σ[i] = Σ[i] * Σ[i]' .+ 1e-3  # Ensure positive definiteness
    end

    return HMMParams(πₖ=πₖ, A=A, μ=μ, Σ=Σ)
end

function build_data(rng::AbstractRNG, model::HiddenMarkovModel, instance::HMMInstance)
    return [vec(rand(rng, model; n=instance.seq_length)[2]) for _ in 1:instance.num_trials]
end

function init_params(rng::AbstractRNG, instance::LDSInstance)
    (; latent_dim, obs_dim, num_trials, seq_length) = instance
    
    # Initialize state transition matrix (A), process noise covariance (Q)
    A = random_rotation_matrix(latent_dim, rng)

    Q = randn(rng, latent_dim, latent_dim)
    Q = Q * Q' .+ 1e-3

    x0 = randn(rng, latent_dim)
    P0 = randn(rng, latent_dim, latent_dim)
    P0 = P0 * P0'  .+ 1e-3

    C = randn(rng, obs_dim, latent_dim)
    R = randn(rng, obs_dim, obs_dim)
    R = R * R'  .+ 1e-3

    return LDSParams(;A=A, Q=Q, x0=x0, P0=P0, C=C, R=R)
end

function build_data(rng::AbstractRNG, model::LinearDynamicalSystem, instance::LDSInstance)
    (; latent_dim, obs_dim, num_trials, seq_length) = instance

    latents, observations = rand(rng, model; ntrials=num_trials, tsteps=seq_length)
    return latents, observations
end