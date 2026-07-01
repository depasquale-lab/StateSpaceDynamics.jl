using StateSpaceDynamics
export init_params, build_data, LDSParams

@kwdef struct LDSParams{T<:Real,V<:AbstractVector{T},M<:AbstractMatrix{T}}
    A::M
    Q::M
    x0::V
    P0::M
    C::M
    R::M
    b::V
    d::V
end

function init_params(rng::AbstractRNG, instance::LDSInstance)
    (; latent_dim, obs_dim) = instance

    # Initialize state transition matrix (A), process noise covariance (Q)
    A = random_rotation_matrix(latent_dim, rng)

    Q = randn(rng, latent_dim, latent_dim)
    Q = Q * Q' .+ 1e-3

    x0 = randn(rng, latent_dim)
    P0 = randn(rng, latent_dim, latent_dim)
    P0 = P0 * P0' .+ 1e-3

    C = randn(rng, obs_dim, latent_dim)
    R = randn(rng, obs_dim, obs_dim)
    R = R * R' .+ 1e-3

    b = randn(rng, latent_dim)
    d = randn(rng, obs_dim)

    return LDSParams(; A=A, Q=Q, x0=x0, P0=P0, C=C, R=R, b=b, d=d)
end

function build_data(rng::AbstractRNG, model::LinearDynamicalSystem, instance::LDSInstance)
    (; num_trials, seq_length) = instance

    latents, observations = rand(rng, model, fill(seq_length, num_trials))
    return latents, observations
end
