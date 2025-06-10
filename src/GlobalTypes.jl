"""
The purpose of this file is to provide a common place for all global types to be defined. This is to avoid circular dependencies between files.
"""

export MixtureModel, EmissionModel, AbstractHMM, DynamicalSystem, AbstractStateModel, AbstractObservationModel

# Create abstract types here 
"""
Abstract type for Mixture Models. I.e. GMM's, etc.
"""

abstract type MixtureModel end

"""
Abstract type for Regression Models. I.e. GaussianRegression, BernoulliRegression, etc.
"""

abstract type RegressionModel end

"""
Abstract type for HMMs 
"""

abstract type AbstractHMM end

"""
Abstract type for Dynamical Systems. I.e. LDS, etc.
"""

abstract type DynamicalSystem end
abstract type AbstractStateModel{T<:Real} end
abstract type AbstractObservationModel{T<:Real} end

"""
Base type hierarchy for emission models.
Each emission model must implement:
- sample()
- loglikelihood()
- fit!()
"""

abstract type EmissionModel end
abstract type RegressionEmission <: EmissionModel end
abstract type AutoRegressiveEmission <: RegressionEmission end


"""
    ForwardBackward{T<:Real}

A mutable struct that encapsulates the forward–backward algorithm outputs for a hidden Markov model (HMM).

# Fields
- `loglikelihoods::Matrix{T}`: Matrix of log-likelihoods for each observation and state.
- `α::Matrix{T}`: The forward probabilities (α) for each time step and state.
- `β::Matrix{T}`: The backward probabilities (β) for each time step and state.
- `γ::Matrix{T}`: The state occupancy probabilities (γ) for each time step and state.
- `ξ::Array{T,3}`: The pairwise state occupancy probabilities (ξ) for consecutive time steps and state pairs.

Typically, `α` and `β` are computed by the forward–backward algorithm to find the likelihood of an observation sequence. `γ` and `ξ` are derived from these calculations to estimate how states transition over time.
"""
mutable struct ForwardBackward{T<:Real}
    loglikelihoods::Matrix{T}
    α::Matrix{T}
    β::Matrix{T}
    γ::Matrix{T}
    ξ::Array{T, 3}
end


""""
    FilterSmooth{T<:Real}

A mutable structure for storing smoothed estimates and associated covariance matrices in a filtering or smoothing algorithm.

# Type Parameters
- `T<:Real`: The numerical type used for all fields (e.g., `Float64`, `Float32`).

# Fields
- `x_smooth::Matrix{T}`  
  The matrix containing smoothed state estimates over time. Each column typically represents the state vector at a given time step.

- `p_smooth::Array{T, 3}`  
  The posterior covariance matrices with dimensions (latent_dim, latent_dim, time_steps)

- `E_z::Array{T, 3}`  
  The expected latent states, size (state_dim, T, n_trials).

- `E_zz::Array{T, 4}`  
  The expected value of z_t * z_t', size (state_dim, state_dim, T, n_trials).

- `E_zz_prev::Array{T, 4}`  
  The expected value of z_t * z_{t-1}', size (state_dim, state_dim, T, n_trials).

# Example
```julia
# Initialize a FilterSmooth object with Float64 type
filter = FilterSmooth{Float64}(
    x_smooth = zeros(10, 100),
    p_smooth = zeros(10, 10, 100),
    E_z = zeros(10, 5, 100),
    E_zz = zeros(10, 10, 5, 100),
    E_zz_prev = zeros(10, 10, 5, 100)
)
"""
mutable struct FilterSmooth{T<:Real}
    x_smooth::Matrix{T}
    p_smooth::Array{T, 3}
    E_z::Array{T, 3}
    E_zz::Array{T, 4}
    E_zz_prev::Array{T, 4}
end