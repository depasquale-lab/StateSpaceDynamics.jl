"""
The purpose of this file is to provide a common place for all global types to be defined.
This is to avoid circular dependencies between files.
"""

export MixtureModel, EmissionModel, DynamicalSystem
export AbstractHMM, AbstractStateModel, AbstractObservationModel

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

"""
Base type hierarchy for regression emission models.
"""
abstract type RegressionEmission <: EmissionModel end

"""
Special case of regression emission models that are autoregressive.
"""
abstract type AutoRegressiveEmission <: RegressionEmission end

"""
    ForwardBackward{T<:Real}

A mutable struct that encapsulates the forward–backward algorithm outputs for a hidden
Markov model (HMM).

# Fields
- `loglikelihoods::Matrix{T}`: Matrix of log-likelihoods for each observation and state.
- `α::Matrix{T}`: The forward probabilities (α) for each time step and state.
- `β::Matrix{T}`: The backward probabilities (β) for each time step and state.
- `γ::Matrix{T}`: The state occupancy probabilities (γ) for each time step and state.
- `ξ::Array{T,3}`: The pairwise state occupancy probabilities (ξ) for consecutive time steps
    and state pairs.

Typically, `α` and `β` are computed by the forward–backward algorithm to find the likelihood
of an observation sequence. `γ` and `ξ` are derived from these calculations to estimate how
states transition over time.
"""
mutable struct ForwardBackward{T<:Real, V<:AbstractVector{T}, M<:AbstractMatrix{T}, MM<:AbstractMatrix{T}}
    loglikelihoods::M
    α::M
    β::M
    γ::M
    ξ::MM
end

function Base.show(io::IO, fb::ForwardBackward; gap="")
    println(io, gap, "Forward Backward Object:")
    println(io, gap, "------------------------")
    println(
        io,
        gap,
        " size(logL) = ($(size(fb.loglikelihoods,1)), $(size(fb.loglikelihoods,2)))",
    )
    println(io, gap, " size(α)    = ($(size(fb.α,1)), $(size(fb.α,2)))")
    println(io, gap, " size(β)    = ($(size(fb.β,1)), $(size(fb.β,2)))")
    println(io, gap, " size(γ)    = ($(size(fb.γ,1)), $(size(fb.γ,2)))")
    println(io, gap, " size(ξ)    = ($(size(fb.ξ,1)), $(size(fb.ξ,2)), $(size(fb.ξ,3)))")

    return nothing
end

""""
    FilterSmooth{T<:Real}

A mutable structure for storing smoothed estimates and associated covariance matrices in a
filtering or smoothing algorithm.

# Type Parameters
- `T<:Real`: The numerical type used for all fields (e.g., `Float64`, `Float32`).

# Fields
- `x_smooth::Matrix{T}` The matrix containing smoothed state estimates over time. Each
    column typically represents the state vector at a given time step.

- `p_smooth::Array{T, 3}` The posterior covariance matrices with dimensions
    (latent_dim, latent_dim, time_steps)

- `E_z::Array{T, 3}` The expected latent states, size (state_dim, T, n_trials).

- `E_zz::Array{T, 4}` The expected value of z_t * z_t', size
    (state_dim, state_dim, T, n_trials).

- `E_zz_prev::Array{T, 4}` The expected value of z_t * z_{t-1}', size
    (state_dim, state_dim, T, n_trials).

# Example
```julia
# Initialize a FilterSmooth object with Float64 type
filter = FilterSmooth{Float64}(
    x_smooth = zeros(10, 100),
    p_smooth = zeros(10, 10, 100),
    E_z = zeros(10, 100, 5),
    E_zz = zeros(10, 10, 100, 5),
    E_zz_prev = zeros(10, 10, 100, 5)
)
"""
mutable struct FilterSmooth{T<:Real}
    x_smooth::Matrix{T}
    p_smooth::Array{T, 3}
    p_smooth_tt1::Array{T, 3}
    E_z::Matrix{T}
    E_zz::Array{T, 3}
    E_zz_prev::Array{T, 3}
    entropy::T
end

function Base.show(io::IO, fs::FilterSmooth; gap="")
    println(io, gap, "Filter Smooth Object:")
    println(io, gap, "---------------------")
    println(io, gap, " size(x_smooth)  = ($(size(fs.x_smooth,1)), $(size(fs.x_smooth,2)))")
    println(
        io,
        gap,
        " size(p_smooth)  = ($(size(fs.p_smooth,1)), $(size(fs.p_smooth,2)), $(size(fs.p_smooth,3)))",
    )
    println(
        io,
        gap,
        " size(E_z)       = ($(size(fs.E_z,1)), $(size(fs.E_z,2)), $(size(fs.E_z,3)))",
    )
    println(
        io,
        gap,
        " size(E_zz)      = ($(size(fs.E_zz,1)), $(size(fs.E_zz,2)), $(size(fs.E_zz,3)), $(size(fs.E_zz,4)))",
    )
    println(
        io,
        gap,
        " size(E_zz_prev) = ($(size(fs.E_zz_prev,1)), $(size(fs.E_zz_prev,2)), $(size(fs.E_zz_prev,3)), $(size(fs.E_zz_prev,4)))",
    )

    return nothing
end

struct TrialFilterSmooth{T<:Real}
  FilterSmooths::Vector{FilterSmooth{T}}
end

Base.getindex(f::TrialFilterSmooth, i::Int) = f.FilterSmooths[i]
Base.setindex!(f::TrialFilterSmooth, value::FilterSmooth{T}, i::Int) where T<:Real = (f.FilterSmooths[i] = value)
Base.length(f::TrialFilterSmooth) = length(f.FilterSmooths)