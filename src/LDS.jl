export DynamicalSystem, LDS, KalmanFilter, KalmanSmoother

"""Linear Dynamical Systems Models e.g. the Kalman Filter, (recurrent) Switching Linear Dynamical Systems, etc."""

abstract type DynamicalSystem end

abstract type Params end

"""LDS Params for Optimization"""
mutable struct LDSParams <: Params
    A::Matrix{Float64}
    C::Matrix{Float64}
    Q::Matrix{Float64}
    R::Matrix{Float64}
    x0::Vector{Float64}
    P0::Matrix{Float64}
end

"""Linear Dynamical System (LDS) Definition"""
mutable struct LDS{T1 <: AbstractArray, 
                   T2 <: AbstractArray, 
                   T3 <: AbstractArray, 
                   T4 <: AbstractArray, 
                   T5 <: AbstractVector, 
                   T6 <: AbstractArray} <: DynamicalSystem
    A::T1  # Transition Matrix
    H::T2  # Observation Matrix
    Q::T3  # Process Noise Covariance
    R::T4  # Observation Noise Covariance
    x0::T5 # Initial State
    P0::T6 # Initial Covariance
    D::Int # Observation Dimension
    emissions::String # Emission Model
end

"""LDS Constructor"""
function LDS(A::Union{T1, Nothing}=nothing, 
    H::Union{T2, Nothing}=nothing,
    Q::Union{T3, Nothing}=nothing,
    R::Union{T4, Nothing}=nothing,
    x0::Union{T5, Nothing}=nothing,
    P0::Union{T6, Nothing}=nothing,
    D::Union{Int, Nothing}=nothing,
    emissions::String="Gaussian") where {T1 <: AbstractArray, T2 <: AbstractArray, T3 <: AbstractArray, T4 <: AbstractArray, T5 <: AbstractVector, T6 <: AbstractArray}
    if any(isnothing, [A, H, Q, R, x0, P0, D])
        throw(ErrorException("You must specify all parameters for the LDS model."))
    else
        return LDS{T1, T2, T3, T4, T5, T6}(A, C, Q, R, x0, P0, D, emissions)
    end
end

function KalmanFilter(l::LDS, y::Matrix{Float64})
    # First pre-allocate the matrices we will need
    T = size(y, 1)
    x = zeros(T, l.D)
    P = zeros(T, l.D, l.D)
    v = zeros(T, l.D)
    F = zeros(T, l.D, l.D)
    K = zeros(T, l.D, l.D)
    # Initialize the first state
    x[1, :] = l.x0
    P[1, :, :] = l.P0
    # Initialize the log_likelihood
    ll = 0.0
    # Now perform the Kalman Filter
    for t in 2:T
        # Prediction step
        x[t, :] = l.A * x[t-1, :]
        P[t, :, :] = l.A * P[t-1, :, :] * l.A' + l.Q
        # Compute the Kalman gain
        F[t, :, :] = l.H * P[t, :, :] * l.H' + l.R
        K[t, :, :] = P[t, :, :] * l.H' / F[t, :, :]
        # Update step
        v[t, :] = y[t, :] - l.H * x[t, :]
        x[t, :] = x[t, :] + K[t, :, :] * v[t, :]
        P[t, :, :] = P[t, :, :] - K[t, :, :] * l.H * P[t, :, :]
        # Update the log-likelihood using Cholesky decomposition
        if !ishermitian(F[t, :, :])
            @warn "F is not symmetric at time $t, this is likely a numerical issue, but worth examining."
            F[t, :, :] = (F[t, :, :] + F[t, :, :]') / 2
        end
        chol_F = cholesky(F[t, :, :])
        ll -= 0.5 * (l.D * log(2 * π) + 2 * sum(log.(diag(chol_F.L))) + v[t, :]' * (chol_F \ v[t, :]))
    end
    return x, P, v, F, K, ll
end

function KalmanSmoother(l::LDS, y::Matrix{Float64})
    # Forward pass (Kalman Filter)
    x, P, v, F, K, ll = KalmanFilter(l, y)
    # Pre-allocate smoother arrays
    xs = copy(x)  # Smoothed state estimates
    Ps = copy(P)  # Smoothed state covariances
    T = size(y, 1)
    # Backward pass
    for t in T-1:-1:1
        # Compute the smoother gain
        L = P[t, :, :] * l.A' / P[t+1, :, :]
        # Update smoothed estimates
        xs[t, :] += L * (xs[t+1, :] - l.A * x[t, :])
        Ps[t, :, :] += L * (Ps[t+1, :, :] - P[t+1, :, :]) * L'
    end
    return xs, Ps
end

function E_step(l::LDS, y::Vector{Float64})
    #TODO: Implement E_step
end

function M_step(l::LDS, y::Vector{Float64})
    #TODO: Implement M_step
end

function KalmanFilterEM(l::LDS, y::Vector{Float64})
end

function KalmanFilterEGD(l::LDS, y::Vector{Float64})
end


# SLDS Definition
mutable struct SLDS <: DynamicalSystem
    #TODO: Implement SLDS
end

#rSLDS Definition
mutable struct rSLDS <: DynamicalSystem
    #TODO: Implement rSLDS
end