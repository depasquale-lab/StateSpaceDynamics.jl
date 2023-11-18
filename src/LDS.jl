
"""Linear Dynamical Systems Models e.g. the Kalman Filter, (recurrent) Switching Linear Dynamical Systems, etc."""

# export statement
export DynamicalSystem, LDS, KalmanFilter, KalmanSmoother

# Abstract types
abstract type DynamicalSystem end

abstract type Params end

"""LDS Params for Optimization"""
mutable struct LDSParams
    A::Union{Matrix{Float64}, Nothing} # Transition Matrix
    H::Union{Matrix{Float64}, Nothing} # Observation Matrix
    B::Union{Matrix{Float64}, Nothing} # Control Matrix
    Q::Union{Matrix{Float64}, Nothing} # Process Noise Covariance
    R::Union{Matrix{Float64}, Nothing} # Observation Noise Covariance
    x0::Union{Vector{Float64}, Nothing} # Initial State
    P0::Union{Matrix{Float64}, Nothing} # Initial Covariance
    obs_dim::Int # Observation Dimension
    latent_dim::Int # Latent Dimension
    emissions::String # Emission Model 
    fit_bool::Vector{Bool} # Boolean vector for which parameters to fit
end

"""Linear Dynamical System (LDS) Definition"""
mutable struct LDS{T1 <: AbstractArray, 
                   T2 <: AbstractArray, 
                   T3 <: AbstractArray, 
                   T4 <: AbstractArray, 
                   T5 <: AbstractVector, 
                   T6 <: AbstractArray,
                   T7 <: AbstractArray} <: DynamicalSystem
    A::T1  # Transition Matrix
    H::T2  # Observation Matrix
    B::T3  # Control Matrix
    Q::T4  # Process Noise Covariance
    R::T5  # Observation Noise Covariance
    x0::T6 # Initial State
    P0::T7 # Initial Covariance
    obs_dim::Int # Observation Dimension
    latent_dim::Int # Latent Dimension
    emissions::String # Emission Model
end

"""LDS Constructor"""
function LDS(params::LDSParams)
    # Initialize optional parameters if they are 'nothing'
    A = params.A !== nothing ? params.A : randn(params.D, params.D)
    H = params.H !== nothing ? params.H : randn(params.D, params.D)
    B = params.B # can be 'nothing' if not provided
    Q = params.Q !== nothing ? params.Q : Matrix{Float64}(I, params.D, params.D)
    R = params.R !== nothing ? params.R : Matrix{Float64}(I, params.D, params.D)
    x0 = params.x0 !== nothing ? params.x0 : randn(params.D)
    P0 = params.P0 !== nothing ? params.P0 : Matrix{Float64}(I, params.D, params.D)

    LDS{typeof(A), typeof(H), typeof(B), typeof(Q), typeof(R), typeof(x0)}(
        A, H, B, Q, R, x0, P0, params.D, params.emissions, params.fit_bool
    )
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
    #TODO: Implement KalmanFilterEM
end

function KalmanFilterEGD(l::LDS, y::Vector{Float64})
end

function loglikelihood(l::LDS, y::Vector{Float64})
    return 
end


# SLDS Definition
mutable struct SLDS <: DynamicalSystem
    #TODO: Implement SLDS
end

#rSLDS Definition
mutable struct rSLDS <: DynamicalSystem
    #TODO: Implement rSLDS
end