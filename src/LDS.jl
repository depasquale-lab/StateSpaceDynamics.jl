abstract type DynamicalSystem end

# LDS Definition
mutable struct LDS <: DynamicalSystem
    #TODO: Implement LDS
    A::Matrix{Float64} # Transition Matrix
    C::Matrix{Float64} # Observation Matrix
    B::Matrix{Float64} # Control Matrix
    Q::Matrix{Float64} # Process Noise Covariance
    R::Matrix{Float64} # Observation Noise Covariance
    x0::Vector{Float64} # Initial State
    P0::Matrix{Float64} # Initial Covariance
    D::Int # Observation Dimension
    emissions::String # Emission Model
end

"""LDS Constructor"""
function LDS(A::Union{Matrix{Float64}, Nothing}=nothing, 
             C::Union{Matrix{Float64}, Nothing}=nothing,
             B::Union{Matrix{Float64}, Nothing}=nothing,
             Q::Union{Matrix{Float64}, Nothing}=nothing,
             R::Union{Matrix{Float64}, Nothing}=nothing,
             x0::Union{Vector{Float64}, Nothing}=nothing,
             P0::Union{Matrix{Float64}, Nothing}=nothing,
             D::Union{Int, Nothing}=nothing,
             emissions::String="Gaussian")
    if any(isnothing, [A, C, B, Q, R, x0, P0, D])
        throw(ErrorException("You must specify all parameters for the LDS model."))
    end
end

function KalmanFilter(l::LDS, y::Vector{Float64})
    # first pre-allocate the matrices we will need
    T = length(y)
    x = zeros(T, l.D)
    P = zeros(T, l.D, l.D)
    v = zeros(T, l.D)
    F = zeros(T, l.D, l.D)
    K = zeros(T, l.D, l.D)
    # initialize the first state
    x[1, :] = l.x0
    P[1, :, :] = l.P0
    # now perform the Kalman Filter
    for t in 2:T
        # prediction step
        x[t, :] = l.A * x[t-1, :]
        P[t, :, :] = l.A * P[t-1, :, :] * l.A' + l.Q
        # compute the Kalman gain
        F[t, :, :] = l.C * P[t, :, :] * l.C' + l.R
        K[t, :, :] = P[t, :, :] * l.C' * inv(F[t, :, :])
        # update step
        v[t, :] = y[t, :] - l.C * x[t, :]
        x[t, :] = x[t, :] + K[t, :, :] * v[t, :]
        P[t, :, :] = P[t, :, :] - K[t, :, :] * l.C * P[t, :, :]
    end
end

function KalmanSmoother(l::LDS, y::Vector{Float64})
end

function KalmanFilterEM(l::LDS, y::Vector{Float64})
end


# SLDS Definition
mutable struct SLDS <: DynamicalSystem
    #TODO: Implement SLDS
end

#rSLDS Definition
mutable struct rSLDS <: DynamicalSystem
    #TODO: Implement rSLDS
end