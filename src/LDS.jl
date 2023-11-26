
"""Linear Dynamical Systems Models e.g. the Kalman Filter, (recurrent) Switching Linear Dynamical Systems, etc."""

# export statement
export DynamicalSystem, LDS, LDSParams, KalmanFilter, KalmanSmoother, loglikelihood

# Abstract types
abstract type DynamicalSystem end

"""Linear Dynamical System (LDS) Definition"""
mutable struct LDS{T1 <: Union{AbstractArray, Nothing}, 
                   T2 <: Union{AbstractArray, Nothing}, 
                   T3 <: Union{AbstractArray, Nothing}, 
                   T4 <: Union{AbstractArray, Nothing}, 
                   T5 <: Union{AbstractArray, Nothing}, 
                   T6 <: Union{AbstractArray, Nothing},
                   T7 <: Union{AbstractArray, Nothing},
                   T8 <: Union{AbstractArray, Nothing}} <: DynamicalSystem
    A::T1  # Transition Matrix
    H::T2  # Observation Matrix
    B::T3  # Control Matrix
    Q::T4  # Process Noise Covariance
    R::T5  # Observation Noise Covariance
    x0::T6 # Initial State
    P0::T7 # Initial Covariance
    inputs::T8 # Inputs
    obs_dim::Int # Observation Dimension
    latent_dim::Int # Latent Dimension
    emissions::String # Emission Model
    fit_bool::Vector{Bool} # Vector of booleans indicating which parameters to fit
end

# Constructor for LDS
function LDS(A::T1, 
             H::T2, 
             B::T3, 
             Q::T4, 
             R::T5, 
             x0::T6, 
             P0::T7,
             inputs::T8,
             obs_dim::Int=1, 
             latent_dim::Int=1, 
             emissions::String="Gaussian",
             fit_bool::Vector{Bool}=[true, true, true, true, true, true, true, true]) where {T1 <: Union{AbstractArray, Nothing}, 
                                                     T2 <: Union{AbstractArray, Nothing}, 
                                                     T3 <: Union{AbstractArray, Nothing}, 
                                                     T4 <: Union{AbstractArray, Nothing}, 
                                                     T5 <: Union{AbstractArray, Nothing}, 
                                                     T6 <: Union{AbstractArray, Nothing},
                                                     T7 <: Union{AbstractArray, Nothing},
                                                     T8 <: Union{AbstractArray, Nothing}}

    # create model
    lds = LDS{T1, T2, T3, T4, T5, T6, T7, T8}(A, H, B, Q, R, x0, P0, inputs, obs_dim, latent_dim, emissions, fit_bool)
    # Initiliaze empty parameters
    initiliaze_missing_parameters!(lds)
    return lds
end

function LDS()
    obs_dim=1
    latent_dim=1
    emissions="Gaussian"
    fit_bool=[true, true, true, true, true, true, true, true]
    A = nothing
    H = nothing
    B = nothing
    Q = nothing
    R = nothing
    x0 = nothing
    P0 = nothing
    inputs = nothing
    return LDS(A, H, B, Q, R, x0, P0, inputs, obs_dim, latent_dim, emissions, fit_bool)
end

# Function that handles initialization of the LDS
function initiliaze_missing_parameters!(lds::LDS)
    # create random defaults for missing parameters
    lds.A = lds.A === nothing ? rand(lds.latent_dim, lds.latent_dim) : lds.A
    lds.H = lds.H === nothing ? rand(lds.obs_dim, lds.latent_dim) : lds.H
    lds.Q = lds.Q === nothing ? rand(lds.latent_dim, lds.latent_dim) : lds.Q
    lds.R = lds.R === nothing ? rand(lds.obs_dim) : lds.R
    lds.x0 = lds.x0 === nothing ? rand(lds.latent_dim) : lds.x0
    lds.P0 = lds.P0 === nothing ? rand(lds.latent_dim, lds.latent_dim) : lds.P0
    if lds.inputs !== nothing
        lds.B = lds.B === nothing ? rand(lds.latent_dim, lds.latent_dim) : lds.B
    end
end

function KalmanFilter(l::LDS, y::AbstractArray)
    # First pre-allocate the matrices we will need
    T = size(y, 1)
    x = zeros(T, l.latent_dim)
    P = zeros(T, l.latent_dim, l.latent_dim)
    v = zeros(T, l.latent_dim)
    F = zeros(T, l.latent_dim, l.latent_dim)
    K = zeros(T, l.latent_dim, l.latent_dim)
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
        K[t, :, :] = (P[t, :, :] * l.H') / F[t, :, :]
        # Update step
        v[t, :] = y[t, :] - l.H * x[t, :]
        x[t, :] = x[t, :] + K[t, :, :] * v[t, :]
        P[t, :, :] = P[t, :, :] - K[t, :, :] * l.H * P[t, :, :]
        # Update the log-likelihood using Cholesky decomposition
        if !ishermitian(F[t, :, :])
            # @warn "F is not symmetric at time $t, this is likely a numerical issue, but worth examining."
            F[t, :, :] = (F[t, :, :] + F[t, :, :]') / 2
        end
        chol_F = cholesky(F[t, :, :])
        ll -= 0.5 * (l.latent_dim * log(2 * π) + 2 * sum(log.(diag(chol_F.L))) + v[t, :]' * (chol_F \ v[t, :]))
    end
    return x, P, v, F, K, ll
end

function KalmanSmoother(l::LDS, y::AbstractArray)
    # Forward pass (Kalman Filter)
    x, P, v, F, K, ll = KalmanFilter(l, y)
    # Pre-allocate smoother arrays
    xs = copy(x)  # Smoothed state estimates
    Ps = copy(P)  # Smoothed state covariances
    T = size(y, 1)
    # Backward pass
    for t in T-1:-1:1
        # Compute the smoother gain
        L = P[t, :, :] * l.A' * (P[t+1, :, :] \ I)
        # Update smoothed estimates
        xs[t, :] += L * (xs[t+1, :] - x[t+1, :])
        Ps[t, :, :] += L * (Ps[t+1, :, :] - P[t+1, :, :]) * L'
    end
    return xs, Ps
end

function E_step(l::LDS, y::AbstractArray)
    # Run the Kalman Smoother
    xs, Ps = KalmanSmoother(l, y)
    # Calculate additional statistics needed for the M-step
    T = size(y, 1)
    Exx = zeros(l.latent_dim, l.latent_dim, T)
    Exx_lag = zeros(l.latent_dim, l.latent_dim, T-1)

    for t in 1:T
        Exx[:, :, t] = Ps[t, :, :] + xs[t, :] * xs[t, :]'
        if t < T
            L = Ps[t, :, :] * l.A' / Ps[t+1, :, :]
            Exx_lag[:, :, t] = L * Ps[t+1, :, :] + xs[t, :] * xs[t+1, :]'
        end
    end

    return xs, Ps, Exx, Exx_lag
end

function M_step(l::LDS, y::Matrix{Float64}, xs::Matrix{Float64}, Ps::Array{Float64, 3}, Exx::Array{Float64, 3}, Exx_lag::Array{Float64, 3})
    T = size(y, 1)

    # Update A, Q
    S0 = sum(Exx[:, :, 1:T-1], dims=3)
    S1 = sum(Exx_lag, dims=3)
    S2 = sum(Exx[:, :, 2:T], dims=3)

    l.A = S1 / S0
    l.Q = (S2 - l.A * S1') / (T - 1)

    # Update H, R
    l.H = (y' * xs) / sum(Exx, dims=3)
    y_hat = l.H * xs'
    residuals = y - y_hat'
    l.R = (residuals' * residuals) / T

    # Update x0, P0
    l.x0 = xs[1, :]
    l.P0 = Ps[1, :, :]

    return l
end

function KalmanFilterEM(l::LDS, y::Vector{Float64})
    #TODO: Implement KalmanFilterEM
end

function KalmanFilterEGD(l::LDS, y::Vector{Float64})
end

function loglikelihood(l::LDS, y::AbstractArray)
    # calculates the loglikelihood of the LDS model
    xs, ps = KalmanFilter(l, y)
    kf_obs_pred = xs * l.H'
    # now we need to calculate the residuals
    residuals = y - kf_obs_pred
    # now we need to calculate the loglikelihood
    ll = sum(logpdf(MvNormal(zeros(l.latent_dim), l.R), residuals'))
    return ll
end

function loglikelihood(l::LDS, xs::Matrix{Float64}, Ps::Array{Float64, 3}, y::Matrix{Float64})
    # calculates the loglikelihood of the LDS model
    kf_obs_pred = xs * l.H'
    # now we need to calculate the residuals
    residuals = y - kf_obs_pred
    # now we need to calculate the loglikelihood
    ll = sum(logpdf(MvNormal(zeros(l.latent_dim), l.R), residuals'))
    return ll
end

# SLDS Definition
mutable struct SLDS <: DynamicalSystem
    #TODO: Implement SLDS
end

#rSLDS Definition
mutable struct rSLDS <: DynamicalSystem
    #TODO: Implement rSLDS
end