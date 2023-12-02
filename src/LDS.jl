
"""Linear Dynamical Systems Models e.g. the Kalman Filter, (recurrent) Switching Linear Dynamical Systems, etc."""

# export statement
export DynamicalSystem, LDS, KalmanFilter, KalmanSmoother, loglikelihood, KalmanFilterOptim!

# Abstract types
abstract type DynamicalSystem end

"""Linear Dynamical System (LDS) Definition"""
mutable struct LDS <: DynamicalSystem
    A::Union{AbstractArray, Nothing}  # Transition Matrix
    H::Union{AbstractArray, Nothing}  # Observation Matrix
    B::Union{AbstractArray, Nothing}  # Control Matrix
    Q::Union{AbstractArray, Nothing}  # Process Noise Covariance
    R::Union{AbstractArray, Nothing}  # Observation Noise Covariance
    x0::Union{AbstractArray, Nothing} # Initial State
    P0::Union{AbstractArray, Nothing} # Initial Covariance
    inputs::Union{AbstractArray, Nothing} # Inputs
    obs_dim::Int # Observation Dimension
    latent_dim::Int # Latent Dimension
    emissions::String # Emission Model
    fit_bool::Vector{Bool} # Vector of booleans indicating which parameters to fit
end

# Default constructor with no parameters
function LDS()
    return LDS(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, 1, 1, "Gaussian", fill(true, 7))
end

# Flexible constructor that handles all three cases
function LDS(; A=nothing, H=nothing, B=nothing, Q=nothing, R=nothing, x0=nothing, P0=nothing, inputs=nothing, obs_dim=1, latent_dim=1, emissions="Gaussian", fit_bool=fill(true, 7))
    lds = LDS(A, H, B, Q, R, x0, P0, inputs, obs_dim, latent_dim, emissions, fit_bool)
    initialize_missing_parameters!(lds)
    return lds
end

# Function to initialize missing parameters
function initialize_missing_parameters!(lds::LDS)
    lds.A = lds.A === nothing ? rand(lds.latent_dim, lds.latent_dim) : lds.A
    lds.H = lds.H === nothing ? rand(lds.obs_dim, lds.latent_dim) : lds.H
    lds.Q = lds.Q === nothing ? I(lds.latent_dim) : lds.Q
    lds.R = lds.R === nothing ? I(lds.obs_dim) : lds.R
    lds.x0 = lds.x0 === nothing ? rand(lds.latent_dim) : lds.x0
    lds.P0 = lds.P0 === nothing ? rand(lds.latent_dim, lds.latent_dim) : lds.P0
    if lds.inputs !== nothing
        lds.B = lds.B === nothing ? rand(lds.latent_dim, size(lds.inputs, 2)) : lds.B
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

function M_step!(l::LDS, y::Matrix{Float64}, xs::Matrix{Float64}, Ps::Array{Float64, 3}, Exx::Array{Float64, 3}, Exx_lag::Array{Float64, 3})
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


function KalmanFilterEM!(l::LDS, y::AbstractArray, max_iter::Int=100, tol::Float64=1e-6)
    # Initialize log-likelihood
    prev_ll = -Inf
    # Run EM
    for i in 1:max_iter
        # E-step
        xs, Ps, Exx, Exx_lag = E_step(l, y)
        # M-step
        M_step!(l, y, xs, Ps, Exx, Exx_lag)
        # Calculate log-likelihood
        ll = loglikelihood(l, y)
        # Check convergence
        if abs(ll - prev_ll) < tol
            break
        end
        prev_ll = ll
    end
    return l, ll
end


function KalmanFilterOptim!(l::LDS, y::AbstractArray)
    # create parameter vector and index vector
    params, param_idx = params_to_vector(l)
    # define objective function
    nll(params) = loglikelihood(params, param_idx, l, y)
    result = optimize(nll, params, BFGS(), Optim.Options(iterations=1000), autodiff=:forward)
    optimal_params = result.minimizer
    # update parameters
    return optimal_params
end


function loglikelihood(params::Vector{T1}, param_idx::Vector{Symbol}, l::LDS, y::AbstractArray) where {T1}
    # Convert the parameter vector back to the LDS struct
    vector_to_params!(l, params, param_idx)
    # The key is to ensure that all operations and functions used here are compatible with ForwardDiff.Dual
    xs, ps = KalmanFilter(l, y)
    kf_obs_pred = xs * l.H'
    residuals = y - kf_obs_pred
    ll = sum(logpdf(MvNormal(zeros(T1, l.latent_dim), l.R), residuals'))
    return ll
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

function loglikelihood(l::LDS, xs::Matrix{Float64}, y::Matrix{Float64})
    # calculates the loglikelihood of the LDS model
    kf_obs_pred = xs * l.H'
    # now we need to calculate the residuals
    residuals = y - kf_obs_pred
    # now we need to calculate the loglikelihood
    ll = sum(logpdf(MvNormal(zeros(l.latent_dim), l.R), residuals'))
    return ll
end

function params_to_vector(l::LDS)
    # Unpack parameters
    @unpack A, H, B, Q, R, x0, P0, inputs, fit_bool = l
    # Initialize parameter vector and index vector
    params = Vector{Float64}()
    params_idx = Vector{Symbol}()
    # List of fields and their corresponding symbols
    fields = [:A, :H, :B, :Q, :R, :x0, :P0]
    # Iterate over each field
    for (i, field) in enumerate(fields)
        if fit_bool[i]
            # Special handling for 'B' when inputs are present
            if field == :B && !isnothing(inputs)
                params = vcat(params, vec(B))
            else
                params = vcat(params, vec(getfield(l, field)))
            end
            push!(params_idx, field)
        end
    end
    return params, params_idx
end

function vector_to_params!(l::LDS, params::Vector{<:ForwardDiff.Dual}, param_idx::Vector{Symbol})
    idx = 1
    for p in param_idx
        current_param = getfield(l, p)
        param_size = length(current_param)
        # Extract the parameter segment from 'params'
        param_segment = params[idx:(idx + param_size - 1)]
        # Convert Dual numbers to Float64 (extract the value part)
        value_segment = map(x -> x.value, param_segment)
        # Reshape and assign the segment back to the field in 'l'
        reshaped_param = reshape(value_segment, size(current_param))
        setfield!(l, p, reshaped_param)
        # Update the index for the next parameter
        idx += param_size
    end
end





# SLDS Definition
mutable struct SLDS <: DynamicalSystem
    #TODO: Implement SLDS
end

#rSLDS Definition
mutable struct rSLDS <: DynamicalSystem
    #TODO: Implement rSLDS
end