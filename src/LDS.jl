
"""Linear Dynamical Systems Models e.g. the Kalman Filter, (recurrent) Switching Linear Dynamical Systems, etc."""

# export statement
export LDS, KalmanFilter, KalmanSmoother, loglikelihood, KalmanFilterOptim!


"""Linear Dynamical System (LDS) Definition"""
mutable struct LDS <: DynamicalSystem
    A::Union{AbstractArray, Nothing}  # Transition Matrix
    H::Union{AbstractArray, Nothing}  # Observation Matrix
    B::Union{AbstractArray, Nothing}  # Control Matrix
    Q::Union{AbstractArray, Nothing}  # Qrocess Noise Covariance
    R::Union{AbstractArray, Nothing}  # Observation Noise Covariance
    x0::Union{AbstractArray, Nothing} # Initial State
    q0::Union{AbstractArray, Nothing} # Initial Covariance
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
function LDS(; A=nothing, H=nothing, B=nothing, Q=nothing, R=nothing, x0=nothing, q0=nothing, inputs=nothing, obs_dim=1, latent_dim=1, emissions="Gaussian", fit_bool=fill(true, 7))
    lds = LDS(A, H, B, Q, R, x0, q0, inputs, obs_dim, latent_dim, emissions, fit_bool)
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
    lds.q0 = lds.q0 === nothing ? rand(lds.latent_dim, lds.latent_dim) : lds.q0
    if lds.inputs !== nothing
        lds.B = lds.B === nothing ? rand(lds.latent_dim, size(lds.inputs, 2)) : lds.B
    end
end

function KalmanFilter(l::LDS, y::AbstractArray)
    # First pre-allocate the matrices we will need
    T, D = size(y)
    x = zeros(T, l.latent_dim)
    q = zeros(T, l.latent_dim, l.latent_dim)
    v = zeros(T, l.latent_dim)
    F = zeros(T, l.latent_dim, l.latent_dim)
    K = zeros(T, l.latent_dim, l.latent_dim)
    # Initialize the first state
    x[1, :] = l.x0
    q[1, :, :] = l.q0
    # Now perform the Kalman Filter
    for t in 2:T
        # Prediction step
        x[t, :] = l.A * x[t-1, :]
        q[t, :, :] = l.A * q[t-1, :, :] * l.A' + l.Q
        # Compute the Kalman gain
        F[t, :, :] = l.H * q[t, :, :] * l.H' + l.R
        K[t, :, :] = (q[t, :, :] * l.H') * pinv(F[t, :, :] + 1e-9 * I(D))
        # Update step
        v[t, :] = y[t, :] - l.H * x[t, :]
        x[t, :] = x[t, :] + K[t, :, :] * v[t, :]
        q[t, :, :] = q[t, :, :] - K[t, :, :] * l.H * q[t, :, :]
    end
    # Compute the log-likelihood
    residuals = y - (x * l.H')
    ll = loglikelihood(residuals, F, T, D)
    return x, q, v, F, K, ll
end


function KalmanSmoother(l::LDS, y::AbstractArray)
    # Forward pass (Kalman Filter)
    x, q, v, F, K, ll = KalmanFilter(l, y)
    # Pre-allocate smoother arrays
    xs = zeros(size(x))  # Smoothed state estimates
    qs = zeros(size(q))  # Smoothed state covariances
    T = size(y, 1)
    time_cov = zeros(size(q))
    # Backward pass
    xs[T, :] = x[T, :]
    qs[T, :, :] = q[T, :, :]
    time_cov[T, :, :] = (I(l.latent_dim) - K[T, :, :] * l.H') * l.A * q[T-1, :, :]
    J_t = 0
    for t in T:-1:2
        # Compute the smoother gain
        J = q[t-1, :, :] * l.A' * pinv(q[t, :, :])
        # Update smoothed estimates
        xs[t-1, :] = x[t-1, :] + J * (xs[t, :] - l.A * x[t-1, :])
        qs[t-1, :, :] = q[t-1, :, :] + J * (qs[t, :, :] - q[t, :, :]) * J'
        # Update time covariance
        time_cov[t-1, :, :] = q[t, :, :] * J' + J_t * (time_cov[t, :, :] - l.A * q[t, :, :]) * J'
        # Save previous smoother gain
        J_t = J
    end
    return xs, qs, time_cov
end

function EStep(l::LDS, y::AbstractArray)
    # get length
    T = size(y, 1)
    # Run Kalman Smoother
    xs, qs, time_cov = KalmanSmoother(l, y)
    # define sufficient statistics
    δ = zeros(size(y[1,:], 1), size(xs[1,:], 1))
    γ = zeros(size(xs[1,:], 1), size(xs[1,:], 1))
    β = zeros(size(xs[1,:], 1), size(xs[1,:], 1))
    # E-step
    for t in 1:T
        # compute sufficient statistics
        δ += y[t, :] * xs[t, :]'
        γ += xs[t, :] * xs[t, :]' + qs[t, :, :]
        if t > 1
            β += xs[t, :] * xs[t-1, :]' + time_cov[t, :, :]
        end
    end
    # calculate γ₁ and γ₂
    γ₁ = γ - xs[T, :] * xs[T,:]' - qs[T, :, :]
    γ₂ = γ - xs[1, :] * xs[1,:]' - qs[1, :, :]
    return xs, qs, δ, γ, γ₁, γ₂, β
end

function MStep!(l::LDS, y::AbstractArray, xs::AbstractArray, qs::AbstractArray, δ::AbstractArray, γ::AbstractArray, γ₁::AbstractArray, γ₂::AbstractArray, β::AbstractArray)
    # get length
    T = size(y, 1)
    # calculate alpha
    α = y' * y

    # update A
    if l.fit_bool[1]
        l.A = β * pinv(γ₁)
    end
    # update H
    if l.fit_bool[2]
        l.H = δ * pinv(γ)
    end
    # update Q
    if l.fit_bool[4]
        l.Q = (γ₂ - (l.A * β')) / (T-1)
    end
    # update R
    if l.fit_bool[5]
        l.R = (α - (l.H * δ')) / T
    end

    # update x0
    if l.fit_bool[6]
        l.x0 = xs[1, :]
    end

    # update q0
    if l.fit_bool[7]
        l.q0 = qs[1, :, :]
    end
end


function KalmanFilterEM!(l::LDS, y::AbstractArray, max_iter::Int=100, tol::Float64=1e-6)
    # Initialize log-likelihood
    prev_ll = -Inf
    # Run EM
    for i in 1:max_iter
        # E-step
        xs, qs, δ, γ, γ₁, γ₂, β = EStep(l, y)
        # M-step
        MStep!(l, y, xs, qs, δ, γ, γ₁, γ₂, β)
        # Calculate log-likelihood
        ll = loglikelihood(l, y)
        println("Log-likelihood at iteration $i: ", ll)
        # Check convergence
        if abs(ll - prev_ll) < tol
            break
        end
        prev_ll = ll
    end
    return l, prev_ll
end


function KalmanFilterOptim!(l::LDS, y::AbstractArray)
    # create parameter vector and index vector
    params, param_idx = params_to_vector(l)
    # define objective function
    nll(params) = -loglikelihood(params, param_idx, l, y)
    result = optimize(nll, params, BFGS(), Optim.Options(iterations=1000), autodiff=:forward)
    optimal_params = result.minimizer
    # update parameters
    return optimal_params
end

"""
Computes the loglikelihood of the LDS model given a a set of observations. This variant is used for optimization.

Args:
    params: Vector of parameters, compatible with ForwardDiff.Dual
    param_idx: Vector of symbols corresponding to the fields in the LDS struct
    l: LDS struct
    y: Matrix of observations

Returns:
    ll: Loglikelihood of the LDS model

"""
function loglikelihood(params::Vector{T1}, param_idx::Vector{Symbol}, l::LDS, y::AbstractArray) where {T1}
    T, D = size(y)
    # Convert the parameter vector back to the LDS struct
    vector_to_params!(l, params, param_idx)
    # The key is to ensure that all operations and functions used here are compatible with ForwardDiff.Dual
    xs, _, _, F, _, _ = KalmanFilter(l, y)
    # calculate the residuals
    residuals = y - (xs * l.H')
    # calculate the loglikelihood
    ll = loglikelihood(residuals, F, T, D)
    return ll
end

"""
Computes the loglikelihood of the LDS model given a a set of observations.

Args:
    l: LDS struct
    y: Matrix of observations

Returns:
    ll: Loglikelihood of the LDS model
"""
function loglikelihood(l::LDS, y::AbstractArray)
    T, D = size(y)
    # calculates the loglikelihood of the LDS model
    xs, _, _, F, _, _ = KalmanFilter(l, y)
    # calculate the residuals
    residuals = y - (xs * l.H')
    # calculate the loglikelihood
    ll = loglikelihood(residuals, F, T, D)
    return ll
end

function loglikelihood(residuals::AbstractArray, F::AbstractArray, T::Int, D::Int)
    ll = 0.0
    # calculate the loglikelihood
    for t in 1:T
        # use cholesky to compute the log determinant of F for stability

        # check if F is symmetric
        if !ishermitian(F[t, :, :])
            F[t, :, :] = (F[t, :, :] + F[t, :, :]') / 2
        end
        # Regularize F to ensure it is positive definite
        if !isposdef(F[t, :, :])
            F[t, :, :] = F[t, :, :] + 1e-9 * I(D)
        end
        chol_F = cholesky(F[t, :, :])
        logdet_F = 2 * sum(log.(diag(chol_F.U)))
        ll += -0.5*(D*log(2*pi)+ logdet_F + (residuals[t, :]' * pinv(F[t, :, :]) * residuals[t, :]))
        if ll == Inf
            break
        end
    end
    return ll
end

"""
Compute p(X|Y) for a given LDS model and a set of observations.
"""
function loglikelihood_X(X::AbstractArray, l::LDS, y::AbstractArray)
    T = size(y, 1)
    # p(q₁)
    ll = -0.5 * (X[1, :])' * pinv(l.q0) * X[1, :]
    # p(qₜ|qₜ₋₁)
    for t in 2:T
        ll += (X[t, :]-l.A*X[t-1, :])' * pinv(l.Q) * (X[t, :]-l.A*X[t-1, :])
    end
    # p(yₜ|qₜ)
    for t in 1:T
        ll += (y[t, :]-l.H*X[t, :])' * pinv(l.R) * (y[t, :]-l.H*X[t, :])
    end
    return ll
end

"""
Calculates the gradient of the loglikelihood of the LDS model given a a set of observations.
"""
function ∇ₗₗ(X::AbstractArray, l::LDS, y::AbstractArray)
    grad = ForwardDiff.gradient(X -> loglikelihood_X(X, l, y), X)
    return grad
end

"""
Calculates the Hessian of the loglikelihood of the LDS model given a a set of observations.
"""
function ∇²ₗₗ(X::AbstractArray, l::LDS, y::AbstractArray)
    hess = ForwardDiff.hessian(X -> loglikelihood_X(X, l, y), X)
    return hess
end

"""
Calculates the most likely set of observations "X" based on the LDS model and a set of observations "y". This is
the Kalman Smoother. This matrix formulation is based on the paper "A new look at state-space models for neural data"
DOI 10.1007/s10827-009-0179-x
"""
function matrix_Kalman_Smoother(l::LDS, y::AbstractArray)
    # create an initial guess for the latent state "X"
    X = randn(size(y, 1), l.latent_dim)
    # Smooth
    X̂ = ∇²ₗₗ(X, l, y) \ ∇ₗₗ(X, l, y)
    return X̂
end


"""
Converts the parameters in the LDS struct to a vector of parameters.

Args:
    l: LDS struct

Returns:
    params: Vector of parameters, compatible with ForwardDiff.Dual
    param_idx: Vector of symbols corresponding to the fields in the LDS struct
"""
function params_to_vector(l::LDS)
    # Unpack parameters
    @unpack A, H, B, Q, R, x0, q0, inputs, fit_bool = l
    # Initialize parameter vector and index vector
    params = Vector{Float64}()
    params_idx = Vector{Symbol}()
    # List of fields and their corresponding symbols
    fields = [:A, :H, :B, :Q, :R, :x0, :q0]
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

"""
Converts a vector of parameters to the corresponding fields in the LDS struct.

Args:
    l: LDS struct
    params: Vector of parameters, compatible with ForwardDiff.Dual
    param_idx: Vector of symbols corresponding to the fields in the LDS struct
"""
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

"""
Poisson Linear Dynamical System (PLDS) Definition

For a description of the model see:
Macke, Jakob H., et al. "Empirical models of spiking in neural populations." 
Advances in neural information processing systems 24 (2011).

Args:
    A: Transition Matrix
    C: Observation Matrix
    Q: Process Noise Covariance
    D: History Control Matrix
    d: Mean Firing Rate Vector
    sₖₜ: Spike History Vector
    x₀: Initial State
    q₀: Initial Covariance 
"""

mutable struct PLDS <: DynamicalSystem
    A::Union{AbstractArray, Nothing}  # Transition Matrix
    C::Union{AbstractArray, Nothing}  # Observation Matrix
    Q::Union{AbstractArray, Nothing}  # Process Noise Covariance
    D::Union{AbstractArray, Nothing}  # History Control Matrix
    d::Union{AbstractArray, Nothing}  # Mean Firing Rate Vector
    sₖₜ::Union{AbstractArray, Nothing}  # Spike History Vector
    x₀::Union{AbstractArray, Nothing} # Initial State
    q₀::Union{AbstractArray, Nothing} # Initial Covariance
    bₜ::Union{AbstractArray, Nothing} # Inputs
    obs_dim::Int # Observation Dimension
    latent_dim::Int # Latent Dimension
    fit_bool::Vector{Bool} # Vector of booleans indicating which parameters to fit
end

# Default constructor with no parameters
function PLDS()
    return PLDS(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, 1, 1, fill(true, 8))
end

# Flexible constructor that handles all three cases
function PLDS(; A=nothing, C=nothing, Q=nothing, D=nothing, d=nothing, sₖₜ=nothing, x₀=nothing, q₀=nothing, inputs=nothing, obs_dim=1, latent_dim=1, fit_bool=fill(true, 8))
    plds = PLDS(A, C, Q, D, d, sₖₜ, x₀, q₀, inputs, obs_dim, latent_dim, fit_bool)
    initialize_missing_parameters!(plds)
    return plds
end

function filter(plds::PLDS, observations::Matrix{Int})
    T = size(observations, 1)
    x = zeros(T, plds.latent_dim)
    q = zeros(T, plds.latent_dim, plds.latent_dim)
    #TODO: Finish later.
end


mutable struct fLDS <: DynamicalSystem
    #TODO: Implement fLDS
end

# SLDS Definition
mutable struct SLDS <: AbstractHMM
    #TODO: Implement SLDS
end

#rSLDS Definition
mutable struct rSLDS <: AbstractHMM
    #TODO: Implement rSLDS
end