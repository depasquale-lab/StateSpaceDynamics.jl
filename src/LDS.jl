
"""Linear Dynamical Systems Models e.g. the Kalman Filter, (recurrent) Switching Linear Dynamical Systems, etc."""

# export statement
export LDS, KalmanFilter, KalmanSmoother, loglikelihood


"""Linear Dynamical System (LDS) Definition"""
mutable struct LDS <: DynamicalSystem
    A::Union{AbstractArray, Nothing}  # Transition Matrix
    H::Union{AbstractArray, Nothing}  # Observation Matrix
    B::Union{AbstractArray, Nothing}  # Control Matrix
    Q::Union{AbstractArray, Nothing}  # Qrocess Noise Covariance
    R::Union{AbstractArray, Nothing}  # Observation Noise Covariance
    x0::Union{AbstractArray, Nothing} # Initial State
    p0::Union{AbstractArray, Nothing} # Initial Covariance
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
function LDS(; A=nothing, H=nothing, B=nothing, Q=nothing, R=nothing, x0=nothing, p0=nothing, inputs=nothing, obs_dim=1, latent_dim=1, emissions="Gaussian", fit_bool=fill(true, 7))
    lds = LDS(A, H, B, Q, R, x0, p0, inputs, obs_dim, latent_dim, emissions, fit_bool)
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
    lds.p0 = lds.p0 === nothing ? rand(lds.latent_dim, lds.latent_dim) : lds.p0
    if lds.inputs !== nothing
        lds.B = lds.B === nothing ? rand(lds.latent_dim, size(lds.inputs, 2)) : lds.B
    end
end

"""
Initiliazes the parameters of the LDS model using PCA.

Args:
    l: LDS struct
    y: Matrix of observations
"""
function pca_init!(l::LDS, y::AbstractArray)
    # get number of observations
    T = size(y, 1)
    # get number of latent dimensions
    K = l.latent_dim
    # get number of observation dimensions
    D = l.obs_dim
    # init a pca model
    ppca = PPCA(y, K)
    # run EM
    fit!(ppca, y)
    # set the parameters
    l.H = ppca.W
    # set the initial state by projecting the first observation onto the latent space
    l.x0 = ppca.z[1, :]
end

function KalmanFilter(l::LDS, y::AbstractArray)
    # First pre-allocate the matrices we will need
    T, D = size(y)
    x_pred = zeros(T, l.latent_dim)
    p_pred = zeros(T, l.latent_dim, l.latent_dim)
    x_filt = zeros(T, l.latent_dim)
    p_filt = zeros(T, l.latent_dim, l.latent_dim)
    v = zeros(T, l.latent_dim)
    F = zeros(T, l.latent_dim, l.latent_dim)
    K = zeros(T, l.latent_dim, l.latent_dim)
    # Initialize the first state
    x_pred[1, :] = l.x0
    p_pred[1, :, :] = l.p0
    x_filt[1, :] = l.x0
    p_filt[1, :, :] = l.p0
    # Now perform the Kalman Filter
    for t in 2:T
        # Prediction step
        x_pred[t, :] = l.A * x_filt[t-1, :]
        p_pred[t, :, :] = l.A * p_filt[t-1, :, :] * l.A' + l.Q
        # Compute the Kalman gain
        F[t, :, :] = l.H * p_pred[t, :, :] * l.H' + l.R
        K[t, :, :] = p_pred[t, :, :] * l.H' * pinv(F[t, :, :])
        # Update step
        v[t, :] = y[t, :] - (l.H * x_pred[t, :])
        x_filt[t, :] = x_pred[t, :] + K[t, :, :] * v[t, :]
        p_filt[t, :, :] = p_pred[t, :, :] - (K[t, :, :] * l.H * p_pred[t, :, :])
    end
    # Compute the log-likelihood
    residuals = y - (x_filt * l.H')
    ll = loglikelihood(residuals, F, T, D)
    return x_filt, p_filt, x_pred, p_pred, v, F, K, ll
end

function KalmanSmoother(l::LDS, y::AbstractArray)
    # Forward pass (Kalman Filter)
    x_filt, p_filt, x_pred, p_pred, v, F, K, ll = KalmanFilter(l, y)
    # Pre-allocate smoother arrays
    x_smooth = zeros(size(x_filt))  # Smoothed state estimates
    p_smooth = zeros(size(p_filt))  # Smoothed state covariances
    J = ones(size(p_filt))  # Smoother gain
    T = size(y, 1)
    # Backward pass
    x_smooth[end, :] = x_filt[T, :]
    p_smooth[end, :, :] = p_filt[T, :, :]
    for t in T:-1:2
        # Compute the smoother gain
        J[t-1, :, :] = p_filt[t-1, :, :] * l.A' * pinv(p_pred[t, :, :])
        # Update smoothed estimates
        x_smooth[t-1, :] = x_filt[t-1, :] + J[t-1, :, :] * (x_smooth[t, :] - l.A * x_filt[t-1, :])
        p_smooth[t-1, :, :] = p_filt[t-1, :, :] + J[t-1, :, :] * (p_smooth[t, :, :] - p_pred[t, :, :]) * J[t-1, :, :]'
    end
    return x_smooth, p_smooth, J
end

"""
Computes the sufficient statistics for the E-step of the EM algorithm. This implementation uses the definitions from
Pattern Recognition and Machine Learning by Christopher Bishop (pg. 642).

This function computes the following statistics:
    E[zₙ] = ̂xₙ
    E[zₙzₙᵀ] = ̂xₙ̂xₙᵀ + ̂pₙ
    E[zₙzₙ₋₁ᵀ] = Jₙ₋₁̂pₙ + ̂xₙ̂xₙ₋₁ᵀ

Args:
    J: Smoother gain
    V: Smoothed state covariances
    μ: Smoothed state estimates
"""
function sufficient_statistics(J::AbstractArray, V::AbstractArray, μ::AbstractArray)
    T = size(μ, 1)
    # Initialize sufficient statistics
    E_z = zeros(T, size(μ, 2))
    E_zz = zeros(T, size(μ, 2), size(μ, 2))
    E_zz_prev = zeros(T, size(μ, 2), size(μ, 2))
    # Compute sufficient statistics
    for t in 1:T
        E_z[t, :] = μ[t, :]
        E_zz[t, :, :] = V[t, :, :] + (μ[t, :] * μ[t, :]')
        if t > 1
            E_zz_prev[t, :, :] =  V[t, :, :] * J[t-1, :, :]' + μ[t, :] * μ[t-1, :]'
        end
    end
    return E_z, E_zz, E_zz_prev
end 

function EStep(l::LDS, y::AbstractArray)
    # run the kalman smoother
    x_smooth, p_smooth, J = KalmanSmoother(l, y)
    # compute the sufficient statistics
    E_z, E_zz, E_zz_prev = sufficient_statistics(J, p_smooth, x_smooth)
    return x_smooth, p_smooth, E_z, E_zz, E_zz_prev
end

function update_initial_state_mean!(l::LDS, E_z::AbstractArray)
    # update the state mean
    if l.fit_bool[1]
        l.x0 = E_z[1, :]
    end
end

function update_initial_state_covariance!(l::LDS, E_z::AbstractArray, E_zz::AbstractArray)
    # update the state covariance
    if l.fit_bool[2]
        l.p0 = E_zz[1, :, :] - (E_z[1, :] * E_z[1, :]') 
    end
end

function update_A!(l::LDS, E_zz::AbstractArray, E_zz_prev::AbstractArray)
    # update the transition matrix
    if l.fit_bool[3]
        l.A = dropdims(sum(E_zz_prev[2:end, :, :], dims=1), dims=1) * pinv(dropdims(sum(E_zz[1:end-1, :, :], dims=1), dims=1))
    end
end

function update_Q!(l::LDS, E_zz::AbstractArray, E_zz_prev::AbstractArray)
    if l.fit_bool[4]
        # get length
        T = size(E_zz, 1)
        # update the process noise covariance
        running_sum = zeros(l.latent_dim, l.latent_dim)
        for t in 2:T
            running_sum += E_zz[t, :, :] - (l.A * E_zz_prev[t, :, :]') - (E_zz_prev[t, :, :] * l.A') + (l.A * E_zz[t-1, :, :] * l.A')
        end 
        l.Q = running_sum / (T - 1)
    end
end

function update_H!(l::LDS, E_z::AbstractArray, E_zz::AbstractArray, y::AbstractArray)
    # update the observation matrix
    if l.fit_bool[5]
        T = size(E_z, 1)
        sum_1 = sum(y[t, :] * E_z[t, :]' for t in 1:T)
        sum_2 = sum(E_zz[t, :, :] for t in 1:T)
        l.H = sum_1 * pinv(sum_2)
    end
end

function update_R!(l::LDS, E_z::AbstractArray, E_zz::AbstractArray, y::AbstractArray)
    T = size(E_z, 1)
    n_dim_obs = size(y, 2)
    res = zeros(n_dim_obs, n_dim_obs)
    for t in 1:T
        res += y[t, :] * y[t, :]' - l.H' * E_z[t, :] * y[t, :]' - y[t, :] * E_z[t, :]' * l.H + l.H' * E_zz[t, :, :] * l.H
    end
    l.R = res / T
end

function MStep!(l::LDS, E_z, E_zz, E_zz_prev, y_n)
    # update the parameters
    update_initial_state_mean!(l, E_z)
    update_initial_state_covariance!(l, E_z, E_zz)
    update_A!(l, E_zz, E_zz_prev)
    update_Q!(l, E_zz, E_zz_prev)
    update_H!(l, E_z, E_zz, y_n)
    update_R!(l, E_z, E_zz, y_n)
end

function KalmanFilterEM!(l::LDS, y::AbstractArray, max_iter::Int=100, tol::Float64=1e-6)
    # Initialize log-likelihood
    prev_ll = -Inf
    # Run EM
    for i in 1:max_iter
        # E-step
        xs, ps, E_z, E_zz, E_zz_prev = EStep(l, y)
        # M-step
        MStep!(l, E_z, E_zz, E_zz_prev, y)
        # Calculate log-likelihood
        println("Log-likelihood at iteration $i: ", ll)
        # Check convergence
        if abs(ll - prev_ll) < tol
            break
        end
        prev_ll = ll
    end
    return l, prev_ll
end

"""
Constructs the Hessian matrix of the loglikelihood of the LDS model given a set of observations. This is used for the direct optimization of the loglikelihood
as advocated by Paninski et al. (2009). The block tridiagonal structure of the Hessian is exploited to reduce the number of parameters that need to be computed, and
to reduce the memory requirements. Together with the gradient, this allows for Kalman Smoothing to be performed by simply solving a linear system of equations:

    ̂xₙ = H \\ ∇

where ̂xₙ is the smoothed state estimate, H is the Hessian matrix, and ∇ is the gradient of the loglikelihood.

Args:
    l: LDS struct
    y: Matrix of observations

Returns:
    H: Hessian matrix of the loglikelihood

"""
function Hessian(l::LDS, y::AbstractArray)
    # get size of observation matrix
    T, _ = size(y)
    # calculate inv q and r so we can do it just once
    inv_R = pinv(l.R)
    inv_Q = pinv(l.Q)
    inv_p0 = pinv(l.p0)
    # calculate the super and sub diagonal entries of the Hessian
    H_sub_entry = inv_Q * l.A
    H_super_entry = Matrix(H_sub_entry')
    # create the sub and super diagonal blocks
    H_sub = [H_sub_entry for i in 1:T-1]
    H_super = [H_super_entry for i in 1:T-1]
    # calculate the diagonal entries of the Hessian
    yt_given_xt = - l.H' * inv_R * l.H
    xt_given_xt_1 = - inv_Q
    xt1_given_xt = - l.A' * inv_Q * l.A
    x_t = - inv_p0
    # create the diagonal block
    H_diag = [(yt_given_xt + xt_given_xt_1 + xt1_given_xt) for i in 1:T]
    # fix the edge cases i.e. T = 1 and T = T
    H_diag[1] = yt_given_xt + xt1_given_xt + x_t
    H_diag[end] = yt_given_xt + xt_given_xt_1
    return block_tridgm(H_diag, H_super, H_sub)
end

function Gradient(l::LDS, y::AbstractArray)
    # get the size of the observation matrix
    T, _ = size(y)
    # get the predicted x's
    x = zeros(T, l.latent_dim)
    x[1, :] = l.x0
    for t in 2:T
        x[t, :] = l.A * x[t-1, :]
    end
    # calculate the inv of Q, R, and p0
    inv_R = pinv(l.R)
    inv_Q = pinv(l.Q)
    inv_p0 = pinv(l.p0)
    # calculate the gradient
    grad = zeros(T, l.latent_dim)
    # calculate the gradient for the first time step
    grad[1, :] = (l.A' * inv_Q * (x[2, :] - l.A * x[1, :])) + (l.H' * inv_R * (y[1, :] - l.H * x[1, :])) - (inv_p0 * (x[1, :] - l.x0))
    # calulate the gradient up until the last time step
    for t in 2:T-1
        grad[t, :] = (l.H' * inv_R * (y[t, :] - l.H * x[t, :])) - (inv_Q * (x[t, :] - l.A * x[t-1, :])) + (l.A * inv_Q * (x[t+1, :] - l.A * x[t, :]))
    end
    # calculate the gradient for the last time step
    grad[T, :] = ( l.H' * inv_R * (y[T, :] - l.H * x[T, :])) - (inv_Q * (x[T, :] - l.A * x[T-1, :]))
    return grad 
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
#TODO: There is an error in thus function.

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

function loglikelihood(l::LDS, x::AbstractArray, y::AbstractArray, S::AbstractArray)
    # check if S is symmetric
    if !ishermitian(S)
        S = (S + S') / 2
    end
    if !isposdef(S)
        println(S)
    end
    return logpdf(MvNormal(l.H * x, S), y)
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
    # calculate inverses
    inv_R = pinv(l.R)
    inv_Q = pinv(l.Q)

    # p(p₁)
    ll = (X[1, :] - l.x0)' * pinv(l.p0) * (X[1, :] - l.x0)

    # p(pₜ|pₜ₋₁) and p(yₜ|pₜ)
    for t in 1:T
        if t > 1
            # add p(pₜ|pₜ₋₁)
            ll += (X[t, :]-l.A*X[t-1, :])' * inv_Q * (X[t, :]-l.A*X[t-1, :])
        end
        # add p(yₜ|pₜ)
        ll += (y[t, :]-l.H*X[t, :])' * inv_R * (y[t, :]-l.H*X[t, :])
    end
    
    return -0.5 * ll
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
    p₀: Initial Covariance 
"""

mutable struct PLDS <: DynamicalSystem
    A::Union{AbstractArray, Nothing}  # Transition Matrix
    C::Union{AbstractArray, Nothing}  # Observation Matrix
    Q::Union{AbstractArray, Nothing}  # Process Noise Covariance
    D::Union{AbstractArray, Nothing}  # History Control Matrix
    d::Union{AbstractArray, Nothing}  # Mean Firing Rate Vector
    sₖₜ::Union{AbstractArray, Nothing}  # Spike History Vector
    x₀::Union{AbstractArray, Nothing} # Initial State
    p₀::Union{AbstractArray, Nothing} # Initial Covariance
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
function PLDS(; A=nothing, C=nothing, Q=nothing, D=nothing, d=nothing, sₖₜ=nothing, x₀=nothing, p₀=nothing, inputs=nothing, obs_dim=1, latent_dim=1, fit_bool=fill(true, 8))
    plds = PLDS(A, C, Q, D, d, sₖₜ, x₀, p₀, inputs, obs_dim, latent_dim, fit_bool)
    initialize_missing_parameters!(plds)
    return plds
end

function filter(plds::PLDS, observations::Matrix{Int})
    T = size(observations, 1)
    x = zeros(T, plds.latent_dim)
    p = zeros(T, plds.latent_dim, plds.latent_dim)
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