
"""Linear Dynamical Systems Models e.g. the Kalman Filter, (recurrent) Switching Linear Dynamical Systems, etc."""

# export statement
export LDS, KalmanFilter, KalmanSmoother, loglikelihood, PoissonLDS
export RTSSmoother, DirectSmoother, KalmanSmoother, KalmanFilterEM!, loglikelihood, marginal_loglikelihood

"""
    LDS

A Linear Dynamical System (LDS) is a model that describes the evolution of a latent state variable xₜ and an observed variable yₜ. The model is defined by the following equations:
    
        xₜ = A * xₜ₋₁ + B * uₜ + wₜ
        yₜ = H * xₜ + vₜ
    
# Fields
- `A::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0)`: Transition Matrix
- `H::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0)`: Observation Matrix
- `B::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0)`: Control Matrix
- `Q::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0)`: Process Noise Covariance
- `R::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0)`: Observation Noise Covariance
- `x0::AbstractVector{<:Real}=Vector{Float64}(undef, 0)`: Initial State
- `p0::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0)`: Initial Covariance
- `inputs::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0)`: Inputs
- `obs_dim::Int`: Observation Dimension
- `latent_dim::Int`: Latent Dimension
- `fit_bool::Vector{Bool}=fill(true, 7)`: Vector of booleans indicating which parameters to fit

# Examples
```julia
using SSM

# Create an LDS model
l = LDS(obs_dim=2, latent_dim=2)
```
"""
mutable struct LDS <: DynamicalSystem
    A::AbstractMatrix{<:Real}  # Transition Matrix
    H::AbstractMatrix{<:Real}  # Observation Matrix
    B::AbstractMatrix{<:Real}  # Control Matrix
    Q::AbstractMatrix{<:Real}  # Process Noise Covariance
    R::AbstractMatrix{<:Real}  # Observation Noise Covariance
    x0::AbstractVector{<:Real} # Initial State
    p0::AbstractMatrix{<:Real} # Initial Covariance
    inputs::AbstractMatrix{<:Real} # Inputs
    obs_dim::Int # Observation Dimension
    latent_dim::Int # Latent Dimension
    fit_bool::Vector{Bool} # Vector of booleans indicating which parameters to fit
end

function LDS(; 
    A::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0),
    H::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0),
    B::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0),
    Q::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0),
    R::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0),
    x0::AbstractVector{<:Real}=Vector{Float64}(undef, 0),
    p0::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0),
    inputs::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0),
    obs_dim::Int,
    latent_dim::Int,
    fit_bool::Vector{Bool}=fill(true, 7)
)
    A = isempty(A) ? rand(latent_dim, latent_dim) : A
    H = isempty(H) ? rand(obs_dim, latent_dim) : H
    B = isempty(B) ? zeros(latent_dim, 1) : B
    Q = isempty(Q) ? Matrix{Float64}(I(latent_dim)) : Q
    R = isempty(R) ? Matrix{Float64}(I(latent_dim)) : R
    x0 = isempty(x0) ? rand(latent_dim) : x0
    p0 = isempty(p0) ? Matrix{Float64}(I(latent_dim)) : p0
    inputs = isempty(inputs) ? zeros(1, 1) : inputs
    
    # Ensure obs_dim and latent_dim are assigned values
    if obs_dim === nothing
        error("You must supply the dimension of the observations.")
    end

    if latent_dim === nothing
        error("You must supply the dimension of the latent states.")
    end

    return LDS(A, H, B, Q, R, x0, p0, inputs, obs_dim, latent_dim, fit_bool)
end

"""
Initiliazes the parameters of the LDS (the observation matrix and the initial state values, i.e., H and x0), model using PPCA.

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

"""
    KalmanFilter(l::LDS, y::AbstractArray)

Runs the Kalman Filter on a given LDS model and a set of observations.

# Arguments
- `l::LDS`: LDS model
- `y::AbstractArray`: Matrix of observations

# Returns
TBC
"""
function KalmanFilter(l::LDS, y::AbstractArray)
    # First pre-allocate the matrices we will need
    T, D = size(y)
    x_pred = zeros(T, l.latent_dim)
    p_pred = zeros(T, l.latent_dim, l.latent_dim)
    x_filt = zeros(T, l.latent_dim)
    p_filt = zeros(T, l.latent_dim, l.latent_dim)
    v = zeros(T, l.obs_dim)
    S = zeros(T, l.obs_dim, l.obs_dim)
    K = zeros(T, l.latent_dim, l.obs_dim)
    # Init the log-likelihood
    ml = 0.0
    # Now perform the Kalman Filter
    for t in 1:T
        if t==1
            # Initialize the first state
            x_pred[1, :] = l.x0
            p_pred[1, :, :] = l.p0
            x_filt[1, :] = l.x0
            p_filt[1, :, :] = l.p0
        else
            # Prediction step
            x_pred[t, :] = l.A * x_filt[t-1, :]
            p_pred[t, :, :] = (l.A * p_filt[t-1, :, :] * l.A') + l.Q
        end
        # Compute the Kalman gain, innovation, and innovation covariance
        v[t, :, :] = y[t, :] - (l.H * x_pred[t, :])
        S[t, :, :] = (l.H * p_pred[t, :, :] * l.H') + l.R
        K[t, :, :] = p_pred[t, :, :] * l.H' / S[t, :, :]
        # Update step
        x_filt[t, :] = x_pred[t, :] + (K[t, :, :] * v[t, :])
        p_filt[t, :, :] = (I(l.latent_dim) - K[t, :, :] * l.H) * p_pred[t, :, :]
        # Update the log-likelihood
        ml += marginal_loglikelihood(l, v[t, :], S[t, :, :])
    end
    return x_filt, p_filt, x_pred, p_pred, v, S, K, ml
end

"""
    RTSSmoother(l::LDS, y::AbstractArray)

    Explanation TBC.

# Arguments
- `l::LDS`: LDS model
- `y::AbstractArray`: Matrix of observations

# Returns
TBC
"""
function RTSSmoother(l::LDS, y::AbstractArray)
    # Forward pass (Kalman Filter)
    x_filt, p_filt, x_pred, p_pred, v, s, K, ml = KalmanFilter(l, y)
    
    # Pre-allocate smoother arrays
    T, n = size(x_filt)
    x_smooth = similar(x_filt)
    p_smooth = similar(p_filt)
    J = similar(p_filt)
    
    # Backward pass
    x_smooth[T, :] = x_filt[T, :]
    p_smooth[T, :, :] = p_filt[T, :, :]
    
    for t in T-1:-1:1
        # Compute the smoother gain
        J[t, :, :] = p_filt[t, :, :] * l.A' / p_pred[t+1, :, :]
        # Update smoothed estimates
        x_smooth[t, :] = x_filt[t, :] + J[t, :, :] * (x_smooth[t+1, :] - l.A * x_filt[t, :])
        # Update smoothed covariance
        p_smooth[t, :, :] = p_filt[t, :, :] + 
                            J[t, :, :] * (p_smooth[t+1, :, :] - p_pred[t+1, :, :]) * J[t, :, :]'
        # Enforce symmetry
        p_smooth[t, :, :] = (p_smooth[t, :, :] + p_smooth[t, :, :]') / 2
    end
    
    return x_smooth, p_smooth, J, ml
end

"""
    DirectSmoother(l::LDS, y::Matrix{<:Real})

This function performs direct smoothing for a linear dynamical system (LDS) given the system parameters `l` and the observed data `y`.

# Arguments
- `l::LDS`: The LDS object representing the system parameters.
- `y::Matrix{<:Real}`: The observed data matrix.

# Returns
- `x::Matrix{<:Real}`: The optimal state estimate.
- `p_smooth::Matrix{<:Real}`: The posterior covariance matrix.
- `inverse_offdiag::Matrix{<:Real}`: The inverse off-diagonal matrix.
- `Q_val::Float64`: The Q-function value.

# Example
"""
function DirectSmoother(l::LDS, y::Matrix{<:Real})
    # Get the length of Y
    T = size(y, 1)
    D = l.latent_dim
    # create starting point for the optimization, as the likelihood is quadratic we can just use zeros
    X₀ = zeros(T*D)
    # create wrappers for the loglikelihood, gradient, and hessian
    
    function nll(vec_x::Vector{<:Real})
        # reshape the vector to a matrix
        x = SSM.interleave_reshape(vec_x, T, D)
        # compute the negative loglikelihood
        return -loglikelihood(x, l, y)
    end

    function g!(g::Vector{<:Real}, vec_x::Vector{<:Real})
        # reshape the vector to a matrix
        x = SSM.interleave_reshape(vec_x, T, D)
        # compute the gradient
        grad = SSM.Gradient(l, y, x)
        # reshape the gradient to a vector
        g .= vec(permutedims(-grad))
    end

    function h!(h::Matrix{<:Real}, vec_x::Vector{<:Real})
        # reshape the vector to a matrix
        x = SSM.interleave_reshape(vec_x, T, D)
        # compute the hessian
        H, _, _, _ = SSM.Hessian(l, y)
        # reshape the hessian to a matrix
        h .= -H
    end

    # set up the optimization problem
    res = optimize(nll, g!, h!, X₀, Newton())

    # get the optimal state
    x = SSM.interleave_reshape(res.minimizer, T, D)

    # get covariances and nearest-neighbor second moments
    H, main, super, sub = SSM.Hessian(l, y)
    p_smooth, inverse_offdiag = SSM.block_tridiagonal_inverse(-sub, -main, -super)

    # concatenate a zero matrix to the inverse off diagonal to match the dimensions of the posterior covariance
    inverse_offdiag = cat(zeros(1, l.latent_dim, l.latent_dim), inverse_offdiag, dims=1)

    # finally clacualte Q-function
    Q_val = SSM.Q(l, x, p_smooth, inverse_offdiag, y)

    return x, p_smooth, inverse_offdiag, Q_val
end

function KalmanSmoother(l::LDS, y::AbstractArray, Smoother::SmoothingMethod=RTSSmoothing())
    return _smooth(l, y, Smoother)
end

function _smooth(l::LDS, y::AbstractArray, ::RTSSmoothing)
    return RTSSmoother(l, y)
end

function _smooth(l::LDS, y::AbstractArray, ::DirectSmoothing)
    return DirectSmoother(l, y)
end


"""
    sufficient_statistics(smoothing_method, μ, P, J_or_Ptt1)

Compute the sufficient statistics for a given smoothing method.

# Arguments
- `smoothing_method`: The smoothing method to use.
- `μ`: A matrix the mean values of the latent variables at each time step.
- `P`: An array of covariance matrices of the latent variables at each time step.
- `J_or_Ptt1`: An array of either the smoother-gain matrices or the cross-covariance matrices between consecutive latent variables.

# Returns
- The computed sufficient statistics. See the specific implementation for details.

"""
function sufficient_statistics(smoothing_method::SmoothingMethod, μ::Matrix{<:Real}, P::Array{<:Real}, J_or_Ptt1::Array{<:Real})
    return _sufficient_statistics(smoothing_method, μ, P, J_or_Ptt1)
end


"""
    _sufficient_statistics(RTSSmoothing, μ::Matrix{<:Real}, P::Array{<:Real}, J::Array{<:Real})

Computes the sufficient statistics for the E-step of the EM algorithm. This implementation uses the definitions from Pattern Recognition and Machine Learning by Christopher Bishop (pg. 642) and Bayesian Filtering and Smoothing by Simo Sarkka and Lennart Svenson.
This version is used with the RTS-Smoother.

This function computes the following statistics:
- `E[zₙ] = ̂xₙ`
- `E[zₙzₙᵀ] = ̂xₙ̂xₙᵀ + ̂pₙ`
- `E[zₙzₙ₋₁ᵀ] = Jₙ₋₁̂pₙ + ̂xₙ̂xₙ₋₁ᵀ`

# Arguments
- `RTSSmoothing`: The RTSSmoothing object.
- `μ::Matrix{<:Real}`: The mean vector of the latent states at each time step.
- `P::Array{<:Real}`: The covariance matrix of the latent states at each time step.
- `J::Array{<:Real}`: The smoother-gain matrix of the latent states at each time step.

# Returns
- `E_z`: The expected value of the latent states.
- `E_zz`: The expected value of the outer product of the latent states.
- `E_zz_prev`: The expected value of the outer product of the current and previous latent states.
"""
function _sufficient_statistics(smoother::RTSSmoothing, μ::Matrix{<:Real}, P::Array{<:Real}, J::Array{<:Real})
    T = size(μ, 1)
    # Initialize sufficient statistics
    E_z = zeros(T, size(μ, 2))
    E_zz = zeros(T, size(μ, 2), size(μ, 2))
    E_zz_prev = zeros(T, size(μ, 2), size(μ, 2))
    # Compute sufficient statistics
    for t in 1:T
        E_z[t, :] = μ[t, :]
        E_zz[t, :, :] = P[t, :, :] + (μ[t, :] * μ[t, :]')
        if t > 1
            E_zz_prev[t, :, :] =  (P[t, :, :] * J[t-1, :, :]') + (μ[t, :] * μ[t-1, :]')
        end
    end
    return E_z, E_zz, E_zz_prev
end

"""
    _sufficient_statistics(DirectSmoothing, μ::Matrix{<:Real}, P::Array{<:Real}, Ptt1::Array{<:Real})

Compute the sufficient statistics for a linear dynamical system.

# Arguments
- `DirectSmoothing`: The DirectSmoothign object.
- `μ::Matrix{<:Real}`: The mean values of the latent variables at each time step.
- `P::Array{<:Real}`: The covariance matrices of the latent variables at each time step.
- `Ptt1::Array{<:Real}`: The cross-covariance matrices between consecutive latent variables.

# Returns
- `E_z`: The expected values of the latent variables.
- `E_zz`: The expected values of the outer product of the latent variables.
- `E_zz_prev`: The expected values of the outer product of the current and previous latent variables.
"""
function _sufficient_statistics(smoother::DirectSmoothing, μ::Matrix{<:Real}, P::Array{<:Real}, Ptt1::Array{<:Real})
    T = size(μ, 1)
    # Initialize sufficient statistics
    E_z = zeros(T, size(μ, 2))
    E_zz = zeros(T, size(μ, 2), size(μ, 2))
    E_zz_prev = zeros(T, size(μ, 2), size(μ, 2))
    # Compute sufficient statistics
    for t in 1:T
        E_z[t, :] = μ[t, :]
        E_zz[t, :, :] = P[t, :, :] + (μ[t, :] * μ[t, :]')
        if t > 1
            E_zz_prev[t, :, :] = Ptt1[t, :, :]' + (μ[t, :] * μ[t-1, :]')
        end
    end
    return E_z, E_zz, E_zz_prev
end

function E_Step(l::LDS, y::AbstractArray, smoother::SmoothingMethod=RTSSmoothing())
    # run the kalman smoother
    x_smooth, p_smooth, J_or_ptt1, ll = KalmanSmoother(l, y, smoother)
    # compute the sufficient statistics
    E_z, E_zz, E_zz_prev = _sufficient_statistics(smoother, x_smooth, p_smooth, J_or_ptt1)
    return x_smooth, p_smooth, E_z, E_zz, E_zz_prev, ll
end

function update_initial_state_mean!(l::DynamicalSystem, E_z::AbstractArray)
    # update the state mean
    if l.fit_bool[1] 
        l.x0 = E_z[1, :]
    end
end

function update_initial_state_covariance!(l::DynamicalSystem, E_z::Matrix{<:Real}, E_zz::Array{<:Real})
    # update the state covariance
    if l.fit_bool[2]
        l.p0 = E_zz[1, :, :] - (E_z[1, :] * E_z[1, :]') 
    end
end

function update_A!(l::DynamicalSystem, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real})
    # update the transition matrix
    if l.fit_bool[3]
        l.A = dropdims(sum(E_zz_prev, dims=1), dims=1) * pinv(dropdims(sum(E_zz[1:end-1, :, :], dims=1), dims=1))
    end
end

function update_Q!(l::DynamicalSystem, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real})
    if l.fit_bool[4]
        N = size(E_zz, 1)
        # Initialize Q_new
        Q_new = zeros(size(l.A))
        # Calculate the sum of expectations
        sum_expectations = zeros(size(l.A))
        for n in 2:N
            sum_expectations += E_zz[n, :, :] - (E_zz_prev[n, :, :] * l.A') - (l.A * E_zz_prev[n, :, :]') + (l.A * E_zz[n-1, :, :] * l.A')
        end
        # Finalize Q_new calculation
        Q_new = (1 / (N - 1)) * sum_expectations
        l.Q = 0.5 * (Q_new + Q_new')
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
    if l.fit_bool[6]
        N = size(E_z, 1)
        # Initialize the update matrix
        update_matrix = zeros(size(l.H))
        # Calculate the sum of terms
        sum_terms = zeros(size(l.H))
        for n in 1:N
            sum_terms += (y[n, :] * y[n, :]') - (l.H * (y[n, :] * E_z[n, :]')') - ((y[n, :] * E_z[n, :]') * l.H') + (l.H * E_zz[n, :, :] * l.H')
        end
        # Finalize the update matrix calculation
        update_matrix = (1 / N) * sum_terms
        l.R = 0.5 * (update_matrix + update_matrix')
    end
end

function M_Step!(l::LDS, E_z::Matrix{<:Real}, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real}, y_n::AbstractMatrix{<:Real})
    # update the parameters
    update_initial_state_mean!(l, E_z)
    update_initial_state_covariance!(l, E_z, E_zz)
    update_A!(l, E_zz, E_zz_prev)
    update_Q!(l, E_zz, E_zz_prev)
    update_H!(l, E_z, E_zz, y_n)
    update_R!(l, E_z, E_zz, y_n)
end

function KalmanFilterEM!(l::LDS, y::AbstractArray, max_iter::Int=1000, tol::Float64=1e-12, smoother::SmoothingMethod=RTSSmoothing())
    # Initialize log-likelihood
    prev_ml = -Inf
    # Create a list to store the log-likelihood
    mls = []
    # Initialize progress bar
    prog = Progress(max_iter; desc="Fitting LDS via EM...")
    # Run EM
    for i in 1:max_iter
        # E-step
        _, _, E_z, E_zz, E_zz_prev, ml = E_Step(l, y, smoother)
        # M-step
        M_Step!(l, E_z, E_zz, E_zz_prev, y)
        # update the log-likelihood
        push!(mls, ml)
        # update the progress bar
        next!(prog)
        # Check convergence
        if abs(ml - prev_ml) < tol
            finish!(prog)
            return mls
        end
        prev_ml = ml
    end
    finish!(prog)
    return mls
end

"""
Constructs the Hessian matrix of the loglikelihood of the LDS model given a set of observations. This is used for the direct optimization of the loglikelihood
as advocated by Paninski et al. (2009). The block tridiagonal structure of the Hessian is exploited to reduce the number of parameters that need to be computed, and
to reduce the memory requirements. Together with the gradient, this allows for Kalman Smoothing to be performed by simply solving a linear system of equations:

    ̂xₙ₊₁ = ̂xₙ - H \\ ∇

where ̂xₙ is the current smoothed state estimate, H is the Hessian matrix, and ∇ is the gradient of the loglikelihood.

Args:
    l: LDS struct
    y: Matrix of observations

Returns:
    H: Hessian matrix of the loglikelihood
"""
function Hessian(l::LDS, y::AbstractArray)
    # precompute results
    T, _ = size(y)
    inv_R = pinv(l.R)
    inv_Q = pinv(l.Q)
    inv_p0 = pinv(l.p0)
    
    # super and sub diagonals
    H_sub_entry = inv_Q * l.A
    H_super_entry = Matrix(H_sub_entry')

    H_sub = Vector{Matrix{Float64}}(undef, T-1)
    H_super = Vector{Matrix{Float64}}(undef, T-1)

    Threads.@threads for i in 1:T-1
        H_sub[i] = H_sub_entry
        H_super[i] = H_super_entry
    end

    # main diagonal
    yt_given_xt = - l.H' * inv_R * l.H
    xt_given_xt_1 = - inv_Q
    xt1_given_xt = - l.A' * inv_Q * l.A
    x_t = - inv_p0

    H_diag = Vector{Matrix{Float64}}(undef, T)
    Threads.@threads for i in 2:T-1
        H_diag[i] = yt_given_xt + xt_given_xt_1 + xt1_given_xt
    end

    # Edge cases 
    H_diag[1] = yt_given_xt + xt1_given_xt + x_t
    H_diag[T] = yt_given_xt + xt_given_xt_1

    return block_tridgm(H_diag, H_super, H_sub), H_diag, H_super, H_sub
end


"""
Constructs the gradient of the loglikelihood of the LDS model given a set of observations. This is used for the direct optimization of the loglikelihood.

Args:
    l: LDS struct
    y: Matrix of observations
    x: Matrix of latent states

Returns:
    grad: Gradient of the loglikelihood
"""
function Gradient(l::LDS, y::AbstractArray, x::AbstractArray)
    # get the size of the observation matrix
    T, _ = size(y)
    # calculate the inv of Q, R, and p0
    inv_R = pinv(l.R)
    inv_Q = pinv(l.Q)
    inv_p0 = pinv(l.p0)
    # calculate the gradient
    grad = zeros(T, l.latent_dim)
    # calculate the gradient for the first time step
    grad[1, :] = (l.A' * inv_Q * (x[2, :] - l.A * x[1, :])) + (l.H' * inv_R * (y[1, :] - l.H * x[1, :])) - (inv_p0 * (x[1, :] - l.x0))
    # calulate the gradient up until the last time step
    Threads.@threads for t in 2:T-1
        grad[t, :] = (l.H' * inv_R * (y[t, :] - l.H * x[t, :])) - (inv_Q * (x[t, :] - l.A * x[t-1, :])) + (l.A' * inv_Q * (x[t+1, :] - l.A * x[t, :]))
    end
    # calculate the gradient for the last time step
    grad[T, :] = (l.H' * inv_R * (y[T, :] - l.H * x[T, :])) - (inv_Q * (x[T, :] - l.A * x[T-1, :]))
    # return a reshaped gradient so that we can match up the dimensions with the Hessian
    return grad
end

"""
    Q(l::LDS, E_z::Matrix{<:Real}, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real}, y::Matrix{<:Real})

Calculate the Q-function of the EM algorithm for an LDS model i.e. the expected complete-data loglikelihood.

# Arguments
- `A::Matrix{<:Real}`: The transition matrix.
- `Q::AbstractMatrix{<:Real}`: The process noise covariance matrix. Parameterized as Q = Q*Q' to ensure positive definiteness.
- `H::Matrix{<:Real}`: The observation matrix.
- `R::AbstractMatrix{<:Real}`: The observation noise covariance matrix. Parameterized as R = R*R' to ensure positive definiteness.
- `P0::AbstractMatrix{<:Real}`: The initial state covariance matrix. Parameterized as P0 = P0*P0' to ensure positive definiteness.
- `x0::Vector{<:Real}`: The initial state.
- `E_z::Matrix{<:Real}`: The expected latent states.
- `E_zz::Array{<:Real}`: The expected value of the latent states x the latent states.
- `E_zz_prev::Array{<:Real}`: The expected value of the latent states x the latent states at the previous time step.
- `y::Matrix{<:Real}`: The observed data.

# Returns
- `Float64`: The Q-function of the EM algorithm.

# Examples
"""
function Q(A::Matrix{<:Real}, Q::AbstractMatrix{<:Real}, H::Matrix{<:Real}, R::AbstractMatrix{<:Real}, P0::AbstractMatrix{<:Real}, x0::Vector{<:Real}, E_z::Matrix{<:Real}, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real}, y::Matrix{<:Real})
    # Convert Q, R, and P0 to proper matrices
    Q = Q * Q'
    R = R * R'
    P0 = P0 * P0'
    # calculate the inverses
    R_inv = pinv(R)
    Q_inv = pinv(Q)
    P0_inv = pinv(P0)
    # Calculate the Q-function
    Q_val = 0.0
    # Calculate the Q-function for the first time step
    Q_val += -0.5 * (logdet(P0) + tr(P0_inv * (E_zz[1, :, :] - (E_z[1, :] * x0') - (x0 * E_z[1, :]') + (x0 * x0'))))
    # Calculate the Q-function for the state model
    for t in axes(E_z, 1)[2:end] # skip the first time step
        # Individual terms
        term1 = E_zz[t, :, :]
        term2 = A * E_zz_prev[t, :, :]'
        term3 = E_zz_prev[t, :, :] * A'
        term4 = A * E_zz[t-1, :, :] * A'
        # Calculate the Q-value
        q_val = -0.5 * (logdet(Q) + tr(Q_inv * (term1 - term2 - term3 + term4)))
        Q_val += q_val
    end
    # Calculate the Q-function for the observation model
    for t in axes(E_z, 1)
        # Individual terms
        term1 = y[t, :] * y[t, :]'
        term2 = H * (E_z[t, :] * y[t, :]')
        term3 = (y[t, :] * E_z[t, :]') * H'
        term4 = H * E_zz[t, :, :] * H'
        # Calculate the Q-value`
        q_val = -0.5 * (logdet(R) + tr(R_inv * (term1 - term2 - term3 + term4)))
        Q_val += q_val
    end
    return Q_val
end

"""
    Q(l::LDS, E_z::Matrix{<:Real}, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real}, y::Matrix{<:Real})

Calculate the Q-function of the EM algorithm for an LDS model i.e. the expected complete-data loglikelihood.

# Arguments
- `l::LDS`: The Linear Dynamical System model.
- `E_z::Matrix{<:Real}`: The expected latent states.
- `E_zz::Array{<:Real}`: The expected value of the latent states x the latent states.
- `E_zz_prev::Array{<:Real}`: The expected value of the latent states x the latent states at the previous time step.
- `y::Matrix{<:Real}`: The observed data.

# Returns
- `Float64`: The Q-function of the EM algorithm.
"""
function Q(l::LDS, E_z::Matrix{<:Real}, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real}, y::Matrix{<:Real})
    # Re-parameterize the covariance matrices of the LDS model
    Q_chol = Matrix(cholesky(l.Q).L)
    R_chol = Matrix(cholesky(l.R).L)
    P0_chol = Matrix(cholesky(l.p0).L)
    # Calculate the Q-function
    Q_val = Q(l.A, Q_chol, l.H, R_chol, P0_chol, l.x0, E_z, E_zz, E_zz_prev, y)
    return Q_val
end

"""
    loglikelihood(x::AbstractArray, l::LDS, y::AbstractArray)

Calculate the log-likelihood of a linear dynamical system (LDS) given the observed data.

# Arguments
- `x::AbstractArray`: The state sequence of the LDS.
- `l::LDS`: The parameters of the LDS.
- `y::AbstractArray`: The observed data.

# Returns
- `ll::Float64`: The complete-data log-likelihood of the LDS.
"""
function loglikelihood(x::AbstractArray, l::LDS, y::AbstractArray)
    T = size(y, 1)
    # calculate inverses
    inv_R = pinv(l.R)
    inv_Q = pinv(l.Q)
    # p(p₁)
    ll = (x[1, :] - l.x0)' * pinv(l.p0) * (x[1, :] - l.x0)
    # p(pₜ|pₜ₋₁) and p(yₜ|pₜ)
    for t in 1:T
        if t > 1
            # add p(pₜ|pₜ₋₁)
            ll += (x[t, :]-l.A*x[t-1, :])' * inv_Q * (x[t, :]-l.A*x[t-1, :])
        end
        # add p(yₜ|pₜ)
        ll += (y[t, :]-l.H*x[t, :])' * inv_R * (y[t, :]-l.H*x[t, :])
    end
    
    return -0.5 * ll
end

"""
    marginal_loglikelihood(l::LDS, v::Matrix{<:Real}, s::Array{<:Real})

Compute the marginal log-likelihood of the observed data given the parameters of a linear dynamical system (LDS).

# Arguments
- `l::LDS`: An instance of the LDS type representing the parameters of the linear dynamical system.
- `v::Matrix{<:Real}`: The innovation matrix.
- `s::Array{<:Real}`: The covariance matrix of innovations.

# Returns
- `Float64`: The marginal log-likelihood of the observed data.
"""
function marginal_loglikelihood(l::LDS, v::AbstractArray, s::AbstractArray)
    return (-0.5) * ((v' * pinv(s) * v) + logdet(s) + l.obs_dim*log(2*pi))
end


"""
    sample(l::LDS, T::Int)

Sample from a Linear Dynamical System (LDS) model.

# Arguments
- `l::LDS`: The Linear Dynamical System model.
- `T::Int`: The number of time steps to sample.

# Returns
- `x::Matrix{Float64}`: The latent state variables.
- `y::Matrix{Float64}`: The observed data.

# Examples
```julia
A = rand(3, 3)
H = rand(4, 3)
Q = I(3)
R = I(4)
x0 = rand(3)
p0 = I(3)
l = LDS(A=A, H=H, Q=Q, R=R, x0=x0, p0=p0)
x, y = sample(l, 100)
```
"""
function sample(l::LDS, T::Int)
    # Initializae arrays
    x = zeros(T, l.latent_dim)
    y = zeros(T, l.obs_dim)
    # Sample the initial state
    x[1, :] = rand(MvNormal(l.x0, l.p0))
    y[1, :] = rand(MvNormal(l.H * x[1, :], l.R))
    # Sample the rest of the states
    for t in 2:T
        x[t, :] = rand(MvNormal(l.A * x[t-1, :], l.Q))
        y[t, :] = rand(MvNormal(l.H * x[t, :], l.R))
    end
    return x, y
end

"""
    mutable struct PoissonLDS <: DynamicalSystem

A Poisson Linear Dynamical System (PLDS).

# Fields
- `A:: AbstractMatrix{<:Real}`: Transition Matrix
- `C:: AbstractMatrix{<:Real}`: Observation Matrix
- `Q:: AbstractMatrix{<:Real}`: Process Noise Covariance
- `D:: AbstractMatrix{<:Real}`: History Control Matrix
- `log_d:: AbstractVector{<:Real}`: Mean Firing Rate Vector. This is in log space to ensure that the mean firing rate is always positive.
- `x0:: AbstractVector{<:Real}`: Initial State
- `p0:: AbstractMatrix{<:Real}`: Initial Covariance
- `refractory_period:: Int`: Refractory Period
- `obs_dim:: Int`: Observation Dimension
- `latent_dim:: Int`: Latent Dimension
- `fit_bool:: Vector{Bool}`: Vector of booleans indicating which parameters to fit.
"""
mutable struct PoissonLDS <: DynamicalSystem
    A:: AbstractMatrix{<:Real} # Transition Matrix
    C:: AbstractMatrix{<:Real} # Observation Matrix
    Q:: AbstractMatrix{<:Real} # Process Noise Covariance
    D:: AbstractMatrix{<:Real} # History Control Matrix
    log_d:: AbstractVector{<:Real} # Mean Firing Rate Vector
    b:: AbstractArray{<:Real} # Latent State Input
    x0:: AbstractVector{<:Real} # Initial State
    p0:: AbstractMatrix{<:Real} # Initial Covariance
    refractory_period:: Int # Refractory Period
    obs_dim:: Int # Observation Dimension
    latent_dim:: Int # Latent Dimension
    fit_bool:: Vector{Bool} # Vector of booleans indicating which parameters to fit
end

"""
    PoissonLDS(A, C, Q, D, d, x₀, p₀, obs_dim, latent_dim, fit_bool)

A Poisson Linear Dynamical System (PLDS).

This model is described in detail in Macke, Jakob H., et al. "Empirical models of spiking in neural populations." 
Advances in Neural Information Processing Systems 24 (2011).

# Arguments
- `A::AbstractMatrix{<:Real}`: Transition matrix.
- `C::AbstractMatrix{<:Real}`: Observation matrix.
- `Q::AbstractMatrix{<:Real}`: Process noise covariance matrix.
- `D::AbstractMatrix{<:Real}`: History control matrix.
- `log_d::AbstractVector{<:Real}`: Mean firing rate vector in log space. This way we can ensure that the mean firing rate is always positive.
- `b::AbstractMatrix{<:Real}`: Latent state input.
- `x₀::AbstractVector{<:Real}`: Initial state vector.
- `p₀::AbstractMatrix{<:Real}`: Initial covariance matrix.
- `refractory_period::Int`: Refractory period.
- `obs_dim::Int`: Observation dimension.
- `latent_dim::Int`: Latent dimension.
- `fit_bool::Vector{Bool}`: Vector of booleans indicating which parameters to fit.

# Examples
```julia
A = rand(3, 3)
C = rand(4, 3)
Q = I(3)
D = rand(3, 4)
log_d = rand(4)
x₀ = rand(3)
p₀ = I(3)
refractory_period = 1
obs_dim = 4
latent_dim = 3
fit_bool = fill(true, 7)

plds = PoissonLDS(A, C, Q, D, d, x₀, p₀, refractory_period, obs_dim, latent_dim, fit_bool)
"""
function PoissonLDS(;
    A::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0),
    C::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0),
    Q::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0),
    D::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0),
    log_d::AbstractVector{<:Real}=Vector{Float64}(undef, 0),
    b::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0),
    x0::AbstractVector{<:Real}=Vector{Float64}(undef, 0),
    p0::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0),
    refractory_period::Int=1,
    obs_dim::Int,
    latent_dim::Int,
    fit_bool::Vector{Bool}=fill(true, 6))

    # Initialize missing parameters
    A = isempty(A) ? 0.1 * rand(latent_dim, latent_dim) : A
    C = isempty(C) ? 0.1 * rand(obs_dim, latent_dim) : C
    Q = isempty(Q) ? convert(Matrix{Float64}, 0.01 * I(latent_dim)) : Q
    D = isempty(D) ? -abs(rand()) * I(obs_dim) : D
    log_d = isempty(log_d) ? abs.(rand(obs_dim)) : log_d
    b = isempty(b) ? Matrix{Float64}(undef, 0, latent_dim) : b
    x0 = isempty(x0) ? rand(latent_dim) : x0
    p0 = isempty(p0) ? convert(Matrix{Float64}, I(latent_dim)) : p0

    # Check that the observation dimension and latent dimension are specified
    if obs_dim === nothing 
        error("Observation dimension must be specified.")
    end

    if latent_dim === nothing
        error("Latent dimension must be specified.")
    end

    PoissonLDS(A, C, Q, D, log_d, b, x0, p0, refractory_period, obs_dim, latent_dim, fit_bool)
end

function logposterior_nonthreaded(x::AbstractMatrix{<:Real}, plds::PoissonLDS, y::Matrix{<:Real}) 
    # Re-parameterize log_d
    d = exp.(plds.log_d)
    # Calculate the log-posterior
    T = size(y, 1)
    # Get an array of prior spikes
    s = countspikes(y, plds.refractory_period)
    # pre compute matrix inverses
    inv_p0 = pinv(plds.p0)
    inv_Q = pinv(plds.Q)
    # calculate the first term
    pygivenx = 0.0
    for t in 1:T
        pygivenx += (y[t, :]' * ((plds.C * x[t, :]) + (plds.D * s[t, :]) + d)) - sum(exp.((plds.C * x[t, :]) + (plds.D * s[t, :]) + d))
    end
    # calculate the second term
    px1 = -0.5 * (x[1, :] - plds.x0)' * inv_p0 * (x[1, :] - plds.x0)
    # calculate the last term
    pxtgivenxt1 = 0.0
    for t in 2:T
        pxtgivenxt1 += -0.5 * (x[t, :] - ((plds.A * x[t-1, :]) + plds.b[t, :]))' * inv_Q * (x[t, :] - ((plds.A * x[t-1, :]) + plds.b[t, :])) 
    end
    # sum the terms
    return pygivenx + px1 + pxtgivenxt1
end

"""
    logposterior(x::Matrix{<:Real}, plds::PoissonLDS, y::Matrix{<:Real})

Calculate the log-posterior of a Poisson Linear Dynamical System (PLDS) given the observed data.

# Arguments
- `x::Matrix{<:Real}`: The latent state variables of the PLDS.
- `plds::PoissonLDS`: The Poisson Linear Dynamical System model.
- `y::Matrix{<:Real}`: The observed data.

# Returns
- `Float64`: The log-posterior of the PLDS.

# Examples
```julia
```
"""
function logposterior(x::AbstractMatrix{<:Real}, plds::PoissonLDS, y::Matrix{<:Real})
    # Convert the log firing rate to firing rate
    d = exp.(plds.log_d)
    T = size(y, 1)
    s = countspikes(y, plds.refractory_period)
    # pre compute matrix inverses
    inv_p0 = pinv(plds.p0)
    inv_Q = pinv(plds.Q)
    # Get the number of time steps
    pygivenx = zeros(T)
    # Calculate p(yₜ|xₜ)
    @threads for t in 1:T
        temp = (plds.C * x[t, :] .+ plds.D * s[t, :] .+ d)
        pygivenx[t] = (y[t, :]' * temp) - sum(exp.(temp))
    end
    pygivenx_sum = sum(pygivenx)
    # Calculate p(x₁)
    px1 = -0.5 * (x[1, :] .- plds.x0)' * inv_p0 * (x[1, :] .- plds.x0)
    # Calculate p(xₜ|xₜ₋₁)
    pxtgivenxt1 = zeros(T-1)
    @threads for t in 2:T
        temp = (x[t, :] .- (plds.A * x[t-1, :] .+ plds.b[t, :]))
        pxtgivenxt1[t-1] = -0.5 * temp' * inv_Q * temp
    end
    pxtgivenxt1_sum = sum(pxtgivenxt1)
    # Return the log-posterior
    return pygivenx_sum + px1 + pxtgivenxt1_sum
end

"""
    loglikelihood(x::Array{<:Real}, plds::PoissonLDS, y::Array{<:Real})

Calculate the complete-data log-likelihood of the observed data given the latent states and the Poisson LDS model.

# Arguments
- `x::Array{<:Real}`: The latent states of the Poisson LDS model.
- `plds::PoissonLDS`: The Poisson LDS model.
- `y::Array{<:Real}`: The observed data.

# Returns
- `ll::Float64`: The log-likelihood of the observed data.

# Example
"""
function loglikelihood(x::Array{<:Real}, plds::PoissonLDS, y::Array{<:Real})
    # confirm the driving inputs are initialized
    if isempty(plds.b)
        plds.b = zeros(size(x, 2), size(x, 3))
    end
    # Calculate the log-likelihood over all trials
    ll = zeros(size(y, 1))
    Threads.@threads for n in axes(y, 1)
        ll[n] = logposterior(x[n, :, :], plds, y[n, :, :])
    end
    return sum(ll)
end

"""
    Q_initial_obs(x0::Vector{<:Real}, sqrt_p0::Matrix{<:Real}, E_z::Array{<:Real}, E_zz::Array{<:Real})

Calculate the Q-function for the initial observation.

# Arguments
- `x0::Vector{<:Real}`: The initial state.
- `sqrt_p0::Matrix{<:Real}`: The square root of the initial state covariance matrix.
- `E_z::Array{<:Real}`: The expected latent states.
- `E_zz::Array{<:Real}`: The expected latent states x the latent states.

# Returns
- `Float64`: The Q-function for the initial observation.
"""
function Q_initial_obs(x0::Vector{<:Real}, sqrt_p0::Matrix{<:Real}, E_z::Array{<:Real}, E_zz::Array{<:Real})
    # reparametrize p0
    p0 = sqrt_p0 * sqrt_p0'
    # Compute Q
    Q_val = 0.0
    trials = size(E_z, 1)
    for k in 1:trials
        Q_val += -0.5 * (logdet(p0) + tr(pinv(p0) * (E_zz[k, 1, :, :] - (E_z[k, 1, :] * x0') - (x0 * E_z[k, 1, :]') + (x0 * x0'))))
    end
    return Q_val
end


"""
    Q_state_model(A::Matrix{<:Real}, sqrt_Q::Matrix{<:Real}, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real})

Calculate the Q-function for the state model.

# Arguments
- `A::Matrix{<:Real}`: The transition matrix.
- `sqrt_Q::Matrix{<:Real}`: The square root of the process noise covariance matrix.
- `E_zz::Array{<:Real}`: The expected latent states x the latent states.
- `E_zz_prev::Array{<:Real}`: The expected latent states x the latent states at the previous time step.

# Returns
- `Float64`: The Q-function for the state model.
"""
function Q_state_model(A::Matrix{<:Real}, sqrt_Q::Matrix{<:Real}, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real})
    # reparametrize Q
    Q = sqrt_Q * sqrt_Q'
    Q_inv = pinv(Q)
    # Compute Q
    Q_val = 0.0
    trials = size(E_zz, 1)
    time_steps = size(E_zz, 2)
    for k in 1:trials
        for t in 2:time_steps
            term1 = E_zz[k, t, :, :]
            term2 = A * E_zz_prev[k, t, :, :]'
            term3 = E_zz_prev[k, t, :, :] * A'
            term4 = A * E_zz[k, t-1, :, :] * A'
            Q_val += -0.5 * (logdet(Q) + tr(Q_inv * (term1 - term2 - term3 + term4)))
        end
    end
    return Q_val
end

"""
    Q_observation_model(C::Matrix{<:Real}, D::Matrix{<:Real}, log_d::Vector{<:Real}, E_z::Array{<:Real}, E_zz::Array{<:Real}, y::Array{<:Real})

Calculate the Q-function for the observation model.

# Arguments
- `C::Matrix{<:Real}`: The observation matrix.
- `D::Matrix{<:Real}`: The history control matrix.
- `log_d::Vector{<:Real}`: The mean firing rate vector in log space.
- `E_z::Array{<:Real}`: The expected latent states.
- `E_zz::Array{<:Real}`: The expected latent states x the latent states.
- `y::Array{<:Real}`: The observed data.

# Returns
- `Float64`: The Q-function for the observation model.
"""
function Q_observation_model(C::Matrix{<:Real}, D::Matrix{<:Real}, log_d::Vector{<:Real}, E_z::Array{<:Real}, P_smooth::Array{<:Real}, y::Array{<:Real})
    # Re-parametrize log_d
    d = exp.(log_d)
    # Compute Q
    Q_val = 0.0
    trials = size(E_z, 1)
    time_steps = size(E_z, 2)
    # calculate CC term
    CC = zeros(size(C, 1), size(C, 2)^2)
    for i in axes(C, 1)
        CC[i, :] .= vec(C[i, :] * C[i, :]')
    end
    # sum over trials
    for k in 1:trials
        #spikes = SSM.countspikes(y[k, :, :])
        # sum over time-points
        for t in 1:time_steps
            # Mean term
            h = (C * E_z[k, t, :]) + d
            # calculate rho
            ρ = 0.5 * CC * vec(P_smooth[k, t, :, :])
            ŷ = exp.(h + ρ)
            # calculate the Q-value
            Q_val += sum((y[k, t, :] .* h) - ŷ)
        end
    end
    return Q_val
end

"""
    Q_function(plds::PoissonLDS, E_z::Array{<:Real}, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real}, y::Array{<:Real})

Calculate the Q-function of the Poisson Linear Dynamical System (PLDS) model.

# Arguments
- `plds::PoissonLDS`: The Poisson Linear Dynamical System model.
- `E_z::Array{<:Real}`: The expected latent states.
- `E_zz::Array{<:Real}`: The expected latent states x the latent states.
- `E_zz_prev::Array{<:Real}`: The expected latent states x the latent states at the previous time step.
- `y::Array{<:Real}`: The observed data.

# Returns
- `Float64`: The Q-function of the PLDS model.
"""
function Q_function(plds::PoissonLDS, E_z::Array{<:Real}, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real}, y::Array{<:Real})
    # reparametrize the covariance matrices
    sqrt_Q = Matrix(cholesky(plds.Q).L)
    sqrt_p0 = Matrix(cholesky(plds.p0).L)
    # calculate the Q-function
    Q_val = Q_initial_obs(plds.x0, sqrt_p0, E_z, E_zz) + Q_state_model(plds.A, sqrt_Q, E_zz, E_zz_prev) + Q_observation_model(plds.C, plds.D, plds.log_d, E_z, E_zz, y)
    return Q_val
end

"""
    directsmooth(plds::PoissonLDS, y::Matrix{<:Real})

Perform direct smoothing on a Poisson linear dynamical system (PLDS) given the observations `y`.

# Arguments
- `plds::PoissonLDS`: The Poisson linear dynamical system.
- `y::Matrix{<:Real}`: The observations matrix.

# Returns
- `x::Matrix{Float64}`: The smoothed latent states matrix.
"""
function directsmooth(plds::PoissonLDS, y::Matrix{<:Real})
    # get the length of the observations
    T = size(y, 1)
    # generate a set of initial latent states that we can pass to a newton step, use dynamic model to generate
    x₀ = zeros(T, plds.latent_dim)
    x₀[1, :] = plds.x0
    for t in 2:T
        x₀[t, :] = plds.A * x₀[t-1, :] + plds.b[t, :]
    end
    # create wrappers for the log-posterior, gradient, and hessian

    function nlp(vec_x::Vector{<:Real})
        # reshape X
        x = SSM.interleave_reshape(vec_x, T, plds.latent_dim)
        return -logposterior(x, plds, y)
    end

    function g!(G::Vector{<:Real}, vec_x::Vector{<:Real})
        # reshape X
        x = SSM.interleave_reshape(vec_x, T, plds.latent_dim)
        # calculate the gradient
        grad = SSM.Gradient(x, plds, y)
        G .= vec(permutedims(-grad))
    end

    function h!(H::Matrix{<:Real}, vec_x::Vector{<:Real})
        # reshape X
        x = SSM.interleave_reshape(vec_x, T, plds.latent_dim)
        # Calcualte Hessian
        hess, _, _, _ = SSM.Hessian(x, plds, y)
        H .= hess
    end

    # set up optimization problem
    res = optimize(nlp, g!, h!, vec(x₀), Newton())

    # reshape the solution
    x = SSM.interleave_reshape(res.minimizer, T, plds.latent_dim)

    # get smoothed covariances and nearest-neighbor second moments
    H, main, super, sub = SSM.Hessian(x, plds, y)
    p_smooth, p_tt1 = SSM.block_tridiagonal_inverse(-sub, -main, -super)

    # add a matrix of zeros so dimensionality agrees later on
    p_tt1 = cat(reshape(zeros(plds.latent_dim, plds.latent_dim), 1, plds.latent_dim, plds.latent_dim), p_tt1, dims=1)

    return x, p_smooth, p_tt1
end

"""
    smooth(plds::PoissonLDS, y::Array{<:Real})

Smooths the latent states for each trial of a Poisson linear dynamical system (PLDS) model.

# Arguments
- `plds::PoissonLDS`: The Poisson linear dynamical system model.
- `y::Array{<:Real}`: The observed data for each trial.

# Returns
- `x_smooth::Array`: The smoothed latent states for each trial.
- `p_smooth::Array`: The smoothed covariance matrices for each trial.

"""
function smooth(plds::PoissonLDS, y::Array{<:Real})
    if isempty(plds.b)
        plds.b = zeros(size(y, 2), plds.latent_dim)
    end
    # smooth the latent states for each trial
    K, T, _ = size(y)
    x_smooth = zeros(K, T, plds.latent_dim)
    p_smooth = zeros(K, T, plds.latent_dim, plds.latent_dim)
    p_tt1 = zeros(K, T, plds.latent_dim, plds.latent_dim)
    for k in 1:K
        # smooth the latent states
        x_sm, p_sm, p_prev = directsmooth(plds, y[k, :, :])
        x_smooth[k, :, :] = x_sm
        p_smooth[k, :, :, :] = p_sm
        p_tt1[k, :, :, :] = p_prev
    end
    return x_smooth, p_smooth, p_tt1
end

"""
    Gradient(x::Matrix{<:Real}, plds::PoissonLDS, y::Matrix{<:Real})

Calculate the gradient of the log-likelihood with respect to the latent states.

# Arguments
- `x::Matrix{<:Real}`: Matrix of latent states.
- `plds::PoissonLDS`: PoissonLDS object containing model parameters.
- `y::Matrix{<:Real}`: Matrix of observed data.

# Returns
- `grad::Matrix{Float64}`: Matrix of gradients of the log-likelihood with respect to the latent states.

# Example
"""
function Gradient(x::Matrix{<:Real}, plds::PoissonLDS, y::Matrix{<:Real})
    # convert log_d to d i.e. non-log space
    d = exp.(plds.log_d)
    # calculate the gradient of the log-likelihood with respect to the latent states
    T = size(y, 1)
    # Get an array of prior spikes
    s = countspikes(y, plds.refractory_period)
    # get set of fixed constants, i.e inverses of matrices
    inv_Q = pinv(plds.Q)
    inv_p0 = pinv(plds.p0)
    # calculate the gradient
    grad = zeros(T, plds.latent_dim)
    # calculate grad of first observation
    grad[1, :] = ((y[1, :]' * plds.C)' - sum(plds.C' * exp.(plds.C * x[1, :] + d), dims=2)) + (plds.A' * inv_Q * (x[2, :] - (plds.A * x[1, :] + plds.b[1, :]))) - (inv_p0 * (x[1, :] - plds.x0))
    # calculate grad of the rest of the observations
    for t in 2:T-1
        grad[t, :] = ((y[t, :]' * plds.C)' - sum(plds.C' * exp.(plds.C * x[t, :] + plds.D * s[t, :] + d), dims=2)) - (inv_Q * (x[t, :] - (plds.A * x[t-1, :] + plds.b[t, :]))) + (plds.A' * inv_Q * (x[t+1, :] - (plds.A * x[t, :] + plds.b[t+1, :])))
    end
    # calculate grad of the last observation
    grad[T, :] = ((y[T, :]' * plds.C)' - sum(plds.C' * exp.(plds.C * x[T, :] + plds.D * s[T, :] + d), dims=2)) - (inv_Q * (x[T, :] - (plds.A * x[T-1, :] + plds.b[T, :])))
    return grad
end

"""
    Hessian(x::Matrix{<:Real}, plds::PoissonLDS, y::Matrix{<:Real})

Compute the Hessian matrix for a Poisson linear dynamical system (PLDS) model w.r.t. the latent state.

# Arguments
- `x::Matrix{<:Real}`: The latent state matrix of shape `(T, n)`, where `T` is the number of time steps and `n` is the dimensionality of the latent state.
- `plds::PoissonLDS`: The Poisson linear dynamical system model.
- `y::Matrix{<:Real}`: The observed data matrix of shape `(T, m)`, where `m` is the dimensionality of the observed data.

# Returns
- `hessian::Matrix`: The Hessian matrix of shape `(Txn, Txn)`.
- `main::Vector`: The main diagonal of the Hessian matrix.
- `H_super::Vector`: The super-diagonal entries of the Hessian matrix.
- `H_sub::Vector`: The sub-diagonal entries of the Hessian matrix.
"""
function Hessian(x::Matrix{<:Real}, plds::PoissonLDS, y::Matrix{<:Real})
    # convert log_d to d i.e. non-log space
    d = exp.(plds.log_d)
    # pre-compute a few things
    T = size(y, 1)
    inv_Q = inv(plds.Q)
    inv_p0 = inv(plds.p0)
    s = SSM.countspikes(y, plds.refractory_period)

    # calculate super and sub diagonals
    H_sub_entry = inv_Q * plds.A
    H_super_entry = Matrix(H_sub_entry')
 
    H_sub = Vector{typeof(H_sub_entry)}(undef, T-1)
    H_super = Vector{typeof(H_super_entry)}(undef, T-1)
 
    Threads.@threads for i in 1:T-1
        H_sub[i] = H_sub_entry
        H_super[i] = H_super_entry
    end
     
     # pre-compute common terms
    xt_given_xt_1 = - inv_Q
    xt1_given_xt = - plds.A' * inv_Q * plds.A
    xt = - inv_p0

    # calculate the main diagonal
    main = Vector{typeof(xt1_given_xt)}(undef, T)

    # helper function to calculate the poisson hessian term
    function calculatepoissonhess(C::Matrix{<:Real}, λ::Vector{<:Real})
        hess = zeros(size(C, 2), size(C, 2))
        for i in 1:size(C, 1)
            hess -= λ[i] * C[i, :] * C[i, :]'
        end
        return hess
    end

    Threads.@threads for t in 1:T
        λ = exp.(plds.C * x[t, :] + plds.D * s[t, :] + d)
        if t == 1
            main[t] = xt + xt1_given_xt + calculatepoissonhess(plds.C, λ)
        elseif t == T
            main[t] = xt_given_xt_1 + calculatepoissonhess(plds.C, λ)
        else
            main[t] = xt_given_xt_1 + xt1_given_xt + calculatepoissonhess(plds.C, λ)
        end
    end
    return Matrix(block_tridgm(main, H_super, H_sub)), main, H_super, H_sub
end

function sufficient_statistics(μ::Array{<:Real}, P::Array{<:Real}, P_tt1::Array{<:Real})
    # Get dimensions
    K, T, D = size(μ)
    # Initialize sufficient statistics
    E_z = zeros(K, T, D)
    E_zz = zeros(K, T, D, D)
    E_zz_prev = zeros(K, T, D, D)
    # Compute sufficient statistics
    Threads.@threads for k in 1:K
        for t in 1:T
            E_z[k, t, :] = μ[k, t, :]
            E_zz[k, t, :, :] = P[k, t, :, :] + (μ[k, t, :] * μ[k, t, :]')
            if t > 1
                E_zz_prev[k, t, :, :] = (P_tt1[k, t, :, :]') + (μ[k, t, :] * μ[k, t-1, :]')
            end
        end
    end
    return E_z, E_zz, E_zz_prev
end

"""
    E_Step(plds::PoissonLDS, y::Array{<:Real})

Perform the E-step of the Poisson Linear Dynamical System (PLDS) algorithm.

# Arguments
- `plds::PoissonLDS`: The Poisson Linear Dynamical System model.
- `y::Array{<:Real}`: The observed data.

# Returns
- `x_smooth`: The smoothed latent states.
- `p_smooth`: The smoothed latent state covariances.
"""
function E_Step(plds::PoissonLDS, y::Array{<:Real})
    # smooth the observations
    x_smooth, p_smooth, p_tt1 = smooth(plds, y)
    # calculate sufficient statistics for all trials
    E_z, E_zz, E_zz_prev = sufficient_statistics(x_smooth, p_smooth, p_tt1)
    return E_z, E_zz, E_zz_prev, x_smooth, p_smooth
end

"""
    update_x0!(plds::PoissonLDS, x_smooth::Array{<:Real})

Update the initial state of a Poisson Linear Dynamical System (PLDS) model.

# Arguments
- `plds::PoissonLDS`: The PLDS model to update.
- `x_smooth::Array{<:Real}`: The smoothed latents.

# Details
- If `plds.fit_bool[1]` is `true`, the initial state `plds.x0` is updated by summing all of the initial states of the smoothed latents and dividing by the number of trials.

"""
function update_initial_state_mean!(plds::PoissonLDS, E_z::Array{<:Real})
    # update the initial state
    if plds.fit_bool[1]
       # sum all of the initial states of the smoothed latents and divide by the number of trials
        plds.x0 = vec(sum(E_z[:, 1, :], dims=1)) / size(E_z, 1)
    end
end

"""
update_initial_state_covariance!(plds::PoissonLDS, E_zz::Array{<:Real}, E_z::Array{<:Real})

Update the initial covariance matrix `p0` of a `PoissonLDS` model.

# Arguments
- `plds::PoissonLDS`: The PoissonLDS model to update.
- `E_zz::Array{<:Real}`: The expected value of the outer product of the latent variables.
- `E_z::Array{<:Real}`: The expected value of the latent variables.

# Details
This function updates the initial covariance matrix `p0` of a `PoissonLDS` model based on the expected values of the latent variables. The update is performed only if the `fit_bool[2]` flag is set to `true`.

The initial covariance matrix `p0` is calculated for each trial by subtracting the outer product of the expected value of the latent variables from the expected value of the outer product of the latent variables. The resulting covariance matrices are then summed and divided by the number of trials to obtain the updated `p0` matrix.

# Example
"""
function update_initial_state_covariance!(plds::PoissonLDS, E_zz::Array{<:Real}, E_z::Array{<:Real})
    # update the initial covariance
    if plds.fit_bool[2]
        # get number of trials
        num_trials = size(E_zz, 1)
        # create a new array of covariance matrices for each trial
        p0 = zeros(plds.latent_dim, plds.latent_dim)
        # calculate the covariance matrix for each trial
        for n in 1:num_trials
            p0 += E_zz[n, 1, :, :] - (plds.x0 * plds.x0')
        end
        # sum the covariance matrices and divide by the number of trials
        p0 = p0 ./ num_trials
        # make sure p0 is symmetric
        plds.p0 = (p0 + p0') / 2
    end
end

function calculate_A(E_zz::Array{<:Real}, E_zz_prev::Array{<:Real})
    # Sum along the first dimension and remove it
    E_zz_sum = dropdims(sum(E_zz[1:end-1, :, :], dims=1), dims=1)
    E_zz_prev_sum = dropdims(sum(E_zz_prev, dims=1), dims=1)
    
    # Solve the system E_zz_sum * A' = E_zz_prev_sum' and then transpose the result
    A = (E_zz_sum' \ E_zz_prev_sum')'
    
    return A
end

"""
    update_A_plds!(plds::PoissonLDS, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real})

Update the transition matrix `A` of a Poisson Linear Dynamical System (PLDS) model.

# Arguments
- `plds::PoissonLDS`: The PLDS model object.
- `E_zz::Array{<:Real}`: The expected value of the latent state `z` at time `t` given the observations up to time `t`.
- `E_zz_prev::Array{<:Real}`: The expected value of the latent state `z` at time `t-1` given the observations up to time `t-1`.

# Details
This function updates the transition matrix `A` of the PLDS model by estimating it for each trial and then averaging the estimates.

# Example
"""
function update_A_plds!(plds::PoissonLDS, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real})
    # update the transition matrix
    if plds.fit_bool[3]
        # get estimates of A for each trial
        A = zeros(plds.latent_dim, plds.latent_dim)
        for n in 1:size(E_zz, 1)
            A += calculate_A(E_zz[n, :, :, :], E_zz_prev[n, :, :, :])
        end
        # Average the estimates
        plds.A = A / size(E_zz, 1)
    end
end

"""
    update_Q_plds!(plds::PoissonLDS, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real})

Update the process noise covariance matrix Q for a Poisson Linear Dynamical System (PLDS).

# Arguments
- `plds::PoissonLDS`: The Poisson Linear Dynamical System object.
- `E_zz::Array{<:Real}`: The expected value of the latent variable z multiplied by its transpose, for the current time step.
- `E_zz_prev::Array{<:Real}`: The expected value of the latent variable z multiplied by its transpose, for the previous time step.

# Details
This function updates the process noise covariance matrix Q for the PLDS. It calculates estimates of Q for each trial, and then averages these estimates to obtain the final Q.

# Example
"""
function update_Q_plds!(plds::PoissonLDS, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real})
    # update the process noise covariance matrix
    if plds.fit_bool[4]
        # get estimates of Q for each trial
        Q = zeros(size(E_zz, 1), plds.latent_dim, plds.latent_dim)
        for n in 1:size(E_zz, 1)
            Q[n, :, :] = update_Q!(plds, E_zz[n, :, :, :], E_zz_prev[n, :, :, :])
        end
        # Average the estimates
        Q = dropdims(sum(Q, dims=1), dims=1) / size(Q, 1)
        # make sure Q is symmetric
        plds.Q = (Q + Q') / 2
    end
end

"""
    update_b!(plds::PoissonLDS, x_smooth::Array{<:Real})

Update the latent state input `b` in a Poisson linear dynamical system (PLDS) model.

# Arguments
- `plds::PoissonLDS`: The Poisson linear dynamical system model.
- `x_smooth::Array{<:Real}`: The smoothed latent state trajectory.

# Details
- This function updates the latent state input `b` in the PLDS model.
- If `plds.fit_bool[6]` is `true`, an array of latent state inputs `b` is created for each trial.
- The latent state input `b` is computed as the difference between the current smoothed latent state `x_smooth[:, :, n]` and the previous smoothed latent state `plds.A * x_smooth[:, :, n-1]`.
- The latent state inputs `b` are summed across trials and divided by the number of trials to obtain the final `b` value.

# Examples
"""
function update_b!(plds::PoissonLDS, x_smooth::Array{<:Real})
    # update the latent state input
    if plds.fit_bool[5]
        # create an array of latent state inputs for each trial
        b = zeros(size(x_smooth, 1), size(x_smooth, 2), size(x_smooth, 3))
        for n in 1:size(x_smooth, 1)
            for t in 2:size(x_smooth, 2)
                b[n, t, :] = x_smooth[n, t, :] - (plds.A * x_smooth[n, t-1, :])
            end
        end
        # sum the latent state inputs and divide by the number of trials
        plds.b = dropdims(sum(b, dims=1), dims=1) / size(b, 1)
    end
end

function update_observation_model!(plds::PoissonLDS, E_z::Array{<:Real}, P_smooth::Array{<:Real}, y::Array{<:Real})
    # update the observation model parameters: C, D, and d
    if plds.fit_bool[6]
        # flatten the parameters so we can pass them to the optimizer as a single vector
        params = vcat(vec(plds.C), plds.log_d)
        # create a helper function that takes a vector of the observation model parameters
        function f(params::Vector{<:Real}, D::Matrix{<:Real}, E_z::Array{<:Real}, P_smooth::Array{<:Real}, y::Array{<:Real})
            # Split params into C and log_d
            C_size = size(D, 1)  # Assuming C has the same number of rows as D
            log_d = params[end-size(D, 2)+1:end]  # Assuming log_d has the same length as the number of columns in D
            C = reshape(params[1:end-size(D, 2)], C_size, :)  # Reshape the remaining params into C
        
            # Call Q_observation_model with the new C and log_d, and other unchanged parameters
            Q_val = Q_observation_model(C, D, log_d, E_z, P_smooth, y)
        
            return -Q_val
        end
        # # create gradient function
        # g! = (g, params) -> grad!(g, params, x_smooth, y, plds)
        # optimize
        result = optimize(params -> f(params, plds.D, E_z, P_smooth, y), params, LBFGS())
        # update the parameters
        plds.C = reshape(result.minimizer[1:plds.obs_dim * plds.latent_dim], plds.obs_dim, plds.latent_dim)
        # plds.D = reshape(result.minimizer[plds.obs_dim * plds.latent_dim + 1:plds.obs_dim * plds.latent_dim + plds.obs_dim * plds.obs_dim], plds.obs_dim, plds.obs_dim)
        plds.log_d = result.minimizer[end-plds.obs_dim+1:end]
    end
end

function M_Step!(plds::PoissonLDS, E_z::Array{<:Real}, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real}, p_smooth::Array{<:Real}, y::Array{<:Real})
    # update the parameters
    update_initial_state_mean!(plds, E_z)
    update_initial_state_covariance!(plds, E_zz, E_z)
    # update_b!(plds, x_smooth) # needs to be updated before A
    update_A_plds!(plds, E_zz, E_zz_prev)
    update_Q_plds!(plds, E_zz, E_zz_prev)
    update_observation_model!(plds, E_z, p_smooth, y)
end

function fit!(plds::PoissonLDS, y::Array{<:Real}, max_iter::Int=100, tol::Float64=1e-3)
    # create a variable to store the log-likelihood
    ec_lls = []
    ll_prev = -Inf
    # create a progress bar
    prog = Progress(max_iter; desc="Fitting PoissonLDS via LaPlace EM: ")
    # iterate through the EM algorithm
    for i in 1:max_iter
        # E-step
        E_z, E_zz, E_zz_prev, x_smooth, p_smooth = E_Step(plds, y)
        # calculate the log-likelihood
        ll = Q_function(plds, E_z, E_zz, E_zz_prev, y)
        push!(ec_lls, ll)
        # Update progress bar
        next!(prog)
        # M-step
        M_Step!(plds, E_z, E_zz, E_zz_prev, p_smooth, y)
        # check for convergence
        if abs(ll - ll_prev) < tol
            finish!(prog)
            return ec_lls
        end
        ll_prev = ll
    end
    ProgressMeter.finish!(prog)
    println("Maximum iterations reached without convergence.")
    return ec_lls
end

"""
    countspikes(y::Matrix{<:Real}, window::Int)
   
Counts the number of "spikes" or "events" within a specified time window to calcualate sₖₜ for the Poisson LDS model.

Args:
- `y::Matrix{<:Real}`: Matrix of observations.
- `window::Int`: Time window for counting spikes.
"""
function countspikes(y::Matrix{<:Real}, window::Int=1)
    # Get size of the observation matrix
    T, D = size(y)
    # Initialize the spike-count matrix
    s = zeros(T, D)
    # Compute the cumulative sum of the observation matrix along the first dimension (time)
    cumsum_y = cumsum(y, dims=1)
    
    # Loop over time points from 2 to T
    for t in 2:T
        if t - window <= 1
            # If the time window is less than or equal to 1, use the cumulative sum directly
            s[t, :] = cumsum_y[t-1, :]
        else
            # Otherwise, calculate the sum of the window by subtracting cumulative sums
            s[t, :] = cumsum_y[t-1, :] .- cumsum_y[t-window-1, :]
        end
    end
    
    return s
end

"""
    sample(plds::PoissonLDS, T::Int, K::Int)

Sample from a Poisson Linear Dynamical System (PLDS) model.

# Arguments
- `plds::PoissonLDS`: The Poisson Linear Dynamical System model.
- `T::Int64`: The number of time steps to sample.
- `k::Int64`: The number of trials to sample.

# Returns
- `x::Array{Float64}`: The latent state variables.
- `y::Array{Float64}`: The observed data.

# Examples
"""
function sample(plds::PoissonLDS, T::Int64, K::Int64)
    # Convert log_d to d i.e. non-log space
    d = exp.(plds.log_d)
    # Pre-allocate arrays
    x = zeros(K, T, plds.latent_dim)
    y = zeros(K, T, plds.obs_dim)
    
    for k in 1:K
        # Sample the initial stated
        x[k, 1, :] = rand(MvNormal(plds.x0, plds.p0))
        y[k, 1, :] = rand.(Poisson.(exp.(plds.C * x[k, 1, :] + d)))
        # Sample the rest of the states
        for t in 2:T
            s = max(1, t - plds.refractory_period)
            spikes = sum(y[k, s:t-1, :], dims=1)'
            x[k, t, :] = rand(MvNormal((plds.A * x[k, t-1, :]) + plds.b[t, :], plds.Q))
            y[k, t, :] = rand.(Poisson.(exp.((plds.C * x[k, t, :]) + (plds.D * spikes) + d)))
        end
    end
    return x, y
end


