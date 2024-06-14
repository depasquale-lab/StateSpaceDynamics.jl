
"""Linear Dynamical Systems Models e.g. the Kalman Filter, (recurrent) Switching Linear Dynamical Systems, etc."""

# export statement
export LDS, KalmanFilter, KalmanSmoother, loglikelihood, PoissonLDS

# constants
const DEFAULT_LATENT_DIM = 2
const DEFAULT_OBS_DIM = 2

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
    fit_bool::Vector{Bool} # Vector of booleans indicating which parameters to fit
end

function LDS(; 
    A::Union{AbstractArray, Nothing}=nothing,
    H::Union{AbstractArray, Nothing}=nothing,
    B::Union{AbstractArray, Nothing}=nothing,
    Q::Union{AbstractArray, Nothing}=nothing,
    R::Union{AbstractArray, Nothing}=nothing,
    x0::Union{AbstractArray, Nothing}=nothing,
    p0::Union{AbstractArray, Nothing}=nothing,
    inputs::Union{AbstractArray, Nothing}=nothing,
    obs_dim::Int=DEFAULT_OBS_DIM,
    latent_dim::Int=DEFAULT_LATENT_DIM,
    fit_bool::Vector{Bool}=fill(true, 7)
)
    LDS(
        A, H, B, Q, R, x0, p0, inputs, obs_dim, latent_dim, fit_bool
    ) |> initialize_missing_parameters!
end

# Function to initialize missing parameters
function initialize_missing_parameters!(lds::LDS)
    lds.A = lds.A === nothing ? rand(lds.latent_dim, lds.latent_dim) : lds.A
    lds.H = lds.H === nothing ? rand(lds.obs_dim, lds.latent_dim) : lds.H
    lds.Q = lds.Q === nothing ? I(lds.latent_dim) : lds.Q
    lds.R = lds.R === nothing ? I(lds.obs_dim) : lds.R
    lds.x0 = lds.x0 === nothing ? rand(lds.latent_dim) : lds.x0
    lds.p0 = lds.p0 === nothing ? Matrix(rand() * I(lds.latent_dim)) : lds.p0
    if lds.inputs !== nothing
        lds.B = lds.B === nothing ? rand(lds.latent_dim, size(lds.inputs, 2)) : lds.B
    end
    return lds
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
        K[t, :, :] = p_pred[t, :, :] * l.H' * pinv(S[t, :, :])
        # Update step
        x_filt[t, :] = x_pred[t, :] + (K[t, :, :] * v[t, :])
        p_filt[t, :, :] = (I(l.latent_dim) - K[t, :, :] * l.H) * p_pred[t, :, :]
        ml += marginal_loglikelihood(l, v[t, :], S[t, :, :])
    end
    return x_filt, p_filt, x_pred, p_pred, v, S, K, ml
end

function RTSSmoother(l::LDS, y::AbstractArray)
    # Forward pass (Kalman Filter)
    x_filt, p_filt, x_pred, p_pred, v, F, K, ml = KalmanFilter(l, y)
    # Pre-allocate smoother arrays
    x_smooth = zeros(size(x_filt))  # Smoothed state estimates
    p_smooth = zeros(size(p_filt))  # Smoothed state covariances
    J = ones(size(p_filt))  # Smoother gain
    T = size(y, 1)
    # Backward pass
    for t in T:-1:2
        if t == T
            x_smooth[end, :] = x_filt[T, :]
            p_smooth[end, :, :] = p_filt[T, :, :]
        end
        # Compute the smoother gain
        J[t-1, :, :] = p_filt[t-1, :, :] * l.A' * pinv(p_pred[t, :, :])
        # Update smoothed estimates
        x_smooth[t-1, :] = x_filt[t-1, :] + J[t-1, :, :] * (x_smooth[t, :] - l.A * x_filt[t-1, :])
        p_smooth[t-1, :, :] = p_filt[t-1, :, :] + J[t-1, :, :] * (p_smooth[t, :, :] - p_pred[t, :, :]) * J[t-1, :, :]'
        # quickly enforce symmetry
        p_smooth[t-1, :, :] = 0.5 * (p_smooth[t-1, :, :] + p_smooth[t-1, :, :]')
    end
    return x_smooth, p_smooth, J, ml
end

function DirectSmoother(l::LDS, y::AbstractArray, tol::Float64=1e-6)
    # Pre-allocate arrays
    T, D = size(y)
    p_smooth = zeros(T, l.latent_dim, l.latent_dim)
    # Compute the precdiction as a starting point for the optimization
    xₜ = zeros(T, l.latent_dim)
    xₜ[1, :] = l.x0
    for t in 2:T
        xₜ[t, :] = l.A * xₜ[t-1, :]
    end
    # Compute the Hessian of the loglikelihood
    H, main, super, sub = Hessian(l, y)
    # compute the inverse of the main diagonal of the Hessian, this is the posterior covariance
    p_smooth, inverse_offdiag = block_tridiagonal_inverse(-sub, -main, -super)
    # now optimize
    for i in 1:5 # this should stop at the first iteration in theory but likely will at iteration 2
        # Compute the gradient
        grad = Gradient(l, y, xₜ)
        # reshape the gradient to a vector to pass to newton_raphson_step_tridg!, we transpose as the way Julia reshapes is by vertically stacking columns as we need to match up observations to the Hessian.
        grad = Matrix{Float64}(reshape(grad', (T*D), 1))
        # Compute the Newton-Raphson step        
        xₜ₊₁ = newton_raphson_step_tridg!(xₜ, H, grad)
        # Check for convergence (uncomment the following lines to enable convergence checking)
        if norm(xₜ₊₁ - xₜ) < tol
            println("Converged at iteration ", i)
            return xₜ₊₁, p_smooth, inverse_offdiag
        else
            println("Norm of gradient iterate difference: ", norm(xₜ₊₁ - xₜ))
        end
        # Update the iterate
        xₜ = xₜ₊₁
    end
    # Print a warning if the routine did not converge
    println("Warning: Newton-Raphson routine did not converge.")
    return xₜ, p_smooth, inverse_offdiag
end

function KalmanSmoother(l::LDS, y::AbstractArray, method::String="RTS")
    if method == "RTS"
        return RTSSmoother(l, y)
    else
        return DirectSmoother(l, y)
    end
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
        l.A = dropdims(sum(E_zz_prev[2:end, :, :], dims=1), dims=1) * pinv(dropdims(sum(E_zz[1:end-1, :, :], dims=1), dims=1))
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

"""
Computes the sufficient statistics for the E-step of the EM algorithm. This implementation uses the definitions from
Pattern Recognition and Machine Learning by Christopher Bishop (pg. 642) and Bayesian Filtering and Smoothing by Simo Sarkka and Lennart Svenson.

This function computes the following statistics:
    E[zₙ] = ̂xₙ
    E[zₙzₙᵀ] = ̂xₙ̂xₙᵀ + ̂pₙ
    E[zₙzₙ₋₁ᵀ] = Jₙ₋₁̂pₙ + ̂xₙ̂xₙ₋₁ᵀ

Args:
    J: Smoother gain
    P: Smoothed state covariances
    μ: Smoothed state estimates
"""
function sufficient_statistics(J::Array{<:Real}, P::Array{<:Real}, μ::Matrix{<:Real})
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

function E_Step(l::LDS, y::AbstractArray)
    # run the kalman smoother
    x_smooth, p_smooth, J, ml = KalmanSmoother(l, y)
    # compute the sufficient statistics
    E_z, E_zz, E_zz_prev = sufficient_statistics(J, p_smooth, x_smooth)
    return x_smooth, p_smooth, E_z, E_zz, E_zz_prev, ml
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

function KalmanFilterEM!(l::LDS, y::AbstractArray, max_iter::Int=1000, tol::Float64=1e-6)
    # Initialize log-likelihood
    prev_ml = -Inf
    # Create a list to store the log-likelihood
    mls = []
    # Initialize progress bar
    prog = Progress(max_iter; desc="Running EM Algorithm...")
    # Run EM
    for i in 1:max_iter
        # E-step
        _, _, E_z, E_zz, E_zz_prev, ml = E_Step(l, y)
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
Compute p(X|Y) for a given LDS model and a set of observations i.e. the loglikelihood.

Args:
    X: Matrix of latent states
    l: LDS struct
    y: Matrix of observations

Returns:
    ll: Loglikelihood of the LDS model given the observations
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
Compute the marginal loglikelihood of a given LDS model and a set of observations.

Args:
    l: LDS struct
    y: Matrix of observations
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
- `d:: AbstractVector{<:Real}`: Mean Firing Rate Vector
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
    d:: AbstractVector{<:Real} # Mean Firing Rate Vector
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
- `d::AbstractVector{<:Real}`: Mean firing rate vector.
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
d = rand(4)
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
    d::AbstractVector{<:Real}=Vector{Float64}(undef, 0),
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
    Q = isempty(Q) ? 0.01 * I(latent_dim) : Q
    D = isempty(D) ? -abs(rand()) * I(obs_dim) : D
    d = isempty(d) ? abs.(rand(obs_dim)) : d
    b = isempty(b) ? Matrix{Float64}(undef, 0, latent_dim) : b
    x0 = isempty(x0) ? rand(latent_dim) : x0
    p0 = isempty(p0) ? I(latent_dim) : p0

    # Check that the observation dimension and latent dimension are specified
    if obs_dim === nothing 
        error("Observation dimension must be specified.")
    end

    if latent_dim === nothing
        error("Latent dimension must be specified.")
    end

    PoissonLDS(A, C, Q, D, d, b, x0, p0, refractory_period, obs_dim, latent_dim, fit_bool)
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
function logposterior(x::Matrix{<:Real}, plds::PoissonLDS, y::Matrix{<:Real}) 
    # Calculate the log-posterior
    T = size(y, 1)
    # Get an array of prior spikes
    s = countspikes(y, plds.refractory_period)
    # calculate the first term
    pygivenx = 0.0
    for t in 1:T
        pygivenx += (y[t, :]' * ((plds.C * x[t, :]) + (plds.D * s[t, :]) + plds.d)) - sum(exp.((plds.C * x[t, :]) + (plds.D * s[t, :]) + plds.d))
    end
    # calculate the second term
    px1 = -0.5 * (x[1, :] - plds.x0)' * pinv(plds.p0) * (x[1, :] - plds.x0)
    # calculate the last term
    pxtgivenxt1 = 0.0
    for t in 2:T
        pxtgivenxt1 += -0.5 * (x[t, :] - ((plds.A * x[t-1, :]) + plds.b[t, :]))' * pinv(plds.Q) * (x[t, :] - ((plds.A * x[t-1, :]) + plds.b[t, :])) 
    end
    # calculate the log-determinants
    log_det_Q = -(T-1)/2 * logdet(plds.Q + (I * 1e-3))
    log_det_p0 = -0.5 * logdet(plds.p0 + (I * 1e-3))
    log_det = log_det_Q + log_det_p0
    # sum the terms
    return pygivenx + px1 + pxtgivenxt1 + log_det
end


"""
    directsmooth(plds::PoissonLDS, y::Matrix{<:Real})

Perform direct smoothing on a Poisson linear dynamical system (PLDS) given the observations `y`.

# Arguments
- `plds::PoissonLDS`: The Poisson linear dynamical system.
- `y::Matrix{<:Real}`: The observations matrix.

# Returns
- `x::Matrix{Float64}`: The smoothed latent states matrix.

# Example
"""
function directsmooth(plds::PoissonLDS, y::Matrix{<:Real}, max_iter::Int=1000, tol::Float64=1e-6)
    # get the length of the observations
    T = size(y, 1)
    # generate a set of initial latent states that we can pass to a newton step
    x = zeros(T, plds.latent_dim)
    # calculate the prediction step
    x[1, :] = plds.x0
    for t in 2:T
        x[t, :] = plds.A * x[t-1, :] + plds.b[t, :]
    end
    # smooth the observations
    for i in 1:max_iter
        # calculate the gradient
        grad = Gradient(x, plds, y)
        # reshape the gradient to a Vector
        grad = Matrix{Float64}(reshape(grad', (T*plds.latent_dim), 1))
        # calculate the Hessian
        H, main, super, sub = Hessian(x, plds, y)
        # calculate the newton raphson step
        x_new = newton_raphson_step_tridg!(x, H, grad)
        # check for convergence
        if norm(x_new - x) < tol
            # println("Converged at iteration ", i)
            H, main, super, sub = Hessian(x_new, plds, y)
            # calculate p_smooth
            p_smooth, p_tt1 = block_tridiagonal_inverse(-sub, -main, -super)
            # add a matrix of zeros so dimensionality agrees later on
            p_tt1 = cat(reshape(zeros(plds.latent_dim, plds.latent_dim), 1, plds.latent_dim, plds.latent_dim), p_tt1, dims=1)
            return x_new, p_smooth, p_tt1
        end
        # update the latent states
        x = x_new
    end
    # print a warning if the routine did not converge
    println("Warning: Newton-Raphson routine did not converge.")
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
        plds.b = zeros(size(x, 1), size(x, 2))
    end
    # Calculate the log-likelihood over all trials
    ll = 0.0
    for n in 1:size(y, 3)
        ll += logposterior(x[:, :, n], plds, y[:, :, n])
    end
    return ll
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
        plds.b = zeros(size(y, 1), plds.latent_dim)
    end
    # smooth the latent states for each trial
    x_smooth = zeros(size(y, 1), plds.latent_dim, size(y, 3))
    p_smooth = zeros(size(y, 1), plds.latent_dim, plds.latent_dim, size(y, 3))
    p_tt1 = zeros(size(y, 1), plds.latent_dim, plds.latent_dim, size(y, 3))
    for n in 1:size(y, 3)
        # smooth the latent states
        x_sm, p_sm, p_prev = directsmooth(plds, y[:, :, n])
        x_smooth[:, :, n] = x_sm
        p_smooth[:, :, :, n] = p_sm
        p_tt1[:, :, :, n] = p_prev
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
    grad[1, :] = ((y[1, :]' * plds.C)' - sum(plds.C' * exp.(plds.C * x[1, :] + plds.d), dims=2)) + (plds.A' * inv_Q * (x[2, :] - (plds.A * x[1, :] + plds.b[1, :]))) - (inv_p0 * (x[1, :] - plds.x0))
    # calculate grad of the rest of the observations
    for t in 2:T-1
        grad[t, :] = ((y[t, :]' * plds.C)' - sum(plds.C' * exp.(plds.C * x[t, :] + plds.D * s[t, :] + plds.d), dims=2)) - (inv_Q * (x[t, :] - (plds.A * x[t-1, :] + plds.b[t, :]))) + (plds.A' * inv_Q * (x[t+1, :] - (plds.A * x[t, :] + plds.b[t+1, :])))
    end
    # calculate grad of the last observation
    grad[T, :] = ((y[T, :]' * plds.C)' - sum(plds.C' * exp.(plds.C * x[T, :] + plds.D * s[T, :] + plds.d), dims=2)) - (inv_Q * (x[T, :] - (plds.A * x[T-1, :] + plds.b[T, :])))
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
    # pre-compute a few things
    T = size(y, 1)
    inv_Q = pinv(plds.Q)
    inv_p0 = pinv(plds.p0)
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
        λ = exp.(plds.C * x[t, :] + plds.D * s[t, :] + plds.d)
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
    T, D, K = size(μ)
    # Initialize sufficient statistics
    E_z = zeros(T, D, K)
    E_zz = zeros(T, D, D, K)
    E_zz_prev = zeros(T, D, D, K)
    # Compute sufficient statistics
    Threads.@threads for k in 1:K
        for t in 1:T
            E_z[t, :, k] = μ[t, :, k]
            E_zz[t, :, :, k] = P[t, :, :, k] + (μ[t, :, k] * μ[t, :, k]')
            if t > 1
                E_zz_prev[t, :, :, k] = (P_tt1[t, :, :, k]') + (μ[t, :, k] * μ[t-1, :, k]')
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
        plds.x0 = vec(sum(E_z[1, :, :], dims=2)) / size(E_z, 3)
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
        num_trials = size(E_zz, 4)
        # create a new array of covariance matrices for each trial
        p0 = zeros(size(E_zz, 2), size(E_zz, 2))
        # calculate the covariance matrix for each trial
        for n in 1:num_trials
            p0 += E_zz[1, :, :, n] - (E_z[1, :, n] * E_z[1, :, n]')
        end
        # sum the covariance matrices and divide by the number of trials
        plds.p0 = p0 ./ num_trials
    end
end

function calculate_A(E_zz::Array{<:Real}, E_zz_prev::Array{<:Real})
    # update the transition matrix
    return dropdims(sum(E_zz_prev[2:end, :, :], dims=1), dims=1) * pinv(dropdims(sum(E_zz[1:end-1, :, :], dims=1), dims=1))
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
        for n in 1:size(E_zz, 4)
            A += calculate_A(E_zz[:, :, :, n], E_zz_prev[:, :, :, n])
        end
        # Average the estimates
        plds.A = A / size(E_zz, 4)
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
        Q = zeros(plds.latent_dim, plds.latent_dim, size(E_zz, 4))
        for n in 1:size(E_zz, 4)
            Q[:, :, n] = update_Q!(plds, E_zz[:, :, :, n], E_zz_prev[:, :, :, n])
        end
        # Average the estimates
        plds.Q = dropdims(sum(Q, dims=3), dims=3) / size(Q, 3)
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
        for n in 1:size(x_smooth, 3)
            for t in 2:size(x_smooth, 1)
                b[t, :, n] = x_smooth[t, :, n] - (plds.A * x_smooth[t-1, :, n])
            end
        end
        # sum the latent state inputs and divide by the number of trials
        plds.b = dropdims(sum(b, dims=3), dims=3) / size(b, 3)
    end
end

function update_observation_model!(plds::PoissonLDS, x_smooth::Array{<:Real}, y::Array{<:Real})
    # update the observation model parameters: C, D, and d
    if plds.fit_bool[6]
        # flatten the parameters so we can pass them to the optimizer as a single vector
        params = vcat(vec(plds.C), vec(plds.D), plds.d)
        # create a helper function that takes a vector of the observation model parameters
        function f(params::Vector{<:Real}, x_smooth::Array{<:Real}, y::Array{<:Real})
            # reshape the parameters
            C = reshape(params[1:plds.obs_dim * plds.latent_dim], plds.obs_dim, plds.latent_dim)
            D = reshape(params[plds.obs_dim * plds.latent_dim + 1:plds.obs_dim * plds.latent_dim + plds.obs_dim * plds.obs_dim], plds.obs_dim, plds.obs_dim)
            d = params[end-plds.obs_dim+1:end]
            # create a PLDS object with the new parameters
            plds_new = PoissonLDS(A=plds.A, C=C, Q=plds.Q, D=D, d=d, x0=plds.x0, p0=plds.p0, refractory_period=plds.refractory_period, obs_dim=plds.obs_dim, latent_dim=plds.latent_dim, fit_bool=plds.fit_bool)
            # calcualte the loglikelihood of the new model
            ll = loglikelihood(x_smooth, plds_new, y)
            return -ll
        end
        # optimize
        result = optimize(params -> f(params, x_smooth, y), params, LBFGS(), autodiff=:forward)
        # update the parameters
        plds.C = reshape(result.minimizer[1:plds.obs_dim * plds.latent_dim], plds.obs_dim, plds.latent_dim)
        plds.D = reshape(result.minimizer[plds.obs_dim * plds.latent_dim + 1:plds.obs_dim * plds.latent_dim + plds.obs_dim * plds.obs_dim], plds.obs_dim, plds.obs_dim)
        plds.d = result.minimizer[end-plds.obs_dim+1:end]
    end
end

function M_Step!(plds::PoissonLDS, E_z::Array{<:Real}, E_zz::Array{<:Real}, E_zz_prev::Array{<:Real}, x_smooth::Array{<:Real}, y::Array{<:Real})
    # update the parameters
    update_initial_state_mean!(plds, E_z)
    update_initial_state_covariance!(plds, E_zz, E_z)
    update_b!(plds, x_smooth) # needs to be updated before A
    update_A_plds!(plds, E_zz, E_zz_prev)
    update_Q_plds!(plds, E_zz, E_zz_prev)
    update_observation_model!(plds, x_smooth, y)
end

function fit!(plds::PoissonLDS, y::Array{<:Real}, max_iter::Int=1000, tol::Float64=1e-6)
    # create a variable to store the log-likelihood
    ec_lls = []
    ll_prev = -Inf
    # iterate through the EM algorithm
    for i in 1:max_iter
        # initialize the latent states
        x_smooth, p_smooth, p_tt1 = smooth(plds, y)
        # calculate the sufficient statistics
        E_z, E_zz, E_zz_prev = sufficient_statistics(x_smooth, p_smooth, p_tt1)
        # M-step
        M_Step!(plds, E_z, E_zz, E_zz_prev, x_smooth, y)
        # E-step
        E_z, E_zz, E_zz_prev, x_smooth, p_smooth = E_Step(plds, y)
        # calculate the log-likelihood
        ll = loglikelihood(x_smooth, plds, y)
        push!(ec_lls, ll)
        println("Iteration: ", i, " Log-likelihood: ", ll)
        # check for convergence
        if abs(ll - ll_prev) < tol
            println("Converged at iteration ", i)
            return ec_lls
        end
        ll_prev = ll
    end
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
    # Count the number of spikes from 2 to T
    for t in 2:T
        start_idx = max(1, t-window)
        s[t, :] = sum(y[start_idx:t-1, :], dims=1)
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
    # Pre-allocate arrays
    x = zeros(T, plds.latent_dim, K)
    y = zeros(T, plds.obs_dim, K)
    
    for k in 1:K
        # Sample the initial stated
        x[1, :, k] = rand(MvNormal(plds.x0, plds.p0))
        y[1, :, k] = rand.(Poisson.(exp.(plds.C * x[1, :, k] + plds.d)))
        # Sample the rest of the states
        for t in 2:T
            s = max(1, t - plds.refractory_period)
            spikes = sum(y[s:t-1, :, k], dims=1)'
            x[t, :, k] = rand(MvNormal((plds.A * x[t-1, :, k]) + plds.b[t, :], plds.Q))
            y[t, :, k] = rand.(Poisson.(exp.((plds.C * x[t, :, k]) + (plds.D * spikes) + plds.d)))
        end
    end
    return x, y
end


