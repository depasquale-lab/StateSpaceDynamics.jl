export GaussianLDS, PoissonLDS, sample, smooth


"""
    GaussianStateModel{T<:Real} <: AbstractStateModel

Represents the state model of a Linear Dynamical System with Gaussian noise.

# Fields
- `A::Matrix{T}`: Transition matrix
- `Q::Matrix{T}`: Process noise covariance
- `x0::Vector{T}`: Initial state
- `P0::Matrix{T}`: Initial state covariance
"""
mutable struct GaussianStateModel{T<:Real} <: AbstractStateModel{T}
    A::Matrix{T}
    Q::Matrix{T}
    x0::Vector{T}
    P0::Matrix{T}
end

"""
    GaussianStateModel(; A, Q, x0, P0, latent_dim)

Construct a GaussianStateModel with the given parameters or random initializations.

# Arguments
- `A::Matrix{T}=Matrix{T}(undef, 0, 0)`: Transition matrix
- `Q::Matrix{T}=Matrix{T}(undef, 0, 0)`: Process noise covariance
- `x0::Vector{T}=Vector{T}(undef, 0)`: Initial state
- `P0::Matrix{T}=Matrix{T}(undef, 0, 0)`: Initial state covariance
- `latent_dim::Int`: Dimension of the latent state (required if any matrix is not provided)
"""
function GaussianStateModel(; 
    A::Matrix{T}=Matrix{T}(undef, 0, 0),
    Q::Matrix{T}=Matrix{T}(undef, 0, 0),
    x0::Vector{T}=Vector{T}(undef, 0),
    P0::Matrix{T}=Matrix{T}(undef, 0, 0),
    latent_dim::Int=0
) where T<:Real
    if latent_dim == 0 && (isempty(A) || isempty(Q) || isempty(x0) || isempty(P0))
        error("Must provide latent_dim if any matrix is not provided")
    end
    
    A = isempty(A) ? randn(T, latent_dim, latent_dim) : A
    Q = isempty(Q) ? Matrix{T}(I, latent_dim, latent_dim) : Q
    x0 = isempty(x0) ? randn(T, latent_dim) : x0
    P0 = isempty(P0) ? Matrix{T}(I, latent_dim, latent_dim) : P0

    GaussianStateModel{T}(A, Q, x0, P0)
end

"""
    GaussianObservationModel{T<:Real} <: AbstractObservationModel

Represents the observation model of a Linear Dynamical System with Gaussian noise.

# Fields
- `C::Matrix{T}`: Observation matrix
- `R::Matrix{T}`: Observation noise covariance
"""
mutable struct GaussianObservationModel{T<:Real} <: AbstractObservationModel{T}
    C::Matrix{T}
    R::Matrix{T}
end

"""
    GaussianObservationModel(; C, R, obs_dim, latent_dim)

Construct a GaussianObservationModel with the given parameters or random initializations.

# Arguments
- `C::Matrix{T}=Matrix{T}(undef, 0, 0)`: Observation matrix
- `R::Matrix{T}=Matrix{T}(undef, 0, 0)`: Observation noise covariance
- `obs_dim::Int`: Dimension of the observations (required if C or R is not provided)
- `latent_dim::Int`: Dimension of the latent state (required if C is not provided)
"""
function GaussianObservationModel(;
    C::Matrix{T}=Matrix{T}(undef, 0, 0),
    R::Matrix{T}=Matrix{T}(undef, 0, 0),
    obs_dim::Int=0,
    latent_dim::Int=0
) where T<:Real
    if obs_dim == 0 && (isempty(C) || isempty(R))
        error("Must provide obs_dim if C or R is not provided")
    end
    if latent_dim == 0 && isempty(C)
        error("Must provide latent_dim if C is not provided")
    end

    C = isempty(C) ? randn(T, obs_dim, latent_dim) : C
    R = isempty(R) ? Matrix{T}(I, obs_dim, obs_dim) : R

    GaussianObservationModel{T}(C, R)
end

"""
    PoissonObservationModel{T<:Real} <: AbstractObservationModel

Represents the observation model of a Linear Dynamical System with Poisson observations.

# Fields
- `C::Matrix{T}`: Observation matrix
- `log_d::Vector{T}`: Mean firing rate vector (log space)
"""
mutable struct PoissonObservationModel{T<:Real} <: AbstractObservationModel{T}
    C::Matrix{T}
    log_d::Vector{T}
end

"""
    PoissonObservationModel(; C, log_d, obs_dim, latent_dim)

Construct a PoissonObservationModel with the given parameters or random initializations.

# Arguments
- `C::Matrix{T}=Matrix{T}(undef, 0, 0)`: Observation matrix
- `log_d::Vector{T}=Vector{T}(undef, 0)`: Mean firing rate vector (log space)
- `obs_dim::Int`: Dimension of the observations (required if any matrix is not provided)
- `latent_dim::Int`: Dimension of the latent state (required if C is not provided)
"""
function PoissonObservationModel(;
    C::Matrix{T}=Matrix{T}(undef, 0, 0),
    log_d::Vector{T}=Vector{T}(undef, 0),
    obs_dim::Int=0,
    latent_dim::Int=0
) where T<:Real
    if obs_dim == 0 && (isempty(C) || isempty(D) || isempty(log_d))
        error("Must provide obs_dim if any matrix is not provided")
    end
    if latent_dim == 0 && isempty(C)
        error("Must provide latent_dim if C is not provided")
    end

    C = isempty(C) ? randn(T, obs_dim, latent_dim) : C
    log_d = isempty(log_d) ? randn(T, obs_dim) : log_d

    PoissonObservationModel{T}(C, log_d)
end

"""
    LinearDynamicalSystem{S<:AbstractStateModel, O<:AbstractObservationModel}

Represents a unified Linear Dynamical System with customizable state and observation models.

# Fields
- `state_model::S`: The state model (e.g., GaussianStateModel)
- `obs_model::O`: The observation model (e.g., GaussianObservationModel or PoissonObservationModel)
- `latent_dim::Int`: Dimension of the latent state
- `obs_dim::Int`: Dimension of the observations
- `fit_bool::Vector{Bool}`: Vector indicating which parameters to fit during optimization
"""
struct LinearDynamicalSystem{S<:AbstractStateModel, O<:AbstractObservationModel}
    state_model::S
    obs_model::O
    latent_dim::Int
    obs_dim::Int
    fit_bool::Vector{Bool}
end

"""
    GaussianLDS(; A, C, Q, R, x0, P0, fit_bool, obs_dim, latent_dim)

Construct a Linear Dynamical System with Gaussian state and observation models.

# Arguments
- `A::Matrix{T}=Matrix{T}(undef, 0, 0)`: Transition matrix
- `C::Matrix{T}=Matrix{T}(undef, 0, 0)`: Observation matrix
- `Q::Matrix{T}=Matrix{T}(undef, 0, 0)`: Process noise covariance
- `R::Matrix{T}=Matrix{T}(undef, 0, 0)`: Observation noise covariance
- `x0::Vector{T}=Vector{T}(undef, 0)`: Initial state
- `P0::Matrix{T}=Matrix{T}(undef, 0, 0)`: Initial state covariance
- `fit_bool::Vector{Bool}=fill(true, 6)`: Vector indicating which parameters to fit during optimization
- `obs_dim::Int`: Dimension of the observations (required if C or R is not provided)
- `latent_dim::Int`: Dimension of the latent state (required if A, Q, x0, P0, or C is not provided)
"""
function GaussianLDS(;
    A::Matrix{T}=Matrix{Float64}(undef, 0, 0),
    C::Matrix{T}=Matrix{Float64}(undef, 0, 0),
    Q::Matrix{T}=Matrix{Float64}(undef, 0, 0),
    R::Matrix{T}=Matrix{Float64}(undef, 0, 0),
    x0::Vector{T}=Vector{Float64}(undef, 0),
    P0::Matrix{T}=Matrix{Float64}(undef, 0, 0),
    fit_bool::Vector{Bool}=fill(true, 6),
    obs_dim::Int=0,
    latent_dim::Int=0
) where T<:Real
    if latent_dim == 0 && (isempty(A) || isempty(Q) || isempty(x0) || isempty(P0) || isempty(C))
        error("Must provide latent_dim if A, Q, x0, P0, or C is not provided")
    end
    if obs_dim == 0 && (isempty(C) || isempty(R))
        error("Must provide obs_dim if C or R is not provided")
    end

    state_model = GaussianStateModel(A=A, Q=Q, x0=x0, P0=P0, latent_dim=latent_dim)
    obs_model = GaussianObservationModel(C=C, R=R, obs_dim=obs_dim, latent_dim=latent_dim)
    LinearDynamicalSystem(state_model, obs_model, latent_dim, obs_dim, fit_bool)
end


"""
    sample(lds::LinearDynamicalSystem{S,O}, T_steps::Int, n_trials::Int) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Sample from a Linear Dynamical System (LDS) model for multiple trials.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System model.
- `T_steps::Int`: The number of time steps to sample for each trial.
- `n_trials::Int`: The number of trials to sample.=

# Returns
- `x::Array{T, 3}`: The latent state variables. Dimensions: (n_trials, T_steps, latent_dim)
- `y::Array{T, 3}`: The observed data. Dimensions: (n_trials, T_steps, obs_dim)

# Examples
```julia
lds = GaussianLDS(obs_dim=4, latent_dim=3)
x, y = sample(lds, 10, 100)  # 10 trials, 100 time steps each
```
"""
function sample(lds::LinearDynamicalSystem{S,O}, T_steps::Int, n_trials::Int) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}
    # Extract model components
    A, Q, x0, P0 = lds.state_model.A, lds.state_model.Q, lds.state_model.x0, lds.state_model.P0
    C, R = lds.obs_model.C, lds.obs_model.R

    # Initialize arrays
    x = Array{T, 3}(undef, n_trials, T_steps, lds.latent_dim)
    y = Array{T, 3}(undef, n_trials, T_steps, lds.obs_dim)

    # Sample for each trial
    for trial in 1:n_trials
        # Sample the initial state
        x[trial, 1, :] = rand(MvNormal(x0, P0))
        y[trial, 1, :] = rand(MvNormal(C * x[trial, 1, :], R))

        # Sample the rest of the states
        for t in 2:T_steps
            x[trial, t, :] = rand(MvNormal(A * x[trial, t-1, :], Q))
            y[trial, t, :] = rand(MvNormal(C * x[trial, t, :], R))
        end
    end

    return x, y
end

"""
    loglikelihood(x::AbstractMatrix{T}, lds::LinearDynamicalSystem{S,O}, y::AbstractMatrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Calculate the complete-data log-likelihood of a linear dynamical system (LDS) given the observed data.

# Arguments
- `x::AbstractMatrix{T}`: The state sequence of the LDS.
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System.
- `y::AbstractMatrix{T}`: The observed data.

# Returns
- `ll::T`: The complete-data log-likelihood of the LDS.
"""
function loglikelihood(x::AbstractMatrix{T}, lds::LinearDynamicalSystem{S,O}, y::AbstractMatrix{U}) where {T<:Real, U<:Real, S<:GaussianStateModel{<:Real}, O<:GaussianObservationModel{<:Real}}
    T_steps = size(y, 1)
   
    # Extract model components
    A, Q, x0, P0 = lds.state_model.A, lds.state_model.Q, lds.state_model.x0, lds.state_model.P0
    C, R = lds.obs_model.C, lds.obs_model.R
   
    # Pre-compute inverses
    inv_R = pinv(R)
    inv_Q = pinv(Q)
    inv_P0 = pinv(P0)
   
    # Initialize log-likelihood with initial state contribution
    dx0 = x[1, :] - x0
    ll = dx0' * inv_P0 * dx0
   
    # State transitions and observations
    for t in 1:T_steps
        if t > 1
            dx = x[t, :] - A * x[t-1, :]
            ll += dx' * inv_Q * dx
        end
        dy = y[t, :] - C * x[t, :]
        ll += dy' * inv_R * dy
    end
   
    return -0.5 * ll
end


"""
    Gradient(lds::LinearDynamicalSystem{S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Compute the gradient of the log-likelihood with respect to the latent states for a linear dynamical system.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System.
- `y::AbstractMatrix{T}`: The observed data.
- `x::AbstractMatrix{T}`: The latent states.

# Returns
- `grad::Matrix{T}`: Gradient of the log-likelihood with respect to the latent states.
"""
function Gradient(lds::LinearDynamicalSystem{S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}
    T_steps, _ = size(y)
   
    # Extract model components
    A, Q, x0, P0 = lds.state_model.A, lds.state_model.Q, lds.state_model.x0, lds.state_model.P0
    C, R = lds.obs_model.C, lds.obs_model.R
   
    # Pre-compute inverses
    inv_R = pinv(R)
    inv_Q = pinv(Q)
    inv_P0 = pinv(P0)
   
    # Initialize gradient
    grad = zeros(T_steps, lds.latent_dim)
   
    # Compute gradient
    grad[1, :] = A' * inv_Q * (x[2, :] - A * x[1, :]) + C' * inv_R * (y[1, :] - C * x[1, :]) - inv_P0 * (x[1, :] - x0)
   
    Threads.@threads for t in 2:T_steps-1
        grad[t, :] = C' * inv_R * (y[t, :] - C * x[t, :]) -
                     inv_Q * (x[t, :] - A * x[t-1, :]) +
                     A' * inv_Q * (x[t+1, :] - A * x[t, :])
    end
   
    grad[T_steps, :] = C' * inv_R * (y[T_steps, :] - C * x[T_steps, :]) -
                       inv_Q * (x[T_steps, :] - A * x[T_steps-1, :])
   
    return grad
end

"""
    Hessian(lds::LinearDynamicalSystem{S,O}, y::AbstractMatrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Construct the Hessian matrix of the log-likelihood of the LDS model given a set of observations.

This function is used for the direct optimization of the log-likelihood as advocated by Paninski et al. (2009). 
The block tridiagonal structure of the Hessian is exploited to reduce the number of parameters that need to be computed,
and to reduce the memory requirements. Together with the gradient, this allows for Kalman Smoothing to be performed 
by simply solving a linear system of equations:

    ̂xₙ₊₁ = ̂xₙ - H \\ ∇

where ̂xₙ is the current smoothed state estimate, H is the Hessian matrix, and ∇ is the gradient of the log-likelihood.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System.
- `y::AbstractMatrix{T}`: Matrix of observations.

# Returns
- `H::Matrix{T}`: Hessian matrix of the log-likelihood.
- `H_diag::Vector{Matrix{T}}`: Main diagonal blocks of the Hessian.
- `H_super::Vector{Matrix{T}}`: Super-diagonal blocks of the Hessian.
- `H_sub::Vector{Matrix{T}}`: Sub-diagonal blocks of the Hessian.
"""
function Hessian(lds::LinearDynamicalSystem{S,O}, y::AbstractMatrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}
    # Extract model components
    A, Q, x0, P0 = lds.state_model.A, lds.state_model.Q, lds.state_model.x0, lds.state_model.P0
    C, R = lds.obs_model.C, lds.obs_model.R

    # Precompute results
    T_steps, _ = size(y)
    inv_R = pinv(R)
    inv_Q = pinv(Q)
    inv_P0 = pinv(P0)

    # Super and sub diagonals
    H_sub_entry = inv_Q * A
    H_super_entry = Matrix(H_sub_entry')
    H_sub = Vector{Matrix{T}}(undef, T_steps-1)
    H_super = Vector{Matrix{T}}(undef, T_steps-1)

    Threads.@threads for i in 1:T_steps-1
        H_sub[i] = H_sub_entry
        H_super[i] = H_super_entry
    end

    # Main diagonal components
    yt_given_xt = -C' * inv_R * C
    xt_given_xt_1 = -inv_Q
    xt1_given_xt = -A' * inv_Q * A
    x_t = -inv_P0

    # Construct main diagonal
    H_diag = Vector{Matrix{T}}(undef, T_steps)
    
    Threads.@threads for i in 2:T_steps-1
        H_diag[i] = yt_given_xt + xt_given_xt_1 + xt1_given_xt
    end

    # Edge cases
    H_diag[1] = yt_given_xt + xt1_given_xt + x_t
    H_diag[T_steps] = yt_given_xt + xt_given_xt_1

    # Construct full Hessian
    H = block_tridgm(H_diag, H_super, H_sub)

    return H, H_diag, H_super, H_sub
end


"""
    smooth(lds::LinearDynamicalSystem{S,O}, y::Matrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

This function performs direct smoothing for a linear dynamical system (LDS) given the system parameters and the observed data.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The LDS object representing the system parameters.
- `y::Matrix{T}`: The observed data matrix.

# Returns
- `x::Matrix{T}`: The optimal state estimate.
- `p_smooth::Array{T, 3}`: The posterior covariance matrix.
- `inverse_offdiag::Array{T, 3}`: The inverse off-diagonal matrix.
- `Q_val::T`: The Q-function value.

# Example
```julia
lds = GaussianLDS(obs_dim=4, latent_dim=3)
y = randn(100, 4)  # 100 time steps, 4 observed dimensions
x, p_smooth, inverse_offdiag, Q_val = DirectSmoother(lds, y)
```
"""
function smooth(lds::LinearDynamicalSystem{S,O}, y::Matrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}
    # Get the length of Y and latent dimension
    time_steps, _ = size(y)
    D = lds.latent_dim

    # Create starting point for the optimization
    X₀ = zeros(T, time_steps * D)

    # Create wrappers for the loglikelihood, gradient, and hessian
    function nll(vec_x::Vector{T})
        x = SSM.interleave_reshape(vec_x, time_steps, D)
        return -loglikelihood(x, lds, y)
    end

    function g!(g::Vector{T}, vec_x::Vector{T})
        x = SSM.interleave_reshape(vec_x, time_steps, D)
        grad = Gradient(lds, y, x)
        g .= vec(permutedims(-grad))
    end

    function h!(h::Matrix{T}, vec_x::Vector{T})
        x = SSM.interleave_reshape(vec_x, time_steps, D)
        H, *, *, _ = Hessian(lds, y)
        h .= -H
    end

    # Set up the optimization problem
    res = optimize(nll, g!, h!, X₀, Newton(), Optim.Options(g_tol=1e-14))

    # get the optimal state
    x = SSM.interleave_reshape(res.minimizer, time_steps, D)

    # Get covariances and nearest-neighbor second moments
    H, main, super, sub = Hessian(lds, y)
    p_smooth, inverse_offdiag = block_tridiagonal_inverse(-sub, -main, -super)

    # Enforce symmetry of p_smooth
    for i in 1:time_steps
        p_smooth[i, :, :] = 0.5 * (p_smooth[i, :, :] + p_smooth[i, :, :]')
    end

    # Concatenate a zero matrix to the inverse off diagonal
    inverse_offdiag = cat(dims=1, zeros(T, 1, D, D), inverse_offdiag)

    return x, p_smooth, inverse_offdiag
end


"""
    smooth(lds::LinearDynamicalSystem{S,O}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

This function performs direct smoothing for a linear dynamical system (LDS) given the system parameters and the observed data for multiple trials.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The LDS object representing the system parameters.
- `y::Array{T,3}`: The observed data array with dimensions (n_trials, time_steps, obs_dim).

# Returns
- `x::Array{T,3}`: The optimal state estimates with dimensions (n_trials, time_steps, latent_dim).
- `p_smooth::Array{T,4}`: The posterior covariance matrices with dimensions (n_trials, time_steps, latent_dim, latent_dim).
- `inverse_offdiag::Array{T,4}`: The inverse off-diagonal matrices with dimensions (n_trials, time_steps, latent_dim, latent_dim).

# Example
```julia
lds = GaussianLDS(obs_dim=4, latent_dim=3)
y = randn(5, 100, 4)  # 5 trials, 100 time steps, 4 observed dimensions
x, p_smooth, inverse_offdiag = smooth(lds, y)
```
"""
function smooth(lds::LinearDynamicalSystem{S,O}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}
    n_trials, time_steps, _ = size(y)
    latent_dim = lds.latent_dim

    # Initialize arrays to store results for all trials
    x_all = Array{T,3}(undef, n_trials, time_steps, latent_dim)
    p_smooth_all = Array{T,4}(undef, n_trials, time_steps, latent_dim, latent_dim)
    inverse_offdiag_all = Array{T,4}(undef, n_trials, time_steps, latent_dim, latent_dim)

    # Process each trial
    for trial in 1:n_trials
        # Extract data for the current trial
        y_trial = y[trial, :, :]

        # Call the single-trial smooth function
        x, p_smooth, inverse_offdiag = smooth(lds, y_trial)

        # Store results for this trial
        x_all[trial, :, :] = x
        p_smooth_all[trial, :, :, :] = p_smooth
        inverse_offdiag_all[trial, :, :, :] = inverse_offdiag
    end

    return x_all, p_smooth_all, inverse_offdiag_all
end

"""
    Q_state(A, Q, P0, x0, E_z, E_zz, E_zz_prev)

Calculate the state component of the Q-function for the EM algorithm in a Linear Dynamical System.

# Arguments
- `A::Matrix{<:Real}`: The state transition matrix.
- `Q::AbstractMatrix{<:Real}`: The process noise covariance matrix (or its Cholesky factor).
- `P0::AbstractMatrix{<:Real}`: The initial state covariance matrix (or its Cholesky factor).
- `x0::Vector{<:Real}`: The initial state mean.
- `E_z::Matrix{<:Real}`: The expected latent states, size (T, state_dim).
- `E_zz::Array{<:Real, 3}`: The expected value of z_t * z_t', size (T, state_dim, state_dim).
- `E_zz_prev::Array{<:Real, 3}`: The expected value of z_t * z_{t-1}', size (T, state_dim, state_dim).

# Returns
- `Q_val::Float64`: The state component of the Q-function.

# Note
- If Q and P0 are provided as Cholesky factors, they will be converted to full matrices.
"""
function Q_state(A::Matrix{<:Real}, Q::AbstractMatrix{<:Real}, P0::AbstractMatrix{<:Real}, 
    x0::Vector{<:Real}, E_z::Matrix{<:Real}, E_zz::Array{<:Real, 3}, E_zz_prev::Array{<:Real, 3})

    # Calculate the inverses
    Q_inv = pinv(Q)
    P0_inv = pinv(P0)

    # Get dimensions
    T_step = size(E_z, 1)

    # Initialize Q_val
    Q_val = 0.0

    # Calculate the Q-function for the first time step
    Q_val += -0.5 * (logdet(P0) + tr(P0_inv * (E_zz[1, :, :] - (E_z[1, :] * x0') - (x0 * E_z[1, :]') + (x0 * x0'))))

    # Calculate the Q-function for the state model
    for t in 2:T_step
        term1 = E_zz[t, :, :]
        term2 = A * E_zz_prev[t, :, :]'
        term3 = E_zz_prev[t, :, :] * A'
        term4 = A * E_zz[t-1, :, :] * A'
        Q_val += -0.5 * (logdet(Q) + tr(Q_inv * (term1 - term2 - term3 + term4)))
    end

    return Q_val
end

"""
    Q_obs(H, R, E_z, E_zz, y)

Calculate the observation component of the Q-function for the EM algorithm in a Linear Dynamical System.

# Arguments
- `H::Matrix{<:Real}`: The observation matrix.
- `R::AbstractMatrix{<:Real}`: The observation noise covariance matrix (or its Cholesky factor).
- `E_z::Matrix{<:Real}`: The expected latent states, size (T, state_dim).
- `E_zz::Array{<:Real, 3}`: The expected value of z_t * z_t', size (T, state_dim, state_dim).
- `y::Matrix{<:Real}`: The observed data, size (T, obs_dim).

# Returns
- `Q_val::Float64`: The observation component of the Q-function.

# Note
- If R is provided as a Cholesky factor, it will be converted to a full matrix.
"""
function Q_obs(H::Matrix{<:Real}, R::AbstractMatrix{<:Real}, E_z::Matrix{<:Real}, 
    E_zz::Array{<:Real, 3}, y::Matrix{<:Real})

    # Calculate the inverse
    R_inv = pinv(R)

    # Get dimensions
    obs_dim = size(H, 1)
    T_step = size(E_z, 1)

    # Initialize Q_val
    Q_val = 0.0

    # Calculate the Q-function for the observation model
    for t in 1:T_step
        term1 = y[t, :] * y[t, :]'
        term2 = H * (E_z[t, :] * y[t, :]')
        term3 = (y[t, :] * E_z[t, :]') * H'
        term4 = H * E_zz[t, :, :] * H'
        Q_val += -0.5 * (obs_dim * log(2π) + logdet(R) + tr(R_inv * (term1 - term2 - term3 + term4)))
    end

    return Q_val
end

"""
    Q(A, Q, H, R, P0, x0, E_z, E_zz, E_zz_prev, y)

Calculate the complete Q-function for the EM algorithm in a Linear Dynamical System.

# Arguments
- `A::Matrix{<:Real}`: The state transition matrix.
- `Q::AbstractMatrix{<:Real}`: The process noise covariance matrix (or its Cholesky factor).
- `H::Matrix{<:Real}`: The observation matrix.
- `R::AbstractMatrix{<:Real}`: The observation noise covariance matrix (or its Cholesky factor).
- `P0::AbstractMatrix{<:Real}`: The initial state covariance matrix (or its Cholesky factor).
- `x0::Vector{<:Real}`: The initial state mean.
- `E_z::Matrix{<:Real}`: The expected latent states, size (T, state_dim).
- `E_zz::Array{<:Real, 3}`: The expected value of z_t * z_t', size (T, state_dim, state_dim).
- `E_zz_prev::Array{<:Real, 3}`: The expected value of z_t * z_{t-1}', size (T, state_dim, state_dim).
- `y::Matrix{<:Real}`: The observed data, size (T, obs_dim).

# Returns
- `Q_val::Float64`: The complete Q-function value.
"""
function Q_function(A::Matrix{<:Real}, Q::AbstractMatrix{<:Real}, H::Matrix{<:Real}, R::AbstractMatrix{<:Real}, 
           P0::AbstractMatrix{<:Real}, x0::Vector{<:Real}, E_z::Matrix{<:Real}, E_zz::Array{<:Real, 3}, 
           E_zz_prev::Array{<:Real, 3}, y::Matrix{<:Real})
    Q_val_state = Q_state(A, Q, P0, x0, E_z, E_zz, E_zz_prev)
    Q_val_obs = Q_obs(H, R, E_z, E_zz, y)
    return Q_val_state + Q_val_obs
end

"""
    Q(lds::LinearDynamicalSystem{S,O}, E_z, E_zz, E_zz_prev, y) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Calculate the Q-function for the EM algorithm using a LinearDynamicalSystem struct.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct containing model parameters.
- `E_z::Matrix{T}`: The expected latent states, size (T_steps, latent_dim).
- `E_zz::Array{T, 3}`: The expected value of z_t * z_t', size (T_steps, latent_dim, latent_dim).
- `E_zz_prev::Array{T, 3}`: The expected value of z_t * z_{t-1}', size (T_steps, latent_dim, latent_dim).
- `y::Matrix{T}`: The observed data, size (T_steps, obs_dim).

# Returns
- `Q_val::T`: The complete Q-function value.

# Note
- This function assumes that the covariance matrices in the LDS struct are full matrices and converts them to Cholesky factors before calculation.
- The type parameter T is inferred from the input types.
"""
function Q_function(lds::LinearDynamicalSystem{S,O}, E_z::Matrix{T}, E_zz::Array{T, 3}, E_zz_prev::Array{T, 3}, y::Matrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}
    # Extract model components
    A, Q = lds.state_model.A, lds.state_model.Q
    H, R = lds.obs_model.C, lds.obs_model.R
    x0, P0 = lds.state_model.x0, lds.state_model.P0
    
    # Re-parameterize the covariance matrices of the LDS model
    Q_chol = Matrix(cholesky(Q).L)
    R_chol = Matrix(cholesky(R).L)
    P0_chol = Matrix(cholesky(P0).L)
    
    # Calculate the Q-function
    Q_val_state = Q_state(A, Q_chol, P0_chol, x0, E_z, E_zz, E_zz_prev)
    Q_val_obs = Q_obs(H, R_chol, E_z, E_zz, y)
    
    return Q_val_state + Q_val_obs
end


"""
    sufficient_statistics(x_smooth::Array{T,3}, p_smooth::Array{T,4}, p_smooth_t1::Array{T,4}) where T <: Real

Compute sufficient statistics for the EM algorithm in a Linear Dynamical System.

# Arguments
- `x_smooth::Array{T,3}`: Smoothed state estimates, size (n_trials, T_steps, state_dim)
- `p_smooth::Array{T,4}`: Smoothed state covariances, size (n_trials, T_steps, state_dim, state_dim)
- `p_smooth_t1::Array{T,4}`: Lag-one covariance smoother, size (n_trials, T_steps, state_dim, state_dim)

# Returns
- `E_z::Array{T,3}`: Expected latent states, size (n_trials, T_steps, state_dim)
- `E_zz::Array{T,4}`: Expected z_t * z_t', size (n_trials, T_steps, state_dim, state_dim)
- `E_zz_prev::Array{T,4}`: Expected z_t * z_{t-1}', size (n_trials, T_steps, state_dim, state_dim)

# Note
- The function computes the expected values for all trials.
- For single-trial data, use inputs with n_trials = 1.
"""
function sufficient_statistics(x_smooth::Array{T,3}, p_smooth::Array{T,4}, p_smooth_t1::Array{T,4}) where T <: Real
    n_trials, T_steps, _ = size(x_smooth)
    
    E_z = copy(x_smooth)
    E_zz = similar(p_smooth)
    E_zz_prev = similar(p_smooth)

    for trial in 1:n_trials
        for t in 1:T_steps
            E_zz[trial, t, :, :] = p_smooth[trial, t, :, :] + x_smooth[trial, t, :] * x_smooth[trial, t, :]'
            if t > 1
                E_zz_prev[trial, t, :, :] = p_smooth_t1[trial, t, :, :] + x_smooth[trial, t, :] * x_smooth[trial, t-1, :]'
            end
        end
        # Set the first time step of E_zz_prev to zeros for each trial
        E_zz_prev[trial, 1, :, :] .= 0
    end

    return E_z, E_zz, E_zz_prev
end

"""
    estep(lds::LinearDynamicalSystem{S,O}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}

Perform the E-step of the EM algorithm for a Linear Dynamical System, treating all input as multi-trial.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `y::Array{T,3}`: Observed data, size (n_trials, T_steps, obs_dim)
    Note: For single-trial data, use y[1:1, :, :] to create a 3D array with n_trials = 1

# Returns
- `E_z::Array{T,3}`: Expected latent states, size (n_trials, T_steps, state_dim)
- `E_zz::Array{T,4}`: Expected z_t * z_t', size (n_trials, T_steps, state_dim, state_dim)
- `E_zz_prev::Array{T,4}`: Expected z_t * z_{t-1}', size (n_trials, T_steps, state_dim, state_dim)
- `x_smooth::Array{T,3}`: Smoothed state estimates, size (n_trials, T_steps, state_dim)
- `p_smooth::Array{T,4}`: Smoothed state covariances, size (n_trials, T_steps, state_dim, state_dim)
- `ml::T`: Total marginal likelihood (log-likelihood) of the data across all trials

# Note
- This function first smooths the data using the `smooth` function, then computes sufficient statistics.
- It treats all input as multi-trial, with single-trial being a special case where n_trials = 1.
"""
function estep(lds::LinearDynamicalSystem{S,O}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}
    # Smooth the data
    x_smooth, p_smooth, inverse_offdiag = smooth(lds, y)

    # Compute sufficient statistics
    E_z, E_zz, E_zz_prev = sufficient_statistics(x_smooth, p_smooth, inverse_offdiag)

    # Calculate the total marginal likelihood across all trials
    ml_total = calculate_elbo(lds, E_z, E_zz, E_zz_prev, p_smooth, y)

    return E_z, E_zz, E_zz_prev, x_smooth, p_smooth, ml_total
end

"""
    calculate_elbo(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, E_zz_prev::Array{T,4}, p_smooth::Array{T,4}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}

Calculate the Evidence Lower Bound (ELBO) for a Linear Dynamical System.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `E_z::Array{T,3}`: Expected latent states, size (n_trials, T_steps, state_dim)
- `E_zz::Array{T,4}`: Expected z_t * z_t', size (n_trials, T_steps, state_dim, state_dim)
- `E_zz_prev::Array{T,4}`: Expected z_t * z_{t-1}', size (n_trials, T_steps, state_dim, state_dim)
- `p_smooth::Array{T,4}`: Smoothed state covariances, size (n_trials, T_steps, state_dim, state_dim)
- `y::Array{T,3}`: Observed data, size (n_trials, T_steps, obs_dim)

# Returns
- `elbo::T`: The Evidence Lower Bound (ELBO) for the LDS.

# Note
- For a GaussianLDS the ELBO is equivalent to the total marginal likelihood
"""
function calculate_elbo(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, E_zz_prev::Array{T,4}, p_smooth::Array{T,4}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}
    A, Q, x0, P0 = lds.state_model.A, lds.state_model.Q, lds.state_model.x0, lds.state_model.P0
    C, R = lds.obs_model.C, lds.obs_model.R
    
    n_trials = size(y, 1)
    elbo = 0.0

    for trial in 1:n_trials
        Q_val = SSM.Q_function(A, Q, C, R, P0, x0, E_z[trial,:,:], E_zz[trial,:,:,:], E_zz_prev[trial,:,:,:], y[trial,:,:])
        
        # Calculate the entropy of q(z) for this trial
        entropy = sum(gaussianentropy(Matrix(p_smooth[trial,t,:,:])) for t in axes(p_smooth, 2))
        elbo += Q_val - entropy
    end

    return elbo
end

"""
    update_initial_state_mean!(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}

Update the initial state mean of the Linear Dynamical System using the average across all trials.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `E_z::Array{T,3}`: Expected latent states, size (n_trials, T_steps, state_dim)

# Note
- This function modifies `lds` in-place.
- The update is only performed if `lds.fit_bool[1]` is true.
- The initial state mean is computed as the average of the first time step across all trials.
"""
function update_initial_state_mean!(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}
    if lds.fit_bool[1]
        x0_new = zeros(lds.latent_dim)
        for i in axes(E_z, 1)
            x0_new += E_z[i, 1, :]
        end
        x0_new = x0_new ./ size(E_z, 1)
        lds.state_model.x0 .= x0_new
    end
end

"""
    update_initial_state_covariance!(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}

Update the initial state covariance of the Linear Dynamical System using the average across all trials.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `E_z::Array{T,3}`: Expected latent states, size (n_trials, T_steps, state_dim)
- `E_zz::Array{T,4}`: Expected z_t * z_t', size (n_trials, T_steps, state_dim, state_dim)

# Note
- This function modifies `lds` in-place.
- The update is only performed if `lds.fit_bool[2]` is true.
- The initial state covariance is computed as the average of the first time step across all trials.
"""
function update_initial_state_covariance!(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}
    if lds.fit_bool[2]
        # Dims and pre-allocate
        n_trials = size(E_z, 1)
        state_dim = size(E_z, 3)
        p0_new = zeros(T, state_dim, state_dim)

        # Calculate the sum of E_zz[1,:,:] across all trials
        for trial in 1:n_trials
            p0_new += E_zz[trial, 1, :, :] - (lds.state_model.x0 * lds.state_model.x0')
        end

        # Average across all trials
        p0_new = p0_new ./ n_trials
        
        # Enforce symmetry
        lds.state_model.P0 = 0.5 * (p0_new + p0_new')
    end
end

"""
    update_A!(lds::LinearDynamicalSystem{S,O}, E_zz::Array{T, 4}, E_zz_prev::Array{T, 4}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}

Update the transition matrix A of the Linear Dynamical System.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `E_zz::Array{T, 4}`: Expected z_t * z_t', size (n_trials, T_steps, state_dim, state_dim)
- `E_zz_prev::Array{T, 4}`: Expected z_t * z_{t-1}', size (n_trials, T_steps, state_dim, state_dim)

# Note
- This function modifies `lds` in-place.
- The update is only performed if `lds.fit_bool[3]` is true.
"""
function update_A!(lds::LinearDynamicalSystem{S,O}, E_zz::Array{T, 4}, E_zz_prev::Array{T, 4}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}
    if lds.fit_bool[3]
        _, _, state_dim, _ = size(E_zz)
        
        # Calculate and sum A across trials
        E_zz_sum = zeros(state_dim, state_dim)
        E_zz_prev_sum = zeros(state_dim, state_dim)

        for trial in axes(E_zz, 1)
            E_zz_sum += dropdims(sum(E_zz[trial, 1:end-1, :, :], dims=1), dims=1)
            E_zz_prev_sum += dropdims(sum(E_zz_prev[trial, :, :, :], dims=1), dims=1)
        end

        # Average across trials
        lds.state_model.A = E_zz_prev_sum / E_zz_sum

    end
end

"""
    update_Q!(lds::LinearDynamicalSystem{S,O}, E_zz::Array{T, 4}, E_zz_prev::Array{T, 4}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}

Update the process noise covariance matrix Q of the Linear Dynamical System.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `E_zz::Array{T, 4}`: Expected z_t * z_t', size (n_trials, T_steps, state_dim, state_dim)
- `E_zz_prev::Array{T, 4}`: Expected z_t * z_{t-1}', size (n_trials, T_steps, state_dim, state_dim)

# Note
- This function modifies `lds` in-place.
- The update is only performed if `lds.fit_bool[4]` is true.
- The result is averaged across all trials.
"""
function update_Q!(lds::LinearDynamicalSystem{S,O}, E_zz::Array{T, 4}, E_zz_prev::Array{T, 4}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}
    if lds.fit_bool[4]
        n_trials, T_steps, state_dim, _ = size(E_zz)
        
        # Initialize Q_new
        Q_new = zeros(T, state_dim, state_dim)
        
        for trial in 1:n_trials
            sum_expectations = zeros(T, state_dim, state_dim)
            for t in 2:T_steps
                sum_expectations += E_zz[trial, t, :, :] - 
                                    (E_zz_prev[trial, t, :, :] * lds.state_model.A') - 
                                    (lds.state_model.A * E_zz_prev[trial, t, :, :]') + 
                                    (lds.state_model.A * E_zz[trial, t-1, :, :] * lds.state_model.A')
            end
            Q_new += sum_expectations
        end
        
        # Average across trials
        Q_new = Q_new / (n_trials * (T_steps - 1))
        
        # Ensure symmetry
        lds.state_model.Q = 0.5 * (Q_new + Q_new')
    end
end

"""
    update_C!(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Update the observation matrix C of the Linear Dynamical System.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `E_z::Array{T,3}`: Expected latent states, size (n_trials, T_steps, state_dim)
- `E_zz::Array{T,4}`: Expected z_t * z_t', size (n_trials, T_steps, state_dim, state_dim)
- `y::Array{T,3}`: Observed data, size (n_trials, T_steps, obs_dim)

# Note
- This function modifies `lds` in-place.
- The update is only performed if `lds.fit_bool[5]` is true.
- The result is averaged across all trials.
"""
function update_C!(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}
    if lds.fit_bool[5]
        n_trials, T_steps, _ = size(y)
        
        sum_yz = zeros(T, size(lds.obs_model.C))
        sum_zz = zeros(T, size(E_zz)[3:4])
        
        for trial in 1:n_trials
            for t in 1:T_steps
                sum_yz += y[trial, t, :] * E_z[trial, t, :]'
                sum_zz += E_zz[trial, t, :, :]
            end
        end
        
        lds.obs_model.C = sum_yz / sum_zz
    end
end


"""
    update_R!(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Update the observation noise covariance matrix R of the Linear Dynamical System.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `E_z::Array{T,3}`: Expected latent states, size (n_trials, T_steps, state_dim)
- `E_zz::Array{T,4}`: Expected z_t * z_t', size (n_trials, T_steps, state_dim, state_dim)
- `y::Array{T,3}`: Observed data, size (n_trials, T_steps, obs_dim)

# Note
- This function modifies `lds` in-place.
- The update is only performed if `lds.fit_bool[6]` is true.
- The result is averaged across all trials.
"""
function update_R!(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}
    if lds.fit_bool[6]
        n_trials, T_steps, obs_dim = size(y)
        
        R_new = zeros(T, obs_dim, obs_dim)
        
        for trial in 1:n_trials
            for t in 1:T_steps
                yt = y[trial, t, :]
                zt = E_z[trial, t, :]
                Zt = E_zz[trial, t, :, :]
                
                R_new += (yt * yt') - (lds.obs_model.C * (yt * zt')') - 
                         ((yt * zt') * lds.obs_model.C') + 
                         (lds.obs_model.C * Zt * lds.obs_model.C')
            end
        end
        
        # Average across all trials and time steps
        R_new = R_new / (n_trials * T_steps)
        
        # Ensure symmetry
        lds.obs_model.R = 0.5 * (R_new + R_new')
    end
end

"""
    mstep!(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, E_zz_prev::Array{T,4}, p_smooth::Array{T, 4}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Perform the M-step of the EM algorithm for a Linear Dynamical System with multi-trial data.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `E_z::Array{T,3}`: Expected latent states, size (n_trials, T_steps, state_dim)
- `E_zz::Array{T,4}`: Expected z_t * z_t', size (n_trials, T_steps, state_dim, state_dim)
- `E_zz_prev::Array{T,4}`: Expected z_t * z_{t-1}', size (n_trials, T_steps, state_dim, state_dim)
- `p_smooth::Array{T,4}`: Smoothed state covariances, size (n_trials, T_steps, state_dim, state_dim) (not used)
- `y::Array{T,3}`: Observed data, size (n_trials, T_steps, obs_dim)

# Note
- This function modifies `lds` in-place by updating all model parameters.
- Updates are performed only for parameters where the corresponding `fit_bool` is true.
- All update functions now handle multi-trial data.
- P_smooth is required but not used in the M-step so that the function signature matches the PoissonLDS version.
"""
function mstep!(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, E_zz_prev::Array{T,4}, p_smooth::Array{T, 4}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}
    # Update parameters
    update_initial_state_mean!(lds, E_z)
    update_initial_state_covariance!(lds, E_z, E_zz)
    update_A!(lds, E_zz, E_zz_prev)
    update_Q!(lds, E_zz, E_zz_prev)
    update_C!(lds, E_z, E_zz, y)
    update_R!(lds, E_z, E_zz, y)
end


"""
    fit!(lds::LinearDynamicalSystem{S,O}, y::Matrix{T}; 
         max_iter::Int=1000, 
         tol::Real=1e-12, 
         smoother::SmoothingMethod=RTSSmoothing()) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Fit a Linear Dynamical System using the Expectation-Maximization (EM) algorithm with Kalman smoothing.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System to be fitted.
- `y::Matrix{T}`: Observed data, size (T_steps, obs_dim).

# Keyword Arguments
- `max_iter::Int=1000`: Maximum number of EM iterations.
- `tol::Real=1e-12`: Convergence tolerance for log-likelihood change.
- `smoother::SmoothingMethod=RTSSmoothing()`: The smoothing method to use in the E-step.

# Returns
- `mls::Vector{T}`: Vector of log-likelihood values for each iteration.

# Note
- This function modifies `lds` in-place.
- The function stops when the change in log-likelihood is less than `tol` or `max_iter` is reached.
- A progress bar is displayed during the fitting process.
"""
function fit!(lds::LinearDynamicalSystem{S,O}, y::Array{T, 3}; 
              max_iter::Int=1000, 
              tol::Real=1e-12, 
              ) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}
    
    # Initialize log-likelihood
    prev_ml = -T(Inf)
    
    # Create a vector to store the log-likelihood values
    mls = Vector{T}()
    sizehint!(mls, max_iter)  # Pre-allocate for efficiency
    
    # Initialize progress bar
    if O <: GaussianObservationModel
        prog = Progress(max_iter; desc="Fitting LDS via EM...", output=stderr)
    elseif O <: PoissonObservationModel
        prog = Progress(max_iter; desc="Fitting Poisson LDS via LaplaceEM...", output=stderr)
    else 
        error("Unknown LDS model type")
    end
    
    # Run EM
    for i in 1:max_iter
        # E-step
        E_z, E_zz, E_zz_prev, x_smooth, p_smooth, ml = estep(lds, y)
        
        # M-step
        mstep!(lds, E_z, E_zz, E_zz_prev, p_smooth, y)
        
        # Update the log-likelihood vector
        push!(mls, ml)
        
        # Update the progress bar
        next!(prog)
        
        # Check convergence
        if abs(ml - prev_ml) < tol
            finish!(prog)
            return mls
        end
        
        prev_ml = ml
    end
    
    # Finish the progress bar if max_iter is reached
    finish!(prog)
    
    return mls
end


"""
    PoissonLDS(; A, C, Q, log_d, x0, P0, refractory_period, fit_bool, obs_dim, latent_dim)

Construct a Linear Dynamical System with Gaussian state and Poisson observation models.

# Arguments
- `A::Matrix{T}=Matrix{T}(undef, 0, 0)`: Transition matrix
- `C::Matrix{T}=Matrix{T}(undef, 0, 0)`: Observation matrix
- `Q::Matrix{T}=Matrix{T}(undef, 0, 0)`: Process noise covariance
- `log_d::Vector{T}=Vector{T}(undef, 0)`: Mean firing rate vector (log space)
- `x0::Vector{T}=Vector{T}(undef, 0)`: Initial state
- `P0::Matrix{T}=Matrix{T}(undef, 0, 0)`: Initial state covariance
- `refractory_period::Int=1`: Refractory period
- `fit_bool::Vector{Bool}=fill(true, 7)`: Vector indicating which parameters to fit during optimization
- `obs_dim::Int`: Dimension of the observations (required if C, D, or log_d is not provided)
- `latent_dim::Int`: Dimension of the latent state (required if A, Q, x0, P0, or C is not provided)
"""
function PoissonLDS(;
    A::Matrix{T}=Matrix{Float64}(undef, 0, 0),
    C::Matrix{T}=Matrix{Float64}(undef, 0, 0),
    Q::Matrix{T}=Matrix{Float64}(undef, 0, 0),
    log_d::Vector{T}=Vector{Float64}(undef, 0),
    x0::Vector{T}=Vector{Float64}(undef, 0),
    P0::Matrix{T}=Matrix{Float64}(undef, 0, 0),
    fit_bool::Vector{Bool}=fill(true, 6),
    obs_dim::Int=0,
    latent_dim::Int=0
) where T<:Real
    if latent_dim == 0 && (isempty(A) || isempty(Q) || isempty(x0) || isempty(P0) || isempty(C))
        error("Must provide latent_dim if A, Q, x0, P0, or C is not provided")
    end
    if obs_dim == 0 && (isempty(C) || isempty(log_d))
        error("Must provide obs_dim if C, or log_d is not provided")
    end

    state_model = GaussianStateModel(A=A, Q=Q, x0=x0, P0=P0, latent_dim=latent_dim)
    obs_model = PoissonObservationModel(C=C, log_d=log_d, obs_dim=obs_dim, latent_dim=latent_dim)
    LinearDynamicalSystem(state_model, obs_model, latent_dim, obs_dim, fit_bool)
end

"""
    sample(lds::LinearDynamicalSystem{S,O}, T_steps::Int, n_trials::Int) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Sample from a Poisson Linear Dynamical System (LDS) model for multiple trials.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System model.
- `T_steps::Int`: The number of time steps to sample for each trial.
- `n_trials::Int`: The number of trials to sample.

# Returns
- `x::Array{T, 3}`: The latent state variables. Dimensions: (n_trials, T_steps, latent_dim)
- `y::Array{Int, 3}`: The observed data. Dimensions: (n_trials, T_steps, obs_dim)

# Examples
```julia
lds = LinearDynamicalSystem(obs_dim=4, latent_dim=3)
x, y = sample(lds, 100, 10)  # 10 trials, 100 time steps each
```
"""
function sample(lds::LinearDynamicalSystem{S,O}, T_steps::Int, n_trials::Int) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}
    # Extract model components
    A, Q = lds.state_model.A, lds.state_model.Q
    C, log_d = lds.obs_model.C, lds.obs_model.log_d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    # Convert log_d to d i.e. non-log space
    d = exp.(log_d)

    # Pre-allocate arrays
    x = zeros(T, n_trials, T_steps, lds.latent_dim)
    y = zeros(T, n_trials, T_steps, lds.obs_dim)
   
    for k in 1:n_trials
        # Sample the initial state
        x[k, 1, :] = rand(MvNormal(x0, P0))
        y[k, 1, :] = rand.(Poisson.(exp.(C * x[k, 1, :] .+ d)))

        # Sample the rest of the states
        for t in 2:T_steps
            x[k, t, :] = rand(MvNormal(A * x[k, t-1, :], Q))
            y[k, t, :] = rand.(Poisson.(exp.(C * x[k, t, :] + d)))
        end
    end

    return x, y
end

"""
    loglikelihood(x::Matrix{T}, lds::LinearDynamicalSystem{S,O}, y::Matrix{T}) where {T<:Real, S<:GaussianStateModel, O<:PoissonObservationModel}

Calculate the complete-data log-likelihood of a Poisson Linear Dynamical System model for a single trial. 

# Arguments
- `x::Matrix{T}`: The latent state variables. Dimensions: (T_steps, latent_dim)
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System model.
- `y::Matrix{T}`: The observed data. Dimensions: (T_steps, obs_dim)

# Returns
- `ll::T`: The log-likelihood value.

# Examples
```juliaestep!
lds = PoissonLDS(obs_dim=4, latent_dim=3)
x, y = sample(lds, 100, 1)  # 1 trial, 100 time steps
ll = loglikelihood(x, lds, y)
```
"""
function loglikelihood(x::AbstractMatrix{T}, plds::LinearDynamicalSystem{S,O}, y::AbstractMatrix{U}) where {U<:Real, T<:Real, S<:GaussianStateModel{<:Real}, O<:PoissonObservationModel{<:Real}}
    # Convert the log firing rate to firing rate
    d = exp.(plds.obs_model.log_d)
    T_steps = size(y, 1)

    # Pre-compute matrix inverses
    inv_p0 = pinv(plds.state_model.P0)
    inv_Q = pinv(plds.state_model.Q)

    # Calculate p(yₜ|xₜ)
    pygivenx_sum = zero(T)
    for t in 1:T_steps
        temp = plds.obs_model.C * x[t, :] .+ d
        pygivenx_sum += dot(y[t, :], temp) - sum(exp.(temp))
    end

    # Calculate p(x₁)
    dx1 = x[1, :] .- plds.state_model.x0
    px1 = -T(0.5) * dot(dx1, inv_p0 * dx1)

    # Calculate p(xₜ|xₜ₋₁)
    pxtgivenxt1_sum = zero(T)
    for t in 2:T_steps
        temp = x[t, :] .- (plds.state_model.A * x[t-1, :])
        pxtgivenxt1_sum += -T(0.5) * dot(temp, inv_Q * temp)
    end

    # Return the log-posterior
    return pygivenx_sum + px1 + pxtgivenxt1_sum
end


"""
    loglikelihood(x::Array{T, 3}, lds::LinearDynamicalSystem{S,O}, y::Array{T, 3}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Calculate the complete-data log-likelihood of a Poisson Linear Dynamical System model for multiple trials.

# Arguments
- `x::Array{T, 3}`: The latent state variables. Dimensions: (n_trials, T_steps, latent_dim)
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System model.
- `y::Array{T, 3}`: The observed data. Dimensions: (n_trials, T_steps, obs_dim)

# Returns
- `ll::T`: The log-likelihood value.

# Examples
```julia
lds = PoissonLDS(obs_dim=4, latent_dim=3)
x, y = sample(lds, 100, 10)  # 10 trials, 100 time steps each
ll = loglikelihood(x, lds, y)
```
"""
function loglikelihood(x::Array{T, 3}, plds::LinearDynamicalSystem{O, S}, y::Array{T, 3}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}
    # Calculate the log-likelihood over all trials
    ll = zeros(size(y, 1))
    Threads.@threads for n in axes(y, 1)
        ll[n] = loglikelihood(x[n, :, :], plds, y[n, :, :])
    end
    return sum(ll)
end

"""
    Gradient(lds::LinearDynamicalSystem{S,O}, y::Matrix{T}, x::Matrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Calculate the gradient of the log-likelihood of a Poisson Linear Dynamical System model for a single trial.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System model.
- `y::Matrix{T}`: The observed data. Dimensions: (T_steps, obs_dim)
- `x::Matrix{T}`: The latent state variables. Dimensions: (T_steps, latent_dim)

# Returns
- `grad::Matrix{T}`: The gradient of the log-likelihood. Dimensions: (T_steps, latent_dim)

# Note
The gradient is computed with respect to the latent states x. Each row of the returned gradient
corresponds to the gradient for a single time step.
"""
function Gradient(lds::LinearDynamicalSystem{S,O}, y::Matrix{T}, x::Matrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}
    # Extract model parameters
    A, Q = lds.state_model.A, lds.state_model.Q
    C, log_d = lds.obs_model.C, lds.obs_model.log_d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    # Convert log_d to d (non-log space)
    d = exp.(log_d)

    # Get number of time steps
    T_steps = size(y, 1)

    # Precompute matrix inverses
    inv_P0 = pinv(P0)
    inv_Q = pinv(Q)

    # Pre-allocate gradient
    grad = zeros(T_steps, lds.latent_dim)

    # Calculate gradient for each time step
    for t in 1:T_steps
        # Common term for all time steps
        common_term = (y[t, :]' * C)' - C' * exp.(C * x[t, :] .+ d)

        if t == 1
            # First time step
            grad[t, :] = common_term + A' * inv_Q * (x[2, :] - A * x[1, :]) - inv_P0 * (x[1, :] - x0)
        elseif t == T_steps
            # Last time step
            grad[t, :] = common_term - inv_Q * (x[t, :] - A * x[t-1, :])
        else
            # Intermediate time steps
            grad[t, :] = common_term + A' * inv_Q * (x[t+1, :] - A * x[t, :]) - inv_Q * (x[t, :] - A * x[t-1, :])
        end
    end

    return grad
end

"""
    Hessian(lds::LinearDynamicalSystem{S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Calculate the Hessian matrix of the log-likelihood for a Poisson Linear Dynamical System.

This function computes the Hessian matrix, which represents the second-order partial derivatives
of the log-likelihood with respect to the latent states. The Hessian is useful for optimization
algorithms and for computing the Fisher information matrix.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System with Poisson observations.
- `y::AbstractMatrix{T}`: The observed data. Dimensions: (T_steps, obs_dim)
- `x::AbstractMatrix{T}`: The current estimate of latent states. Dimensions: (T_steps, latent_dim)

# Returns
- `H::Matrix{T}`: The full Hessian matrix.
- `H_diag::Vector{Matrix{T}}`: The main diagonal blocks of the Hessian.
- `H_super::Vector{Matrix{T}}`: The super-diagonal blocks of the Hessian.
- `H_sub::Vector{Matrix{T}}`: The sub-diagonal blocks of the Hessian.

# Note
The function exploits the block tridiagonal structure of the Hessian for efficiency.
The returned full Hessian `H` and its components can be used for various purposes,
such as Newton's method in optimization or for computing confidence intervals.

"""
function Hessian(lds::LinearDynamicalSystem{S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}
    # Extract model components
    A, Q = lds.state_model.A, lds.state_model.Q
    C, log_d = lds.obs_model.C, lds.obs_model.log_d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    # Convert log_d to d i.e. non-log space
    d = exp.(log_d)

    # Pre-compute a few things
    T_steps, _ = size(y)
    inv_Q = pinv(Q)
    inv_P0 = pinv(P0)

    # Calculate super and sub diagonals
    H_sub_entry = inv_Q * A
    H_super_entry = Matrix(H_sub_entry')

    H_sub = Vector{typeof(H_sub_entry)}(undef, T_steps-1)
    H_super = Vector{typeof(H_super_entry)}(undef, T_steps-1)

    Threads.@threads for i in 1:T_steps-1
        H_sub[i] = H_sub_entry
        H_super[i] = H_super_entry
    end

    # Pre-compute common terms
    xt_given_xt_1 = -inv_Q
    xt1_given_xt = -A' * inv_Q * A
    x_t = -inv_P0

    # Helper function to calculate the Poisson Hessian term
    function calculate_poisson_hess(C::Matrix{T}, λ::Vector{T}) where T<:Real
        hess = zeros(T, size(C, 2), size(C, 2))
        for i in 1:size(C, 1)
            hess -= λ[i] * C[i, :] * C[i, :]'
        end
        return hess
    end

    # Calculate the main diagonal
    H_diag = Vector{Matrix{T}}(undef, T_steps)

    Threads.@threads for t in 1:T_steps
        λ = exp.(C * x[t, :] .+ d)
        if t == 1
            H_diag[t] = x_t + xt1_given_xt + calculate_poisson_hess(C, λ)
        elseif t == T_steps
            H_diag[t] = xt_given_xt_1 + calculate_poisson_hess(C, λ)
        else
            H_diag[t] = xt_given_xt_1 + xt1_given_xt + calculate_poisson_hess(C, λ)
        end
    end

    # Construct full Hessian
    H = block_tridgm(H_diag, H_super, H_sub)

    return H, H_diag, H_super, H_sub
end

"""
    Q_state(A::Matrix{T}, Q::Matrix{T}, P0::Matrix{T}, x0::Vector{T}, E_z::Array{T, 3}, E_zz::Array{T, 4}, E_zz_prev::Array{T, 4}) where T<:Real

Calculates the Q-function for the state model over multiple trials.

# Arguments
- `A::Matrix{T}`: The transition matrix.
- `Q::Matrix{T}`: The process noise covariance matrix.
- `P0::Matrix{T}`: The initial state covariance matrix.
- `x0::Vector{T}`: The initial state mean.
- `E_z::Array{T, 3}`: The expected latent states.
- `E_zz::Array{T, 4}`: The expected latent states x the latent states.
- `E_zz_prev::Array{T, 4}`: The expected latent states x the previous latent states.

# Returns
- `Float64`: The Q-function for the state model.
"""
function Q_state(A::Matrix{T}, Q::Matrix{T}, P0::Matrix{T}, x0::Vector{T}, E_z::Array{T, 3}, E_zz::Array{T, 4}, E_zz_prev::Array{T, 4}) where T<:Real
    # Calculate the Q-function for the state model
    Q_val = 0.0

    # Calcualte over trials
    for k in axes(E_z, 1)
        Q_val += Q_state(A, Q, P0, x0, E_z[k, :, :], E_zz[k, :, :, :], E_zz_prev[k, :, :, :])
    end

    return Q_val
end

"""
    Q_observation_model(C::Matrix{<:Real}, D::Matrix{<:Real}, log_d::Vector{<:Real}, E_z::Array{<:Real}, E_zz::Array{<:Real}, y::Array{<:Real})

Calculate the Q-function for the observation model.

# Arguments
- `C::Matrix{<:Real}`: The observation matrix.
- `log_d::Vector{<:Real}`: The mean firing rate vector in log space.
- `E_z::Array{<:Real}`: The expected latent states.
- `E_zz::Array{<:Real}`: The expected latent states x the latent states.
- `y::Array{<:Real}`: The observed data.

# Returns
- `Float64`: The Q-function for the observation model.
"""
function Q_observation_model(C::AbstractMatrix{T}, log_d::AbstractVector{T}, E_z::AbstractArray{U, 3}, P_smooth::AbstractArray{U, 4}, y::Array{U, 3}) where {T<:Real, U<:Real}
    # Re-parametrize log_d
    d = exp.(log_d)
    
    # Compute Q
    Q_val = zero(T)
    trials = size(E_z, 1)
    time_steps = size(E_z, 2)
    
    # calculate CC term
    CC = zeros(T, size(C, 1), size(C, 2)^2)
    for i in axes(C, 1)
        CC[i, :] .= vec(C[i, :] * C[i, :]')
    end
    
    # sum over trials
    for k in 1:trials
        # sum over time-points
        for t in 1:time_steps
            # Mean term
            h = (C * E_z[k, t, :]) .+ d
            
            # calculate rho
            ρ = T(0.5) .* CC * vec(P_smooth[k, t, :, :])
            
            ŷ = exp.(h .+ ρ)
            
            # calculate the Q-value
            Q_val += sum((y[k, t, :] .* h) .- ŷ)
        end
    end
    
    return Q_val
end

"""
    Q_function(A::Matrix{T}, Q::Matrix{T}, C::Matrix{T}, log_d::Vector{T}, x0::Vector{T}, P0::Matrix{T}, E_z::Matrix{T}, E_zz::Array{T, 3}, E_zz_prev::Array{T, 3}, P_smooth::Array{T, 3}, y::Matrix{T})

Calculate the Q-function for the Linear Dynamical System.

# Arguments
- `A::Matrix{T}`: The transition matrix.
- `Q::Matrix{T}`: The process noise covariance matrix.
- `C::Matrix{T}`: The observation matrix.
- `log_d::Vector{T}`: The mean firing rate vector in log space.
- `x0::Vector{T}`: The initial state mean.
- `P0::Matrix{T}`: The initial state covariance matrix.
- `E_z::Matrix{T}`: The expected latent states.
- `E_zz::Array{T, 3}`: The expected latent states x the latent states.
- `E_zz_prev::Array{T, 3}`: The expected latent states x the previous latent states.
- `P_smooth::Array{T, 3}`: The smoothed state covariances.
- `y::Matrix{T}`: The observed data.

# Returns
- `Float64`: The Q-function for the Linear Dynamical System.
"""
function Q_function(A::Matrix{T}, Q::Matrix{T}, C::Matrix{T}, log_d::Vector{T}, x0::Vector{T}, P0::Matrix{T}, E_z::Array{T, 3}, E_zz::Array{T, 4}, E_zz_prev::Array{T, 4}, P_smooth::Array{T, 4}, y::Array{T, 3}) where T<:Real
    # Calculate the Q-function for the state model
    Q_state = SSM.Q_state(A, Q, P0, x0, E_z, E_zz, E_zz_prev)
    # Calculate the Q-function for the observation model
    Q_obs = Q_observation_model(C, log_d, E_z, P_smooth, y)
    return Q_state + Q_obs
end

"""
    calculate_elbo(plds::LinearDynamicalSystem{S,O}, E_z::Array{T, 3}, E_zz::Array{T, 4}, 
                   E_zz_prev::Array{T, 4}, P_smooth::Array{T, 4}, y::Array{T, 3}) where 
                   {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Calculate the Evidence Lower Bound (ELBO) for a Poisson Linear Dynamical System (PLDS).

# Arguments
- `plds::LinearDynamicalSystem{S,O}`: The PLDS model.
- `E_z::Array{T, 3}`: Expected values of latent states. Dimensions: (trials, time steps, state dimension).
- `E_zz::Array{T, 4}`: Expected values of latent state outer products. Dimensions: (trials, time steps, state dimension, state dimension).
- `E_zz_prev::Array{T, 4}`: Expected values of latent state outer products with previous time step. Dimensions: (trials, time steps-1, state dimension, state dimension).
- `P_smooth::Array{T, 4}`: Smoothed covariance matrices. Dimensions: (trials, time steps, state dimension, state dimension).
- `y::Array{T, 3}`: Observed data. Dimensions: (trials, time steps, observation dimension).

# Returns
- `elbo::Float64`: The calculated Evidence Lower Bound.

# Description
This function computes the ELBO for a PLDS model, which consists of two main components:
1. The expected complete log-likelihood (ECLL), calculated using the Q_function.
2. The entropy of the variational distribution, calculated using gaussian entropy.

The ELBO is then computed as: ELBO = ECLL - Entropy.

# Note
Ensure that the dimensions of input arrays match the expected dimensions as described in the arguments section.
"""
function calculate_elbo(plds::LinearDynamicalSystem{S,O}, E_z::Array{T, 3}, E_zz::Array{T, 4}, 
                        E_zz_prev::Array{T, 4}, P_smooth::Array{T, 4}, y::Array{T, 3}) where 
                        {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}
    # Set up parameters
    A, Q, x0, p0 = plds.state_model.A, plds.state_model.Q, plds.state_model.x0, plds.state_model.P0
    C, log_d = plds.obs_model.C, plds.obs_model.log_d

    # Calculate the expected complete log-likelihood
    ecll = Q_function(A, Q, C, log_d, x0, p0, E_z, E_zz, E_zz_prev, P_smooth, y)

    # Calculate the entropy across all trials and timepoints
    entropy = 0.0
    for trial in 1:size(E_z, 1)
        for tStep in 1:size(E_z, 2)
            cov_matrix = P_smooth[trial, tStep, :, :] # Extract covariance matrix
            entropy += SSM.gaussianentropy(cov_matrix)
        end
    end

    # Return the ELBO
    return ecll - entropy
end

"""
    smooth(lds::LinearDynamicalSystem{S,O}, y::Matrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Perform smoothing for a Poisson Linear Dynamical System (LDS).

# Arguments

- `lds::LinearDynamicalSystem{S,O}`: The Poisson Linear Dynamical System model.
- `y::Matrix{T}`: The observed data. Dimensions: (T_steps, obs_dim)

# Returns
- `x::Matrix{T}`: The smoothed state estimates. Dimensions: (T_steps, latent_dim)
- `p_smooth::Array{T, 3}`: The smoothed state covariances. Dimensions: (T_steps, latent_dim, latent_dim)
- `inverse_offdiag::Array{T, 3}`: The inverse off-diagonal matrices. Dimensions: (T_steps, latent_dim, latent_dim)

# Note
This function uses Newton's method for optimization and computes the Hessian
and gradient specifically for the Poisson observation model.
"""
function smooth(lds::LinearDynamicalSystem{S,O}, y::Matrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}
    # Get the length of Y and latent dimension
    time_steps, _ = size(y)
    D = lds.latent_dim

    # Create starting point for the optimization
    X₀ = zeros(T, time_steps * D)

    # Create wrappers for the loglikelihood, gradient, and hessian
    function nll(vec_x::Vector{T})
        x = SSM.interleave_reshape(vec_x, time_steps, D)
        return -loglikelihood(x, lds, y)
    end

    function g!(g::Vector{T}, vec_x::Vector{T})
        x = SSM.interleave_reshape(vec_x, time_steps, D)
        grad = Gradient(lds, y, x)
        g .= vec(permutedims(-grad))
    end

    function h!(h::Matrix{T}, vec_x::Vector{T})
        x = SSM.interleave_reshape(vec_x, time_steps, D)
        H, _, _, _ = Hessian(lds, y, x)
        h .= -H
    end

    # Set up the optimization problem
    res = optimize(nll, g!, h!, X₀, Newton())

    # Get the optimal state
    x = SSM.interleave_reshape(res.minimizer, time_steps, D)

    # Get covariances and nearest-neighbor second moments
    H, main, super, sub = Hessian(lds, y, x)
    p_smooth, inverse_offdiag = block_tridiagonal_inverse(-sub, -main, -super)

    # Enforce symmetry of p_smooth
    for i in 1:time_steps
        p_smooth[i, :, :] = 0.5 * (p_smooth[i, :, :] + p_smooth[i, :, :]')
    end

    # Concatenate a zero matrix to the inverse off diagonal
    inverse_offdiag = cat(dims=1, zeros(T, 1, D, D), inverse_offdiag)

    return x, p_smooth, inverse_offdiag
end

"""
    smooth(lds::LinearDynamicalSystem{S,O}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Smooth the latent states for each trial of a Poisson Linear Dynamical System (PLDS) model.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Poisson Linear Dynamical System model.
- `y::Array{T,3}`: The observed data for each trial. Dimensions: (n_trials, T_steps, obs_dim)

# Returns
- `x_smooth::Array{T,3}`: The smoothed latent states for each trial. Dimensions: (n_trials, T_steps, latent_dim)
- `p_smooth::Array{T,4}`: The smoothed covariance matrices for each trial. Dimensions: (n_trials, T_steps, latent_dim, latent_dim)
- `p_tt1::Array{T,4}`: The lag-one covariance smoother for each trial. Dimensions: (n_trials, T_steps, latent_dim, latent_dim)

# Note
This function applies smoothing to each trial independently using the `smooth` function for a single trial.
"""
function smooth(lds::LinearDynamicalSystem{S,O}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}
    # Get dimensions
    n_trials, T_steps, obs_dim = size(y)
    latent_dim = lds.latent_dim

    # Pre-allocate arrays for results
    x_smooth = zeros(T, n_trials, T_steps, latent_dim)
    p_smooth = zeros(T, n_trials, T_steps, latent_dim, latent_dim)
    p_tt1 = zeros(T, n_trials, T_steps, latent_dim, latent_dim)

    # Smooth the latent states for each trial
    for k in 1:n_trials
        # Smooth the latent states for a single trial
        x_sm, p_sm, p_prev = smooth(lds, y[k, :, :])
        
        # Store the results
        x_smooth[k, :, :] = x_sm
        p_smooth[k, :, :, :] = p_sm
        p_tt1[k, :, :, :] = p_prev
    end

    return x_smooth, p_smooth, p_tt1
end

"""
    gradient_observation_model!(grad::AbstractVector{T}, C::AbstractMatrix{T}, log_d::AbstractVector{T}, E_z::AbstractArray{T}, P_smooth::AbstractArray{T}, y::Array{T}) where T<:Real

Compute the gradient of the Q-function with respect to the observation model parameters (C and log_d) for a Poisson Linear Dynamical System.

# Arguments
- `grad::AbstractVector{T}`: Pre-allocated vector to store the computed gradient.
- `C::AbstractMatrix{T}`: The observation matrix. Dimensions: (obs_dim, latent_dim)
- `log_d::AbstractVector{T}`: The log of the baseline firing rates. Dimensions: (obs_dim,)
- `E_z::AbstractArray{T}`: The expected latent states. Dimensions: (trials, time_steps, latent_dim)
- `P_smooth::AbstractArray{T}`: The smoothed state covariances. Dimensions: (trials, time_steps, latent_dim, latent_dim)
- `y::Array{T}`: The observed data. Dimensions: (trials, time_steps, obs_dim)

# Note
This function modifies `grad` in-place. The gradient is computed for the negative Q-function,
as we're minimizing -Q in optimization routines.
"""
function gradient_observation_model!(grad::AbstractVector{T}, C::AbstractMatrix{T}, log_d::AbstractVector{T}, E_z::AbstractArray{T}, P_smooth::AbstractArray{T}, y::Array{T}) where T<:Real
    d = exp.(log_d)
    obs_dim, latent_dim = size(C)
    trials, time_steps, _ = size(E_z)

    fill!(grad, zero(T))

    for k in 1:trials
        for t in 1:time_steps
            z_t = E_z[k, t, :]
            P_t = P_smooth[k, t, :, :]
            y_t = y[k, t, :]

            h = C * z_t .+ d
            ρ = T(0.5) .* [sum(C[i, :] .* (P_t * C[i, :])) for i in 1:obs_dim]
            λ = exp.(h .+ ρ)

            # Gradient w.r.t. C (flattened in column-major order)
            for i in 1:obs_dim
                for j in 1:latent_dim
                    idx = (j-1)*obs_dim + i
                    grad[idx] += y_t[i] * z_t[j] - λ[i] * (z_t[j] + sum(C[i, :] .* P_t[:, j]))
                end
            end

            # Gradient w.r.t. log_d (with scaling factor)
            grad[end-obs_dim+1:end] .+= (y_t .- λ) .* d
        end
    end

    # Negate the gradient because we're minimizing -Q
    grad .*= -1
end

"""
    update_observation_model!(plds::LinearDynamicalSystem{S,O}, E_z::Array{T, 3}, P_smooth::Array{T, 4}, y::Array{T, 3}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Update the observation model parameters of a Poisson Linear Dynamical System using gradient-based optimization.

# Arguments
- `plds::LinearDynamicalSystem{S,O}`: The Poisson Linear Dynamical System model.
- `E_z::Array{T, 3}`: The expected latent states. Dimensions: (n_trials, T_steps, latent_dim)
- `P_smooth::Array{T, 4}`: The smoothed state covariances. Dimensions: (n_trials, T_steps, latent_dim, latent_dim)
- `y::Array{T, 3}`: The observed data. Dimensions: (n_trials, T_steps, obs_dim)

# Note
This function modifies `plds` in-place by updating the observation model parameters (C and log_d).
The optimization is performed only if `plds.fit_bool[5]` is true.
"""
function update_observation_model!(plds::LinearDynamicalSystem{S,O}, E_z::Array{T, 3}, P_smooth::Array{T, 4}, y::Array{T, 3}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}
    if plds.fit_bool[5]
        params = vcat(vec(plds.obs_model.C), plds.obs_model.log_d)
        
        function f(params::Vector{T})
            C_size = plds.obs_dim * plds.latent_dim
            log_d = params[end-plds.obs_dim+1:end]
            C = reshape(params[1:C_size], plds.obs_dim, plds.latent_dim)
            return -Q_observation_model(C, log_d, E_z, P_smooth, y)
        end
        
        function g!(grad::Vector{T}, params::Vector{T})
            C_size = plds.obs_dim * plds.latent_dim
            log_d = params[end-plds.obs_dim+1:end]
            C = reshape(params[1:C_size], plds.obs_dim, plds.latent_dim)
            gradient_observation_model!(grad, C, log_d, E_z, P_smooth, y)
        end
        
        result = optimize(f, g!, params, LBFGS())
        
        # Update the parameters
        C_size = plds.obs_dim * plds.latent_dim
        plds.obs_model.C = reshape(result.minimizer[1:C_size], plds.obs_dim, plds.latent_dim)
        plds.obs_model.log_d = result.minimizer[end-plds.obs_dim+1:end]
    end
end

"""
    mstep!(plds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, E_zz_Prev{T,4}, p_smooth{T,4}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Perform the M-step of the EM algorithm for a Poisson Linear Dynamical System with multi-trial data.

# Arguments
- `plds::LinearDynamicalSystem{S,O}`: The Poisson Linear Dynamical System struct.
- `E_z::Array{T,3}`: Expected latent states, size (n_trials, T_steps, state_dim)
- `E_zz::Array{T,4}`: Expected z_t * z_t', size (n_trials, T_steps, state_dim, state_dim)
- `E_zz_prev::Array{T,4}`: Expected z_t * z_{t-1}', size (n_trials, T_steps, state_dim, state_dim)
- `p_smooth::Array{T,4}`: Smoothed state covariances, size (n_trials, T_steps, state_dim, state_dim)
- `y::Array{T,3}`: Observed data, size (n_trials, T_steps, obs_dim)

# Note
- This function modifies `plds` in-place by updating all model parameters.
"""
function mstep!(plds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, E_zz_prev::Array{T,4}, p_smooth::Array{T,4}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}
    # Update parameters
    update_initial_state_mean!(plds, E_z)
    update_initial_state_covariance!(plds, E_z, E_zz)
    update_A!(plds, E_zz, E_zz_prev)
    update_Q!(plds, E_zz, E_zz_prev)
    update_observation_model!(plds, E_z, p_smooth, y)
end
