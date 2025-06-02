export GaussianLDS, PoissonLDS, sample, smooth, fit!
import ForwardDiff

"""
    GaussianStateModel{T<:Real} <: AbstractStateModel

Represents the state model of a Linear Dynamical System with Gaussian noise.

# Fields
- `A::Matrix{T}`: Transition matrix
- `Q::Matrix{T}`: Process noise covariance
- `x0::Vector{T}`: Initial state
- `P0::Matrix{T}`: Initial state covariance
"""
mutable struct GaussianStateModel{T<:Float64} <: AbstractStateModel{T}
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
- `latent_dim::Int`: Dimension of the latent state (required if any matrix is not provided.)
"""
function GaussianStateModel(;
    A::Matrix{T}=Matrix{T}(undef, 0, 0),
    Q::Matrix{T}=Matrix{T}(undef, 0, 0),
    x0::Vector{T}=Vector{T}(undef, 0),
    P0::Matrix{T}=Matrix{T}(undef, 0, 0),
    latent_dim::Int=0,
) where {T<:Float64}
    if latent_dim == 0 && (isempty(A) || isempty(Q) || isempty(x0) || isempty(P0))
        throw(ArgumentError("Must provide latent_dim if any matrix is not provided."))
    end

    A = isempty(A) ? randn(T, latent_dim, latent_dim) : A
    Q = isempty(Q) ? Matrix{T}(I, latent_dim, latent_dim) : Q
    x0 = isempty(x0) ? randn(T, latent_dim) : x0
    P0 = isempty(P0) ? Matrix{T}(I, latent_dim, latent_dim) : P0

    return GaussianStateModel{T}(A, Q, x0, P0)
end

function GaussianStateModel(
    A::AbstractMatrix{T},
    Q::AbstractMatrix{T},
    x0::AbstractVector{T},
    P0::AbstractMatrix{T},
) where {T<:Real}
    A64 = to_f64(A)
    Q64 = to_f64(Q)
    x064 = to_f64(x0)
    P064 = to_f64(P0)
    GaussianStateModel(A64, Q64, x064, P064)
end

"""
    GaussianObservationModel{T<:Real} <: AbstractObservationModel

Represents the observation model of a Linear Dynamical System with Gaussian noise.

# Fields
- `C::Matrix{T}`: Observation matrix
- `R::Matrix{T}`: Observation noise covariance
"""
mutable struct GaussianObservationModel{T<:Float64} <: AbstractObservationModel{T}
    C::Matrix{T}
    R::Matrix{T}
end

"""
    GaussianObservationModel(; C, R, obs_dim, latent_dim)

Construct a GaussianObservationModel with the given parameters or random initializations.

# Arguments
- `C::Matrix{T}=Matrix{T}(undef, 0, 0)`: Observation matrix
- `R::Matrix{T}=Matrix{T}(undef, 0, 0)`: Observation noise covariance
- `obs_dim::Int`: Dimension of the observations (required if C or R is not provided.)
- `latent_dim::Int`: Dimension of the latent state (required if C is not provided.)
"""
function GaussianObservationModel(;
    C::Matrix{T}=Matrix{T}(undef, 0, 0),
    R::Matrix{T}=Matrix{T}(undef, 0, 0),
    obs_dim::Int=0,
    latent_dim::Int=0,
) where {T<:Float64}
    if obs_dim == 0 && (isempty(C) || isempty(R))
        throw(ArgumentError("Must provide obs_dim if C or R is not provided."))
    end
    if latent_dim == 0 && isempty(C)
        throw(ArgumentError("Must provide latent_dim if C is not provided."))
    end

    C = isempty(C) ? randn(T, obs_dim, latent_dim) : C
    R = isempty(R) ? Matrix{T}(I, obs_dim, obs_dim) : R

    return GaussianObservationModel{T}(C, R)
end

function GaussianObservationModel(
    C::AbstractMatrix{T}, 
    R::AbstractMatrix{T}, 
    obs_dim::Int,
    latent_dim::Int, 
) where {T<:Real}
    c64 = to_f64(C)
    R64 = to_f64(R)
    GaussianObservationModel(c64, R64, obs_dim, latent_dim)
end

"""
    PoissonObservationModel{T<:Real} <: AbstractObservationModel

Represents the observation model of a Linear Dynamical System with Poisson observations.

# Fields
- `C::Matrix{T}`: Observation matrix
- `log_d::Vector{T}`: Mean firing rate vector (log space)
"""
mutable struct PoissonObservationModel{T<:Float64} <: AbstractObservationModel{T}
    C::Matrix{T}
    log_d::Vector{T}
end

"""
    PoissonObservationModel(; C, log_d, obs_dim, latent_dim)

Construct a PoissonObservationModel with the given parameters or random initializations.

# Arguments
- `C::Matrix{T}=Matrix{T}(undef, 0, 0)`: Observation matrix
- `log_d::Vector{T}=Vector{T}(undef, 0)`: Mean firing rate vector (log space)
- `obs_dim::Int`: Dimension of the observations (required if any matrix is not provided.)
- `latent_dim::Int`: Dimension of the latent state (required if C is not provided.)
"""
function PoissonObservationModel(; 
    C::Matrix{T}=Matrix{T}(undef,0,0),
    log_d::Vector{T}=Vector{T}(undef,0),
    obs_dim::Int=0,
    latent_dim::Int=0
) where {T<:Float64}
    if obs_dim == 0 && (isempty(C) || isempty(log_d))
        throw(ArgumentError("Must provide obs_dim if C or log_d is not provided."))
    end
    if latent_dim == 0 && isempty(C)
        throw(ArgumentError("Must provide latent_dim if C is not provided."))
    end

    C = isempty(C) ? randn(T, obs_dim, latent_dim) : C
    log_d = isempty(log_d) ? randn(T, obs_dim) : log_d

    return PoissonObservationModel{T}(C, log_d)
end

function PoissonObservationModel(
    C::AbstractMatrix{T}, 
    log_d::AbstractVector{T}, 
    obs_dim::Int,
    latent_dim::Int
) where {T<:Real}
    c64 = to_f64(C)
    log_d64 = to_f64(log_d)
    PoissonObservationModel(c64, log_d64, obs_dim, latent_dim)
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
struct LinearDynamicalSystem{S<:AbstractStateModel,O<:AbstractObservationModel}
    state_model::S
    obs_model::O
    latent_dim::Int
    obs_dim::Int
    fit_bool::Vector{Bool}
end

"""
    stateparams(lds::LinearDynamicalSystem{S,O}) where {S<:AbstractStateModel,O<:AbstractObservationModel}

Extract the state parameters from a Linear Dynamical System.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System.

# Returns
- `params::Vector{Vector{Real}}`: Vector of state parameters.
"""
function stateparams(
    lds::LinearDynamicalSystem{S,O}
) where {S<:AbstractStateModel,O<:AbstractObservationModel}
    if isa(lds.state_model, GaussianStateModel)
        return [
            lds.state_model.A, lds.state_model.Q, lds.state_model.x0, lds.state_model.P0
        ]
    end
end

"""
    obsparams(lds::LinearDynamicalSystem{S,O}) where {S<:AbstractStateModel,O<:AbstractObservationModel}

Extract the observation parameters from a Linear Dynamical System.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System.

# Returns
- `params::Vector{Vector{Real}}`: Vector of observation parameters.
"""
function obsparams(
    lds::LinearDynamicalSystem{S,O}
) where {S<:AbstractStateModel,O<:AbstractObservationModel}
    if isa(lds.obs_model, GaussianObservationModel)
        return [lds.obs_model.C, lds.obs_model.R]
    elseif isa(lds.obs_model, PoissonObservationModel)
        return [lds.obs_model.C, lds.obs_model.log_d]
    end
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
- `obs_dim::Int`: Dimension of the observations (required if C or R is not provided.)
- `latent_dim::Int`: Dimension of the latent state (required if A, Q, x0, P0, or C is not provided.)
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
    latent_dim::Int=0,
) where {T<:Float64}
    if latent_dim == 0 &&
        (isempty(A) || isempty(Q) || isempty(x0) || isempty(P0) || isempty(C))
        throw(ArgumentError("Must provide latent_dim if any matrix is not provided."))
    end
    if obs_dim == 0 && (isempty(C) || isempty(R))
        throw(ArgumentError("Must provide obs_dim if C or R is not provided."))
    end

    state_model = GaussianStateModel(; A=A, Q=Q, x0=x0, P0=P0, latent_dim=latent_dim)
    obs_model = GaussianObservationModel(; C=C, R=R, obs_dim=obs_dim, latent_dim=latent_dim)
    return LinearDynamicalSystem(state_model, obs_model, latent_dim, obs_dim, fit_bool)
end

"""
    initialize_FilterSmooth(model::LinearDynamicalSystem, num_obs::Int) -> FilterSmooth{T}

Initialize a `FilterSmooth` object for a given linear dynamical system model and number of observations.

# Arguments
- `model::LinearDynamicalSystem`:  
  The linear dynamical system model containing system parameters, including the latent dimensionality (`latent_dim`).

- `num_obs::Int`:  
  The number of observations (time steps) for which to initialize the smoothing filters.

# Returns
- `FilterSmooth{T}`:  
  A `FilterSmooth` instance with all fields initialized to zero arrays. The dimensions of the arrays are determined by the number of states (`latent_dim`) from the model and the specified number of observations (`num_obs`).

# Example
```julia
# Assume `model` is an instance of LinearDynamicalSystem with latent_dim = 10
num_observations = 100
filter_smooth = initialize_FilterSmooth(model, num_observations)

# `filter_smooth` now contains zero-initialized arrays for smoothing operations
"""
function initialize_FilterSmooth(model::LinearDynamicalSystem, num_obs::Int)
    num_states = model.latent_dim
    FilterSmooth(
        zeros(num_states, num_obs),
        zeros(num_states, num_states, num_obs),
        zeros(num_states, num_obs, 1),
        zeros(num_states, num_states, num_obs, 1),
    zeros(num_states, num_states, num_obs, 1)
    )
end

"""
    sample(lds::LinearDynamicalSystem{S,O}, T_steps::Int, n_trials::Int) where {T<:Float64, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Sample from a Linear Dynamical System (LDS) model for multiple trials.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System model.
- `T_steps::Int`: The number of time steps to sample for each trial.
- `n_trials::Int`: The number of trials to sample.=

# Returns
- `x::Array{T, 3}`: The latent state variables. Dimensions: (latent_dim, T_Steps, n_trials)
- `y::Array{T, 3}`: The observed data. Dimensions: (obs_dim, T_steps, n_trials)

# Examples
```julia
lds = GaussianLDS(obs_dim=4, latent_dim=3)
x, y = sample(lds, 10, 100)  # 10 trials, 100 time steps each
```
"""
function sample(
    lds::LinearDynamicalSystem{S,O}, 
    T_steps::Int, 
    n_trials::Int
) where {T<:Float64,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    A, Q, x0, P0 = lds.state_model.A,
    lds.state_model.Q, lds.state_model.x0,
    lds.state_model.P0
    C, R = lds.obs_model.C, lds.obs_model.R

    x = Array{T,3}(undef, lds.latent_dim, T_steps, n_trials)
    y = Array{T,3}(undef, lds.obs_dim, T_steps, n_trials)

    for trial in 1:n_trials
        x[:, 1, trial] = rand(MvNormal(x0, P0))

        x1_view = @view x[:, 1, trial]
        y[:, 1, trial] = rand(MvNormal(C * x1_view, R))

        for t in 2:T_steps
            x_prev = @view x[:, t-1, trial]
            x[:, t, trial] = rand(MvNormal(A * x_prev, Q))

            x_current = @view x[:, t, trial]
            y[:, t, trial] = rand(MvNormal(C * x_current, R))
        end
    end

    return x, y
end

"""
    loglikelihood(x::AbstractMatrix{T}, lds::LinearDynamicalSystem{S,O}, y::AbstractMatrix{T}) where {T<:Float64, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Calculate the complete-data log-likelihood of a linear dynamical system (LDS) given the observed data.

# Arguments
- `x::AbstractMatrix{T}`: The state sequence of the LDS.
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System.
- `y::AbstractMatrix{T}`: The observed data.
- `w::Vector{Float64}`: coeffcients to weight the data.

# Returns
- `ll::T`: The complete-data log-likelihood of the LDS.
"""
function loglikelihood(
    x::AbstractMatrix{T}, 
    lds::LinearDynamicalSystem{S,O}, 
    y::AbstractMatrix{U},
    w::Vector{Float64}=ones(size(y, 2))
) where {T<:Real,U<:Real,W<:Float64,S<:GaussianStateModel{W},O<:GaussianObservationModel{W}}
    T_steps = size(y, 2)
    A, Q, x0, P0 = lds.state_model.A,
    lds.state_model.Q, lds.state_model.x0,
    lds.state_model.P0
    C, R = lds.obs_model.C, lds.obs_model.R 

    # Pre-compute Cholesky factors instead of inverses
    R_chol = cholesky(Symmetric(R)).U
    Q_chol = cholesky(Symmetric(Q)).U
    P0_chol = cholesky(Symmetric(P0)).U

    # Initial state contribution
    dx0 = view(x, :, 1) - x0
    # Replace dx0' * inv_P0 * dx0 with equivalent using Cholesky
    ll = sum(abs2, P0_chol \ dx0)

    # Create temporaries with the same element type as x
    temp_dx = zeros(T, size(x, 1))
    temp_dy = zeros(promote_type(T, U), size(y, 1))

    @inbounds for t in 1:T_steps
        if t > 1
            mul!(temp_dx, A, view(x, :, t-1), -1.0, false)
            temp_dx .+= view(x, :, t)
            # Replace temp_dx' * inv_Q * temp_dx
            ll += sum(abs2, Q_chol \ temp_dx)
        end
        mul!(temp_dy, C, view(x, :, t), -1.0, false)
        temp_dy .+= view(y, :, t)
        # Replace temp_dy' * inv_R * temp_dy
        ll += w[t] * sum(abs2, R_chol \ temp_dy)
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
- `w::Vector{Float64}`: coeffcients to weight the data.

# Returns
- `grad::Matrix{T}`: Gradient of the log-likelihood with respect to the latent states.
"""
function Gradient(
    lds::LinearDynamicalSystem{S,O}, y::Matrix{T}, x::Matrix{T},
    w::Vector{Float64}=ones(size(y, 2))
) where {T<:Real,U<:Float64, S<:GaussianStateModel{U},O<:GaussianObservationModel{U}}
    # Dims etc.
    latent_dim, T_steps = size(x)
    obs_dim, _ = size(y)
    # Model Parameters
    A, Q, x0, P0 = lds.state_model.A,
    lds.state_model.Q, lds.state_model.x0,
    lds.state_model.P0
    C, R = lds.obs_model.C, lds.obs_model.R

    # Compute Cholesky factors
    R_chol = cholesky(Symmetric(R))
    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))

    # Pre-compute common matrix products
    # Instead of C' * inv(R), we can use: C' * (R_chol \ I)
    # or equivalently: (R_chol \ C)'
    C_inv_R = (R_chol \ C)'
    A_inv_Q = (Q_chol \ A)'

    grad = zeros(T, latent_dim, T_steps)

    # First time step
    dx1 = x[:, 1] - x0
    dx2 = x[:, 2] - A * x[:, 1]
    dy1 = y[:, 1] - C * x[:, 1]

    grad[:, 1] .= A_inv_Q * dx2 + w[1] * C_inv_R * dy1 - (P0_chol \ dx1)

    # Middle time steps
    @inbounds for t in 2:(T_steps - 1)

        x_prev = @view x[:, t - 1]
        x_curr = @view x[:, t]
        x_next = @view x[:, t + 1]
        y_curr = @view y[:, t]

        dxt = x_curr .- A * x_prev
        dxt_next = x_next .- A * x_curr
        dyt = y_curr .- C * x_curr

        grad[:, t] .= w[t] * C_inv_R * dyt - (Q_chol \ dxt) + (A_inv_Q * dxt_next)
    end

    # Last time step
    dxT = x[:, T_steps] - A * x[:, T_steps - 1]
    dyT = y[:, T_steps] - C * x[:, T_steps]

    grad[:, T_steps] .= w[T_steps] * (C_inv_R * dyT) - (Q_chol \ dxT)

    return grad
end

"""
    Hessian(lds::LinearDynamicalSystem{S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

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
- `x::AbstractMatrix{T}`: Matrix of latent states.
- `w::Vector{Float64}`: coeffcients to weight the data.

# Returns
- `H::Matrix{T}`: Hessian matrix of the log-likelihood.
- `H_diag::Vector{Matrix{T}}`: Main diagonal blocks of the Hessian.
- `H_super::Vector{Matrix{T}}`: Super-diagonal blocks of the Hessian.
- `H_sub::Vector{Matrix{T}}`: Sub-diagonal blocks of the Hessian.

# Note 
- `x` is not used in this function, but is required to match the function signature of other Hessian calculations e.g., in PoissonLDS.
"""
function Hessian(
    lds::LinearDynamicalSystem{S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T},
    w::Vector{Float64}=ones(size(y, 2))
) where {T<:Real,U<:Float64, S<:GaussianStateModel{U},O<:GaussianObservationModel{U}}
    A, Q, x0, P0 = lds.state_model.A,
    lds.state_model.Q, lds.state_model.x0,
    lds.state_model.P0
    C, R = lds.obs_model.C, lds.obs_model.R

    T_steps = size(y, 2)

    # Pre-compute inverses
    inv_R = Symmetric(inv(R))
    inv_Q = Symmetric(inv(Q))
    inv_P0 = Symmetric(inv(P0))

    # Pre-allocate all blocks
    H_sub = Vector{Matrix{T}}(undef, T_steps - 1)
    H_super = Vector{Matrix{T}}(undef, T_steps - 1)
    H_diag = Vector{Matrix{T}}(undef, T_steps)

    # Off-diagonal terms
    H_sub_entry = inv_Q * A
    H_super_entry = Matrix(H_sub_entry')

    # Calculate main diagonal terms
    yt_given_xt = -C' * inv_R * C
    xt_given_xt_1 = -inv_Q
    xt1_given_xt = -A' * inv_Q * A
    x_t = -inv_P0

    # Build off-diagonals
    @inbounds for i in 1:(T_steps - 1)
        H_sub[i] = H_sub_entry
        H_super[i] = H_super_entry
    end

    # Build main diagonal
    H_diag[1] = w[1] * yt_given_xt + xt1_given_xt + x_t
    @inbounds for i in 2:(T_steps - 1)
        H_diag[i] = w[i] * yt_given_xt + xt_given_xt_1 + xt1_given_xt
    end
    H_diag[T_steps] = w[T_steps] * (yt_given_xt) + xt_given_xt_1

    H = StateSpaceDynamics.block_tridgm(H_diag, H_super, H_sub)

    return H, H_diag, H_super, H_sub
end

"""
    smooth(lds::LinearDynamicalSystem{S,O}, y::Matrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

This function performs direct smoothing for a linear dynamical system (LDS) given the system parameters and the observed data.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The LDS object representing the system parameters.
- `y::Matrix{T}`: The observed data matrix.
- `w::Vector{Float64}`: coeffcients to weight the data.

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
function smooth(
    lds::LinearDynamicalSystem{S,O}, y::Matrix{T}, w::Vector{Float64}=ones(size(y, 2))
) where {T<:Real,S<:GaussianStateModel{<:Float64},O<:AbstractObservationModel{<:Float64}}
    T_steps, D = size(y, 2), lds.latent_dim

    # set initial "solution" and preallocate x_reshape
    X₀ = zeros(T, D * T_steps)

    function nll(vec_x::Vector{T})
        x = reshape(vec_x, D, T_steps)
        return -loglikelihood(x, lds, y, w)
    end

    function g!(g::Vector{T}, vec_x::Vector{T})
        x = reshape(vec_x, D, T_steps)
        grad = Gradient(lds, y, x, w)
        return g .= vec(-grad)
    end

    function h!(h::AbstractSparseMatrix, vec_x::Vector{T})
        x = reshape(vec_x, D, T_steps)
        H, _, _, _ = Hessian(lds, y, x, w)
        copyto!(h, -H)
        return nothing
    end

    # set up initial values
    initial_f = nll(X₀)

    inital_g = similar(X₀)
    g!(inital_g, X₀)

    initial_h = spzeros(Float64, length(X₀), length(X₀))
    h!(initial_h, X₀)

    # set up a TwiceDifferentiable object i guess?
    td = TwiceDifferentiable(nll, g!, h!, X₀, initial_f, inital_g, initial_h)

    # set up Optim.Options
    opts = Optim.Options(; g_abstol=1e-8, x_abstol=1e-8, f_abstol=1e-8, iterations=100)

    # Go!
    res = optimize(td, X₀, Newton(; linesearch=LineSearches.BackTracking()), opts)

    # Profit
    x = reshape(res.minimizer, D, T_steps)

    H, main, super, sub = Hessian(lds, y, x, w)

    # Get the second moments of the latent state path, use static matrices if the latent dimension is small
    if lds.latent_dim > 10
        p_smooth, inverse_offdiag = block_tridiagonal_inverse(-sub, -main, -super)
    else
        p_smooth, inverse_offdiag = block_tridiagonal_inverse_static(-sub, -main, -super)
    end

    # Calculate the entropy, see utilities.jl for the function
    gauss_entropy = gaussian_entropy(Symmetric(H))

    # Symmetrize the covariance matrices
    @inbounds for i in 1:T_steps
        p_smooth[:, :, i] .= 0.5 .* (p_smooth[:, :, i] .+ p_smooth[:, :, i]')
    end

    # Add a zero matrix for later compatibility
    inverse_offdiag = cat(zeros(T, D, D), inverse_offdiag; dims=3)

    return x, p_smooth, inverse_offdiag, gauss_entropy
end

"""
    smooth(lds::LinearDynamicalSystem{S,O}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

This function performs direct smoothing for a linear dynamical system (LDS) given the system parameters and the observed data for multiple trials.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The LDS object representing the system parameters.
- `y::Array{T,3}`: The observed data array with dimensions (obs_dim, tiem_steps, n_trials).

# Returns
- `x::Array{T,3}`: The optimal state estimates with dimensions (n_trials, time_steps, latent_dim).
- `p_smooth::Array{T,4}`: The posterior covariance matrices with dimensions (latent_dim, latent_dim, time_steps, n_trials).
- `inverse_offdiag::Array{T,4}`: The inverse off-diagonal matrices with dimensions (latent_dim, latent_dim, time_steps, n_trials).

# Example
```julia
lds = GaussianLDS(obs_dim=4, latent_dim=3)
y = randn(5, 100, 4)  # 5 trials, 100 time steps, 4 observed dimension
x, p_smooth, inverse_offdiag = smooth(lds, y)
```
"""
function smooth(
    lds::LinearDynamicalSystem{S,O}, y::Array{T,3}
) where {T<:Real,S<:GaussianStateModel{<:Float64},O<:AbstractObservationModel{<:Float64}}
    obs_dim, T_steps, n_trials = size(y)
    latent_dim = lds.latent_dim

    # Fast path for single trial case
    if n_trials == 1
        x_sm, p_sm, p_prev, ent = smooth(lds, y[:, :, 1])
        # Return directly in the required shape without additional copying
        return reshape(x_sm, latent_dim, T_steps, 1),
               reshape(p_sm, latent_dim, latent_dim, T_steps, 1),
               reshape(p_prev, latent_dim, latent_dim, T_steps, 1),
               ent
    end

    # Pre-allocate output arrays
    x_smooth = Array{T,3}(undef, latent_dim, T_steps, n_trials)
    p_smooth = Array{T,4}(undef, latent_dim, latent_dim, T_steps, n_trials)
    inverse_offdiag = Array{T,4}(undef, latent_dim, latent_dim, T_steps, n_trials)
    total_entropy = 0.0

    @threads for trial in 1:n_trials
        x_sm, p_sm, p_prev, ent = smooth(lds, y[:, :, trial])
        total_entropy += ent
        x_smooth[:, :, trial] .= x_sm
        p_smooth[:, :, :, trial] .= p_sm
        inverse_offdiag[:, :, :, trial] .= p_prev
    end

    return x_smooth, p_smooth, inverse_offdiag, total_entropy
end

"""
    Q_state(A, Q, P0, x0, E_z, E_zz, E_zz_prev)
Calculate the state component of the Q-function for the EM algorithm in a Linear Dynamical System.
# Arguments
- `A::Matrix{<:Real}`: The state transition matrix.
- `Q::AbstractMatrix{<:Real}`: The process noise covariance matrix (or its Cholesky factor).
- `P0::AbstractMatrix{<:Real}`: The initial state covariance matrix (or its Cholesky factor).
- `x0::Vector{<:Real}`: The initial state mean.
- `E_z::Matrix{<:Real}`: The expected latent states, size (state_dim, T).
- `E_zz::Array{<:Real, 3}`: The expected value of z_t * z_t', size (state_dim, state_dim, T).
- `E_zz_prev::Array{<:Real, 3}`: The expected value of z_t * z_{t-1}', size (state_dim, state_dim, T).
# Returns
- `Q_val::Float64`: The state component of the Q-function.
"""
function Q_state(
    A::AbstractMatrix{<:Float64},
    Q::AbstractMatrix{T},
    P0::AbstractMatrix{T},
    x0::AbstractVector{T},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
) where {T<:Real}
    T_step = size(E_z, 2)
    state_dim = size(A, 1)
    
    # Pre-compute constants and decompositions once
    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))
    log_det_Q = logdet(Q_chol)
    log_det_P0 = logdet(P0_chol)
    
    # Pre-allocate temp matrix
    temp = zeros(state_dim, state_dim)
    
    # First time step (handled separately)
    mul!(temp, E_z[:, 1], x0', -1.0, 0.0)  # -E_z[:,1] * x0'
    temp .+= view(E_zz, :, :, 1)           # Add E_zz[:,:,1]
    temp .-= x0 * E_z[:, 1]'               # Subtract x0 * E_z[:,1]'
    temp .+= x0 * x0'                      # Add x0 * x0'
    Q_val = -0.5 * (log_det_P0 + tr(P0_chol \ temp))
    
    # Pre-allocate sums for t ≥ 2
    sum_E_zz_current = zeros(state_dim, state_dim)
    sum_E_zz_prev_cross = zeros(state_dim, state_dim)
    sum_E_zz_prev_time = zeros(state_dim, state_dim)
    
    # Compute sums with views
    @inbounds for t in 2:T_step
        sum_E_zz_current .+= view(E_zz, :, :, t)
        sum_E_zz_prev_cross .+= view(E_zz_prev, :, :, t)
        sum_E_zz_prev_time .+= view(E_zz, :, :, t-1)
    end
    
    # Compute transition term
    copyto!(temp, sum_E_zz_current)
    mul!(temp, A, sum_E_zz_prev_cross', -1.0, 1.0)
    temp .-= sum_E_zz_prev_cross * A'
    mul!(temp, A, sum_E_zz_prev_time * A', 1.0, 1.0)
    
    # Add remaining time steps
    Q_val += -0.5 * ((T_step - 1) * log_det_Q + tr(Q_chol \ temp))
    
    return Q_val
end


"""
    Q_obs(H, R, E_z, E_zz, y)

Calculate the a single time step observation component of the Q-function for the EM algorithm in a Linear Dynamical System before the R^-1 is accounted for.

# Arguments
- `H::Matrix{<:Real}`: The observation matrix.
- `R::AbstractMatrix{<:Real}`: The observation noise covariance matrix (or its Cholesky factor).
- `E_z::Vector{<:Real}`: The expected latent states at time t, size (state_dim).
- `E_zz::Matrix{<:Real}`: The expected value of z_t * z_t' at time t, size (state_dim, state_dim).
- `y::Vector{<:Real}`: The observed data at time t, size (obs_dim).

# Returns
- `q::Float64`: The observation component at time t of the Q-function prior to R^-1.

"""
function Q_obs(
    H::AbstractMatrix{T},
    E_z::AbstractVector{T},
    E_zz::AbstractMatrix{T},
    y::AbstractVector{T},
) where {T<:Real}

    obs_dim = size(H, 1)

    # Pre-allocate statistics
    sum_yy = zeros(obs_dim, obs_dim)
    sum_yz = zeros(obs_dim, size(E_z, 1))
    
    mul!(sum_yy, y, y', 1.0, 1.0)
    mul!(sum_yz, y, E_z', 1.0, 1.0)

    # Pre-allocate and compute final expression
    temp = similar(sum_yy)
    copyto!(temp, sum_yy)
    mul!(temp, H, sum_yz', -1.0, 1.0)
    temp .-= sum_yz * H'
    mul!(temp, H * E_zz, H', 1.0, 1.0)
        
    return temp

end


"""
    Q_obs(H, R, E_z, E_zz, y)

Calculate the observation component of the Q-function for the EM algorithm in a Linear Dynamical System.

# Arguments
- `H::Matrix{<:Real}`: The observation matrix.
- `R::AbstractMatrix{<:Real}`: The observation noise covariance matrix (or its Cholesky factor).
- `E_z::Matrix{<:Real}`: The expected latent states, size (state_dim, T).
- `E_zz::Array{<:Real, 3}`: The expected value of z_t * z_t', size (state_dim, state_dim, T).
- `y::Matrix{<:Real}`: The observed data, size (obs_dim, T).

# Returns
- `Q_val::Float64`: The observation component of the Q-function.

"""
function Q_obs(
    H::AbstractMatrix{T},
    R::AbstractMatrix{T},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    y::AbstractMatrix{T},
    weights::Vector{Float64}=ones(size(y, 2))
) where {T<:Real}
    obs_dim = size(H, 1)
    T_step = size(E_z, 2)
    
    # Pre-compute constants
    R_chol = cholesky(Symmetric(R))
    log_det_R = logdet(R_chol)
    const_term = obs_dim * log(2π)
    
    #Pre-allocate statistics
    temp = zeros(obs_dim, obs_dim)
    
    # Use views in the loop
    @views for t in axes(y, 2)
        temp += weights[t] * Q_obs(H, E_z[:,t], E_zz[:,:,t], y[:,t])
    end

    # Weight the constant terms by the sum of weights
    total_weight = sum(weights)
    
    Q_val = -0.5 * (total_weight * (const_term + log_det_R) + tr(R_chol \ temp))
    
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
- `E_z::Matrix{<:Real}`: The expected latent states, size (state_dim, T).
- `E_zz::Array{<:Real, 3}`: The expected value of z_t * z_t', size (state_dim, state_dim, T).
- `E_zz_prev::Array{<:Real, 3}`: The expected value of z_t * z_{t-1}', size (state_dim, state_dim, T).
- `y::Matrix{<:Real}`: The observed data, size (obs_dim, T).

# Returns
- `Q_val::Float64`: The complete Q-function value.
"""
function Q_function(
    A::AbstractMatrix{T},
    Q::AbstractMatrix{T},
    C::AbstractMatrix{T},
    R::AbstractMatrix{T},
    P0::AbstractMatrix{T},
    x0::AbstractVector{T},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
    y::AbstractMatrix{T},
    weights::Vector{Float64}=ones(size(y, 2))
) where {T<:Real}
    Q_val_state = Q_state(A, Q, P0, x0, E_z, E_zz, E_zz_prev)
    Q_val_obs = Q_obs(C, R, E_z, E_zz, y, weights)
    return Q_val_state + Q_val_obs
end

"""
    sufficient_statistics(x_smooth::Array{T,3}, p_smooth::Array{T,4}, p_smooth_t1::Array{T,4}) where T <: Real

Compute sufficient statistics for the EM algorithm in a Linear Dynamical System.

# Arguments
- `x_smooth::Array{T,3}`: Smoothed state estimates, size (state_dim, state_dim, T_steps, n_trials)
- `p_smooth::Array{T,4}`: Smoothed state covariances, size (state_dim, state_dim, T_steps, n_trials, state_dim)
- `p_smooth_t1::Array{T,4}`: Lag-one covariance smoother, size (state_dim, state_dim, T_steps, n_trials, state_dim)

# Returns
- `E_z::Array{T,3}`: Expected latent states, size (state_dim, state_dim, T_steps, n_trials)
- `E_zz::Array{T,4}`: Expected z_t * z_t', size (state_dim, state_dim, T_steps, n_trials, state_dim)
- `E_zz_prev::Array{T,4}`: Expected z_t * z_{t-1}', size (state_dim, state_dim, T_steps, n_trials, state_dim)

# Note
- The function computes the expected values for all trials.
- For single-trial data, use inputs with n_trials = 1.
"""
function sufficient_statistics(
    x_smooth::Array{T,3}, p_smooth::Array{T,4}, p_smooth_t1::Array{T,4}
) where {T<:Real}
    latent_dim, T_steps, n_trials = size(x_smooth)

    E_z = copy(x_smooth)
    E_zz = similar(p_smooth)
    E_zz_prev = similar(p_smooth)

    for trial in 1:n_trials
        @inbounds for t in 1:T_steps
            E_zz[:, :, t, trial] .=
                p_smooth[:, :, t, trial] + x_smooth[:, t, trial] * x_smooth[:, t, trial]'
            if t > 1
                E_zz_prev[:, :, t, trial] .=
                    p_smooth_t1[:, :, t, trial] +
                    x_smooth[:, t, trial] * x_smooth[:, t - 1, trial]'
            end
        end
        E_zz_prev[:, :, 1, trial] .= 0
    end

    return E_z, E_zz, E_zz_prev
end

"""
    estep(lds::LinearDynamicalSystem{S,O}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}

Perform the E-step of the EM algorithm for a Linear Dynamical System, treating all input as multi-trial.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `y::Array{T,3}`: Observed data, size (obs_dim, T_steps, n_trials)
    Note: For single-trial data, use y[1:1, :, :] to create a 3D array with n_trials = 1

# Returns
- `E_z::Array{T,3}`: Expected latent states, size (state_dim, state_dim, T_steps, n_trials)
- `E_zz::Array{T,4}`: Expected z_t * z_t', size (state_dim, state_dim, T_steps, n_trials, state_dim)
- `E_zz_prev::Array{T,4}`: Expected z_t * z_{t-1}', size (state_dim, state_dim, T_steps, n_trials, state_dim)
- `x_smooth::Array{T,3}`: Smoothed state estimates, size (state_dim, state_dim, T_steps, n_trials)
- `p_smooth::Array{T,4}`: Smoothed state covariances, size (state_dim, state_dim, T_steps, n_trials, state_dim)
- `ml::T`: Total marginal likelihood (log-likelihood) of the data across all trials

# Note
- This function first smooths the data using the `smooth` function, then computes sufficient statistics.
- It treats all input as multi-trial, with single-trial being a special case where n_trials = 1.
"""
function estep(
    lds::LinearDynamicalSystem{S,O}, y::Array{T,3}
) where {T<:Real,U<:Float64, S<:GaussianStateModel{U},O<:AbstractObservationModel{U}}
    # smooth
    x_smooth, p_smooth, inverse_offdiag, total_entropy = smooth(lds, y)

    # calculate sufficient statistics
    E_z, E_zz, E_zz_prev = sufficient_statistics(x_smooth, p_smooth, inverse_offdiag)

    # calculate elbo
    ml_total = calculate_elbo(lds, E_z, E_zz, E_zz_prev, p_smooth, y, total_entropy)

    return E_z, E_zz, E_zz_prev, x_smooth, p_smooth, ml_total
end

"""
    calculate_elbo(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, E_zz_prev::Array{T,4}, p_smooth::Array{T,4}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}

Calculate the Evidence Lower Bound (ELBO) for a Linear Dynamical System.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `E_z::Array{T,3}`: Expected latent states, size (state_dim, state_dim, T_steps, n_trials)
- `E_zz::Array{T,4}`: Expected z_t * z_t', size (state_dim, state_dim, T_steps, n_trials, state_dim)
- `E_zz_prev::Array{T,4}`: Expected z_t * z_{t-1}', size (state_dim, state_dim, T_steps, n_trials, state_dim)
- `p_smooth::Array{T,4}`: Smoothed state covariances, size (state_dim, state_dim, T_steps, n_trials, state_dim)
- `y::Array{T,3}`: Observed data, size (obs_dim, T_steps, n_trials)

# Returns
- `elbo::T`: The Evidence Lower Bound (ELBO) for the LDS.

# Note
- For a GaussianLDS the ELBO is equivalent to the total marginal likelihood
"""
function calculate_elbo(
    lds::LinearDynamicalSystem{S,O},
    E_z::Array{T,3},
    E_zz::Array{T,4},
    E_zz_prev::Array{T,4},
    p_smooth::Array{T,4},
    y::Array{T,3},
    total_entropy::Float64,
    weights::Vector{Float64}=ones(size(y, 2))
) where {T<:Real,U<:Float64,S<:GaussianStateModel{U},O<:GaussianObservationModel{U}}
    n_trials = size(y, 3)
    Q_vals = zeros(T, n_trials)

    # Thread over trials
    @threads for trial in 1:n_trials
        Q_vals[trial] = StateSpaceDynamics.Q_function(
            lds.state_model.A,
            lds.state_model.Q,
            lds.obs_model.C,
            lds.obs_model.R,
            lds.state_model.P0,
            lds.state_model.x0,
            view(E_z, :, :, trial),
            view(E_zz, :, :, :, trial),
            view(E_zz_prev, :, :, :, trial),
            view(y, :, :, trial),
            weights
        )
    end

    return sum(Q_vals) - total_entropy
end


"""
    update_initial_state_mean!(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}

Update the initial state mean of the Linear Dynamical System using the average across all trials.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `E_z::Array{T,3}`: Expected latent states, size (state_dim, state_dim, T_steps, n_trials)

# Note
- This function modifies `lds` in-place.
- The update is only performed if `lds.fit_bool[1]` is true.
- The initial state mean is computed as the average of the first time step across all trials.
"""
function update_initial_state_mean!(
    lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, 
) where {T<:Real,U<:Float64, S<:GaussianStateModel{U},O<:AbstractObservationModel{U}}
    if lds.fit_bool[1]
        x0_new = zeros(lds.latent_dim)
        for i in axes(E_z, 3)
            x0_new .+= E_z[:, 1, i]
        end
        lds.state_model.x0 .= x0_new ./ size(E_z, 3)
    end
end

"""
    update_initial_state_covariance!(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}

Update the initial state covariance of the Linear Dynamical System using the average across all trials.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `E_z::Array{T,3}`: Expected latent states, size (state_dim, state_dim, T_steps, n_trials)
- `E_zz::Array{T,4}`: Expected z_t * z_t', size (state_dim, state_dim, T_steps, n_trials, state_dim)

# Note
- This function modifies `lds` in-place.
- The update is only performed if `lds.fit_bool[2]` is true.
- The initial state covariance is computed as the average of the first time step across all trials.
"""
function update_initial_state_covariance!(
    lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, 
) where {T<:Real,U<:Float64, S<:GaussianStateModel{U},O<:AbstractObservationModel{U}}
    if lds.fit_bool[2]
        n_trials = size(E_z, 3)
        state_dim = size(E_z, 1)
        p0_new = zeros(T, state_dim, state_dim)

        for trial in 1:n_trials
            p0_new .+= E_zz[:, :, 1, trial] - (lds.state_model.x0 * lds.state_model.x0')
        end

        p0_new ./= n_trials
        p0_new .= 0.5 * (p0_new + p0_new')

        # Set the new P0 matrix
        lds.state_model.P0 = p0_new
    end
end

"""
    update_A!(lds::LinearDynamicalSystem{S,O}, E_zz::Array{T, 4}, E_zz_prev::Array{T, 4}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}

Update the transition matrix A of the Linear Dynamical System.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `E_zz::Array{T, 4}`: Expected z_t * z_t', size (state_dim, state_dim, T_steps, n_trials)
- `E_zz_prev::Array{T, 4}`: Expected z_t * z_{t-1}', size (state_dim, state_dim, T_steps, n_trials)

# Note
- This function modifies `lds` in-place.
- The update is only performed if `lds.fit_bool[3]` is true.
"""

function update_A!(
    lds::LinearDynamicalSystem{S,O}, E_zz::Array{T,4}, E_zz_prev::Array{T,4}
) where {T<:Real,S<:GaussianStateModel{<:Float64},O<:AbstractObservationModel{<:Float64}}
    if lds.fit_bool[3]
        state_dim = size(E_zz, 1)

        E_zz_sum = zeros(state_dim, state_dim)
        E_zz_prev_sum = zeros(state_dim, state_dim)

        for trial in axes(E_zz, 4)
            E_zz_sum .+= sum(E_zz[:, :, 1:(end - 1), trial]; dims=3)
            E_zz_prev_sum .+= sum(E_zz_prev[:, :, :, trial]; dims=3)
        end

        lds.state_model.A = E_zz_prev_sum / E_zz_sum
    end
end

"""
    update_Q!(lds::LinearDynamicalSystem{S,O}, E_zz::Array{T, 4}, E_zz_prev::Array{T, 4}) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}

Update the process noise covariance matrix Q of the Linear Dynamical System.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `E_zz::Array{T, 4}`: Expected z_t * z_t', size (state_dim, state_dim, T_steps, n_trials)
- `E_zz_prev::Array{T, 4}`: Expected z_t * z_{t-1}', size (state_dim, state_dim, T_steps, n_trials)

# Note
- This function modifies `lds` in-place.
- The update is only performed if `lds.fit_bool[4]` is true.
- The result is averaged across all trials.
"""
function update_Q!(
    lds::LinearDynamicalSystem{S,O}, E_zz::Array{T,4}, E_zz_prev::Array{T,4}
) where {T<:Real,U<:Float64, S<:GaussianStateModel{U},O<:AbstractObservationModel{U}}
    if lds.fit_bool[4]
        n_trials, T_steps = size(E_zz, 4), size(E_zz, 3)
        state_dim = size(E_zz, 1)
        Q_new = zeros(T, state_dim, state_dim)
        A = lds.state_model.A

        for trial in 1:n_trials
            @inbounds for t in 2:T_steps
                # Get current state covariance and previous-current cross covariance
                Σt = E_zz[:, :, t, trial]          # E[z_t z_t']
                Σt_prev = E_zz[:, :, t - 1, trial]   # E[z_{t-1} z_{t-1}']
                Σt_cross = E_zz_prev[:, :, t, trial] # E[z_t z_{t-1}']

                # Compute innovation: actual state minus predicted state
                # Q = E[(z_t - Az_{t-1})(z_t - Az_{t-1})']
                innovation_cov = Σt - Σt_cross * A' - A * Σt_cross' + A * Σt_prev * A'

                # This is equivalent to E[(z_t - Az_{t-1})(z_t - Az_{t-1})']
                Q_new .+= innovation_cov
            end
        end

        Q_new ./= (n_trials * (T_steps - 1))
        Q_new = 0.5 * (Q_new + Q_new')  # Symmetrize

        # Set the new Q matrix
        lds.state_model.Q = Q_new
    end
end

"""
    update_C!(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Update the observation matrix C of the Linear Dynamical System.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `E_z::Array{T,3}`: Expected latent states, size (state_dim, state_dim, T_steps, n_trials)
- `E_zz::Array{T,4}`: Expected z_t * z_t', size (state_dim, state_dim, T_steps, n_trials)
- `y::Array{T,3}`: Observed data, size (obs_dim, T_steps, n_trials)

# Note
- This function modifies `lds` in-place.
- The update is only performed if `lds.fit_bool[5]` is true.
- The result is averaged across all trials.
"""
function update_C!(
    lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, y::Array{T,3},
    w::Vector{Float64}=ones(size(y, 2))
) where {T<:Real,U<:Float64, S<:GaussianStateModel{U},O<:GaussianObservationModel{U}}
    if lds.fit_bool[5]
        n_trials, T_steps = size(y, 3), size(y, 2)

        sum_yz = zeros(T, size(lds.obs_model.C))
        sum_zz = zeros(T, size(E_zz)[1:2])

        for trial in 1:n_trials
            @inbounds for t in 1:T_steps

                y_view   = @view y[:, t, trial]  
                Ez_view  = @view E_z[:, t, trial]    
                Ezz_view = @view E_zz[:, :, t, trial]
                
                sum_yz .+= w[t] * (y_view * Ez_view')
                sum_zz .+= w[t] * Ezz_view
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
- `E_z::Array{T,3}`: Expected latent states, size (state_dim, state_dim, T_steps, n_trials)
- `E_zz::Array{T,4}`: Expected z_t * z_t', size (state_dim, state_dim, T_steps, n_trials)
- `y::Array{T,3}`: Observed data, size (obs_dim, T_steps, n_trials)

# Note
- This function modifies `lds` in-place.
- The update is only performed if `lds.fit_bool[6]` is true.
- The result is averaged across all trials.
"""
function update_R!(
    lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, y::Array{T,3},
    w::Vector{Float64}=ones(size(y, 2))
) where {T<:Real,U<:Float64, S<:GaussianStateModel{U},O<:GaussianObservationModel{U}}
    if lds.fit_bool[6]
        obs_dim, T_steps, n_trials = size(y)
        R_new = zeros(T, obs_dim, obs_dim)
        C = lds.obs_model.C
        
        # Pre-allocate all temporary arrays
        innovation = zeros(T, obs_dim)
        Czt = zeros(T, obs_dim)
        temp_matrix = zeros(T, obs_dim, size(C, 2))  # For storing C * state_uncertainty
        
        # Reorganize as sum of outer products
        for trial in 1:n_trials
            @inbounds for t in 1:T_steps
                # Compute innovation using pre-allocated arrays
                yt = @view y[:, t, trial]
                zt = @view E_z[:, t, trial]
                mul!(Czt, C, zt)
                @. innovation = (yt - Czt)
                
                # Add innovation outer product
                #BLAS.ger!(one(T), innovation, innovation, R_new)
                mul!(R_new, innovation, innovation', w[t], one(T))  # R_new += w[t] * innovation * innovation'
                
                # Add correction term efficiently:
                # First compute state_uncertainty = Σ_t - z_t*z_t'
                state_uncertainty = view(E_zz, :, :, t, trial) - zt * zt'
                
                # Then compute C * state_uncertainty * C' in steps:
                mul!(temp_matrix, C, state_uncertainty)  # temp = C * state_uncertainty
                mul!(R_new, temp_matrix, C', w[t], one(T))  # R_new += w[t] * C * state_uncertainty * C'
            end
        end
        
        R_new ./= (n_trials * T_steps)
        R_new .= 0.5 * (R_new + R_new')  # Symmetrize

        # Set the new R matrix
        lds.obs_model.R = R_new
    end
end

"""
    mstep!(lds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, E_zz_prev::Array{T,4}, p_smooth::Array{T, 4}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Perform the M-step of the EM algorithm for a Linear Dynamical System with multi-trial data.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System struct
- `E_z::Array{T,3}`: Expected latent states, size (state_dim, state_dim, T_steps, n_trials)
- `E_zz::Array{T,4}`: Expected z_t * z_t', size (state_dim, state_dim, T_steps, n_trials)
- `E_zz_prev::Array{T,4}`: Expected z_t * z_{t-1}', size (state_dim, state_dim, T_steps, n_trials)
- `p_smooth::Array{T,4}`: Smoothed state covariances, size (state_dim, state_dim, T_steps, n_trials) (not used)
- `y::Array{T,3}`: Observed data, size (obs_dim, T_steps, n_trials)

# Note
- This function modifies `lds` in-place by updating all model parameters.
- Updates are performed only for parameters where the corresponding `fit_bool` is true.
- All update functions now handle multi-trial data.
- P_smooth is required but not used in the M-step so that the function signature matches the PoissonLDS version.
"""
function mstep!(
    lds::LinearDynamicalSystem{S,O},
    E_z::Array{T,3},
    E_zz::Array{T,4},
    E_zz_prev::Array{T,4},
    p_smooth::Array{T,4},
    y::Array{T,3},
    w::Vector{Float64}=ones(size(y, 2))
) where {T<:Real,U<:Float64, S<:GaussianStateModel{U},O<:GaussianObservationModel{U}}
    # get initial parameters
    old_params = vec(stateparams(lds))
    old_params = [old_params; vec(obsparams(lds))]

    # Update parameters
    update_initial_state_mean!(lds, E_z)
    update_initial_state_covariance!(lds, E_z, E_zz)
    update_A!(lds, E_zz, E_zz_prev)
    update_Q!(lds, E_zz, E_zz_prev)
    update_C!(lds, E_z, E_zz, y)
    update_R!(lds, E_z, E_zz, y)

    # get new params
    new_params = vec(stateparams(lds))
    new_params = [new_params; vec(obsparams(lds))]

    # calculate norm of parameter changes
    norm_change = norm(new_params - old_params)
    return norm_change
end

"""
    fit!(lds::LinearDynamicalSystem{S,O}, y::Matrix{T}; 
         max_iter::Int=1000, 
         tol::Real=1e-12, 
         ) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Fit a Linear Dynamical System using the Expectation-Maximization (EM) algorithm with Kalman smoothing.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System to be fitted.
- `y::Matrix{T}`: Observed data, size (obs_dim, T_steps).

# Keyword Arguments
- `max_iter::Int=1000`: Maximum number of EM iterations.
- `tol::Real=1e-12`: Convergence tolerance for log-likelihood change.

# Returns
- `mls::Vector{T}`: Vector of log-likelihood values for each iteration.
"""
function fit!(
    lds::LinearDynamicalSystem{S,O}, y::Array{T,3}; max_iter::Int=1000, tol::Real=1e-12
) where {T<:Real,U<:Float64, S<:GaussianStateModel{U},O<:AbstractObservationModel{U}}

    # Initialize log-likelihood
    prev_ml = -T(Inf)

    # Create a vector to store the log-likelihood values
    mls = Vector{T}()
    param_diff = Vector{T}()

    sizehint!(mls, max_iter)  # Pre-allocate for efficiency

    # Initialize progress bar
    if O <: GaussianObservationModel
        prog = Progress(max_iter; desc="Fitting LDS via EM...", barlen=50, showspeed=true)
    elseif O <: PoissonObservationModel
        prog = Progress(
            max_iter; desc="Fitting Poisson LDS via LaPlaceEM...", barlen=50, showspeed=true
        )
    else
        error("Unknown LDS model type")
    end

    # Run EM
    for i in 1:max_iter
        # E-step
        E_z, E_zz, E_zz_prev, x_smooth, p_smooth, ml = estep(lds, y)

        # M-step
        Δparams = mstep!(lds, E_z, E_zz, E_zz_prev, p_smooth, y)
        # Update the log-likelihood vector and parameter difference
        push!(mls, ml)
        push!(param_diff, Δparams)

        # Update the progress bar
        next!(prog)

        # Check convergence
        if abs(ml - prev_ml) < tol
            finish!(prog)
            return mls, param_diff
        end

        prev_ml = ml
    end

    # Finish the progress bar if max_iter is reached
    finish!(prog)

    return mls, param_diff
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
- `obs_dim::Int`: Dimension of the observations (required if C, D, or log_d is not provided.)
- `latent_dim::Int`: Dimension of the latent state (required if A, Q, x0, P0, or C is not provided.)
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
    latent_dim::Int=0,
) where {T<:Float64}
    if latent_dim == 0 &&
        (isempty(A) || isempty(Q) || isempty(x0) || isempty(P0) || isempty(C))
        throw(
            ArgumentError("Must provide latent_dim if A, Q, x0, P0, or C is not provided.")
        )
    end
    if obs_dim == 0 && (isempty(C) || isempty(log_d))
        ethrow(ArgumentError("Must provide obs_dim if C or log_d is not provided."))
    end

    state_model = GaussianStateModel(; A=A, Q=Q, x0=x0, P0=P0, latent_dim=latent_dim)
    obs_model = PoissonObservationModel(; 
        C=C, log_d=log_d, obs_dim=obs_dim, latent_dim=latent_dim
    )
    return LinearDynamicalSystem(state_model, obs_model, latent_dim, obs_dim, fit_bool)
end

"""
    sample(lds::LinearDynamicalSystem{S,O}, T_steps::Int, n_trials::Int) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Sample from a Poisson Linear Dynamical System (LDS) model for multiple trials.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System model.
- `T_steps::Int`: The number of time steps to sample for each trial.
- `n_trials::Int`: The number of trials to sample.

# Returns
- `x::Array{T, 3}`: The latent state variables. Dimensions: (latent_dim, T_Steps, n_trials)
- `y::Array{Int, 3}`: The observed data. Dimensions: (obs_dim, T_steps, n_trials)

# Examples
```julia
lds = LinearDynamicalSystem(obs_dim=4, latent_dim=3)
x, y = sample(lds, 100, 10)  # 10 trials, 100 time steps each
```
"""
function sample(
    lds::LinearDynamicalSystem{S,O}, T_steps::Int, n_trials::Int
) where {T<:Float64,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    # Extract model components
    A, Q = lds.state_model.A, lds.state_model.Q
    C, log_d = lds.obs_model.C, lds.obs_model.log_d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    # Convert log_d to d i.e. non-log space
    d = exp.(log_d)

    # Pre-allocate arrays (now in column-major order)
    x = zeros(T, lds.latent_dim, T_steps, n_trials)
    y = zeros(T, lds.obs_dim, T_steps, n_trials)

    for k in 1:n_trials
        # Sample the initial state
        x[:, 1, k] = rand(MvNormal(x0, P0))

        x1_view = @view x[:, 1, k]
        y[:, 1, k] = rand.(Poisson.(exp.(C * x1_view .+ d)))

        # Sample the rest of the states
        for t in 2:T_steps
            x_prev = @view x[:, t - 1, k]
            x[:, t, k] = rand(MvNormal(A * x_prev, Q))

            x_curr = @view x[:, t, k]
            y[:, t, k] = rand.(Poisson.(exp.(C * x_curr + d)))
        end
    end

    return x, y
end

"""
    loglikelihood(x::Matrix{T}, lds::LinearDynamicalSystem{S,O}, y::Matrix{T}) where {T<:Real, S<:GaussianStateModel, O<:PoissonObservationModel}

Calculate the complete-data log-likelihood of a Poisson Linear Dynamical System model for a single trial. 

# Arguments
- `x::Matrix{T}`: The latent state variables. Dimensions: (latent_dim, T_steps)
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System model.
- `y::Matrix{T}`: The observed data. Dimensions: (obs_dim, T_steps)
- `w::Vector{T}`: Weights for each observation in the log-likelihood calculation. Not currently used.

# Returns
- `ll::T`: The log-likelihood value.

# Examples
```juliaestep!
lds = PoissonLDS(obs_dim=4, latent_dim=3)
x, y = sample(lds, 100, 1)  # 1 trial, 100 time steps
ll = loglikelihood(x, lds, y)
```
"""
function loglikelihood(
    x::AbstractMatrix{T}, 
    plds::LinearDynamicalSystem{S,O}, 
    y::AbstractMatrix{U}, 
    w::Vector{U}=ones(size(y, 2))
) where {T<:Real,U<:Real,W<:Float64,S<:GaussianStateModel{W},O<:PoissonObservationModel{W}}
    # Convert the log firing rate to firing rate
    d = exp.(plds.obs_model.log_d)
    T_steps = size(y, 2)

    # Pre-compute matrix inverses
    inv_p0 = inv(plds.state_model.P0)
    inv_Q = inv(plds.state_model.Q)

    # Calculate p(yₜ|xₜ)
    pygivenx_sum = zero(T)
    @inbounds for t in 1:T_steps
        x_view = @view x[:, t]
        y_view = @view y[:, t]
        temp = plds.obs_model.C * x_view .+ d
        pygivenx_sum += dot(y_view, temp) - sum(exp.(temp))
    end

    # Calculate p(x₁)
    dx1 = x[:, 1] .- plds.state_model.x0
    px1 = -T(0.5) * dot(dx1, inv_p0 * dx1)

    # Calculate p(xₜ|xₜ₋₁)
    pxtgivenxt1_sum = zero(T)
    @inbounds for t in 2:T_steps
        x_prev = @view x[:, t - 1]         
        x_curr = @view x[:, t]
        temp = x_curr .- (plds.state_model.A * x_prev)
        pxtgivenxt1_sum += -T(0.5) * dot(temp, inv_Q * temp)
    end

    # Return the log-posterior
    return pygivenx_sum + px1 + pxtgivenxt1_sum
end

"""
    loglikelihood(x::Array{T, 3}, lds::LinearDynamicalSystem{S,O}, y::Array{T, 3}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Calculate the complete-data log-likelihood of a Poisson Linear Dynamical System model for multiple trials.

# Arguments
- `x::Array{T, 3}`: The latent state variables. Dimensions: (latent_dim, T_Steps, n_trials)
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System model.
- `y::Array{T, 3}`: The observed data. Dimensions: (obs_dim, T_steps, n_trials)

# Returns
- `ll::T`: The log-likelihood value.

# Examples
```julia
lds = PoissonLDS(obs_dim=4, latent_dim=3)
x, y = sample(lds, 100, 10)  # 10 trials, 100 time steps each
ll = loglikelihood(x, lds, y)
```
"""
function loglikelihood(
    x::Array{T,3}, 
    plds::LinearDynamicalSystem{O,S}, 
    y::Array{T,3}
) where {T<:Real,U<:Float64,S<:GaussianStateModel{U},O<:PoissonObservationModel{U}}
    # Calculate the log-likelihood over all trials
    ll = zeros(size(y, 3))
    @threads for n in axes(y, 3)
        x_view = @view x[:, :, n]
        y_view = @view y[:, :, n]
        ll[n] .= loglikelihood(x_view, plds, y_view)
    end
    return sum(ll)
end

"""
    Gradient(lds::LinearDynamicalSystem{S,O}, y::Matrix{T}, x::Matrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Calculate the gradient of the log-likelihood of a Poisson Linear Dynamical System model for a single trial.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System model.
- `y::Matrix{T}`: The observed data. Dimensions: (obs_dim, T_steps)
- `x::Matrix{T}`: The latent state variables. Dimensions: (latent_dim, T_steps)
- `w::Vector{T}`: Weights for each observation in the log-likelihood calculation. Not currently used.

# Returns
- `grad::Matrix{T}`: The gradient of the log-likelihood. Dimensions: (latent_dim, T_steps)

# Note
The gradient is computed with respect to the latent states x. Each row of the returned gradient
corresponds to the gradient for a single time step.
"""
function Gradient(
    lds::LinearDynamicalSystem{S,O}, y::Matrix{T}, x::Matrix{T}, w::Vector{T}=ones(size(y, 2))
) where {T<:Real,U<:Float64, S<:GaussianStateModel{U},O<:PoissonObservationModel{U}}
    # Extract model parameters
    A, Q = lds.state_model.A, lds.state_model.Q
    C, log_d = lds.obs_model.C, lds.obs_model.log_d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    # Convert log_d to d (non-log space)
    d = exp.(log_d)

    # Get number of time steps
    T_steps = size(y, 2)

    # Precompute matrix inverses
    inv_P0 = inv(P0)
    inv_Q = inv(Q)

    # Pre-allocate gradient
    grad = zeros(lds.latent_dim, T_steps)

    # Calculate gradient for each time step
    @inbounds for t in 1:T_steps

        x_t = @view x[:, t]
        y_t = @view y[:, t]

        # Common term for all time steps
        temp = exp.(C * x_t .+ d)
        common_term = C' * (y_t - temp)

        if t == 1
            # First time step
            x_1 = x_t                        
            x_2 = @view x[:, 2]
            grad[:, t] .=
                common_term + A' * inv_Q * (x_2 .- A * x_1) - inv_P0 * (x_1 .- x0)
        elseif t == T_steps
            # Last time step
            x_curr = x_t                      
            x_prev = @view x[:, T_steps - 1]
            grad[:, t] .= common_term - inv_Q * (x_curr .- A * x_prev)
        else
            # Intermediate time steps
            x_prev = @view x[:, t - 1]      
            x_next = @view x[:, t + 1] 
            grad[:, t] .=
                common_term + A' * inv_Q * (x_next .- A * x_t) -
            inv_Q * (x_t .- A * x_prev)
        end
    end

    return grad
end

"""
    Hessian(lds::LinearDynamicalSystem{S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Calculate the Hessian matrix of the log-likelihood for a Poisson Linear Dynamical System.

This function computes the Hessian matrix, which represents the second-order partial derivatives
of the log-likelihood with respect to the latent states.

# Arguments
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System with Poisson observations.
- `y::AbstractMatrix{T}`: The observed data. Dimensions: (obs_dim, T_steps)
- `x::AbstractMatrix{T}`: The current estimate of latent states. Dimensions: (latent_dim, T_steps)
- `w::Vector{T}`: Weights for each observation in the log-likelihood calculation. Not currently used.

# Returns
- `H::Matrix{T}`: The full Hessian matrix.
- `H_diag::Vector{Matrix{T}}`: The main diagonal blocks of the Hessian.
- `H_super::Vector{Matrix{T}}`: The super-diagonal blocks of the Hessian.
- `H_sub::Vector{Matrix{T}}`: The sub-diagonal blocks of the Hessian.

"""
function Hessian(
    lds::LinearDynamicalSystem{S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}, w::Vector{T}=ones(size(y, 2))
) where {T<:Real,U<:Float64, S<:GaussianStateModel{U},O<:PoissonObservationModel{U}}
    # Extract model components
    A, Q = lds.state_model.A, lds.state_model.Q
    C, log_d = lds.obs_model.C, lds.obs_model.log_d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    # Convert log_d to d i.e. non-log space
    d = exp.(log_d)

    # Pre-compute a few things
    T_steps = size(y, 2)
    inv_Q = pinv(Q)
    inv_P0 = pinv(P0)

    # Calculate super and sub diagonals
    H_sub_entry = inv_Q * A
    H_super_entry = permutedims(H_sub_entry)

    H_sub = Vector{typeof(H_sub_entry)}(undef, T_steps - 1)
    H_super = Vector{typeof(H_super_entry)}(undef, T_steps - 1)

    @inbounds for i in 1:(T_steps - 1)
        H_sub[i] = H_sub_entry
        H_super[i] = H_super_entry
    end

    # Pre-compute common terms
    xt_given_xt_1 = -inv_Q
    xt1_given_xt = -A' * inv_Q * A
    x_t = -inv_P0

    # Helper function to calculate the Poisson Hessian term
    function calculate_poisson_hess(C::Matrix{T}, λ::Vector{T}) where {T<:Real}
        return -C' * Diagonal(λ) * C
    end

    # Calculate the main diagonal
    H_diag = Vector{Matrix{T}}(undef, T_steps)

    @inbounds for t in 1:T_steps
        x_view = @view x[:, t]
        λ = exp.(C * x_view .+ d)
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
function Q_state(
    A::Matrix{T},
    Q::Matrix{T},
    P0::Matrix{T},
    x0::Vector{T},
    E_z::Array{T,3},
    E_zz::Array{T,4},
    E_zz_prev::Array{T,4},
) where {T<:Real}
    # Calculate the Q-function for the state model
    Q_val = 0.0

    # Calcualte over trials
    @threads for k in axes(E_z, 3)
        Q_val += Q_state(
            A, Q, P0, x0, E_z[:, :, k], E_zz[:, :, :, k], E_zz_prev[:, :, :, k]
        )
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
function Q_observation_model(
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    E_z::AbstractArray{U,3},
    P_smooth::AbstractArray{U,4},
    y::Array{U,3},
) where {T<:Real,U<:Real}
    # Get dimensions
    obs_dim, state_dim = size(C)

    # Re-parametrize log_d
    d = exp.(log_d)

    # Compute Q
    Q_val = zero(T)
    trials = size(E_z, 3)
    time_steps = size(E_z, 2)

    # Pre-allocate
    h = Vector{T}(undef, obs_dim)
    ρ = Vector{T}(undef, obs_dim)
    CC = Matrix{T}(undef, obs_dim, state_dim^2)

    # calculate CC term
    CC = zeros(T, size(C, 1), size(C, 2)^2)
    for i in axes(C, 1)
        CC[i, :] .= vec(C[i, :] * C[i, :]')
    end

    # sum over trials
    @threads for k in 1:trials
        # sum over time-points
        @inbounds for t in 1:time_steps
            # Mean term
            h .= (C * E_z[:, t, k]) .+ d

            # calculate rho
            ρ .= T(0.5) .* CC * vec(P_smooth[:, :, t, k])

            ŷ = exp.(h .+ ρ)

            # calculate the Q-value
            Q_val += sum((y[:, t, k] .* h) .- ŷ)
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
function Q_function(
    A::Matrix{T},
    Q::Matrix{T},
    C::Matrix{T},
    log_d::Vector{T},
    x0::Vector{T},
    P0::Matrix{T},
    E_z::Array{T,3},
    E_zz::Array{T,4},
    E_zz_prev::Array{T,4},
    P_smooth::Array{T,4},
    y::Array{T,3},
) where {T<:Real}
    # Calculate the Q-function for the state model
    Q_state = StateSpaceDynamics.Q_state(A, Q, P0, x0, E_z, E_zz, E_zz_prev)
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
- `E_z::Array{T, 3}`: Expected values of latent states. Dimensions: (state_dim, t_steps, n_trials).
- `E_zz::Array{T, 4}`: Expected values of latent state outer products. Dimensions: (state_dim, state_dim, t_steps, n_trials).
- `E_zz_prev::Array{T, 4}`: Expected values of latent state outer products with previous time step. Dimensions: (state dimension, state dimension, t_steps-1, n_trials).
- `P_smooth::Array{T, 4}`: Smoothed covariance matrices. Dimensions: (state dimension, state dimension, t_steps, n_trials).
- `y::Array{T, 3}`: Observed data. Dimensions: (obs_dim, t_steps, n_trials).

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
function calculate_elbo(
    plds::LinearDynamicalSystem{S,O},
    E_z::Array{T,3},
    E_zz::Array{T,4},
    E_zz_prev::Array{T,4},
    P_smooth::Array{T,4},
    y::Array{T,3},
    total_entropy::Float64,
) where {T<:Real,U<:Float64, S<:GaussianStateModel{U},O<:PoissonObservationModel{U}}
    # Set up parameters
    A, Q, x0, p0 = plds.state_model.A,
    plds.state_model.Q, plds.state_model.x0,
    plds.state_model.P0
    C, log_d = plds.obs_model.C, plds.obs_model.log_d

    # Calculate the expected complete log-likelihood
    ecll = Q_function(A, Q, C, log_d, x0, p0, E_z, E_zz, E_zz_prev, P_smooth, y)

    # Return the ELBO
    return ecll - total_entropy
end

"""
    gradient_observation_model!(grad::AbstractVector{T}, C::AbstractMatrix{T}, log_d::AbstractVector{T}, E_z::AbstractArray{T}, P_smooth::AbstractArray{T}, y::Array{T}) where T<:Real

Compute the gradient of the Q-function with respect to the observation model parameters (C and log_d) for a Poisson Linear Dynamical System.

# Arguments
- `grad::AbstractVector{T}`: Pre-allocated vector to store the computed gradient.
- `C::AbstractMatrix{T}`: The observation matrix. Dimensions: (obs_dim, latent_dim)
- `log_d::AbstractVector{T}`: The log of the baseline firing rates. Dimensions: (obs_dim,)
- `E_z::AbstractArray{T}`: The expected latent states. Dimensions: (latent_dim, t_steps, n_trials)
- `P_smooth::AbstractArray{T}`: The smoothed state covariances. Dimensions: (latent_dim, latent_dim, t_steps, n_trials)
- `y::Array{T}`: The observed data. Dimensions: (obs_dim, t_steps, N-trials)

# Note
This function modifies `grad` in-place. The gradient is computed for the negative Q-function,
as we're minimizing -Q in optimization routines.
"""
function gradient_observation_model!(
    grad::AbstractVector{T},
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    E_z::AbstractArray{T},
    P_smooth::AbstractArray{T},
    y::Array{T},
) where {T<:Real}
    d = exp.(log_d)
    obs_dim, latent_dim = size(C)
    latent_dim, time_steps, trials = size(E_z)
    
    # Pre-allocate shared temporary arrays
    h = zeros(T, obs_dim)
    ρ = zeros(T, obs_dim)
    λ = zeros(T, obs_dim)
    CP_row = zeros(T, latent_dim)  # Single row buffer for CP computations
    
    fill!(grad, zero(T))
    
    @threads for k in 1:trials
        # Local temporary arrays for each thread
        local_grad = zeros(T, length(grad))
        
        for t in 1:time_steps
            # Use views for all array slices
            z_t = @view E_z[:, t, k]
            P_t = @view P_smooth[:, :, t, k]
            y_t = @view y[:, t, k]
            
            # Compute h = C * z_t + d in-place
            mul!(h, C, z_t)
            h .+= d
            
            # Compute ρ more efficiently using local storage
            @inbounds for i in 1:obs_dim
                # Compute one row of CP at a time
                mul!(CP_row, P_t', C[i, :])
                ρ[i] = T(0.5) * dot(C[i, :], CP_row)
            end
            
            # Compute λ in-place
            @. λ = exp(h + ρ)
            
            # Gradient computation with fewer allocations
            @inbounds for j in 1:latent_dim
                Pj = @view P_t[:, j]
                for i in 1:obs_dim
                    idx = (j - 1) * obs_dim + i
                    CP_term = dot(C[i, :], Pj)
                    local_grad[idx] += y_t[i] * z_t[j] - λ[i] * (z_t[j] + CP_term)
                end
            end
            
            # Update log_d gradient
            @views local_grad[(end - obs_dim + 1):end] .+= (y_t .- λ) .* d
        end
        
        # Thread-safe update of global gradient
        grad .+= local_grad
    end
    
    return grad .*= -1
end

"""
    update_observation_model!(plds::LinearDynamicalSystem{S,O}, E_z::Array{T, 3}, P_smooth::Array{T, 4}, y::Array{T, 3}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Update the observation model parameters of a Poisson Linear Dynamical System using gradient-based optimization.

# Arguments
- `plds::LinearDynamicalSystem{S,O}`: The Poisson Linear Dynamical System model.
- `E_z::Array{T, 3}`: The expected latent states. Dimensions: (latent_dim, T_Steps, n_trials)
- `P_smooth::Array{T, 4}`: The smoothed state covariances. Dimensions: (latent_dim, T_Steps, n_trials, latent_dim)
- `y::Array{T, 3}`: The observed data. Dimensions: (obs_dim, T_steps, n_trials)

# Note
This function modifies `plds` in-place by updating the observation model parameters (C and log_d).
The optimization is performed only if `plds.fit_bool[5]` is true.
"""
function update_observation_model!(
    plds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, P_smooth::Array{T,4}, y::Array{T,3}
) where {T<:Real,U<:Float64,S<:GaussianStateModel{U},O<:PoissonObservationModel{U}}
    if plds.fit_bool[5]

        params = vcat(vec(plds.obs_model.C), plds.obs_model.log_d)

        function f(params::Vector{T})
            C_size = plds.obs_dim * plds.latent_dim
            log_d = params[(end - plds.obs_dim + 1):end]
            C = reshape(params[1:C_size], plds.obs_dim, plds.latent_dim)
            return -Q_observation_model(C, log_d, E_z, P_smooth, y)
        end

        function g!(grad::Vector{T}, params::Vector{T})
            C_size = plds.obs_dim * plds.latent_dim
            log_d = params[(end - plds.obs_dim + 1):end]
            C = reshape(params[1:C_size], plds.obs_dim, plds.latent_dim)
            return gradient_observation_model!(grad, C, log_d, E_z, P_smooth, y)
        end

        opts = Optim.Options(
            x_reltol=1e-12,
            x_abstol=1e-12,
            g_abstol=1e-12,
            f_reltol=1e-12,
            f_abstol=1e-12,
        )

        # use CG result as inital guess for LBFGS
        result = optimize(f, g!, params, LBFGS(;linesearch=LineSearches.MoreThuente()), opts)

        # Update the parameters
        C_size = plds.obs_dim * plds.latent_dim
        plds.obs_model.C = reshape(
            result.minimizer[1:C_size], plds.obs_dim, plds.latent_dim
        )
        plds.obs_model.log_d = result.minimizer[(end - plds.obs_dim + 1):end]
    end
end

"""
    mstep!(plds::LinearDynamicalSystem{S,O}, E_z::Array{T,3}, E_zz::Array{T,4}, E_zz_Prev{T,4}, p_smooth{T,4}, y::Array{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Perform the M-step of the EM algorithm for a Poisson Linear Dynamical System with multi-trial data.

# Arguments
- `plds::LinearDynamicalSystem{S,O}`: The Poisson Linear Dynamical System struct.
- `E_z::Array{T,3}`: Expected latent states, size (state_dim, state_dim, T_steps, n_trials)
- `E_zz::Array{T,4}`: Expected z_t * z_t', size (state_dim, state_dim, T_steps, n_trials)
- `E_zz_prev::Array{T,4}`: Expected z_t * z_{t-1}', size (state_dim, state_dim, T_steps, n_trials)
- `p_smooth::Array{T,4}`: Smoothed state covariances, size (state_dim, state_dim, T_steps, n_trials)
- `y::Array{T,3}`: Observed data, size (obs_dim, T_steps, n_trials)

# Note
- This function modifies `plds` in-place by updating all model parameters.
"""
function mstep!(
    plds::LinearDynamicalSystem{S,O},
    E_z::Array{T,3},
    E_zz::Array{T,4},
    E_zz_prev::Array{T,4},
    p_smooth::Array{T,4},
    y::Array{T,3},
) where {T<:Real,U<:Float64, S<:GaussianStateModel{U},O<:PoissonObservationModel{U}}
    # Get old params
    old_params = vec(stateparams(plds))
    old_params = vcat(old_params, vec(obsparams(plds)))
    # Update parameters
    update_initial_state_mean!(plds, E_z)
    update_initial_state_covariance!(plds, E_z, E_zz)
    update_A!(plds, E_zz, E_zz_prev)
    update_Q!(plds, E_zz, E_zz_prev)
    update_observation_model!(plds, E_z, p_smooth, y)
    # Get new params
    new_params = vec(stateparams(plds))
    new_params = vcat(new_params, vec(obsparams(plds)))

    norm_params = norm(new_params - old_params)
    return norm_params
end
