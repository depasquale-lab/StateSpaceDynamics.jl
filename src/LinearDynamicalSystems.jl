export LinearDynamicalSystem, GaussianStateModel, GaussianObservationModel, PoissonObservationModel, rand, smooth, fit!

"""
    GaussianStateModel{T<:Real. M<:AbstractMatrix{T}, V<:AbstractVector{T}}}

Represents the state model of a Linear Dynamical System with Gaussian noise.

# Fields 
- `A::M`: Transition matrix (size `latent_dim×latent_dim`). 
- `Q::M`: Process noise covariance matrix 
- `x0::V`: Initial state vector (length `latent_dim`).
- `P0::M`: Initial state covariance matrix (size `latent_dim×latent_dim
"""
Base.@kwdef mutable struct GaussianStateModel{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}} <: AbstractStateModel{T}
    A::M
    Q::M
    x0::V
    P0::M 
end

"""
    GaussianObservationModel{T<:Real, M<:AbstractMatrix{T}} 

Represents the observation model of a Linear Dynamical System with Gaussian noise.

# Fields
- `C::M`: Observation matrix of size `(obs_dim × latent_dim)`. Maps latent states into observation space. 
- `R::M`: Observation noise covariance of size `(obs_dim × obs_dim)`. 
"""
Base.@kwdef mutable struct GaussianObservationModel{T<:Real, M<:AbstractMatrix{T}} <: AbstractObservationModel{T}
    C::M
    R::M
end

"""
    PoissonObservationModel{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}} <: AbstractObservationModel{T}

Represents the observation model of a Linear Dynamical System with Poisson observations.

# Fields
- `C::AbstractMatrix{T}`: Observation matrix of size `(obs_dim × latent_dim)`. Maps latent states into observation space.
- `log_d::AbstractVector{T}`: Mean firing rate vector (log space) of size `(obs_dim × obs_dim)`. 
"""
Base.@kwdef mutable struct PoissonObservationModel{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}} <: AbstractObservationModel{T}
    C::M
    log_d::V
end

"""
    LinearDynamicalSystem{T<:Real, S<:AbstractStateModel{T}, O<:AbstractObservationModel{T}}

Represents a unified Linear Dynamical System with customizable state and observation models.

# Fields
- `state_model::S`: The state model (e.g., GaussianStateModel)
- `obs_model::O`: The observation model (e.g., GaussianObservationModel or PoissonObservationModel)
- `latent_dim::Int`: Dimension of the latent state
- `obs_dim::Int`: Dimension of the observations
- `fit_bool::Vector{Bool}`: Vector indicating which parameters to fit during optimization
"""
Base.@kwdef struct LinearDynamicalSystem{T<:Real, S<:AbstractStateModel{T}, O<:AbstractObservationModel{T}}
    state_model::S
    obs_model::O
    latent_dim::Int
    obs_dim::Int
    fit_bool::Vector{Bool}
end

"""
    stateparams(lds::LinearDynamicalSystem{T,S,O}) 

Extract the state parameters from a Linear Dynamical System.
"""
function stateparams(
    lds::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    if isa(lds.state_model, GaussianStateModel)
        return [
            lds.state_model.A, lds.state_model.Q, lds.state_model.x0, lds.state_model.P0
        ]
    end
end

"""
    obsparams(lds::LinearDynamicalSystem{S,O}) 

Extract the observation parameters from a Linear Dynamical System.
"""
function obsparams(
    lds::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
    if isa(lds.obs_model, GaussianObservationModel)
        return [lds.obs_model.C, lds.obs_model.R]
    elseif isa(lds.obs_model, PoissonObservationModel)
        return [lds.obs_model.C, lds.obs_model.log_d]
    end
end


"""
    initialize_FilterSmooth(model, num_obs) 
   
Initialize a `FilterSmooth` object for a given linear dynamical system model and number of observations.
"""
function initialize_FilterSmooth(
    model::LinearDynamicalSystem{T,S,O}, 
    num_obs::Int
) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}
    num_states = model.latent_dim
    FilterSmooth{T}(
        zeros(T, num_states, num_obs),                    # x_smooth
        zeros(T, num_states, num_states, num_obs),        # p_smooth  
        zeros(T, num_states, num_states, num_obs),        # p_smooth_tt1
        zeros(T, num_states, num_obs),                    # E_z
        zeros(T, num_states, num_states, num_obs),        # E_zz
        zeros(T, num_states, num_states, num_obs),        # E_zz_prev
        zero(T)                                           # entropy
    )
end

function initialize_FilterSmooth(
    model::LinearDynamicalSystem{T,S,O},
    tsteps::Int,
    ntrials::Int
) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}
    filter_smooths = [initialize_FilterSmooth(model, tsteps) for _ in 1:ntrials]
    return TrialFilterSmooth(filter_smooths)
end

function Random.rand(rng::AbstractRNG, lds::LinearDynamicalSystem{T,S,O}; tsteps::Int, ntrials::Int) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    A, Q, x0, P0 = lds.state_model.A, lds.state_model.Q, lds.state_model.x0, lds.state_model.P0
    C, R = lds.obs_model.C, lds.obs_model.R

    x = Array{T,3}(undef, lds.latent_dim, tsteps, ntrials)
    y = Array{T,3}(undef, lds.obs_dim, tsteps, ntrials)

    for trial in 1:ntrials
        x[:, 1, trial] = rand(rng, MvNormal(x0, P0))
        y[:, 1, trial] = rand(rng, MvNormal(C * x[:, 1, trial], R))

        for t in 2:tsteps
            x[:, t, trial] = rand(rng, MvNormal(A * x[:, t - 1, trial], Q))
            y[:, t, trial] = rand(rng, MvNormal(C * x[:, t, trial], R))
        end
    end

    return x, y
end

# For Poisson LDS
function Random.rand(rng::AbstractRNG, lds::LinearDynamicalSystem{T,S,O}; tsteps::Int, ntrials::Int) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    # Extract model components
    A, Q = lds.state_model.A, lds.state_model.Q
    C, log_d = lds.obs_model.C, lds.obs_model.log_d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    # Convert log_d to d i.e. non-log space
    d = exp.(log_d)

    # Pre-allocate arrays
    x = zeros(T, lds.latent_dim, tsteps, ntrials)
    y = zeros(T, lds.obs_dim, tsteps, ntrials)

    for k in 1:ntrials
        # Sample the initial state
        x[:, 1, k] = rand(rng, MvNormal(x0, P0))
        y[:, 1, k] = rand.(rng, Poisson.(exp.(C * x[:, 1, k] .+ d)))

        # Sample the rest of the states
        for t in 2:tsteps
            x[:, t, k] = rand(rng, MvNormal(A * x[:, t - 1, k], Q))
            y[:, t, k] = rand.(rng, Poisson.(exp.(C * x[:, t, k] + d)))
        end
    end

    return x, y
end

"""
    Random.rand(lds::LinearDynamicalSystem; tsteps::Int, ntrials::Int)
    Random.rand(rng::AbstractRNG, lds::LinearDynamicalSystem; tsteps::Int, ntrials::Int)

Sample from a Linear Dynamical System.
"""
function Random.rand(lds::LinearDynamicalSystem; kwargs...)
    return rand(Random.default_rng(), lds; kwargs...)
end

"""
    loglikelihood(x::AbstractMatrix{T}, lds::LinearDynamicalSystem{S,O}, y::AbstractMatrix{T}) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Calculate the complete-data log-likelihood of a linear dynamical system (LDS) given the observed data.

# Arguments
- `x::AbstractMatrix{T}`: The state sequence of the LDS.
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System.
- `y::AbstractMatrix{T}`: The observed data.
- `w::AbstractVector{T}`: coeffcients to weight the data.

# Returns
- `ll::T`: The complete-data log-likelihood of the LDS.
"""
function loglikelihood(
    x::AbstractMatrix{U}, lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T},
    w::Union{Nothing,AbstractVector{T}} = nothing
) where {U<:Real, T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

    if w === nothing
        w = ones(eltype(y), size(y, 2))
    elseif eltype(w) !== eltype(y)
        error("weights must be Vector{$(eltype(y))}; Got Vector{$(eltype(w))}")
    end

    tsteps = size(y, 2)
    A, Q, x0, P0 = lds.state_model.A, lds.state_model.Q, lds.state_model.x0, lds.state_model.P0
    C, R = lds.obs_model.C, lds.obs_model.R

    # Use symmetric Cholesky to allow AD with dual types
    R_chol = cholesky(Symmetric(R)).U
    Q_chol = cholesky(Symmetric(Q)).U
    P0_chol = cholesky(Symmetric(P0)).U

    dx0 = view(x, :, 1) .- x0
    ll = sum(abs2, P0_chol \ dx0)

    # Allocate temporary arrays with correct type
    temp_dx = zeros(eltype(x), size(x, 1))
    temp_dy = zeros(eltype(x), size(y, 1))
    
    # Pre-allocate solve result vectors (reuse these!)
    temp_solve_Q = zeros(eltype(x), size(x, 1))
    temp_solve_R = zeros(eltype(x), size(y, 1))

    for t in 1:tsteps
        if t > 1
            mul!(temp_dx, A, view(x, :, t-1), -one(eltype(x)), false)
            temp_dx .+= view(x, :, t)
            
            # In-place solve: temp_solve_Q = Q_chol \ temp_dx
            ldiv!(temp_solve_Q, Q_chol, temp_dx)
            ll += sum(abs2, temp_solve_Q)
        end
        
        mul!(temp_dy, C, view(x, :, t), -one(eltype(x)), false)
        temp_dy .+= view(y, :, t)
        
        # In-place solve: temp_solve_R = R_chol \ temp_dy  
        ldiv!(temp_solve_R, R_chol, temp_dy)
        ll += w[t] * sum(abs2, temp_solve_R)
    end

    return -eltype(x)(0.5) * ll
end


"""
    Gradient(lds, y, x) 

Compute the gradient of the log-likelihood with respect to the latent states for a linear dynamical system.
"""
function Gradient(
    lds::LinearDynamicalSystem{T,S,O}, 
    y::AbstractMatrix{T}, 
    x::AbstractMatrix{T},
    w::Union{Nothing,AbstractVector{T}} = nothing
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    if w === nothing
        w = ones(T, size(y,2)) 
    end
    # Dims etc.
    latent_dim, tsteps = size(x)
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

    grad = zeros(T, latent_dim, tsteps)

    # Pre-allocate dxt, dxt_next, dyt, and two temps for efficiency
    dxt       = zeros(T, latent_dim)
    dxt_next  = zeros(T, latent_dim)
    dyt       = zeros(T, obs_dim)
    tmp1      = zeros(T, latent_dim)  # for w[t] * C_inv_R * dyt
    tmp2      = zeros(T, latent_dim)  # for A_inv_Q * dxt_next


    # First time step
    dxt .= x[:, 1] - x0
    dxt_next .= x[:, 2] - A * x[:, 1]
    dyt .= y[:, 1] - C * x[:, 1]

    grad[:, 1] .= A_inv_Q * dxt_next + w[1] * C_inv_R * dyt - (P0_chol \ dxt)

    @views for t in 2:(tsteps - 1)
        # dxt = x[:, t] - A * x[:, t-1]
        mul!(dxt, A, x[:, t - 1])
        dxt .= x[:, t] .- dxt

        # dxt_next = x[:, t+1] - A * x[:, t]
        mul!(dxt_next, A, x[:, t])
        dxt_next .= x[:, t + 1] .- dxt_next

        # dyt = y[:, t] - C * x[:, t]
        mul!(dyt, C, x[:, t])
        dyt .= y[:, t] .- dyt

        # tmp1 = w[t] * C_inv_R * dyt
        mul!(tmp1, C_inv_R, dyt)
        tmp1 .*= w[t]

        # tmp2 = A_inv_Q * dxt_next
        mul!(tmp2, A_inv_Q, dxt_next)

        # grad[:, t] = tmp1 - (Q_chol \ dxt) + tmp2
        grad[:, t] .= tmp1 .- (Q_chol \ dxt) .+ tmp2
    end

    # Last time step
    dxt .= x[:, tsteps] - A * x[:, tsteps - 1]
    dyt .= y[:, tsteps] - C * x[:, tsteps]

    grad[:, tsteps] .= w[tsteps] * (C_inv_R * dyt) - (Q_chol \ dxt)

    return grad
end

"""
    Hessian(lds, y, x) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Construct the Hessian matrix of the log-likelihood of the LDS model given a set of observations.

This function is used for the direct optimization of the log-likelihood as advocated by Paninski et al. (2009). 
The block tridiagonal structure of the Hessian is exploited to reduce the number of parameters that need to be computed,
and to reduce the memory requirements. Together with the gradient, this allows for Kalman Smoothing to be performed 
by simply solving a linear system of equations:

    ̂xₙ₊₁ = ̂xₙ - H \\ ∇

where ̂xₙ is the current smoothed state estimate, H is the Hessian matrix, and ∇ is the gradient of the log-likelihood.

# Note 
- `x` is not used in this function, but is required to match the function signature of other Hessian calculations e.g., in PoissonLDS.
"""
function Hessian(
    lds::LinearDynamicalSystem{T,S,O}, 
    y::AbstractMatrix{T}, 
    x::AbstractMatrix{T},
    w::Union{Nothing,AbstractVector{T}} = nothing
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    if w === nothing
        w = ones(T, size(y,2)) 
    end 

    A, Q, x0, P0 = lds.state_model.A,
    lds.state_model.Q, lds.state_model.x0,
    lds.state_model.P0
    C, R = lds.obs_model.C, lds.obs_model.R

    tsteps = size(y, 2)

    # Pre-compute inverses
    inv_R = Symmetric(inv(R))
    inv_Q = Symmetric(inv(Q))
    inv_P0 = Symmetric(inv(P0))

    # Off-diagonal terms
    H_sub_entry = inv_Q * A
    H_super_entry = Matrix(H_sub_entry')

    H_sub = [H_sub_entry for _ in 1:(tsteps - 1)]
    H_super = [H_super_entry for _ in 1:(tsteps - 1)]

    # Calculate main diagonal terms
    yt_given_xt = -C' * inv_R * C
    xt_given_xt_1 = -inv_Q
    xt1_given_xt = -A' * inv_Q * A
    x_t = -inv_P0

    # Pre-compute constant terms to avoid repeated allocation
    middle_base = xt_given_xt_1 + xt1_given_xt  # This is constant for middle entries
    last_base = xt_given_xt_1                   # This is the term for the last entry

    H_diag = [Matrix{T}(undef, size(A, 1), size(A, 1)) for _ in 1:tsteps]

    # Build main diagonal efficiently
    @. H_diag[1] = w[1] * yt_given_xt + xt1_given_xt + x_t

    # avoid temporaries in the loop
    for i in 2:(tsteps - 1)
        @. H_diag[i] = w[i] * yt_given_xt + middle_base
    end

    @. H_diag[tsteps] = w[tsteps] * yt_given_xt + last_base

    H = StateSpaceDynamics.block_tridgm(H_diag, H_super, H_sub)

    return H, H_diag, H_super, H_sub
end

"""
    smooth(lds, y)

This function performs direct smoothing for a linear dynamical system (LDS) given the system parameters and the observed data for a single trial

# Arguments
- `lds::LinearDynamicalSystem{T,S,O}`: The LDS object representing the system parameters.
- `y::AbstractMatrix{T}`: The observed data matrix.
- `w::Union{Nothing,AbstractVector{T}}`: coeffcients to weight the data.

# Returns
- `x::AbstractMatrix{T}`: The optimal state estimate.
- `p_smooth::Array{T, 3}`: The posterior covariance matrix.
- `inverse_offdiag::Array{T, 3}`: The inverse off-diagonal matrix.
- `Q_val::T`: The Q-function value.
"""
function smooth(lds::LinearDynamicalSystem, y::AbstractMatrix{T}, w::Union{Nothing, AbstractVector{T}}=nothing) where {T}
    fs = initialize_FilterSmooth(lds, size(y, 2))
    smooth!(lds, fs, y, w)
    return fs.x_smooth, fs.p_smooth
end

function smooth(lds::LinearDynamicalSystem, y::AbstractArray{T,3},
                w::Union{Nothing,AbstractVector{T}}=nothing) where {T}
    tfs = initialize_FilterSmooth(lds, size(y,2), size(y,3))
    smooth!(lds, tfs, y)

    D  = lds.latent_dim
    Tt = size(y, 2)
    N  = size(y, 3)

    xs = Array{T,3}(undef, D, Tt, N)
    Ps = Array{T,4}(undef, D, D, Tt, N)

    for n in 1:N
        fs = tfs.FilterSmooths[n]
        xs[:,:,n]      .= fs.x_smooth
        Ps[:,:,:,n]    .= fs.p_smooth
    end
    return xs, Ps
end


function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    fs::FilterSmooth{T},
    y::AbstractMatrix{T}, 
    w::Union{Nothing,AbstractVector{T}} = nothing
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    if w === nothing
        w = ones(T, size(y,2)) 
    elseif eltype(w) !== T
        error("weights must be Vector{$(T)}; Got Vector{$(eltype(w))}")
    end

    tsteps, D = size(y, 2), lds.latent_dim

    # use old fs if it exists, by default is zeros if no iteration of EM has occurred
    X₀ = Vector{T}(vec(fs.E_z))

    function nll(vec_x::AbstractVector{T})
        x = reshape(vec_x, D, tsteps)
        return -loglikelihood(x, lds, y, w)
    end

    function g!(g::Vector{T}, vec_x::Vector{T})
        x = reshape(vec_x, D, tsteps)
        grad = Gradient(lds, y, x, w)
        return g .= vec(-grad)
    end

    function h!(h::SparseMatrixCSC{T}, vec_x::Vector{T}) where {T<:Real}
        x = reshape(vec_x, D, tsteps)
        H, _, _, _ = Hessian(lds, y, x, w)
        mul!(h, -1.0, H)  # h .= -H, in-place 
        return nothing
    end

    # set up initial values
    initial_f = nll(X₀)

    inital_g = similar(X₀)
    g!(inital_g, X₀)

    initial_h = spzeros(T, length(X₀), length(X₀))
    h!(initial_h, X₀)

    # set up a TwiceDifferentiable object i guess?
    td = TwiceDifferentiable(nll, g!, h!, X₀, initial_f, inital_g, initial_h)

    # set up Optim.Options
    opts = Optim.Options(; g_abstol=1e-8, x_abstol=1e-8, f_abstol=1e-8, iterations=100)

    # Go!
    res = optimize(td, X₀, Newton(;linesearch=LineSearches.BackTracking()), opts)

    # Profit
    fs.x_smooth .= reshape(res.minimizer, D, tsteps)

    H, main, super, sub = Hessian(lds, y, fs.x_smooth, w)

    # Get the second moments of the latent state path, use static matrices if the latent dimension is small
    if lds.latent_dim > 10
        p_smooth_result, p_smooth_tt1_result = block_tridiagonal_inverse(-sub, -main, -super)
        fs.p_smooth .= p_smooth_result
        fs.p_smooth_tt1[:, :, 2:end] .= p_smooth_tt1_result
    else
        p_smooth_result, p_smooth_tt1_result = block_tridiagonal_inverse_static(-sub, -main, -super, Val(lds.latent_dim))
        fs.p_smooth .= p_smooth_result
        fs.p_smooth_tt1[:, :, 2:end] .= p_smooth_tt1_result
    end

    # Calculate the entropy, see Utilities.jl for the function
    fs.entropy = gaussian_entropy(Symmetric(H))

    # Symmetrize the covariance matrices
    @views for i in 1:tsteps
        fs.p_smooth[:, :, i] .= 0.5 .* (fs.p_smooth[:, :, i] .+ fs.p_smooth[:, :, i]')
    end

    return fs
end

"""
    smooth(lds, y) 

This function performs direct smoothing for a linear dynamical system (LDS) given the system parameters and the observed data for multiple trials.

# Arguments
- `lds::LinearDynamicalSystem{T,S,O}`: The LDS object representing the system parameters.
- `y::AbstractArray{T,3}`: The observed data array with dimensions (obs_dim, tsteps, ntrials).

# Returns
- `x::AbstractArray{T,3}`: The optimal state estimates with dimensions (ntrials, tsteps, latent_dim).
- `p_smooth::AbstractArray{T,4}`: The posterior covariance matrices with dimensions (latent_dim, latent_dim, tsteps, ntrials).
- `inverse_offdiag::AbstractArray{T,4}`: The inverse off-diagonal matrices with dimensions (latent_dim, latent_dim, tsteps, ntrials).
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    ntrials = size(y, 3)
    
    if ntrials == 1
        # Single trial - no threading overhead
        smooth!(lds, tfs[1], y[:, :, 1])
    else
        # Multiple trials - use threading
        @views @threads for trial in 1:ntrials
            smooth!(lds, tfs[trial], y[:, :, trial])
        end
    end
    
    return tfs
end

"""
    Q_state(A, Q, P0, x0, E_z, E_zz, E_zz_prev) 

Calculate the state component of the Q-function for the EM algorithm in a Linear Dynamical System.
"""
function Q_state(
    A::AbstractMatrix{T},
    Q::AbstractMatrix{T},
    P0::AbstractMatrix{T},
    x0::AbstractVector{T},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
) where {T<:Real}
    tstep = size(E_z, 2)
    state_dim = size(A, 1)
    
    # Pre-compute constants and decompositions once
    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))
    log_det_Q = logdet(Q_chol)
    log_det_P0 = logdet(P0_chol)
    
    # Pre-allocate temp matrix
    temp = zeros(T, state_dim, state_dim)
    
    # First time step (handled separately)
    mul!(temp, E_z[:, 1], x0', T(-1.0), T(0.0))  # -E_z[:,1] * x0'
    temp .+= view(E_zz, :, :, 1)           # Add E_zz[:,:,1]
    temp .-= x0 * E_z[:, 1]'               # Subtract x0 * E_z[:,1]'
    temp .+= x0 * x0'                      # Add x0 * x0'
    Q_val = T(-0.5) * (log_det_P0 + tr(P0_chol \ temp))
    
    # Pre-allocate sums for t ≥ 2
    sum_E_zz_current = zeros(T, state_dim, state_dim)
    sum_E_zz_prev_cross = zeros(T, state_dim, state_dim)
    sum_E_zz_prev_time = zeros(T, state_dim, state_dim)
    
    # Compute sums with views
    for t in 2:tstep
        sum_E_zz_current .+= view(E_zz, :, :, t)
        sum_E_zz_prev_cross .+= view(E_zz_prev, :, :, t)
        sum_E_zz_prev_time .+= view(E_zz, :, :, t-1)
    end
    
    # Compute transition term
    copyto!(temp, sum_E_zz_current)
    mul!(temp, A, sum_E_zz_prev_cross', T(-1.0), T(1.0))
    temp .-= sum_E_zz_prev_cross * A'
    mul!(temp, A, sum_E_zz_prev_time * A', T(1.0), T(1.0))
    
    # Add remaining time steps
    Q_val += T(-0.5) * ((tstep - 1) * log_det_Q + tr(Q_chol \ temp))
    
    return Q_val
end


"""
    Q_obs(H, R, E_z, E_zz, y) 

Calculate the a single time step observation component of the Q-function for the EM algorithm in a Linear Dynamical System before the R^-1 is accounted for.
"""
function Q_obs!(
    result::AbstractMatrix{T},
    H::AbstractMatrix{T},
    E_z::AbstractVector{T},
    E_zz::AbstractMatrix{T},
    y::AbstractVector{T},
    buffers
) where {T<:Real}
    
    # Unpack buffers (no allocation!)
    sum_yy, sum_yz, temp_result, work1, work2 = buffers
    
    # All operations use pre-allocated buffers
    mul!(sum_yy, y, y')
    
    # Efficient outer product: sum_yz = y * E_z'
    fill!(sum_yz, zero(T))
    BLAS.ger!(one(T), y, E_z, sum_yz)  # sum_yz += y * E_z'
    
    # Build result using buffers
    copyto!(result, sum_yy)
    mul!(result, H, sum_yz', -one(T), one(T))   # result -= H * sum_yz'
    mul!(work1, sum_yz, H')                     # work1 = sum_yz * H'  
    result .-= work1                            # result -= work1
    mul!(work2, E_zz, H')                       # work2 = E_zz * H'
    mul!(result, H, work2, one(T), one(T))      # result += H * work2
    
    return result
end


"""
    Q_obs(H, R, E_z, E_zz, y) where {T<:Real}

Calculate the observation component of the Q-function for the EM algorithm in a Linear Dynamical System.
"""
function Q_obs(
    H::AbstractMatrix{T},
    R::AbstractMatrix{T},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    y::AbstractMatrix{T},
    w::Union{Nothing,AbstractVector{T}} = nothing
) where {T<:Real}
    if w === nothing
        w = ones(T, size(y,2)) 
    end 

    obs_dim = size(H, 1)
    latent_dim = size(E_z, 1)
    tstep = size(E_z, 2)
    
    # Pre-compute constants
    R_chol = cholesky(Symmetric(R))
    log_det_R = logdet(R_chol)
    const_term = obs_dim * log(2π)
    
    # Pre-allocate ALL buffers once (reuse across all timesteps!)
    temp = zeros(T, obs_dim, obs_dim)
    work_matrix = zeros(T, obs_dim, obs_dim)
    
    # Buffers for the lower-level Q_obs
    buffers = (
        sum_yy = zeros(T, obs_dim, obs_dim),
        sum_yz = zeros(T, obs_dim, latent_dim),
        temp_result = zeros(T, obs_dim, obs_dim),
        work1 = zeros(T, obs_dim, obs_dim),
        work2 = zeros(T, latent_dim, obs_dim)
    )
    
    # Use views in the loop - now with buffer passing
    @views for t in axes(y, 2)
        # Pass buffers to lower-level function
        Q_obs!(work_matrix, H, E_z[:,t], E_zz[:,:,t], y[:,t], buffers)
        
        # Scale in-place and accumulate in-place
        work_matrix .*= w[t]
        temp .+= work_matrix
    end

    # Weight the constant terms by the sum of weights
    total_weight = sum(w)
    
    Q_val = T(-0.5) * (total_weight * (const_term + log_det_R) + tr(R_chol \ temp))
    
    return Q_val
end


"""
    Q(A, Q, H, R, P0, x0, E_z, E_zz, E_zz_prev, y) 

Calculate the complete Q-function for the EM algorithm in a Linear Dynamical System.
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
    w::Union{Nothing,AbstractVector{T}} = nothing
) where {T<:Real}
    if w === nothing
        w = ones(T, size(y,2)) 
    end
     
    Q_val_state = Q_state(A, Q, P0, x0, E_z, E_zz, E_zz_prev)
    Q_val_obs = Q_obs(C, R, E_z, E_zz, y, w)
    return Q_val_state + Q_val_obs
end

"""
    sufficient_statistics(x_smooth, p_smooth, p_smooth_t1) 

Compute sufficient statistics for the EM algorithm in a Linear Dynamical System.

# Note
- The function computes the expected values for all trials.
- For single-trial data, use inputs with ntrials = 1.
"""
function sufficient_statistics!(fs::FilterSmooth{T}) where {T<:Real}
    latent_dim, tsteps = size(fs.x_smooth)
    
    # E_z is just a copy of x_smooth
    fs.E_z .= fs.x_smooth
    
    # Compute E_zz and E_zz_prev in-place
    @views for t in 1:tsteps
        # E_zz[:,:,t] = p_smooth[:,:,t] + x_smooth[:,t] * x_smooth[:,t]'
        mul!(fs.E_zz[:, :, t], fs.x_smooth[:, t:t], fs.x_smooth[:, t:t]')
        fs.E_zz[:, :, t] .+= fs.p_smooth[:, :, t]
        
        if t > 1
            # E_zz_prev[:,:,t] = p_smooth_tt1[:,:,t] + x_smooth[:,t] * x_smooth[:,t-1]'
            mul!(fs.E_zz_prev[:, :, t], fs.x_smooth[:, t:t], fs.x_smooth[:, t-1:t-1]')
            fs.E_zz_prev[:, :, t] .+= fs.p_smooth_tt1[:, :, t]
        else
            fs.E_zz_prev[:, :, 1] .= 0
        end
    end
end

function sufficient_statistics!(tfs::TrialFilterSmooth{T}) where {T<:Real}
    ntrials = length(tfs.FilterSmooths)

    if ntrials == 1
        sufficient_statistics!(tfs[1])
    else 
        @threads for i in 1:ntrials
            sufficient_statistics!(tfs[i])
        end
    end
end


"""
    estep(lds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T,3}) 

Perform the E-step of the EM algorithm for a Linear Dynamical System, treating all input as multi-trial.

# Note
- This function first smooths the data using the `smooth` function, then computes sufficient statistics.
- It treats all input as multi-trial, with single-trial being a special case where ntrials = 1.
"""
function estep!(lds, tfs, y)
    smooth!(lds, tfs, y)  
    sufficient_statistics!(tfs)
    elbo = calculate_elbo(lds, tfs, y) 
    return elbo 
end


"""
    calculate_elbo(lds, E_z, E_zz, E_zz_prev, p_smooth, y, total_entropy) 

Calculate the Evidence Lower Bound (ELBO) for a Linear Dynamical System.
"""
function calculate_elbo(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing, AbstractVector{T}} = nothing, 
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    if w === nothing
        w = ones(T, size(y,2)) 
    end

    ntrials = size(y, 3)
    Q_vals = zeros(T, ntrials)

    # Calculate total entropy from individual FilterSmooth objects
    total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths)

    # Thread over trials
    @threads for trial in 1:ntrials
        fs = tfs[trial]  # Get the FilterSmooth for this trial
        Q_vals[trial] = Q_function(
            lds.state_model.A,
            lds.state_model.Q,
            lds.obs_model.C,
            lds.obs_model.R,
            lds.state_model.P0,
            lds.state_model.x0,
            fs.E_z,                    
            fs.E_zz,                   
            fs.E_zz_prev,
            view(y, :, :, trial),
            w
        )
    end

    return sum(Q_vals) - total_entropy
end


"""
    update_initial_state_mean!(lds::LinearDynamicalSystem{T,S,O, E_z::AbstractArray{T,3}) 

Update the initial state mean of the Linear Dynamical System using the average across all trials.
"""
function update_initial_state_mean!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, 
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    if lds.fit_bool[1]
        ntrials = length(tfs.FilterSmooths)
        x0_new = zeros(T, lds.latent_dim)
        
        for trial in 1:ntrials
            fs = tfs[trial]
            x0_new .+= fs.E_z[:, 1]
        end
        
        lds.state_model.x0 .= x0_new ./ ntrials
    end
end

"""
    update_initial_state_covariance!(lds::LinearDynamicalSystem{T,S,O}, E_z::AbstractArray{T,3}, E_zz::AbstractArray{T,4})

Update the initial state covariance of the Linear Dynamical System using the average across all trials.
"""
function update_initial_state_covariance!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, 
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    if lds.fit_bool[2]
        ntrials = length(tfs.FilterSmooths)
        state_dim = lds.latent_dim
        p0_new = zeros(T, state_dim, state_dim)

        for trial in 1:ntrials
            fs = tfs[trial]
            p0_new .+= fs.E_zz[:, :, 1] - (lds.state_model.x0 * lds.state_model.x0')
        end

        p0_new ./= ntrials
        p0_new .= 0.5 * (p0_new + p0_new')

        # Set the new P0 matrix
        lds.state_model.P0 = p0_new
    end
end

"""
    update_A!(lds::LinearDynamicalSystem{T,S,O}, E_zz::AbstractArray{T,4}, E_zz_prev::AbstractArray{T,4}) 

Update the transition matrix A of the Linear Dynamical System.

"""
function update_A!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    if lds.fit_bool[3]
        state_dim = lds.latent_dim
        ntrials = length(tfs.FilterSmooths)

        E_zz_sum = zeros(T, state_dim, state_dim)
        E_zz_prev_sum = zeros(T, state_dim, state_dim)

        for trial in 1:ntrials
            fs = tfs[trial]
            tsteps = size(fs.E_zz, 3)
            
            # Sum over time steps (excluding last for E_zz)
            E_zz_sum .+= sum(@view(fs.E_zz[:, :, 1:(end-1)]); dims=3)
            E_zz_prev_sum .+= sum(fs.E_zz_prev; dims=3)
        end

        lds.state_model.A = E_zz_prev_sum / E_zz_sum
    end
end

"""
    update_Q!(lds::LinearDynamicalSystem{T,S,O}, E_zz::AbstractArray{T,4}, E_zz_prev::AbstractArray{T,4})

Update the process noise covariance matrix Q of the Linear Dynamical System.

"""
function update_Q!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    if lds.fit_bool[4]
        ntrials = length(tfs.FilterSmooths)
        state_dim = lds.latent_dim
        A = lds.state_model.A
        
        Q_new = zeros(T, state_dim, state_dim)
        
        # Pre-allocate working matrices (reuse these!)
        temp1 = Matrix{T}(undef, state_dim, state_dim)  # For Σt_cross * A'
        temp2 = Matrix{T}(undef, state_dim, state_dim)  # For A * Σt_cross'
        temp3 = Matrix{T}(undef, state_dim, state_dim)  # For A * Σt_prev
        temp4 = Matrix{T}(undef, state_dim, state_dim)  # For (A * Σt_prev) * A'
        innovation_cov = Matrix{T}(undef, state_dim, state_dim)
        
        total_time_steps = 0

        for trial in 1:ntrials
            fs = tfs[trial]
            tsteps = size(fs.E_zz, 3)
            
            @views for t in 2:tsteps
                Σt       = fs.E_zz[:, :, t]
                Σt_prev  = fs.E_zz[:, :, t-1]
                Σt_cross = fs.E_zz_prev[:, :, t]

                # Break down the complex expression using pre-allocated temps
                mul!(temp1, Σt_cross, A')      # temp1 = Σt_cross * A'
                mul!(temp2, A, Σt_cross')      # temp2 = A * Σt_cross'
                mul!(temp3, A, Σt_prev)        # temp3 = A * Σt_prev
                mul!(temp4, temp3, A')         # temp4 = (A * Σt_prev) * A'
                
                # innovation_cov = Σt - temp1 - temp2 + temp4
                @. innovation_cov = Σt - temp1 - temp2 + temp4
                
                Q_new .+= innovation_cov
            end
            total_time_steps += (tsteps - 1)
        end

        Q_new ./= total_time_steps
        Q_new .= 0.5 * (Q_new + Q_new')  # Symmetrize in-place

        lds.state_model.Q = Q_new
    end
end

"""
    update_C!(lds::LinearDynamicalSystem{T,S,O}, E_z::AbstractArray{T,3}, E_zz::AbstractArray{T,4}, y::AbstractArray{T,3}) 

Update the observation matrix C of the Linear Dynamical System.

"""
function update_C!(
    lds::LinearDynamicalSystem{T,S,O}, 
    tfs::TrialFilterSmooth{T}, 
    y::AbstractArray{T,3},
    w::Union{Nothing, AbstractVector{T}} = nothing
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    if w === nothing
        w = ones(T, size(y,2)) 
    end

    if lds.fit_bool[5]
        ntrials, tsteps = size(y, 3), size(y, 2)

        sum_yz = zeros(T, size(lds.obs_model.C))
        sum_zz = zeros(T, lds.latent_dim, lds.latent_dim)

        # Pre-allocate working matrices (reuse these!)
        work_yz = zeros(T, size(lds.obs_model.C))
        work_zz = zeros(T, lds.latent_dim, lds.latent_dim)

        for trial in 1:ntrials
            fs = tfs[trial]
            @views for t in 1:tsteps
                # Efficient outer product: work_yz = y[:, t, trial] * fs.E_z[:, t]'
                mul!(work_yz, y[:, t, trial], fs.E_z[:, t]')
                
                # Scale in-place and accumulate
                work_yz .*= w[t]
                sum_yz .+= work_yz
                
                # Copy and scale the covariance matrix
                work_zz .= fs.E_zz[:, :, t]
                work_zz .*= w[t]
                sum_zz .+= work_zz
            end
        end

        lds.obs_model.C = sum_yz / sum_zz
    end
end

"""
    update_R!(lds::LinearDynamicalSystem{T,S,O}, E_z::AbstractArray{T,3}, E_zz::AbstractArray{T,4}, y::AbstractArray{T,3}) 

Update the observation noise covariance matrix R of the Linear Dynamical System.

"""
function update_R!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, y::AbstractArray{T,3},
    w::Union{Nothing, AbstractVector{T}} = nothing
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    if w === nothing
        w = ones(T, size(y,2)) 
    end
    
    if lds.fit_bool[6]
        obs_dim, tsteps, ntrials = size(y)
        R_new = zeros(T, obs_dim, obs_dim)
        C = lds.obs_model.C
        
        # Pre-allocate all temporary arrays
        innovation = zeros(T, obs_dim)
        Czt = zeros(T, obs_dim)
        temp_matrix = zeros(T, obs_dim, size(C, 2))  # For storing C * state_uncertainty
        
        # Pre-allocate matrices for state_uncertainty calculation
        outer_product = zeros(T, lds.latent_dim, lds.latent_dim)  # For z_t * z_t'
        state_uncertainty = zeros(T, lds.latent_dim, lds.latent_dim)  # For Σ_t - z_t*z_t'
        
        # Reorganize as sum of outer products
        for trial in 1:ntrials
            fs = tfs[trial]
            @views for t in 1:tsteps
                # Compute innovation using pre-allocated arrays
                mul!(Czt, C, fs.E_z[:, t])
                @. innovation = (y[:, t, trial] - Czt)
                
                # Add innovation outer product
                mul!(R_new, innovation, innovation', w[t], one(T))  # R_new += w[t] * innovation * innovation'
                
                # Compute state_uncertainty efficiently:
                # First: outer_product = z_t * z_t'
                mul!(outer_product, fs.E_z[:, t], fs.E_z[:, t]')
                
                # Second: state_uncertainty = Σ_t - z_t*z_t'
                state_uncertainty .= fs.E_zz[:, :, t]
                state_uncertainty .-= outer_product
                
                # Then compute C * state_uncertainty * C' in steps:
                mul!(temp_matrix, C, state_uncertainty)  # temp = C * state_uncertainty
                mul!(R_new, temp_matrix, C', w[t], one(T))  # R_new += w[t] * C * state_uncertainty * C'
            end
        end
        
        R_new ./= (ntrials * tsteps)
        R_new .= 0.5 * (R_new + R_new')  # Symmetrize

        # Set the new R matrix
        lds.obs_model.R = R_new
    end
end

"""
    mstep!(lds, E_z, E_zz, E_zz_prev, p_smooth, y) 

Perform the M-step of the EM algorithm for a Linear Dynamical System with multi-trial data.
"""
function mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing, AbstractVector{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    if w === nothing
        w = ones(T, size(y,2)) 
    end

    # get initial parameters
    old_params = vec(stateparams(lds))
    old_params = [old_params; vec(obsparams(lds))]

    # Update parameters
    update_initial_state_mean!(lds, tfs)
    update_initial_state_covariance!(lds, tfs)
    update_A!(lds, tfs)
    update_Q!(lds, tfs)
    update_C!(lds, tfs, y)
    update_R!(lds, tfs, y)

    # get new params
    new_params = vec(stateparams(lds))
    new_params = [new_params; vec(obsparams(lds))]

    # calculate norm of parameter changes
    norm_change = norm(new_params - old_params)
    return norm_change
end

"""
    fit!(lds, y; max_iter::Int=1000, tol::Real=1e-12) 
    where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Fit a Linear Dynamical System using the Expectation-Maximization (EM) algorithm with Kalman smoothing over multiple trials

# Arguments
- `lds::LinearDynamicalSystem{T,S,O}`: The Linear Dynamical System to be fitted.
- `y::AbstractArray{T,3}`: Observed data, size(obs_dim, T_steps, n_trials)

# Keyword Arguments
- `max_iter::Int=1000`: Maximum number of EM iterations.
- `tol::T=1e-12`: Convergence tolerance for log-likelihood change.

# Returns
- `mls::Vector{T}`: Vector of log-likelihood values for each iteration.
- `param_diff::Vector{T}`: Vector of parameter deltas for each EM iteration. 
"""
function fit!(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3};
    max_iter::Int=1000, tol::Float64=1e-12
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    if eltype(y) !== T
        error("Observed data must be of type $(T); Got $(eltype(y)))")
    end
    # Initialize log-likelihood
    prev_ml = -T(Inf)

    # Create a vector to store the log-likelihood values
    mls = Vector{T}()
    param_diff = Vector{T}()

    sizehint!(mls, max_iter)  # Pre-allocate for efficiency

    # Create a FilterSmooth object
    tfs = initialize_FilterSmooth(lds, size(y, 2), size(y,3))

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
        ml = estep!(lds, tfs, y)

        # M-step
        Δparams = mstep!(lds, tfs, y)
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
    loglikelihood(x::AbstractMatrix{U}, plds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}) 

Calculate the complete-data log-likelihood of a Poisson Linear Dynamical System model for a single trial. 

# Arguments
- `x::AbstractMatrix{T}`: The latent state variables. Dimensions: (latent_dim, tsteps)
- `lds::LinearDynamicalSystem{T,S,O}`: The Linear Dynamical System model.
- `y::AbstractMatrix{T}`: The observed data. Dimensions: (obs_dim, tsteps)
- `w::Vector{T}`: Weights for each observation in the log-likelihood calculation. Not currently used.

# Returns
- `ll::T`: The log-likelihood value.

# Ref 
- loglikelihood(x::AbstractArray{T,3}, plds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T,3})
"""
function loglikelihood(
    x::AbstractMatrix{U}, 
    plds::LinearDynamicalSystem{T,S,O}, 
    y::AbstractMatrix{T}, 
    w::Union{Nothing,AbstractVector{T}}=nothing,
) where {U<:Real,T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    if w === nothing
        w = ones(U, size(y,2)) 
    elseif eltype(w) !== T
        error("weights must be Vector{$(U)}; Got Vector{$(eltype(w))}")
    end

    # Convert the log firing rate to firing rate
    d = exp.(plds.obs_model.log_d)
    tsteps = size(y, 2)

    # Pre-compute matrix inverses
    inv_p0 = inv(plds.state_model.P0)
    inv_Q = inv(plds.state_model.Q)

    # Get dimensions
    C = plds.obs_model.C
    A = plds.state_model.A
    x0 = plds.state_model.x0
    obs_dim, latent_dim = size(C)

    # Pre-allocate ALL temporary vectors
    Cx_temp = Vector{eltype(x)}(undef, obs_dim)      # For C * x[:, t]
    obs_temp = Vector{eltype(x)}(undef, obs_dim)     # For C * x[:, t] + d
    Ax_temp = Vector{eltype(x)}(undef, latent_dim)   # For A * x[:, t-1]
    state_diff = Vector{eltype(x)}(undef, latent_dim) # For x[:, t] - A * x[:, t-1]
    quad_temp = Vector{eltype(x)}(undef, latent_dim)  # For inv_Q * state_diff

    # Calculate p(yₜ|xₜ)
    pygivenx_sum = zero(T)  
    @views for t in 1:tsteps
        # Replace: temp .= C * x[:, t] .+ d
        mul!(Cx_temp, C, x[:, t])           # Cx_temp = C * x[:, t]
        obs_temp .= Cx_temp .+ d            # obs_temp = C * x[:, t] + d
        
        # Compute dot(y[:, t], obs_temp) - sum(exp, obs_temp)
        dot_product = dot(y[:, t], obs_temp)
        exp_sum = zero(eltype(x))
        for i in 1:obs_dim
            exp_sum += exp(obs_temp[i])
        end
        pygivenx_sum += dot_product - exp_sum
    end

    # Calculate p(x₁) - avoid creating dx1 temporary
    # Replace: dx1 = @view(x[:, 1]) .- plds.state_model.x0; px1 = -U(0.5) * dot(dx1, inv_p0 * dx1)
    state_diff .= x[:, 1] .- x0             # state_diff = x[:, 1] - x0
    mul!(quad_temp, inv_p0, state_diff)     # quad_temp = inv_p0 * state_diff
    px1 = -U(0.5) * dot(state_diff, quad_temp)

    # Calculate p(xₜ|xₜ₋₁)
    pxtgivenxt1_sum = zero(U)
    @views for t in 2:tsteps
        # Replace: temp .= x[:, t] .- (A * x[:, t - 1])
        mul!(Ax_temp, A, x[:, t - 1])       # Ax_temp = A * x[:, t-1]
        state_diff .= x[:, t] .- Ax_temp    # state_diff = x[:, t] - A * x[:, t-1]
        
        # Replace: pxtgivenxt1_sum += -U(0.5) * dot(temp, inv_Q * temp)
        mul!(quad_temp, inv_Q, state_diff)  # quad_temp = inv_Q * state_diff
        pxtgivenxt1_sum += -U(0.5) * dot(state_diff, quad_temp)
    end

    # Return the log-posterior
    return pygivenx_sum + px1 + pxtgivenxt1_sum
end

"""
    loglikelihood(x::AbstractArray{T,3}, plds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T,3})

Calculate the complete-data log-likelihood of a Poisson Linear Dynamical System model for multiple trials. 
"""
function loglikelihood(
    x::AbstractArray{T,3}, 
    plds::LinearDynamicalSystem{T,S,O}, 
    y::AbstractArray{T,3}
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    # Calculate the log-likelihood over all trials
    ll = zeros(T, size(y, 3))
    @threads for n in axes(y, 3)
        ll[n] .= loglikelihood(x[:, :, n], plds, y[:, :, n])
    end
    return sum(ll)
end


"""
    Gradient(lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}) 

Calculate the gradient of the log-likelihood of a Poisson Linear Dynamical System model for a single trial.
"""
function Gradient(
    lds::LinearDynamicalSystem{T,S,O}, 
    y::AbstractMatrix{T}, 
    x::AbstractMatrix{T}, 
    w::Union{Nothing, AbstractVector{T}}=nothing, 
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    if w === nothing
        w = ones(T, size(y, 2))
    end 
    
    # Extract model parameters
    A, Q = lds.state_model.A, lds.state_model.Q
    C, log_d = lds.obs_model.C, lds.obs_model.log_d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    # Convert log_d to d (non-log space)
    d = exp.(log_d)

    # Get dimensions
    tsteps = size(y, 2)
    latent_dim = lds.latent_dim
    obs_dim = lds.obs_dim

    # Precompute matrix inverses
    inv_P0 = inv(P0)
    inv_Q = inv(Q)

    # Pre-allocate gradient
    grad = zeros(T, latent_dim, tsteps)

    # Pre-allocate ALL temporary vectors (reused across timesteps)
    Cx_t = Vector{T}(undef, obs_dim)           # C * x[:, t]
    exp_term = Vector{T}(undef, obs_dim)       # exp(C * x[:, t] + d)
    innovation = Vector{T}(undef, obs_dim)     # y[:, t] - exp_term
    common_term = Vector{T}(undef, latent_dim) # C' * innovation
    
    # Temporary vectors for state dynamics terms
    Ax_t = Vector{T}(undef, latent_dim)        # A * x[:, t]
    Ax_prev = Vector{T}(undef, latent_dim)     # A * x[:, t-1]
    state_diff = Vector{T}(undef, latent_dim)  # Various state differences
    temp_grad = Vector{T}(undef, latent_dim)   # Temporary for accumulating gradient parts

    # Calculate gradient for each time step
    @views for t in 1:tsteps
        # Compute observation term efficiently
        # temp = exp.(C * x[:, t] .+ d)
        mul!(Cx_t, C, x[:, t])                 # Cx_t = C * x[:, t]
        for i in 1:obs_dim
            exp_term[i] = exp(Cx_t[i] + d[i])  # exp_term = exp(C * x[:, t] + d)
        end
        
        # common_term = C' * (y[:, t] - temp)
        innovation .= y[:, t] .- exp_term      # innovation = y[:, t] - exp_term
        mul!(common_term, C', innovation)      # common_term = C' * innovation

        if t == 1
            # First time step: common_term + A' * inv_Q * (x[:, 2] - A * x[:, t]) - inv_P0 * (x[:, t] - x0)
            
            # Compute A * x[:, t]
            mul!(Ax_t, A, x[:, t])
            
            # Compute x[:, 2] - A * x[:, t]
            state_diff .= x[:, 2] .- Ax_t
            
            # Compute inv_Q * (x[:, 2] - A * x[:, t])
            mul!(temp_grad, inv_Q, state_diff)
            
            # Compute A' * inv_Q * (x[:, 2] - A * x[:, t])
            mul!(grad[:, t], A', temp_grad)
            
            # Add common_term
            grad[:, t] .+= common_term
            
            # Subtract inv_P0 * (x[:, t] - x0)
            state_diff .= x[:, t] .- x0
            mul!(temp_grad, inv_P0, state_diff)
            grad[:, t] .-= temp_grad
            
        elseif t == tsteps
            # Last time step: common_term - inv_Q * (x[:, t] - A * x[:, t-1])
            
            # Compute A * x[:, t-1]
            mul!(Ax_prev, A, x[:, t-1])
            
            # Compute x[:, t] - A * x[:, t-1]
            state_diff .= x[:, t] .- Ax_prev
            
            # Compute inv_Q * (x[:, t] - A * x[:, t-1])
            mul!(temp_grad, inv_Q, state_diff)
            
            # grad[:, t] = common_term - inv_Q * (...)
            grad[:, t] .= common_term .- temp_grad
            
        else
            # Intermediate time steps: 
            # common_term + A' * inv_Q * (x[:, t+1] - A * x[:, t]) - inv_Q * (x[:, t] - A * x[:, t-1])
            
            # First part: A' * inv_Q * (x[:, t+1] - A * x[:, t])
            mul!(Ax_t, A, x[:, t])                   # Ax_t = A * x[:, t]
            state_diff .= x[:, t+1] .- Ax_t          # state_diff = x[:, t+1] - A * x[:, t]
            mul!(temp_grad, inv_Q, state_diff)       # temp_grad = inv_Q * state_diff
            mul!(grad[:, t], A', temp_grad)          # grad[:, t] = A' * temp_grad
            
            # Add common_term
            grad[:, t] .+= common_term
            
            # Second part: - inv_Q * (x[:, t] - A * x[:, t-1])
            mul!(Ax_prev, A, x[:, t-1])              # Ax_prev = A * x[:, t-1]
            state_diff .= x[:, t] .- Ax_prev         # state_diff = x[:, t] - A * x[:, t-1]
            mul!(temp_grad, inv_Q, state_diff)       # temp_grad = inv_Q * state_diff
            grad[:, t] .-= temp_grad                 # grad[:, t] -= temp_grad
        end
    end

    return grad
end

"""
    Hessian(lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}) 

Calculate the Hessian matrix of the log-likelihood for a Poisson Linear Dynamical System.
"""
function Hessian(
    lds::LinearDynamicalSystem{T,S,O}, 
    y::AbstractMatrix{T}, 
    x::AbstractMatrix{T}, 
    w::Union{Nothing, AbstractVector{T}}=nothing, 
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    if w === nothing
        w=ones(T, size(y, 2))
    end 
    # Extract model components
    A, Q = lds.state_model.A, lds.state_model.Q
    C, log_d = lds.obs_model.C, lds.obs_model.log_d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    # Convert log_d to d i.e. non-log space
    d = exp.(log_d)

    # Pre-compute a few things
    tsteps = size(y, 2)
    inv_Q = inv(Symmetric(Q))
    inv_P0 = inv(Symmetric(P0))

    # Calculate super and sub diagonals
    H_sub_entry = inv_Q * A
    H_super_entry = permutedims(H_sub_entry)

    # Fill the super and sub diagonals
    H_sub = [H_sub_entry for _ in 1:(tsteps - 1)]
    H_super = [H_super_entry for _ in 1:(tsteps - 1)]   

    λ = zeros(T, size(C, 1))
    z = similar(λ)
    poisson_tmp = Matrix{T}(undef, size(C, 2), size(C, 2))
    H_diag = [Matrix{T}(undef, size(x, 1), size(x, 1)) for _ in 1:tsteps]

    # minnimal allocation Hessian helper function
    function calculate_poisson_hess!(out::Matrix{T}, C::Matrix{T}, λ::Vector{T}) where T
        n, p = size(C)
        @inbounds for j in 1:p, i in 1:p
            acc = zero(T)
            for k in 1:n
                acc += C[k, i] * λ[k] * C[k, j]
            end
            out[i, j] = -acc
        end
    end

    # Pre-computed values for the Hessian
    xt_given_xt_1 = -inv_Q
    xt1_given_xt = -A' * inv_Q * A
    x_t = -inv_P0

    Q_middle = xt1_given_xt + xt_given_xt_1
    Q_first = x_t + xt1_given_xt
    Q_last = xt_given_xt_1

    @views for t in 1:tsteps
        mul!(z, C, x[:, t])  # z = C * x[:, t]
        @. λ = exp(z + d)

        if t == 1
            H_diag[t] .= Q_first
        elseif t == tsteps
            H_diag[t] .= Q_last
        else
            H_diag[t] .= Q_middle
        end

        calculate_poisson_hess!(poisson_tmp, C, λ)
        H_diag[t] .+= poisson_tmp
    end

    H = block_tridgm(H_diag, H_super, H_sub)

    return H, H_diag, H_super, H_sub
end

"""
    Q_state(A, Q, P0, x0, E_z, E_zz, E_zz_prev) 

Calculates the Q-function for the state model over multiple trials.
"""
function Q_state(
    A::AbstractMatrix{T},
    Q::AbstractMatrix{T},
    P0::AbstractMatrix{T},
    x0::AbstractVector{T},
    E_z::AbstractArray{T,3},
    E_zz::AbstractArray{T,4},
    E_zz_prev::AbstractArray{T,4},
) where {T<:Real}
    # Calculate the Q-function for the state model
    Q_val = 0.0

    # Calcualte over trials
    @views @threads for k in axes(E_z, 3)
        Q_val += Q_state(
            A, Q, P0, x0, E_z[:, :, k], E_zz[:, :, :, k], E_zz_prev[:, :, :, k]
        )
    end

    return Q_val
end

"""
    Q_observation_model(C, D, log_d, E_z, E_zz, y)

Calculate the Q-function for the observation model.
"""
function Q_observation_model(
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    E_z::AbstractArray{U,3},
    P_smooth::AbstractArray{U,4},
    y::AbstractArray{U,3},
) where {T<:Real,U<:Real}
    obs_dim, state_dim = size(C)

    d = exp.(log_d)
    Q_val = zero(T)
    trials = size(E_z, 3)
    tsteps = size(E_z, 2)

    h = Vector{T}(undef, obs_dim)
    ρ = Vector{T}(undef, obs_dim)
    CC = zeros(T, obs_dim, state_dim^2)

    for i in 1:obs_dim
        CC[i, :] .= vec(C[i, :] * C[i, :]')
    end

    @threads for k in 1:trials
        @views for t in 1:tsteps
            Ez_t = view(E_z, :, t, k)
            P_t  = view(P_smooth, :, :, t, k)
            y_t  = view(y, :, t, k)

            mul!(h, C, Ez_t)          # h = C * E_z[:, t, k]
            h .+= d

            ρ .= T(0.5) .* CC * vec(P_t)
            ŷ = exp.(h .+ ρ)

            Q_val += sum(y_t .* h .- ŷ)
        end
    end

    return Q_val
end

"""
    Q_observation_model(C, log_d, E_z, E_zz, y)

Calculate the Q-function for the observation model for a single trial.
"""
function Q_observation_model(
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    E_z::AbstractMatrix{T},
    p_smooth::AbstractArray{T,3},
    y::AbstractMatrix{T},
) where {T<:Real}
    obs_dim, state_dim = size(C)
    d = exp.(log_d)
    Q_val = zero(T)
    tsteps = size(y, 2)

    h = Vector{T}(undef, obs_dim)
    ρ = Vector{T}(undef, obs_dim)
    temp_vec = Vector{T}(undef, state_dim)  # Reusable temporary for P_t * c_i

    @views for t in 1:tsteps
        Ez_t = E_z[:, t]
        P_t = p_smooth[:, :, t]
        y_t = y[:, t]

        # h = C * Ez_t + d
        mul!(h, C, Ez_t)
        h .+= d

        # Compute ρ[i] = 0.5 * c_i' * P_t * c_i without allocations
        # This replaces: ρ .= T(0.5) .* CC * vec(P_t)
        for i in 1:obs_dim
            c_i = view(C, i, :)  # Row i of C as a vector view
            mul!(temp_vec, P_t, c_i)  # temp_vec = P_t * c_i
            ρ[i] = T(0.5) * dot(c_i, temp_vec)  # ρ[i] = c_i' * P_t * c_i
        end
        
        # Compute ŷ = exp(h + ρ) in-place, reusing ρ as ŷ
        # This replaces: ŷ = exp.(h .+ ρ)
        for i in 1:obs_dim
            ρ[i] = exp(h[i] + ρ[i])  # ρ now stores ŷ values
        end
        
        # Compute sum(y_t .* h .- ŷ) without temporary arrays
        # This replaces: Q_val += sum(y_t .* h .- ŷ)
        for i in 1:obs_dim
            Q_val += y_t[i] * h[i] - ρ[i] 
        end
    end

    return Q_val
end

"""
    Q_observation_model(C, log_d, tfs, y)

Calculate the Q-function for the observation model across all trials using TrialFilterSmooth.
"""
function Q_observation_model(
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
) where {T<:Real}
    trials = length(tfs.FilterSmooths)
    Q_vals = zeros(T, trials)
    
    @threads for k in 1:trials
        fs = tfs[k]  # Get FilterSmooth for this trial
        Q_vals[k] = Q_observation_model(
            C, log_d, fs.E_z, fs.p_smooth, view(y, :, :, k)
        )
    end
    
    return sum(Q_vals)
end

"""
    Q_function(A, Q, C, log_d, x0, P0, E_z, E_zz, E_zz_prev, y)

Calculate the Q-function for a single trial of a Poisson Linear Dynamical System.
"""
function Q_function(
    A::AbstractMatrix{T},
    Q::AbstractMatrix{T},
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    x0::AbstractVector{T},
    P0::AbstractMatrix{T},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
    y::AbstractMatrix{T},
) where {T<:Real}
    # Calculate the Q-function for the state model (single trial)
    Q_state_val = Q_state(A, Q, P0, x0, E_z, E_zz, E_zz_prev)
    
    # Calculate the Q-function for the observation model (single trial)
    Q_obs_val = Q_observation_model(C, log_d, E_z, E_zz, y)
    
    return Q_state_val + Q_obs_val
end


"""
    calculate_elbo(plds, tfs, y)
                   
Calculate the Evidence Lower Bound (ELBO) for a Poisson Linear Dynamical System.
"""
function calculate_elbo(
    plds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    # Set up parameters
    A, Q, x0, P0 = plds.state_model.A, plds.state_model.Q, 
                   plds.state_model.x0, plds.state_model.P0
    C, log_d = plds.obs_model.C, plds.obs_model.log_d

    ntrials = size(y, 3)
    Q_vals = zeros(T, ntrials)

    # Calculate total entropy from individual FilterSmooth objects
    total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths)

    # Thread over trials (like Gaussian version)
    @threads for trial in 1:ntrials
        fs = tfs[trial]  # Get the FilterSmooth for this trial
        Q_vals[trial] = Q_function(
            A, Q, C, log_d, x0, P0,
            fs.E_z,                    
            fs.E_zz,                   
            fs.E_zz_prev,
            view(y, :, :, trial)
        )
    end

    return sum(Q_vals) - total_entropy
end

"""
    gradient_observation_model_single_trial!(grad, C, log_d, E_z, p_smooth, y)

Compute the gradient for a single trial and add it to the accumulated gradient.
"""
function gradient_observation_model_single_trial!(
    grad::AbstractVector{T},
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    E_z::AbstractMatrix{T},
    p_smooth::AbstractArray{T,3},
    y::AbstractMatrix{T},
) where {T<:Real}
    d = exp.(log_d)
    obs_dim, latent_dim = size(C)
    tsteps = size(y, 2)
    
    # Pre-allocate temporary arrays
    h = Vector{T}(undef, obs_dim)
    ρ = Vector{T}(undef, obs_dim)
    λ = Vector{T}(undef, obs_dim)
    
    # Key optimization: pre-compute C * P_smooth_t once per timestep
    CP = Matrix{T}(undef, obs_dim, latent_dim)
    
    @views for t in 1:tsteps
        E_z_t = E_z[:, t]
        P_smooth_t = p_smooth[:, :, t]
        y_t = y[:, t]
        
        # Compute h = C * z_t + d
        mul!(h, C, E_z_t)
        h .+= d
        
        # Pre-compute CP = C * P_smooth_t (this is the expensive operation)
        mul!(CP, C, P_smooth_t)
        
        # Compute ρ efficiently 
        for i in 1:obs_dim
            ρ[i] = T(0.5) * dot(C[i, :], CP[i, :])
        end
        
        # Compute λ = exp(h + ρ)
        for i in 1:obs_dim
            λ[i] = exp(h[i] + ρ[i])
        end
        
        # Gradient computation
        for j in 1:latent_dim
            for i in 1:obs_dim
                idx = (j - 1) * obs_dim + i
                # This is now fast: CP[i,j] instead of dot(C[i, :], P_smooth_t[:, j])
                grad[idx] += y_t[i] * E_z_t[j] - λ[i] * (E_z_t[j] + CP[i, j])
            end
        end
        
        # Update log_d gradient
        @views grad[(end - obs_dim + 1):end] .+= (y_t .- λ) .* d
    end
end

"""
    gradient_observation_model!(grad, C, log_d, tfs, y)

Compute the gradient of the Q-function with respect to the observation model parameters using TrialFilterSmooth.
"""
function gradient_observation_model!(
    grad::AbstractVector{T},
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
) where {T<:Real}
    trials = length(tfs.FilterSmooths)
    
    fill!(grad, zero(T))
    
    # Accumulate gradients from all trials
    @threads for k in 1:trials
        fs = tfs[k]  # Get FilterSmooth for this trial
        # Local gradient for this trial
        local_grad = zeros(T, length(grad))
        
        gradient_observation_model_single_trial!(
            local_grad, C, log_d, fs.E_z, fs.p_smooth, view(y, :, :, k)
        )
        
        # Thread-safe update of global gradient
        grad .+= local_grad
    end
    
    return grad .*= -1
end


"""
    update_observation_model!(plds, tfs, y)

Update the observation model parameters of a PLDS model.
"""
function update_observation_model!(
    plds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3}
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    if plds.fit_bool[5]
        params = vcat(vec(plds.obs_model.C), plds.obs_model.log_d)

        function f(params::Vector{T})
            C_size = plds.obs_dim * plds.latent_dim
            log_d = params[(end - plds.obs_dim + 1):end]
            C = reshape(params[1:C_size], plds.obs_dim, plds.latent_dim)
            return -Q_observation_model(C, log_d, tfs, y)
        end

        function g!(grad::Vector{T}, params::Vector{T})
            C_size = plds.obs_dim * plds.latent_dim
            log_d = params[(end - plds.obs_dim + 1):end]
            C = reshape(params[1:C_size], plds.obs_dim, plds.latent_dim)
            return gradient_observation_model!(grad, C, log_d, tfs, y)
        end

        opts = Optim.Options(
            x_reltol=1e-12,
            x_abstol=1e-12,
            g_abstol=1e-12,
            f_reltol=1e-12,
            f_abstol=1e-12,
        )

        # use CG result as initial guess for LBFGS
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
    mstep!(plds::LinearDynamicalSystem{T,S,O}, E_z::AbstractArray{T,3}, E_zz::AbstractArray{T,4}, E_zz_Prev{T,4}, p_smooth{T,4}, y::AbstractArray{T,3}) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Perform the M-step of the EM algorithm for a Poisson Linear Dynamical System with multi-trial data.
"""
function mstep!(
    plds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    # Get old params
    old_params = vec(stateparams(plds))
    old_params = vcat(old_params, vec(obsparams(plds)))
    
    # Update state parameters
    update_initial_state_mean!(plds, tfs)
    update_initial_state_covariance!(plds, tfs)
    update_A!(plds, tfs)
    update_Q!(plds, tfs)
    
    # Update observation parameters
    update_observation_model!(plds, tfs, y)
    
    # Get new params
    new_params = vec(stateparams(plds))
    new_params = vcat(new_params, vec(obsparams(plds)))

    norm_params = norm(new_params - old_params)
    return norm_params
end
