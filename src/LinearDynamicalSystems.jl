export LinearDynamicalSystem
export GaussianStateModel, GaussianObservationModel, PoissonObservationModel
export rand, smooth, fit!

"""
    GaussianStateModel{T<:Real. M<:AbstractMatrix{T}, V<:AbstractVector{T}}}

Represents the state model of a Linear Dynamical System with Gaussian noise.

# Fields
- `A::M`: Transition matrix (size `latent_dim×latent_dim`).
- `Q::M`: Process noise covariance matrix
- `b::V`: Bias vector (length `latent_dim`).
- `x0::V`: Initial state vector (length `latent_dim`).
- `P0::M`: Initial state covariance matrix (size `latent_dim×latent_dim`).
"""
Base.@kwdef mutable struct GaussianStateModel{
    T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}
} <: AbstractStateModel{T}
    A::M
    Q::M
    b::V
    x0::V
    P0::M
end

function Base.show(io::IO, gsm::GaussianStateModel; gap="")
    println(io, gap, "Gaussian State Model:")
    println(io, gap, "---------------------")

    if size(gsm.A, 1) > 4 || size(gsm.A, 2) > 4
        println(io, gap, " State Parameters:")
        println(io, gap, "  size(A)  = ($(size(gsm.A,1)), $(size(gsm.A,2)))")
        println(io, gap, "  size(Q)  = ($(size(gsm.Q,1)), $(size(gsm.Q,2)))")
        println(io, gap, " Initial State:")
        println(io, gap, "  size(b)  = ($(length(gsm.b)), )")
        println(io, gap, "  size(x0) = ($(length(gsm.x0)), )")
        println(io, gap, "  size(P0) = ($(size(gsm.P0,1)), $(size(gsm.P0,2)))")
    else
        println(io, gap, " State Parameters:")
        println(io, gap, "  A  = $(round.(gsm.A, sigdigits=3))")
        println(io, gap, "  Q  = $(round.(gsm.Q, sigdigits=3))")
        println(io, gap, " Initial State:")
        println(io, gap, "  b  = $(round.(gsm.b, digits=2))")
        println(io, gap, "  x0 = $(round.(gsm.x0, digits=2))")
        println(io, gap, "  P0 = $(round.(gsm.P0, sigdigits=3))")
    end

    return nothing
end

"""
    GaussianObservationModel{T<:Real, M<:AbstractMatrix{T}}

Represents the observation model of a Linear Dynamical System with Gaussian noise.

# Fields
- `C::M`: Observation matrix of size `(obs_dim × latent_dim)`. Maps latent states into
    observation space.
- `R::M`: Observation noise covariance of size `(obs_dim × obs_dim)`.
- `d::V`: Bias vector of length `(obs_dim)`.
"""
Base.@kwdef mutable struct GaussianObservationModel{
    T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}
} <: AbstractObservationModel{T}
    C::M
    R::M
    d::V
end

function Base.show(io::IO, gom::GaussianObservationModel; gap="")
    println(io, gap, "Gaussian Observation Model:")
    println(io, gap, "---------------------------")

    if size(gom.C, 1) > 3 || size(gom.C, 2) > 3
        println(io, gap, " size(C) = ($(size(gom.C,1)), $(size(gom.C,2)))")
        println(io, gap, " size(R) = ($(size(gom.R,1)), $(size(gom.R,2)))")
        println(io, gap, " size(d) = ($(length(gom.d)),)")
    else
        println(io, gap, " C = $(round.(gom.C, digits=2))")
        println(io, gap, " R = $(round.(gom.R, digits=2))")
        println(io, gap, " d = $(round.(gom.d, digits=2))")
    end

    return nothing
end

"""
    PoissonObservationModel{
        T<:Real,
        M<:AbstractMatrix{T},
        V<:AbstractVector{T}
    } <: AbstractObservationModel{T}

Represents the observation model of a Linear Dynamical System with Poisson observations.

# Fields
- `C::AbstractMatrix{T}`: Observation matrix of size `(obs_dim × latent_dim)`. Maps latent
    states into observation space.
- `log_d::AbstractVector{T}`: Mean firing rate vector (log space) of length `(obs_dim)`.
"""
Base.@kwdef mutable struct PoissonObservationModel{
    T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}
} <: AbstractObservationModel{T}
    C::M
    log_d::V
end

function Base.show(io::IO, pom::PoissonObservationModel; gap="")
    nobs, nstate = size(pom.C)

    println(io, gap, "Poisson Observation Model:")
    println(io, gap, "--------------------------")

    if nobs > 4 || nstate > 4
        println(io, gap, " size(C)     = ($nobs, $nstate)")
        println(io, gap, " size(log_d) = ($(length(pom.log_d)),)")
    else
        println(io, gap, " C       = $(round.(pom.C, digits=2))")
        println(io, gap, " log_d   = $(round.(pom.log_d, sigdigits = 3))")
        println(io, gap, " d       = $(round.(exp.(pom.log_d), digits = 2))")
    end

    return nothing
end

"""
    LinearDynamicalSystem{T<:Real, S<:AbstractStateModel{T}, O<:AbstractObservationModel{T}}

Represents a unified Linear Dynamical System with customizable state and observation models.

# Fields
- `state_model::S`: The state model (e.g., GaussianStateModel)
- `obs_model::O`: The observation model (e.g., GaussianObservationModel or
    PoissonObservationModel)
- `latent_dim::Int`: Dimension of the latent state
- `obs_dim::Int`: Dimension of the observations
- `fit_bool::Vector{Bool}`: Vector indicating which parameters to fit during optimization
"""
Base.@kwdef struct LinearDynamicalSystem{
    T<:Real,S<:AbstractStateModel{T},O<:AbstractObservationModel{T}
}
    state_model::S
    obs_model::O
    latent_dim::Int
    obs_dim::Int
    fit_bool::Vector{Bool}
end

function LinearDynamicalSystem(
    state_model::S,
    obs_model::O;
    fit_bool::Union{Vector{Bool}, Nothing} = nothing
) where {T<:Real, S<:AbstractStateModel{T}, O<:AbstractObservationModel{T}}
    
    # Infer dimensions from matrices
    latent_dim = size(state_model.A, 1)
    obs_dim = size(obs_model.C, 1)
    
    # Set default fit_bool based on observation model type
    if fit_bool === nothing
        if obs_model isa PoissonObservationModel
            # For Poisson: [x0, P0, A&b, Q, C&log_d] (5 parameters)
            fit_bool = [true, true, true, true, true]
        else
            # For Gaussian: [x0, P0, A&b, Q, C&d, R] (6 parameters)
            fit_bool = [true, true, true, true, true, true]
        end
    end
    
    # Create the LDS
    lds = LinearDynamicalSystem{T, S, O}(
        state_model, obs_model, latent_dim, obs_dim, fit_bool
    )
    
    # Validate the constructed LDS
    if !isvalid_LDS(lds)
        error("Invalid LinearDynamicalSystem parameters")
    end
    
    return lds
end

function Base.show(io::IO, lds::LinearDynamicalSystem; gap="")
    println(io, gap, "Linear Dynamical System:")
    println(io, gap, "------------------------")
    Base.show(io, lds.state_model; gap=gap * " ")
    Base.show(io, lds.obs_model; gap=gap * " ")
    println(io, gap, " Parameters to update:")
    println(io, gap, " ---------------------")

    if lds.obs_model isa PoissonObservationModel
        # C and log_d are either both updated or neither
        prms = ["x0", "P0", "A (and b)", "Q", "C, log_d"][lds.fit_bool[1:5]]
    else
        prms = ["x0", "P0", "A (and b)", "Q", "C", "R"][lds.fit_bool[1:6]]
    end

    println(io, gap, "  $(join(prms, ", "))")
    return nothing
end

"""
    initialize_FilterSmooth(model, num_obs)

Initialize a `FilterSmooth` object for a given linear dynamical system model and number of
observations.
"""
function initialize_FilterSmooth(
    model::LinearDynamicalSystem{T,S,O}, num_obs::Int
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    num_states = model.latent_dim
    return FilterSmooth(
        zeros(T, num_states, num_obs),
        zeros(T, num_states, num_states, num_obs),
        zeros(T, num_states, num_obs, 1),
        zeros(T, num_states, num_states, num_obs, 1),
        zeros(T, num_states, num_states, num_obs, 1),
    )
end

function _extract_state_params(state_model::GaussianStateModel{T}) where T
    return (A = state_model.A, Q = state_model.Q, b = state_model.b, 
            x0 = state_model.x0, P0 = state_model.P0)
end

function _extract_obs_params(obs_model::GaussianObservationModel{T}) where T
    return (C = obs_model.C, R = obs_model.R, d = obs_model.d)
end

function _extract_obs_params(obs_model::PoissonObservationModel{T}) where T
    return (C = obs_model.C, log_d = obs_model.log_d, d = exp.(obs_model.log_d))
end

function _get_all_params_vec(lds::LinearDynamicalSystem{T,S,O}) where {T<:Real,S<:AbstractStateModel{T},O<:AbstractObservationModel{T}}
    state_params = _extract_state_params(lds.state_model)
    obs_params = _extract_obs_params(lds.obs_model)
    
    # Convert named tuples to vectors and concatenate
    state_vec = vcat(vec(state_params.A), vec(state_params.Q), vec(state_params.b), 
                     vec(state_params.x0), vec(state_params.P0))
    
    if lds.obs_model isa GaussianObservationModel
        obs_vec = vcat(vec(obs_params.C), vec(obs_params.R), vec(obs_params.d))
    else # PoissonObservationModel
        obs_vec = vcat(vec(obs_params.C), vec(obs_params.log_d))
    end
    
    return vcat(state_vec, obs_vec)
end

function _sample_trial!(rng, x_trial, y_trial, state_params, obs_params, obs_model::GaussianObservationModel)
    tsteps = size(x_trial, 2)
    
    # Initial state
    x_trial[:, 1] = rand(rng, MvNormal(state_params.x0, state_params.P0))
    y_trial[:, 1] = rand(rng, MvNormal(obs_params.C * x_trial[:, 1] + obs_params.d, obs_params.R))
    
    # Subsequent states
    for t in 2:tsteps
        x_trial[:, t] = rand(rng, MvNormal(state_params.A * x_trial[:, t-1] + state_params.b, state_params.Q))
        y_trial[:, t] = rand(rng, MvNormal(obs_params.C * x_trial[:, t] + obs_params.d, obs_params.R))
    end
end

function _sample_trial!(rng, x_trial, y_trial, state_params, obs_params, obs_model::PoissonObservationModel)
    tsteps = size(x_trial, 2)
    
    # Initial state
    x_trial[:, 1] = rand(rng, MvNormal(state_params.x0, state_params.P0))
    y_trial[:, 1] = rand.(rng, Poisson.(exp.(obs_params.C * x_trial[:, 1] + obs_params.d)))
    
    # Subsequent states
    for t in 2:tsteps
        x_trial[:, t] = rand(rng, MvNormal(state_params.A * x_trial[:, t-1] + state_params.b, state_params.Q))
        y_trial[:, t] = rand.(rng, Poisson.(exp.(obs_params.C * x_trial[:, t] + obs_params.d)))
    end
end

function Random.rand(
    rng::AbstractRNG, 
    lds::LinearDynamicalSystem{T,S,O}; 
    tsteps::Int, 
    ntrials::Int = 1
) where {T<:Real, S<:GaussianStateModel{T}, O<:AbstractObservationModel{T}}
    
    # Extract parameters once using a more systematic approach
    state_params = _extract_state_params(lds.state_model)
    obs_params = _extract_obs_params(lds.obs_model)
    
    # Pre-allocate based on observation model type
    x = Array{T,3}(undef, lds.latent_dim, tsteps, ntrials)
    y = Array{T,3}(undef, lds.obs_dim, tsteps, ntrials)
    
    # Sample trials (potentially in parallel for large ntrials)
    if ntrials > 10  # Threshold for parallelization
        Threads.@threads for trial in 1:ntrials
            _sample_trial!(rng, view(x, :, :, trial), view(y, :, :, trial), 
                          state_params, obs_params, lds.obs_model)
        end
    else
        for trial in 1:ntrials
            _sample_trial!(rng, view(x, :, :, trial), view(y, :, :, trial), 
                          state_params, obs_params, lds.obs_model)
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
    loglikelihood(
        x::AbstractMatrix{T},
        lds::LinearDynamicalSystem{S,O},
        y::AbstractMatrix{T}
    ) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Calculate the complete-data log-likelihood of a linear dynamical system (LDS) given the
observed data.

# Arguments
- `x::AbstractMatrix{T}`: The state sequence of the LDS.
- `lds::LinearDynamicalSystem{S,O}`: The Linear Dynamical System.
- `y::AbstractMatrix{T}`: The observed data.

# Returns
- `ll::T`: The complete-data log-likelihood of the LDS.
"""
function loglikelihood(
    x::AbstractMatrix{U},
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
) where {U<:Real,T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    A, Q, x0, P0 = lds.state_model.A,
    lds.state_model.Q, lds.state_model.x0,
    lds.state_model.P0
    C, R, b, d = lds.obs_model.C, lds.obs_model.R, lds.state_model.b, lds.obs_model.d

    R_chol = cholesky(Symmetric(R)).U
    Q_chol = cholesky(Symmetric(Q)).U
    P0_chol = cholesky(Symmetric(P0)).U

    dx0 = view(x, :, 1) .- x0
    ll = sum(abs2, P0_chol \ dx0)

    temp_dx = zeros(eltype(x), size(x, 1))
    temp_dy = zeros(eltype(x), size(y, 1))

    for t in 1:tsteps
        if t > 1
            mul!(temp_dx, A, view(x, :, t-1), -one(eltype(x)), false)
            temp_dx .+= view(x, :, t) .- b
            ll += sum(abs2, Q_chol \ temp_dx)
        end

        mul!(temp_dy, C, view(x, :, t), -one(eltype(x)), false)
        temp_dy .+= view(y, :, t) .- d
        ll += sum(abs2, R_chol \ temp_dy)
    end

    return -eltype(x)(0.5) * ll
end

"""
    Gradient(lds, y, x)

Compute the gradient of the log-likelihood with respect to the latent states for a linear
dynamical system.
"""
function Gradient(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    latent_dim, tsteps = size(x)
    A, Q, x0, P0 = lds.state_model.A,
    lds.state_model.Q, lds.state_model.x0,
    lds.state_model.P0
    C, R, b, d = lds.obs_model.C, lds.obs_model.R, lds.state_model.b, lds.obs_model.d

    R_chol = cholesky(Symmetric(R))
    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))

    C_inv_R = (R_chol \ C)'      # C' * inv(R)
    A_inv_Q = (Q_chol \ A)'      # A' * inv(Q)

    grad = zeros(T, latent_dim, tsteps)

    # First time step
    dx1 = x[:, 1] .- x0
    dx2 = x[:, 2] .- (A * x[:, 1] .+ b)
    dy1 = y[:, 1] .- (C * x[:, 1] .+ d)
    grad[:, 1] .= A_inv_Q * dx2 + C_inv_R * dy1 - (P0_chol \ dx1)

    # Middle steps
    @views for t in 2:(tsteps - 1)
        grad[:, t] .=
            C_inv_R * (y[:, t] .- (C * x[:, t] .+ d)) -
            (Q_chol \ (x[:, t] .- (A * x[:, t - 1] .+ b))) +
            (A_inv_Q * (x[:, t + 1] .- (A * x[:, t] .+ b)))
    end

    # Last time step
    dxT = x[:, tsteps] .- (A * x[:, tsteps - 1] .+ b)
    dyT = y[:, tsteps] .- (C * x[:, tsteps] .+ d)
    grad[:, tsteps] .= (C_inv_R * dyT) - (Q_chol \ dxT)

    return grad
end

"""
    Hessian(lds, y, x) where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Construct the Hessian matrix of the log-likelihood of the LDS model given a set of
observations.

This function is used for the direct optimization of the log-likelihood as advocated by
Paninski et al. (2009). The block tridiagonal structure of the Hessian is exploited to
reduce the number of parameters that need to be computed, and to reduce the memory
requirements. Together with the gradient, this allows for Kalman Smoothing to be performed
by simply solving a linear system of equations:

    ̂xₙ₊₁ = ̂xₙ - H \\ ∇

where ` ̂xₙ` is the current smoothed state estimate, `H` is the Hessian matrix, and `∇` is the
gradient of the log-likelihood.

# Note
- `x` is not used in this function, but is required to match the function signature of other
    Hessian calculations e.g., in PoissonLDS.
"""
function Hessian(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    x::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    A, Q, x0, P0 = lds.state_model.A,
    lds.state_model.Q, lds.state_model.x0,
    lds.state_model.P0
    C, R = lds.obs_model.C, lds.obs_model.R

    tsteps = size(y, 2)

    # Pre-compute inverses
    inv_R = Symmetric(inv(R))
    inv_Q = Symmetric(inv(Q))
    inv_P0 = Symmetric(inv(P0))

    # Pre-allocate all blocks
    H_sub = Vector{Matrix{T}}(undef, tsteps - 1)
    H_super = Vector{Matrix{T}}(undef, tsteps - 1)
    H_diag = Vector{Matrix{T}}(undef, tsteps)

    # Off-diagonal terms
    H_sub_entry = inv_Q * A
    H_super_entry = Matrix(H_sub_entry')

    # Calculate main diagonal terms
    yt_given_xt = -C' * inv_R * C
    xt_given_xt_1 = -inv_Q
    xt1_given_xt = -A' * inv_Q * A
    x_t = -inv_P0

    # Build off-diagonals
    for i in 1:(tsteps - 1)
        H_sub[i] = H_sub_entry
        H_super[i] = H_super_entry
    end

    # Build main diagonal
    H_diag[1] = yt_given_xt + xt1_given_xt + x_t

    for i in 2:(tsteps - 1)
        H_diag[i] = yt_given_xt + xt_given_xt_1 + xt1_given_xt
    end

    H_diag[tsteps] = yt_given_xt + xt_given_xt_1
    H = StateSpaceDynamics.block_tridgm(H_diag, H_super, H_sub)

    return H, H_diag, H_super, H_sub
end

"""
    smooth(lds, y)

This function performs direct smoothing for a linear dynamical system (LDS) given the system
parameters and the observed data for a single trial

# Arguments
- `lds::LinearDynamicalSystem{T,S,O}`: The LDS object representing the system parameters.
- `y::AbstractMatrix{T}`: The observed data matrix.

# Returns
- `x::AbstractMatrix{T}`: The optimal state estimate.
- `p_smooth::Array{T, 3}`: The posterior covariance matrix.
- `inverse_offdiag::Array{T, 3}`: The inverse off-diagonal matrix.
- `Q_val::T`: The Q-function value.
"""
function smooth(
    lds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    tsteps, D = size(y, 2), lds.latent_dim
    X₀ = zeros(T, D * tsteps)

    function nll(vec_x::Vector{T})
        x = reshape(vec_x, D, tsteps)
        return -loglikelihood(x, lds, y)
    end

    function g!(g::Vector{T}, vec_x::Vector{T})
        x = reshape(vec_x, D, tsteps)
        grad = Gradient(lds, y, x)
        return g .= vec(-grad)
    end

    function h!(h::SparseMatrixCSC{T}, vec_x::Vector{T}) where {T<:Real}
        x = reshape(vec_x, D, tsteps)
        H, _, _, _ = Hessian(lds, y, x)
        mul!(h, -1.0, H)
        return nothing
    end

    # Initial values setup
    initial_f = nll(X₀)
    inital_g = similar(X₀)
    g!(inital_g, X₀)
    initial_h = spzeros(T, length(X₀), length(X₀))
    h!(initial_h, X₀)

    # Set up TwiceDifferentiable object I guess?
    td = TwiceDifferentiable(nll, g!, h!, X₀, initial_f, inital_g, initial_h)
    opts = Optim.Options(; g_abstol=1e-8, x_abstol=1e-8, f_abstol=1e-8, iterations=100)

    # Go!
    res = optimize(td, X₀, Newton(; linesearch=LineSearches.BackTracking()), opts)

    x = reshape(res.minimizer, D, tsteps)
    H, main, super, sub = Hessian(lds, y, x)

    if lds.latent_dim > 10
        p_smooth, inverse_offdiag = block_tridiagonal_inverse(-sub, -main, -super)
    else
        p_smooth, inverse_offdiag = block_tridiagonal_inverse_static(-sub, -main, -super)
    end

    # See Utilities.jl for definition
    gauss_entropy = gaussian_entropy(Symmetric(H))

    # Symmetrize
    @views for i in 1:tsteps
        p_smooth[:, :, i] .= 0.5 .* (p_smooth[:, :, i] .+ p_smooth[:, :, i]')
    end

    inverse_offdiag = cat(zeros(T, D, D), inverse_offdiag; dims=3)

    return x, p_smooth, inverse_offdiag, gauss_entropy
end

"""
    smooth(lds, y)

This function performs direct smoothing for a linear dynamical system (LDS) given the system
parameters and the observed data for multiple trials.

# Arguments
- `lds::LinearDynamicalSystem{T,S,O}`: The LDS object representing the system parameters.
- `y::AbstractArray{T,3}`: The observed data array with dimensions (obs_dim, tsteps, ntrials).

# Returns
- `x::AbstractArray{T,3}`: The optimal state estimates with dimensions
    (ntrials, tsteps, latent_dim).
- `p_smooth::AbstractArray{T,4}`: The posterior covariance matrices with dimensions
    (latent_dim, latent_dim, tsteps, ntrials).
- `inverse_offdiag::AbstractArray{T,4}`: The inverse off-diagonal matrices with dimensions
    (latent_dim, latent_dim, tsteps, ntrials).
"""
function smooth(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T,3}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    obs_dim, tsteps, ntrials = size(y)
    latent_dim = lds.latent_dim

    # Fast path for single trial
    if ntrials == 1
        x_sm, p_sm, p_prev, ent = smooth(lds, y[:, :, 1])
        return (
            reshape(x_sm, latent_dim, tsteps, 1),
            reshape(p_sm, latent_dim, latent_dim, tsteps, 1),
            reshape(p_prev, latent_dim, latent_dim, tsteps, 1),
            ent,
        )
    end

    x_smooth = Array{T,3}(undef, latent_dim, tsteps, ntrials)
    p_smooth = Array{T,4}(undef, latent_dim, latent_dim, tsteps, ntrials)
    inverse_offdiag = Array{T,4}(undef, latent_dim, latent_dim, tsteps, ntrials)
    entropies = zeros(T, ntrials)  # ← per-trial buffer

    @views @threads for trial in 1:ntrials
        x_sm, p_sm, p_prev, ent = smooth(lds, y[:, :, trial])
        x_smooth[:, :, trial] .= x_sm
        p_smooth[:, :, :, trial] .= p_sm
        inverse_offdiag[:, :, :, trial] .= p_prev
        entropies[trial] = ent        # thread-safe
    end

    return x_smooth, p_smooth, inverse_offdiag, sum(entropies)
end

"""
    Q_state(A, b, Q, P0, x0, E_z, E_zz, E_zz_prev)

State Q-term for an LDS with affine dynamics x_t ~ N(A x_{t-1} + b, Q).
Matches the style of `Q_state` but includes the bias contributions.
"""
function Q_state(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    Q::AbstractMatrix{T},
    P0::AbstractMatrix{T},
    x0::AbstractVector{T},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
) where {T<:Real}
    tstep = size(E_z, 2)
    D = size(A, 1)

    Q_chol = cholesky(Symmetric(Q))
    P0_chol = cholesky(Symmetric(P0))
    log_det_Q = logdet(Q_chol)
    log_det_P0 = logdet(P0_chol)

    # initial-state part (unchanged)
    temp = zeros(T, D, D)
    mul!(temp, E_z[:, 1], x0', T(-1), T(0))
    temp .+= @view E_zz[:, :, 1]
    temp .-= x0 * E_z[:, 1]'
    temp .+= x0 * x0'
    Q_val = T(-0.5) * (log_det_P0 + tr(P0_chol \ temp))

    # transition part with bias
    sum_E_zz = zeros(T, D, D)
    sum_E_zzm1 = zeros(T, D, D)
    sum_E_cross = zeros(T, D, D)
    sum_mu_t = zeros(T, D)
    sum_mu_tm1 = zeros(T, D)

    for t in 2:tstep
        sum_E_zz .+= @view E_zz[:, :, t]
        sum_E_zzm1 .+= @view E_zz[:, :, t - 1]
        sum_E_cross .+= @view E_zz_prev[:, :, t]
        sum_mu_t .+= @view E_z[:, t]
        sum_mu_tm1 .+= @view E_z[:, t - 1]
    end

    copyto!(temp, sum_E_zz)
    mul!(temp, A, sum_E_cross', T(-1), T(1))
    temp .-= sum_E_cross * A'
    mul!(temp, A, sum_E_zzm1 * A', T(1), T(1))
    # bias terms
    temp .-= sum_mu_t * b'
    temp .-= b * sum_mu_t'
    temp .+= A * (sum_mu_tm1 * b')    # A μ_{t-1} bᵀ
    temp .+= (b * sum_mu_tm1') * A'   # b μ_{t-1}ᵀ Aᵀ
    temp .+= (tstep - 1) * (b * b')

    Q_val += T(-0.5) * ((tstep - 1) * log_det_Q + tr(Q_chol \ temp))
    return Q_val
end

"""
    Q_obs(C, d, E_z, E_zz, y)

Single time-step observation component of the Q-function for
y_t ~ 𝓝(C x_t + d, R), before applying R^{-1} and constants.
"""
function Q_obs(
    C::AbstractMatrix{T},
    d::AbstractVector{T},
    E_z::AbstractVector{T},
    E_zz::AbstractMatrix{T},
    y::AbstractVector{T},
) where {T<:Real}
    obs_dim = size(C, 1)

    # Work with residualized observation: ỹ = y - d
    ytil = y .- d

    # Pre-allocate statistics
    sum_yy = zeros(T, obs_dim, obs_dim)
    sum_yz = zeros(T, obs_dim, size(E_z, 1))

    mul!(sum_yy, ytil, ytil', 1.0, 1.0)   # (y-d)(y-d)' 
    mul!(sum_yz, ytil, E_z', 1.0, 1.0)    # (y-d) E[z]'

    # temp = (y-d)(y-d)' - C (y-d)E[z]' - (y-d)E[z]' C' + C E[z z'] C'
    temp = similar(sum_yy)
    copyto!(temp, sum_yy)
    mul!(temp, C, sum_yz', -1.0, 1.0)
    temp .-= sum_yz * C'
    mul!(temp, C * E_zz, C', 1.0, 1.0)

    return temp
end

"""
    Q_obs(C, d, R, E_z, E_zz, y)

Full observation Q-term for Gaussian LDS over all time steps.
"""
function Q_obs(
    C::AbstractMatrix{T},
    d::AbstractVector{T},
    R::AbstractMatrix{T},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    y::AbstractMatrix{T},
) where {T<:Real}
    obs_dim = size(C, 1)
    tsteps = size(y, 2)

    R_chol = cholesky(Symmetric(R))
    log_det_R = logdet(R_chol)
    const_term = obs_dim * log(2π)

    temp = zeros(T, obs_dim, obs_dim)
    @views for t in axes(y, 2)
        temp .+= Q_obs(C, d, E_z[:, t], E_zz[:, :, t], y[:, t])
    end

    return T(-0.5) * (tsteps * (const_term + log_det_R) + tr(R_chol \ temp))
end

"""
    Q_function(A, b, Q, C, d, R, P0, x0, E_z, E_zz, E_zz_prev, y)

Complete Q-function for Gaussian LDS:
x_t ~ 𝓝(A x_{t-1} + b, Q),  y_t ~ 𝓝(C x_t + d, R).
"""
function Q_function(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    Q::AbstractMatrix{T},
    C::AbstractMatrix{T},
    d::AbstractVector{T},
    R::AbstractMatrix{T},
    P0::AbstractMatrix{T},
    x0::AbstractVector{T},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
    y::AbstractMatrix{T},
) where {T<:Real}
    Q_val_state = Q_state(A, b, Q, P0, x0, E_z, E_zz, E_zz_prev)
    Q_val_obs = Q_obs(C, d, R, E_z, E_zz, y)
    return Q_val_state + Q_val_obs
end

"""
    sufficient_statistics(x_smooth, p_smooth, p_smooth_t1)

Compute sufficient statistics for the EM algorithm in a Linear Dynamical System.

# Note
- The function computes the expected values for all trials.
- For single-trial data, use inputs with ntrials = 1.
"""
function sufficient_statistics(
    x_smooth::AbstractArray{T,3},
    p_smooth::AbstractArray{T,4},
    p_smooth_t1::AbstractArray{T,4},
) where {T<:Real}
    latent_dim, tsteps, ntrials = size(x_smooth)

    E_z = copy(x_smooth)
    E_zz = similar(p_smooth)
    E_zz_prev = similar(p_smooth)

    for trial in 1:ntrials
        @views for t in 1:tsteps
            xt = view(x_smooth, :, t, trial)
            pt = view(p_smooth,:,:,t,trial)
            E_zz[:, :, t, trial] .= pt .+ xt * xt'
            if t > 1
                xtm1 = view(x_smooth, :, t - 1, trial)
                pt1 = view(p_smooth_t1,:,:,t,trial)
                E_zz_prev[:, :, t, trial] .= pt1 .+ xt * xtm1'
            end
        end

        @views E_zz_prev[:, :, 1, trial] .= 0
    end

    return E_z, E_zz, E_zz_prev
end

"""
    estep(lds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T,3})

Perform the E-step of the EM algorithm for a Linear Dynamical System, treating all input as
multi-trial.

# Note
- This function first smooths the data using the `smooth` function, then computes sufficient
    statistics.
- It treats all input as multi-trial, with single-trial being a special case where
    `ntrials = 1`.
"""
function estep(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T,3}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    # smooth
    x_smooth, p_smooth, inverse_offdiag, total_entropy = smooth(lds, y)

    # calculate sufficient statistics
    E_z, E_zz, E_zz_prev = sufficient_statistics(x_smooth, p_smooth, inverse_offdiag)

    return E_z, E_zz, E_zz_prev, x_smooth, p_smooth, total_entropy
end

"""
    calculate_elbo(lds, E_z, E_zz, E_zz_prev, p_smooth, y, total_entropy)

Calculate the Evidence Lower Bound (ELBO) for a Linear Dynamical System.
"""
function calculate_elbo(
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractArray{T,3},
    E_zz::AbstractArray{T,4},
    E_zz_prev::AbstractArray{T,4},
    p_smooth::AbstractArray{T,4},
    y::AbstractArray{T,3},
    total_entropy::AbstractFloat,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    ntrials = size(y, 3)
    Q_vals = zeros(T, ntrials)

    @threads for trial in 1:ntrials
        Q_vals[trial] = StateSpaceDynamics.Q_function(
            lds.state_model.A,
            lds.state_model.b,
            lds.state_model.Q,
            lds.obs_model.C,
            lds.obs_model.d,
            lds.obs_model.R,
            lds.state_model.P0,
            lds.state_model.x0,
            view(E_z,:,:,trial),
            view(E_zz,:,:,:,trial),
            view(E_zz_prev,:,:,:,trial),
            view(y,:,:,trial),
        )
    end

    return sum(Q_vals) - total_entropy
end

"""
    update_initial_state_mean!(lds::LinearDynamicalSystem{T,S,O, E_z::AbstractArray{T,3})

Update the initial state mean of the Linear Dynamical System using the average across all
trials.
"""
function update_initial_state_mean!(
    lds::LinearDynamicalSystem{T,S,O}, E_z::AbstractArray{T,3}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    if lds.fit_bool[1]
        x0_new = zeros(lds.latent_dim)
        @views for i in axes(E_z, 3)
            x0_new .+= E_z[:, 1, i]
        end
        lds.state_model.x0 .= x0_new ./ size(E_z, 3)
    end
end

"""
    update_initial_state_covariance!(
        lds::LinearDynamicalSystem{T,S,O},
        E_z::AbstractArray{T,3},
        E_zz::AbstractArray{T,4}
    )

Update the initial state covariance of the Linear Dynamical System using the average across
all trials.
"""
function update_initial_state_covariance!(
    lds::LinearDynamicalSystem{T,S,O}, E_z::AbstractArray{T,3}, E_zz::AbstractArray{T,4}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    if lds.fit_bool[2]
        ntrials = size(E_z, 3)
        state_dim = size(E_z, 1)
        p0_new = zeros(T, state_dim, state_dim)

        for trial in 1:ntrials
            p0_new .+=
                @view(E_zz[:, :, 1, trial]) - (lds.state_model.x0 * lds.state_model.x0')
        end

        p0_new ./= ntrials
        p0_new .= 0.5 * (p0_new + p0_new')

        # Set the new P0 matrix
        lds.state_model.P0 = p0_new
    end

    return nothing
end

"""
    update_A!(
        lds::LinearDynamicalSystem{T,S,O},
        E_zz::AbstractArray{T,4},
        E_zz_prev::AbstractArray{T,4}
    )

Update the transition matrix A of the Linear Dynamical System.

# Note
- This function modifies `lds` in-place.
- The update is only performed if `lds.fit_bool[3]` is true.
"""
function update_A_b!(
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractArray{T,3},
    E_zz::AbstractArray{T,4},
    E_zz_prev::AbstractArray{T,4},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[3] || return nothing
    D = size(E_z, 1)
    Sxz = zeros(T, D, D + 1)
    Szz = zeros(T, D + 1, D + 1)

    tsteps = size(E_z, 2)
    ntrials = size(E_z, 3)

    @views for k in 1:ntrials, t in 2:tsteps
        # Sxz
        Sxz[:, 1:D] .+= E_zz_prev[:, :, t, k]   # E[x_t x_{t-1}ᵀ]
        Sxz[:, D + 1] .+= E_z[:, t, k]            # E[x_t] * 1
        # Szz for z_{t-1} = [x_{t-1}; 1]
        Szz[1:D, 1:D] .+= E_zz[:, :, t - 1, k]    # E[x_{t-1} x_{t-1}ᵀ]
        Szz[1:D, D + 1] .+= E_z[:, t - 1, k]
        Szz[D + 1, 1:D] .+= E_z[:, t - 1, k]
        Szz[D + 1, D + 1] += one(T)
    end

    AB = Sxz / Szz
    lds.state_model.A = AB[:, 1:D]
    lds.state_model.b = AB[:, D + 1]
    return nothing
end

"""
    update_Q!(
        lds::LinearDynamicalSystem{T,S,O},
        E_zz::AbstractArray{T,4},
        E_zz_prev::AbstractArray{T,4}
    )

Update the process noise covariance matrix Q of the Linear Dynamical System.

# Note
- This function modifies `lds` in-place.
- The update is only performed if `lds.fit_bool[4]` is true.
- The result is averaged across all trials.
"""
function update_Q!(
    lds::LinearDynamicalSystem{T,S,O},
    E_zz::AbstractArray{T,4},
    E_zz_prev::AbstractArray{T,4},
    E_z::AbstractArray{T,3},
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[4] || return nothing

    ntrials, tsteps = size(E_zz, 4), size(E_zz, 3)
    D = size(E_zz, 1)
    Q_new = zeros(T, D, D)
    A = lds.state_model.A
    b = lds.state_model.b

    @views for k in 1:ntrials, t in 2:tsteps
        Σt = E_zz[:, :, t, k]          # E[x_t x_tᵀ]
        Σtm1 = E_zz[:, :, t - 1, k]        # E[x_{t-1} x_{t-1}ᵀ]
        Σcross = E_zz_prev[:, :, t, k]     # E[x_t x_{t-1}ᵀ]
        μt = E_z[:, t, k]
        μtm1 = E_z[:, t - 1, k]

        # E[(x_t - A x_{t-1} - b)(x_t - A x_{t-1} - b)ᵀ]
        # (all are second-moment terms, consistent with E_zz and E_zz_prev)
        Q_new .+= Σt
        Q_new .-= Σcross * A'
        Q_new .-= A * Σcross'
        Q_new .+= A * Σtm1 * A'
        Q_new .-= μt * b'
        Q_new .-= b * μt'
        Q_new .+= A * μtm1 * b'
        Q_new .+= b * μtm1' * A'
        Q_new .+= b * b'
    end

    Q_new ./= (ntrials * (tsteps - 1))
    Q_new .= 0.5 .* (Q_new .+ Q_new')   # keep it symmetric
    lds.state_model.Q = Q_new
    return nothing
end

"""
    update_C!(
        lds::LinearDynamicalSystem{T,S,O},
        E_z::AbstractArray{T,3},
        E_zz::AbstractArray{T,4},
        y::AbstractArray{T,3}
    )

Update the observation matrix C of the Linear Dynamical System.

# Note
- This function modifies `lds` in-place.
- The update is only performed if `lds.fit_bool[5]` is true.
- The result is averaged across all trials.
"""
function update_C_d!(
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractArray{T,3},
    E_zz::AbstractArray{T,4},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    lds.fit_bool[5] || return nothing
    if w === nothing
        ;
        w = ones(T, size(y, 2));
    end

    D = size(E_z, 1)
    p = size(y, 1)
    Syz = zeros(T, p, D + 1)
    Szz = zeros(T, D + 1, D + 1)

    tsteps = size(E_z, 2)
    ntrials = size(E_z, 3)

    @views for k in 1:ntrials, t in 1:tsteps
        wt = w[t]
        μ = E_z[:, t, k]
        Σ = E_zz[:, :, t, k]        # E[x_t x_tᵀ]

        # Syz accumulates y_t [μ;1]ᵀ
        Syz[:, 1:D] .+= wt * (y[:, t, k] * μ')
        Syz[:, D + 1] .+= wt * y[:, t, k]

        # Szz accumulates E[[x;1][x;1]ᵀ]
        Szz[1:D, 1:D] .+= wt * Σ
        Szz[1:D, D + 1] .+= wt * μ
        Szz[D + 1, 1:D] .+= wt * μ
        Szz[D + 1, D + 1] += wt
    end

    CD = Syz / Szz
    lds.obs_model.C = CD[:, 1:D]
    lds.obs_model.d = CD[:, D + 1]
    return nothing
end

"""
    update_R!(
        lds::LinearDynamicalSystem{T,S,O},
        E_z::AbstractArray{T,3},
        E_zz::AbstractArray{T,4},
        y::AbstractArray{T,3}
    )

Update the observation noise covariance matrix R of the Linear Dynamical System.

# Note
- This function modifies `lds` in-place.
- The update is only performed if `lds.fit_bool[6]` is true.
- The result is averaged across all trials.
"""
function update_R!(
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractArray{T,3},
    E_zz::AbstractArray{T,4},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    if w === nothing
        w = ones(T, size(y, 2))
    end
    lds.fit_bool[6] || return nothing

    obs_dim, tsteps, ntrials = size(y)
    R_new = zeros(T, obs_dim, obs_dim)
    C = lds.obs_model.C
    d = lds.obs_model.d

    innovation = zeros(T, obs_dim)
    Czt = zeros(T, obs_dim)
    temp_matrix = zeros(T, obs_dim, size(C, 2))

    @views for trial in 1:ntrials, t in 1:tsteps
        wt = w[t]

        mul!(Czt, C, E_z[:, t, trial])
        @. innovation = (y[:, t, trial] - (Czt + d))
        mul!(R_new, innovation, innovation', wt, one(T))

        state_uncertainty = E_zz[:, :, t, trial] - (E_z[:, t, trial]) * (E_z[:, t, trial])'
        mul!(temp_matrix, C, state_uncertainty)
        mul!(R_new, temp_matrix, C', wt, one(T))
    end

    # use sum(w) * ntrials as the proper denominator
    R_new ./= (sum(w) * ntrials)
    R_new .= 0.5 .* (R_new .+ R_new')
    lds.obs_model.R = R_new
    return nothing
end

"""
    mstep!(lds, E_z, E_zz, E_zz_prev, p_smooth, y)

Perform the M-step of the EM algorithm for a Linear Dynamical System with multi-trial data.
"""
function mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractArray{T,3},
    E_zz::AbstractArray{T,4},
    E_zz_prev::AbstractArray{T,4},
    p_smooth::AbstractArray{T,4},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    if w === nothing
        w = ones(T, size(y, 2))
    end

    # Get initial parameters using new approach
    old_params = _get_all_params_vec(lds)

    # Update parameters
    update_initial_state_mean!(lds, E_z)
    update_initial_state_covariance!(lds, E_z, E_zz)
    update_A_b!(lds, E_z, E_zz, E_zz_prev)
    update_Q!(lds, E_zz, E_zz_prev, E_z)
    update_C_d!(lds, E_z, E_zz, y, w)
    update_R!(lds, E_z, E_zz, y, w)

    # Get new parameters using new approach
    new_params = _get_all_params_vec(lds)

    # Parameter delta
    norm_change = norm(new_params - old_params)
    return norm_change
end

"""
    fit!(lds, y; max_iter::Int=1000, tol::Real=1e-12)
    where {T<:Real, S<:GaussianStateModel{T}, O<:GaussianObservationModel{T}}

Fit a Linear Dynamical System using the Expectation-Maximization (EM) algorithm with Kalman
smoothing over multiple trials

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
    max_iter::Int=1000,
    tol::Float64=1e-12,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    if eltype(y) !== T
        error("Observed data must be of type $(T); Got $(eltype(y)))")
    end
    # Initialize log-likelihood
    prev_elbo = -T(Inf)

    # Create a vector to store the log-likelihood values
    elbos = Vector{T}()
    param_diff = Vector{T}()

    sizehint!(elbos, max_iter)  # Pre-allocate for efficiency

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
        E_z, E_zz, E_zz_prev, x_smooth, p_smooth, entropy = estep(lds, y)

        # M-step
        Δparams = mstep!(lds, E_z, E_zz, E_zz_prev, p_smooth, y)

        elbo = calculate_elbo(lds, E_z, E_zz, E_zz_prev, p_smooth, y, entropy)

        # Update the log-likelihood vector and parameter difference
        push!(elbos, elbo)
        push!(param_diff, Δparams)

        # Update the progress bar
        next!(prog)

        # Check convergence
        if abs(elbo - prev_elbo) < tol
            finish!(prog)
            return elbos, param_diff
        end

        prev_elbo = elbo
    end

    # Finish the progress bar if max_iter is reached
    finish!(prog)

    return elbos, param_diff
end

"""
    loglikelihood(
        x::AbstractMatrix{U},
        plds::LinearDynamicalSystem{T,S,O},
        y::AbstractMatrix{T}
    )

Calculate the complete-data log-likelihood of a Poisson Linear Dynamical System model for a
single trial.

# Arguments
- `x::AbstractMatrix{T}`: The latent state variables. Dimensions: (latent_dim, tsteps)
- `lds::LinearDynamicalSystem{T,S,O}`: The Linear Dynamical System model.
- `y::AbstractMatrix{T}`: The observed data. Dimensions: (obs_dim, tsteps)
- `w::Vector{T}`: Weights for each observation in the log-likelihood calculation. Not
    currently used.

# Returns
- `ll::T`: The log-likelihood value.

# Ref
- loglikelihood(
    x::AbstractArray{T,3},
    plds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3}
)
"""
function loglikelihood(
    x::AbstractMatrix{U},
    plds::LinearDynamicalSystem{T,S,O},
    y::AbstractMatrix{T},
    w::Union{Nothing,AbstractVector{T}}=nothing,
) where {U<:Real,T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    if w === nothing
        w = ones(U, size(y, 2))
    elseif eltype(w) !== T
        error("weights must be Vector{$(U)}; Got Vector{$(eltype(w))}")
    end

    # Convert the log firing rate to firing rate
    d = exp.(plds.obs_model.log_d)
    tsteps = size(y, 2)

    # Pre-compute matrix inverses
    inv_p0 = inv(plds.state_model.P0)
    inv_Q = inv(plds.state_model.Q)

    # Calculate p(yₜ|xₜ)
    C = plds.obs_model.C
    obs_dim, latent_dim = size(C)
    temp = Vector{eltype(x)}(undef, obs_dim) # Temporary vector for calculations

    pygivenx_sum = zero(T)

    @views for t in 1:tsteps
        temp .= C * x[:, t] .+ d
        pygivenx_sum += dot(y[:, t], temp) - sum(exp, temp)
    end

    # Calculate p(x₁)
    dx1 = @view(x[:, 1]) .- plds.state_model.x0
    px1 = -U(0.5) * dot(dx1, inv_p0 * dx1)

    # Calculate p(xₜ|xₜ₋₁)
    pxtgivenxt1_sum = zero(U)
    A = plds.state_model.A
    b = plds.state_model.b
    temp = Vector{eltype(x)}(undef, latent_dim)  # Temporary vector for calculations

    @views for t in 2:tsteps
        temp .= x[:, t] .- (A * x[:, t - 1] .+ b)
        pxtgivenxt1_sum += -U(0.5) * dot(temp, inv_Q * temp)
    end

    # Return the log-posterior
    return pygivenx_sum + px1 + pxtgivenxt1_sum
end

"""
    loglikelihood(x::AbstractArray{T,3}, plds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T,3})

Calculate the complete-data log-likelihood of a Poisson Linear Dynamical System model for multiple trials.
"""
function loglikelihood(
    x::AbstractArray{T,3}, plds::LinearDynamicalSystem{T,S,O}, y::AbstractArray{T,3}
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
    w::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    if w === nothing
        w = ones(T, size(y, 2))
    end

    # Extract model parameters
    A, Q, b = lds.state_model.A, lds.state_model.Q, lds.state_model.b
    C, log_d = lds.obs_model.C, lds.obs_model.log_d
    x0, P0 = lds.state_model.x0, lds.state_model.P0

    # Convert log_d to d (non-log space)
    d = exp.(log_d)

    # Get number of time steps
    tsteps = size(y, 2)

    # Precompute matrix inverses
    inv_P0 = inv(P0)
    inv_Q = inv(Q)

    # Pre-allocate gradient
    grad = zeros(T, lds.latent_dim, tsteps)

    # Calculate gradient for each time step
    @views for t in 1:tsteps

        # Common term for all time steps
        temp = exp.(C * x[:, t] .+ d)
        common_term = C' * (y[:, t] - temp)

        if t == 1
            # First time step
            grad[:, t] .=
                common_term + A' * inv_Q * (x[:, 2] .- A * x[:, t] .+ b) -
                inv_P0 * (x[:, t] .- x0)
        elseif t == tsteps
            # Last time step
            grad[:, t] .= common_term - inv_Q * (x[:, t] .- A * x[:, tsteps - 1] .+ b)
        else
            # Intermediate time steps
            grad[:, t] .=
                common_term + A' * inv_Q * (x[:, t + 1] .- A * x[:, t] .+ b) -
                inv_Q * (x[:, t] .- A * x[:, t - 1] .+ b)
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
    w::Union{Nothing,AbstractVector{T}}=nothing,
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

    H_sub = Vector{typeof(H_sub_entry)}(undef, tsteps - 1)
    H_super = Vector{typeof(H_super_entry)}(undef, tsteps - 1)

    for i in 1:(tsteps - 1)
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
    H_diag = Vector{Matrix{T}}(undef, tsteps)

    @views for t in 1:tsteps
        λ = exp.(C * x[:, t] .+ d)
        if t == 1
            H_diag[t] = x_t + xt1_given_xt + calculate_poisson_hess(C, λ)
        elseif t == tsteps
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
    Q_state(A, Q, P0, x0, E_z, E_zz, E_zz_prev)

Calculates the Q-function for the state model over multiple trials.
"""
function Q_state(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    Q::AbstractMatrix{T},
    P0::AbstractMatrix{T},
    x0::AbstractVector{T},
    E_z::AbstractArray{T,3},
    E_zz::AbstractArray{T,4},
    E_zz_prev::AbstractArray{T,4},
) where {T<:Real}
    # Calculate the Q-function for the state model
    vals = zeros(T, size(E_z, 3))

    @views @threads for k in axes(E_z, 3)
        vals[k] = Q_state(
            A, b, Q, P0, x0, E_z[:, :, k], E_zz[:, :, :, k], E_zz_prev[:, :, :, k]
        )
    end

    return sum(vals)
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
            P_t = view(P_smooth,:,:,t,k)
            y_t = view(y, :, t, k)

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
    Q_function(A, Q, C, log_d, x0, P0, E_z, E_zz, E_zz_prev, P_smooth, y)

Calculate the Q-function for the Linear Dynamical System.
"""
function Q_function(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    Q::AbstractMatrix{T},
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    x0::AbstractVector{T},
    P0::AbstractMatrix{T},
    E_z::AbstractArray{T,3},
    E_zz::AbstractArray{T,4},
    E_zz_prev::AbstractArray{T,4},
    P_smooth::AbstractArray{T,4},
    y::AbstractArray{T,3},
) where {T<:Real}
    # Calculate the Q-function for the state model
    Q_state = StateSpaceDynamics.Q_state(A, b, Q, P0, x0, E_z, E_zz, E_zz_prev)
    # Calculate the Q-function for the observation model
    Q_obs = Q_observation_model(C, log_d, E_z, P_smooth, y)

    return Q_state + Q_obs
end

"""
    calculate_elbo(
        plds::LinearDynamicalSystem{T,S,O},
        E_z::AbstractArray{T, 3},
        E_zz::AbstractArray{T, 4},
        E_zz_prev::AbstractArray{T, 4},
        P_smooth::AbstractArray{T, 4},
        y::AbstractArray{T, 3},
        total_entropy::T
    ) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}

Calculate the Evidence Lower Bound (ELBO) for a Poisson Linear Dynamical System (PLDS).

# Note
Ensure that the dimensions of input arrays match the expected dimensions as described in the
arguments section.
"""
function calculate_elbo(
    plds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractArray{T,3},
    E_zz::AbstractArray{T,4},
    E_zz_prev::AbstractArray{T,4},
    P_smooth::AbstractArray{T,4},
    y::AbstractArray{T,3},
    total_entropy::AbstractFloat,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    # Set up parameters
    A, b, Q, x0, p0 = plds.state_model.A,
    plds.state_model.b, plds.state_model.Q, plds.state_model.x0,
    plds.state_model.P0
    C, log_d = plds.obs_model.C, plds.obs_model.log_d

    # Calculate the expected complete log-likelihood
    ecll = Q_function(A, b, Q, C, log_d, x0, p0, E_z, E_zz, E_zz_prev, P_smooth, y)

    # Return the ELBO
    return ecll - total_entropy
end

"""
    gradient_observation_model!(
        grad::AbstractVector{T},
        C::AbstractMatrix{T},
        log_d::AbstractVector{T},
        E_z::AbstractArray{T},
        P_smooth::AbstractArray{T},
        y::AbstractArray{T},
    ) where {T<:Real}

Compute the gradient of the Q-function with respect to the observation model parameters
(C and log_d) for a Poisson Linear Dynamical System.
"""
function gradient_observation_model!(
    grad::AbstractVector{T},
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    E_z::AbstractArray{T},
    P_smooth::AbstractArray{T},
    y::AbstractArray{T},
) where {T<:Real}
    d = exp.(log_d)
    obs_dim, latent_dim = size(C)
    latent_dim, tsteps, trials = size(E_z)

    # Pre-allocate shared temporary arrays
    h = zeros(T, obs_dim)
    ρ = zeros(T, obs_dim)
    λ = zeros(T, obs_dim)
    CP_row = zeros(T, latent_dim)  # Single row buffer for CP computations

    fill!(grad, zero(T))

    @threads for k in 1:trials
        # Local temporary arrays for each thread
        local_grad = zeros(T, length(grad))

        @views for t in 1:tsteps

            # Compute h = C * z_t + d in-place
            mul!(h, C, E_z[:, t, k])
            @. h .+= d

            P_t = P_smooth[:, :, t, k]
            # Compute ρ more efficiently using local storage
            for i in 1:obs_dim
                # Compute one row of CP at a time
                mul!(CP_row, P_t', C[i, :])
                ρ[i] = T(0.5) * dot(C[i, :], CP_row)
            end

            # Compute λ in-place
            @. λ = exp(h + ρ)

            # Gradient computation with fewer allocations
            @views for j in 1:latent_dim
                for i in 1:obs_dim
                    idx = (j - 1) * obs_dim + i
                    CP_term = dot(C[i, :], P_t[:, j])
                    y_t = y[:, t, k]
                    z_t = E_z[:, t, k]
                    local_grad[idx] += y_t[i]*z_t[j] - λ[i]*(z_t[j] + CP_term)
                end
            end

            # Update log_d gradient
            @views local_grad[(end - obs_dim + 1):end] .+= (y[:, t, k] .- λ) .* d
        end

        # Thread-safe update of global gradient
        grad .+= local_grad
    end

    return grad .*= -1
end

"""
    update_observation_model!(
        plds::LinearDynamicalSystem{T,S,O},
        E_z::Array{T, 3},
        P_smooth::Array{T, 4},
        y::Array{T, 3}
    ) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Update the observation model parameters of a Poisson Linear Dynamical System using
gradient-based optimization.

# Note
- This function modifies `plds` in-place by updating the observation model parameters
    (C and log_d).
- The optimization is performed only if `plds.fit_bool[5]` is true.
"""
function update_observation_model!(
    plds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractArray{T,3},
    P_smooth::AbstractArray{T,4},
    y::AbstractArray{T,3},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
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

        opts = Optim.Options(;
            x_reltol=1e-12, x_abstol=1e-12, g_abstol=1e-12, f_reltol=1e-12, f_abstol=1e-12
        )

        # use CG result as inital guess for LBFGS
        result = optimize(
            f, g!, params, LBFGS(; linesearch=LineSearches.HagerZhang()), opts
        )

        # Update the parameters
        C_size = plds.obs_dim * plds.latent_dim
        plds.obs_model.C = reshape(
            result.minimizer[1:C_size], plds.obs_dim, plds.latent_dim
        )
        plds.obs_model.log_d = result.minimizer[(end - plds.obs_dim + 1):end]
    end

    return nothing
end

"""
    mstep!(
        plds::LinearDynamicalSystem{T,S,O},
        E_z::AbstractArray{T,3},
        E_zz::AbstractArray{T,4},
        E_zz_Prev{T,4},
        p_smooth{T,4},
        y::AbstractArray{T,3}
    ) where {T<:Real, S<:GaussianStateModel{T}, O<:PoissonObservationModel{T}}

Perform the M-step of the EM algorithm for a Poisson Linear Dynamical System with
multi-trial data.
"""
function mstep!(
    plds::LinearDynamicalSystem{T,S,O},
    E_z::AbstractArray{T,3},
    E_zz::AbstractArray{T,4},
    E_zz_prev::AbstractArray{T,4},
    p_smooth::AbstractArray{T,4},
    y::AbstractArray{T,3},
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    
    # Get old params using new approach
    old_params = _get_all_params_vec(plds)

    # State-side updates
    update_initial_state_mean!(plds, E_z)
    update_initial_state_covariance!(plds, E_z, E_zz)
    update_A_b!(plds, E_z, E_zz, E_zz_prev)
    update_Q!(plds, E_zz, E_zz_prev, E_z)

    # Update obs model
    update_observation_model!(plds, E_z, p_smooth, y)

    # Return parameter delta
    new_params = _get_all_params_vec(plds)
    return norm(new_params - old_params)
end
