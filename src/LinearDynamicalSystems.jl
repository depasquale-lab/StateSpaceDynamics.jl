export LinearDynamicalSystem
export GaussianStateModel, GaussianObservationModel, PoissonObservationModel
export rand, smooth, fit!
export IWPrior

"""
    IWPrior{T<:Real, M<:AbstractMatrix}

Inverse-Wishart prior for a covariance matrix Σ ~ IW(Ψ, ν), with density
p(Σ) ∝ |Σ|^{-(ν + d + 1)/2} exp(-½ tr(Ψ Σ^{-1})) for d = size(Σ,1).

# Fields
- `Ψ::M`: Scale matrix (d×d, SPD).
- `ν::T`: Degrees of freedom (must satisfy `ν > d + 1` for a proper mode).

# Notes
- The MAP update for a posterior IW(Ψ + S, ν + n) is `(Ψ + S) / (ν + n + d + 1)`.
"""
Base.@kwdef struct IWPrior{T<:Real,M<:AbstractMatrix}
    Ψ::M
    ν::T
end

# helpers for new priors on cov matrices
@inline function iw_map(
    Ψ::AbstractMatrix{T}, ν::T, S::AbstractMatrix{T}, n::T, d::Int
) where {T}
    return (Ψ .+ S) ./ (ν + n + d + one(T))
end

@inline function iw_logprior_term(Σ::AbstractMatrix{T}, prior::IWPrior{T}) where {T}
    D = size(Σ, 1)
    Ψ, ν = prior.Ψ, prior.ν
    # log|Σ| via Cholesky
    F = cholesky(Symmetric(Σ))
    logdetΣ = 2sum(log, diag(F.U))
    # tr(Ψ Σ^{-1}) via triangular solves
    X = F \ Ψ                 # solves Σ * X = Ψ
    return -T(0.5) * ((ν + D + one(T)) * logdetΣ + tr(X))
end

"""
    GaussianStateModel{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}

Represents the state model of a Linear Dynamical System with Gaussian noise.

# Fields
- `A::M`: Transition matrix (size `latent_dim × latent_dim`).
- `Q::M`: Process noise covariance matrix.
- `b::V`: Bias vector (length `latent_dim`).
- `x0::V`: Initial state mean (length `latent_dim`).
- `P0::M`: Initial state covariance (size `latent_dim × latent_dim`).
- `Q_prior::Union{Nothing,IWPrior{T}} = nothing`: Optional Inverse-Wishart prior on `Q`. If set, MAP updates use its mode.
- `P0_prior::Union{Nothing,IWPrior{T}} = nothing`: Optional Inverse-Wishart prior on `P0`. If set, MAP updates use its mode.
"""
Base.@kwdef mutable struct GaussianStateModel{
    T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}
} <: AbstractStateModel{T}
    A::M
    Q::M
    b::V
    x0::V
    P0::M
    Q_prior::Union{Nothing,IWPrior{T}} = nothing
    P0_prior::Union{Nothing,IWPrior{T}} = nothing
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
    GaussianObservationModel{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}

Represents the observation model of a Linear Dynamical System with Gaussian noise.

# Fields
- `C::M`: Observation matrix of size `(obs_dim × latent_dim)`. Maps latent states into
    observation space.
- `R::M`: Observation noise covariance of size `(obs_dim × obs_dim)`.
- `d::V`: Bias vector of length `(obs_dim)`.
- `R_prior::Union{Nothing, IWPrior{T}} = nothing`: Optional Inverse-Wishart prior for `R`.
"""
Base.@kwdef mutable struct GaussianObservationModel{
    T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}
} <: AbstractObservationModel{T}
    C::M
    R::M
    d::V
    R_prior::Union{Nothing,IWPrior{T}} = nothing
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

# Conveneince cosntructors
function GaussianStateModel(
    A::M, Q::M, b::V, x0::V, P0::M
) where {T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
    return GaussianStateModel{T,M,V}(; A=A, Q=Q, b=b, x0=x0, P0=P0,
                                      Q_prior=nothing, P0_prior=nothing)
end

function GaussianObservationModel(
    C::M, R::M, d::V
) where {T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
    return GaussianObservationModel{T,M,V}(; C=C, R=R, d=d, R_prior=nothing)
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
    state_model::S, obs_model::O; fit_bool::Union{Vector{Bool},Nothing}=nothing
) where {T<:Real,S<:AbstractStateModel{T},O<:AbstractObservationModel{T}}

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
    lds = LinearDynamicalSystem{T,S,O}(
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

function _extract_state_params(state_model::GaussianStateModel{T}) where {T}
    return (
        A=state_model.A,
        Q=state_model.Q,
        b=state_model.b,
        x0=state_model.x0,
        P0=state_model.P0,
    )
end

"""
    initialize_FilterSmooth(model, num_obs) 
   
Initialize a `FilterSmooth` object for a given linear dynamical system model and number of observations.
"""
function initialize_FilterSmooth(
    model::LinearDynamicalSystem{T,S,O}, num_obs::Int
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    num_states = model.latent_dim
    return FilterSmooth{T}(
        zeros(T, num_states, num_obs),                    # x_smooth
        zeros(T, num_states, num_states, num_obs),        # p_smooth  
        zeros(T, num_states, num_states, num_obs),        # p_smooth_tt1
        zeros(T, num_states, num_obs),                    # E_z
        zeros(T, num_states, num_states, num_obs),        # E_zz
        zeros(T, num_states, num_states, num_obs),        # E_zz_prev
        zero(T),                                           # entropy
    )
end

function initialize_FilterSmooth(
    model::LinearDynamicalSystem{T,S,O}, tsteps::Int, ntrials::Int
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    filter_smooths = [initialize_FilterSmooth(model, tsteps) for _ in 1:ntrials]
    return TrialFilterSmooth(filter_smooths)
end

function _extract_obs_params(obs_model::GaussianObservationModel{T}) where {T}
    return (C=obs_model.C, R=obs_model.R, d=obs_model.d)
end

function _extract_obs_params(obs_model::PoissonObservationModel{T}) where {T}
    return (C=obs_model.C, log_d=obs_model.log_d, d=exp.(obs_model.log_d))
end

function _get_all_params_vec(
    lds::LinearDynamicalSystem{T,S,O}
) where {T<:Real,S<:AbstractStateModel{T},O<:AbstractObservationModel{T}}
    state_params = _extract_state_params(lds.state_model)
    obs_params = _extract_obs_params(lds.obs_model)

    # Convert named tuples to vectors and concatenate
    state_vec = vcat(
        vec(state_params.A),
        vec(state_params.Q),
        vec(state_params.b),
        vec(state_params.x0),
        vec(state_params.P0),
    )

    if lds.obs_model isa GaussianObservationModel
        obs_vec = vcat(vec(obs_params.C), vec(obs_params.R), vec(obs_params.d))
    else # PoissonObservationModel
        obs_vec = vcat(vec(obs_params.C), vec(obs_params.log_d))
    end

    return vcat(state_vec, obs_vec)
end

function _sample_trial!(
    rng, x_trial, y_trial, state_params, obs_params, obs_model::GaussianObservationModel
)
    tsteps = size(x_trial, 2)

    # Initial state
    x_trial[:, 1] = rand(rng, MvNormal(state_params.x0, state_params.P0))
    y_trial[:, 1] = rand(
        rng, MvNormal(obs_params.C * x_trial[:, 1] + obs_params.d, obs_params.R)
    )

    # Subsequent states
    for t in 2:tsteps
        x_trial[:, t] = rand(
            rng,
            MvNormal(state_params.A * x_trial[:, t - 1] + state_params.b, state_params.Q),
        )
        y_trial[:, t] = rand(
            rng, MvNormal(obs_params.C * x_trial[:, t] + obs_params.d, obs_params.R)
        )
    end
end

function _sample_trial!(
    rng, x_trial, y_trial, state_params, obs_params, obs_model::PoissonObservationModel
)
    tsteps = size(x_trial, 2)

    # Initial state
    x_trial[:, 1] = rand(rng, MvNormal(state_params.x0, state_params.P0))
    y_trial[:, 1] = rand.(rng, Poisson.(exp.(obs_params.C * x_trial[:, 1] + obs_params.d)))

    # Subsequent states
    for t in 2:tsteps
        x_trial[:, t] = rand(
            rng,
            MvNormal(state_params.A * x_trial[:, t - 1] + state_params.b, state_params.Q),
        )
        y_trial[:, t] = rand.(
            rng, Poisson.(exp.(obs_params.C * x_trial[:, t] + obs_params.d))
        )
    end
end

function Random.rand(
    rng::AbstractRNG, lds::LinearDynamicalSystem{T,S,O}; tsteps::Int, ntrials::Int=1
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}

    # Extract parameters once using a more systematic approach
    state_params = _extract_state_params(lds.state_model)
    obs_params = _extract_obs_params(lds.obs_model)

    # Pre-allocate based on observation model type
    x = Array{T,3}(undef, lds.latent_dim, tsteps, ntrials)
    y = Array{T,3}(undef, lds.obs_dim, tsteps, ntrials)

    # Sample trials (potentially in parallel for large ntrials)
    if ntrials > 10  # Threshold for parallelization
        Threads.@threads for trial in 1:ntrials
            _sample_trial!(
                rng,
                view(x,:,:,trial),
                view(y,:,:,trial),
                state_params,
                obs_params,
                lds.obs_model,
            )
        end
    else
        for trial in 1:ntrials
            _sample_trial!(
                rng,
                view(x,:,:,trial),
                view(y,:,:,trial),
                state_params,
                obs_params,
                lds.obs_model,
            )
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
- `ll::Vector{T}`: The complete-data log-likelihood of the LDS at each timestep.
"""
function loglikelihood(
    x::AbstractMatrix{U}, lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}
) where {U<:Real,T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    tsteps = size(y, 2)
    A, Q, x0, P0 = lds.state_model.A,
    lds.state_model.Q, lds.state_model.x0,
    lds.state_model.P0
    C, R, b, d = lds.obs_model.C, lds.obs_model.R, lds.state_model.b, lds.obs_model.d

    R_chol = cholesky(Symmetric(R)).U
    Q_chol = cholesky(Symmetric(Q)).U
    P0_chol = cholesky(Symmetric(P0)).U

    ll_vec = Vector{eltype(x)}(undef, tsteps)

    # Pre-allocate all temporary arrays
    temp_dx = zeros(eltype(x), size(x, 1))
    temp_dy = zeros(eltype(x), size(y, 1))
    temp_solve_Q = zeros(eltype(x), size(x, 1))
    temp_solve_R = zeros(eltype(x), size(y, 1))

    for t in 1:tsteps
        ll_t = zero(eltype(x))

        # Initial state contribution (only at t=1)
        if t == 1
            dx0 = view(x, :, 1) .- x0
            ll_t += sum(abs2, P0_chol \ dx0)
        end

        # Dynamics contribution (t > 1)
        if t > 1
            mul!(temp_dx, A, view(x, :, t-1), -one(eltype(x)), false)
            temp_dx .+= view(x, :, t) .- b
            ldiv!(temp_solve_Q, Q_chol, temp_dx)
            ll_t += sum(abs2, temp_solve_Q)
        end

        # Emission contribution
        mul!(temp_dy, C, view(x, :, t), -one(eltype(x)), false)
        temp_dy .+= view(y, :, t) .- d
        ldiv!(temp_solve_R, R_chol, temp_dy)
        ll_t += sum(abs2, temp_solve_R)

        ll_vec[t] = -eltype(x)(0.5) * ll_t
    end

    return ll_vec
end

"""
    Gradient(lds, y, x)

Compute the gradient of the log-likelihood with respect to the latent states for a linear
dynamical system.
"""
function Gradient(
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    latent_dim, tsteps = size(x)
    obs_dim = size(y, 1)

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

    # Pre-allocate all temporary arrays for efficiency
    dxt = zeros(T, latent_dim)
    dxt_next = zeros(T, latent_dim)
    dyt = zeros(T, obs_dim)
    tmp1 = zeros(T, latent_dim)  # for C_inv_R * dyt
    tmp2 = zeros(T, latent_dim)  # for A_inv_Q * dxt_next
    tmp3 = zeros(T, latent_dim)  # for Q_chol \ dxt

    # First time step
    dxt .= x[:, 1] .- x0
    mul!(dxt_next, A, x[:, 1])
    dxt_next .= x[:, 2] .- dxt_next .- b
    mul!(dyt, C, x[:, 1])
    dyt .= y[:, 1] .- dyt .- d

    mul!(tmp1, C_inv_R, dyt)
    mul!(tmp2, A_inv_Q, dxt_next)
    ldiv!(tmp3, P0_chol, dxt)

    grad[:, 1] .= tmp1 .+ tmp2 .- tmp3

    # Middle steps
    @views for t in 2:(tsteps - 1)
        # dxt = x[:, t] - A * x[:, t-1] - b
        mul!(dxt, A, x[:, t - 1])
        dxt .= x[:, t] .- dxt .- b

        # dxt_next = x[:, t+1] - A * x[:, t] - b
        mul!(dxt_next, A, x[:, t])
        dxt_next .= x[:, t + 1] .- dxt_next .- b

        # dyt = y[:, t] - C * x[:, t] - d
        mul!(dyt, C, x[:, t])
        dyt .= y[:, t] .- dyt .- d

        # tmp1 = C_inv_R * dyt
        mul!(tmp1, C_inv_R, dyt)

        # tmp2 = A_inv_Q * dxt_next
        mul!(tmp2, A_inv_Q, dxt_next)

        # tmp3 = Q_chol \ dxt
        ldiv!(tmp3, Q_chol, dxt)

        # grad[:, t] = tmp1 - tmp3 + tmp2
        grad[:, t] .= tmp1 .- tmp3 .+ tmp2
    end

    # Last time step
    mul!(dxt, A, x[:, tsteps - 1])
    dxt .= x[:, tsteps] .- dxt .- b
    mul!(dyt, C, x[:, tsteps])
    dyt .= y[:, tsteps] .- dyt .- d

    mul!(tmp1, C_inv_R, dyt)
    ldiv!(tmp3, Q_chol, dxt)

    grad[:, tsteps] .= tmp1 .- tmp3

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
    lds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}, x::AbstractMatrix{T}
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
    H = block_tridgm(H_diag, H_super, H_sub)

    return H, H_diag, H_super, H_sub
end

"""
    smooth(lds, y::AbstractMatrix)

Direct smoothing for a single trial.

# Arguments
- `lds::LinearDynamicalSystem`: The model.
- `y::AbstractMatrix`: Observations (obs_dim × tsteps).

# Returns
- `x_smooth::AbstractMatrix`: Smoothed latent means (latent_dim × tsteps).
- `p_smooth::Array{T,3}`: Smoothed latent covariances (latent_dim × latent_dim × tsteps).
"""
function smooth(lds::LinearDynamicalSystem, y::AbstractMatrix{T}) where {T}
    fs = initialize_FilterSmooth(lds, size(y, 2))
    smooth!(lds, fs, y)
    return fs.x_smooth, fs.p_smooth
end

function smooth(lds::LinearDynamicalSystem, y::AbstractArray{T,3}) where {T}
    tfs = initialize_FilterSmooth(lds, size(y, 2), size(y, 3))
    smooth!(lds, tfs, y)

    D = lds.latent_dim
    Tt = size(y, 2)
    N = size(y, 3)

    xs = Array{T,3}(undef, D, Tt, N)
    Ps = Array{T,4}(undef, D, D, Tt, N)

    for n in 1:N
        fs = tfs.FilterSmooths[n]
        xs[:, :, n] .= fs.x_smooth
        Ps[:, :, :, n] .= fs.p_smooth
    end
    return xs, Ps
end

function smooth!(
    lds::LinearDynamicalSystem{T,S,O}, fs::FilterSmooth{T}, y::AbstractMatrix{T}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    tsteps, D = size(y, 2), lds.latent_dim

    # use old fs if it exists, by default is zeros if no iteration of EM has occurred
    X₀ = Vector{T}(vec(fs.E_z))

    function nll(vec_x::AbstractVector{T})
        x = reshape(vec_x, D, tsteps)
        return -sum(loglikelihood(x, lds, y))
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

    # Profit
    fs.x_smooth .= reshape(res.minimizer, D, tsteps)

    H, main, super, sub = Hessian(lds, y, fs.x_smooth)

    # Get the second moments of the latent state path, use static matrices if the latent dimension is small
    if lds.latent_dim > 10
        p_smooth_result, p_smooth_tt1_result = block_tridiagonal_inverse(
            -sub, -main, -super
        )
        fs.p_smooth .= p_smooth_result
        fs.p_smooth_tt1[:, :, 2:end] .= p_smooth_tt1_result
    else
        p_smooth_result, p_smooth_tt1_result = block_tridiagonal_inverse_static(
            -sub, -main, -super, Val(lds.latent_dim)
        )
        fs.p_smooth .= p_smooth_result
        fs.p_smooth_tt1[:, :, 2:end] .= p_smooth_tt1_result
    end

    # Calculate the entropy, see Utilities.jl for the function
    fs.entropy = gaussian_entropy(Symmetric(H))

    # Symmetrize
    @views for i in 1:tsteps
        fs.p_smooth[:, :, i] .= 0.5 .* (fs.p_smooth[:, :, i] .+ fs.p_smooth[:, :, i]')
    end

    return fs
end

"""
    smooth!(lds, tfs, y::AbstractArray{T,3})

Direct smoothing for multiple trials.

# Arguments
- `lds::LinearDynamicalSystem`: The model.
- `tfs::TrialFilterSmooth`: Preallocated container (one per trial).
- `y::Array{T,3}`: Observations (obs_dim × tsteps × ntrials).

# Side effects
- Fills each `FilterSmooth` in `tfs` with `x_smooth`, `p_smooth`, `p_smooth_tt1`, `E_z`, `E_zz`, `E_zz_prev`, `entropy`.

# Returns
- `tfs`: The same `TrialFilterSmooth`, populated.
"""
function smooth!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, y::AbstractArray{T,3}
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
    Q_obs!(C, d, E_z, E_zz, y)

Single time-step observation component of the Q-function for
y_t ~ 𝓝(C x_t + d, R), before applying R^{-1} and constants.
"""
function Q_obs!(
    result::AbstractMatrix{T},
    C::AbstractMatrix{T},
    d::AbstractVector{T},
    E_z::AbstractVector{T},
    E_zz::AbstractMatrix{T},
    y::AbstractVector{T},
    buffers,
) where {T<:Real}

    # Unpack buffers
    ytil, sum_yy, sum_yz, work1, work2 = buffers

    # Residualize: ytil = y - d (pre-allocated buffer)
    ytil .= y .- d

    # All operations use pre-allocated buffers
    mul!(sum_yy, ytil, ytil')

    # Efficient outer product: sum_yz = ytil * E_z'
    fill!(sum_yz, zero(T))
    BLAS.ger!(one(T), ytil, E_z, sum_yz)

    # Build result using buffers
    copyto!(result, sum_yy)
    mul!(result, C, sum_yz', -one(T), one(T))   # result -= C * sum_yz'
    mul!(work1, sum_yz, C')                      # work1 = sum_yz * C'  
    result .-= work1                             # result -= work1
    mul!(work2, E_zz, C')                        # work2 = E_zz * C'
    mul!(result, C, work2, one(T), one(T))       # result += C * work2

    return result
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
    latent_dim = size(E_z, 1)
    tsteps = size(y, 2)

    # Pre-compute constants
    R_chol = cholesky(Symmetric(R))
    log_det_R = logdet(R_chol)
    const_term = obs_dim * log(2π)

    # Pre-allocate ALL buffers once (reuse across all timesteps!)
    temp = zeros(T, obs_dim, obs_dim)
    work_matrix = zeros(T, obs_dim, obs_dim)

    # Buffers for the lower-level Q_obs! (including ytil for bias)
    buffers = (
        ytil=zeros(T, obs_dim),
        sum_yy=zeros(T, obs_dim, obs_dim),
        sum_yz=zeros(T, obs_dim, latent_dim),
        work1=zeros(T, obs_dim, obs_dim),
        work2=zeros(T, latent_dim, obs_dim),
    )

    # Use views in the loop - now with buffer passing
    @views for t in axes(y, 2)
        # Pass buffers to lower-level function (with bias d)
        Q_obs!(work_matrix, C, d, E_z[:, t], E_zz[:, :, t], y[:, t], buffers)

        # Accumulate in-place
        temp .+= work_matrix
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
    calculate_elbo(lds, E_z, E_zz, E_zz_prev, p_smooth, y, total_entropy)

Calculate the Evidence Lower Bound (ELBO) for a Linear Dynamical System. 
Adds constant-free IW log-prior terms for `Q` and `P0` when priors are set, 
so the ELBO tracks the MAP objective.
"""
function calculate_elbo(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, y::AbstractArray{T,3}
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    ntrials = size(y, 3)
    Q_vals = zeros(T, ntrials)

    # Calculate total entropy from individual FilterSmooth objects
    total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths)

    # Thread over trials
    @threads for trial in 1:ntrials
        fs = tfs[trial]  # Get the FilterSmooth for this trial
        Q_vals[trial] = Q_function(
            lds.state_model.A,
            lds.state_model.b,
            lds.state_model.Q,
            lds.obs_model.C,
            lds.obs_model.d,
            lds.obs_model.R,
            lds.state_model.P0,
            lds.state_model.x0,
            fs.E_z,
            fs.E_zz,
            fs.E_zz_prev,
            view(y,:,:,trial),
        )
    end

    # prior terms (included once)
    prior_term = zero(T)
    if lds.state_model.Q_prior !== nothing
        prior_term += iw_logprior_term(lds.state_model.Q, lds.state_model.Q_prior)
    end
    if lds.state_model.P0_prior !== nothing
        prior_term += iw_logprior_term(lds.state_model.P0, lds.state_model.P0_prior)
    end
    if (lds.obs_model isa GaussianObservationModel) && (lds.obs_model.R_prior !== nothing)
        prior_term += iw_logprior_term(lds.obs_model.R, lds.obs_model.R_prior)
    end

    return sum(Q_vals) + prior_term - total_entropy
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
            mul!(
                fs.E_zz_prev[:, :, t], fs.x_smooth[:, t:t], fs.x_smooth[:, (t - 1):(t - 1)]'
            )
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
    estep(lds::LinearDynamicalSystem{T,S,O},tfs::TrialFilterSmooth, y::AbstractArray{T,3})

Perform the E-step of the EM algorithm for a Linear Dynamical System, treating all input as
multi-trial.

# Note
- This function first smooths the data using the `smooth` function, then computes sufficient
    statistics.
- It treats all input as multi-trial, with single-trial being a special case where
    `ntrials = 1`.
"""
function estep!(
    lds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth, y::AbstractArray{T,3}
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    smooth!(lds, tfs, y)
    sufficient_statistics!(tfs)
    elbo = calculate_elbo(lds, tfs, y)
    return elbo
end

"""
    update_initial_state_mean!(
                        lds::LinearDynamicalSystem{T,S,O, 
                        tfs::TrialFilterSmooth,
                        w::Union{Nothing,AbstractVector{<:AbstractVector{T}}} = nothing
                    )

Update the initial state mean of the Linear Dynamical System using the average across all
trials.
"""
function update_initial_state_mean!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    if lds.fit_bool[1]
        ntrials = length(tfs.FilterSmooths)
        x0_new = zeros(T, lds.latent_dim)
        total_weight = zero(T)

        for trial in 1:ntrials
            fs = tfs[trial]
            weight = isnothing(w) ? one(T) : w[trial][1]  # Weight at t=1

            x0_new .+= weight .* fs.E_z[:, 1]
            total_weight += weight
        end

        lds.state_model.x0 .= x0_new ./ total_weight
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
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[2] || return nothing

    D = lds.latent_dim
    S0_sum = zeros(T, D, D)
    total_weight = zero(T)

    for trial in 1:length(tfs.FilterSmooths)
        fs = tfs[trial]
        wt = isnothing(w) ? one(T) : w[trial][1]  # weight at t=1
        S0_sum .+= wt .* (fs.E_zz[:, :, 1] - (lds.state_model.x0 * lds.state_model.x0'))
        total_weight += wt
    end

    P0_hat = if lds.state_model.P0_prior === nothing
        S0_sum ./ total_weight
    else
        Ψ, ν = lds.state_model.P0_prior.Ψ, lds.state_model.P0_prior.ν
        iw_map(Ψ, ν, S0_sum, total_weight, D)
    end

    P0_hat .= 0.5 .* (P0_hat .+ P0_hat')  # symmetrize
    lds.state_model.P0 = P0_hat
    return nothing
end

"""
    update_A!(
        lds::LinearDynamicalSystem{T,S,O},
        E_zz::AbstractArray{T,4},
        E_zz_prev::AbstractArray{T,4}
    )

Update the transition matrix A of the Linear Dynamical System.

"""
function update_A_b!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[3] || return nothing
    D = lds.latent_dim
    ntrials = length(tfs)

    # Accumulate statistics for [A b] jointly
    Sxz = zeros(T, D, D + 1)
    Szz = zeros(T, D + 1, D + 1)

    for trial in 1:ntrials
        fs = tfs[trial]
        tsteps = size(fs.E_z, 2)
        weights = isnothing(w) ? nothing : w[trial]

        @views for t in 2:tsteps
            weight = isnothing(weights) ? one(T) : weights[t]

            # Sxz accumulation
            Sxz[:, 1:D] .+= weight .* fs.E_zz_prev[:, :, t]
            Sxz[:, D + 1] .+= weight .* fs.E_z[:, t]

            # Szz for augmented state z_{t-1} = [x_{t-1}; 1]
            Szz[1:D, 1:D] .+= weight .* fs.E_zz[:, :, t - 1]
            Szz[1:D, D + 1] .+= weight .* fs.E_z[:, t - 1]
            Szz[D + 1, 1:D] .+= weight .* fs.E_z[:, t - 1]
            Szz[D + 1, D + 1] += weight
        end
    end

    # Solve jointly: [A b] = Sxz / Szz
    AB = Sxz / Szz
    lds.state_model.A = AB[:, 1:D]
    lds.state_model.b = AB[:, D + 1]

    return nothing
end

"""
    update_Q!(
        lds::LinearDynamicalSystem{T,S,O},
        tfs::TrialFilterSmooth,
        w::Union{Nothing,AbstractVector{<:AbstractVector{T}}} = nothing
    )

Update the process noise covariance matrix Q of the Linear Dynamical System.

"""
function update_Q!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
    lds.fit_bool[4] || return nothing
    ntrials = length(tfs)
    state_dim = lds.latent_dim
    A = lds.state_model.A
    b = lds.state_model.b
    Q_sum = zeros(T, state_dim, state_dim)

    # Pre-allocate working matrices
    temp1 = Matrix{T}(undef, state_dim, state_dim)
    temp2 = Matrix{T}(undef, state_dim, state_dim)
    temp3 = Matrix{T}(undef, state_dim, state_dim)
    temp4 = Matrix{T}(undef, state_dim, state_dim)
    temp5 = Vector{T}(undef, state_dim)
    innovation_cov = Matrix{T}(undef, state_dim, state_dim)

    total_weight = zero(T)

    for trial in 1:ntrials
        fs = tfs[trial]
        tsteps = size(fs.E_zz, 3)
        weights = isnothing(w) ? nothing : w[trial]

        @views for t in 2:tsteps
            weight = isnothing(weights) ? one(T) : weights[t]

            Σt = fs.E_zz[:, :, t]
            Σtm1 = fs.E_zz[:, :, t - 1]
            Σcross = fs.E_zz_prev[:, :, t]
            μt = fs.E_z[:, t]
            μtm1 = fs.E_z[:, t - 1]

            # Compute using pre-allocated temps
            mul!(temp1, Σcross, A')
            mul!(temp2, A, Σcross')
            mul!(temp3, A, Σtm1)
            mul!(temp4, temp3, A')

            @. innovation_cov = Σt - temp1 - temp2 + temp4

            # Add bias terms
            innovation_cov .-= μt * b'
            innovation_cov .-= b * μt'
            mul!(temp5, A, μtm1)
            innovation_cov .+= temp5 * b'
            innovation_cov .+= b * temp5'
            innovation_cov .+= b * b'

            Q_sum .+= weight .* innovation_cov
            total_weight += weight
        end
    end

    if lds.state_model.Q_prior === nothing
        Q_hat = Q_sum ./ total_weight
    else
        Ψ, ν = lds.state_model.Q_prior.Ψ, lds.state_model.Q_prior.ν
        Q_hat = iw_map(Ψ, ν, Q_sum, total_weight, state_dim)
    end

    Q_hat .= 0.5 .* (Q_hat .+ Q_hat')   # symmetrize
    lds.state_model.Q = Q_hat
    return nothing
end

"""
    update_C_d!(
        lds::LinearDynamicalSystem{T,S,O},
        tfs::TrialFilterSmooth{T},
        y::AbstractArray{T,3},
        w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing
    )

Update the observation matrix C and bias d of the Linear Dynamical System.
"""
function update_C_d!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    lds.fit_bool[5] || return nothing

    ntrials = length(tfs)
    tsteps = size(y, 2)
    D = lds.latent_dim
    p = lds.obs_dim

    # Accumulate statistics for [C d] jointly
    Syz = zeros(T, p, D + 1)
    Szz = zeros(T, D + 1, D + 1)

    # Pre-allocate working matrices (reuse across all trials and timesteps!)
    work_yz = Matrix{T}(undef, p, D)  # For y * μ'
    work_outer = Matrix{T}(undef, D, D)  # For weighted Σ

    for trial in 1:ntrials
        fs = tfs[trial]
        weights = isnothing(w) ? nothing : w[trial]
        @views for t in 1:tsteps
            wt = isnothing(weights) ? one(T) : weights[t]

            μ = fs.E_z[:, t]
            Σ = fs.E_zz[:, :, t]  # E[x_t x_tᵀ]
            yt = y[:, t, trial]

            # Syz accumulates y_t * μ'  (outer product), weighted
            fill!(work_yz, zero(T))
            BLAS.ger!(wt, yt, μ, work_yz)   # work_yz += wt * yt * μ'
            Syz[:, 1:D] .+= work_yz
            Syz[:, D + 1] .+= wt .* yt      # bias column

            # Szz accumulates E[[x;1][x;1]ᵀ] weighted
            work_outer .= Σ
            work_outer .*= wt
            Szz[1:D, 1:D] .+= work_outer
            Szz[1:D, D + 1] .+= wt .* μ
            Szz[D + 1, 1:D] .+= wt .* μ
            Szz[D + 1, D + 1] += wt
        end
    end

    # Solve jointly: [C d] = Syz / Szz
    CD = Syz / Szz
    lds.obs_model.C = CD[:, 1:D]
    lds.obs_model.d = CD[:, D + 1]

    return nothing
end

"""
    update_R!(
        lds::LinearDynamicalSystem{T,S,O},
        tfs::TrialFilterSmooth{T},
        y::AbstractArray{T,3},
        w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing
    )

Update the observation noise covariance matrix R of the Linear Dynamical System.
"""
function update_R!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    lds.fit_bool[6] || return nothing

    obs_dim = lds.obs_dim
    tsteps = size(y, 2)
    ntrials = length(tfs)

    R_new = zeros(T, obs_dim, obs_dim)
    C = lds.obs_model.C
    d = lds.obs_model.d

    total_weight = zero(T)

    # Pre-allocate all temporary arrays (reuse across all trials and timesteps!)
    innovation = Vector{T}(undef, obs_dim)
    Czt = Vector{T}(undef, obs_dim)
    temp_matrix = Matrix{T}(undef, obs_dim, lds.latent_dim)
    outer_product = Matrix{T}(undef, lds.latent_dim, lds.latent_dim)
    state_uncertainty = Matrix{T}(undef, lds.latent_dim, lds.latent_dim)

    for trial in 1:ntrials
        fs = tfs[trial]
        weights = isnothing(w) ? nothing : w[trial]

        @views for t in 1:tsteps
            wt = isnothing(weights) ? one(T) : weights[t]

            # Compute innovation = y - (C*z_t + d) using pre-allocated arrays
            mul!(Czt, C, fs.E_z[:, t])
            @. innovation = y[:, t, trial] - (Czt + d)

            # Add weighted innovation outer product
            BLAS.ger!(wt, innovation, innovation, R_new)

            # Compute state_uncertainty = E[zz] - E[z]E[z]' efficiently
            mul!(outer_product, fs.E_z[:, t], fs.E_z[:, t]')
            state_uncertainty .= fs.E_zz[:, :, t]
            state_uncertainty .-= outer_product

            # Add weighted C * state_uncertainty * C'
            mul!(temp_matrix, C, state_uncertainty)
            mul!(R_new, temp_matrix, C', wt, one(T))

            total_weight += wt
        end
    end

    # Apply prior and normalize
    if lds.obs_model.R_prior === nothing
        R_hat = R_new ./ total_weight
    else
        Ψ, ν = lds.obs_model.R_prior.Ψ, lds.obs_model.R_prior.ν
        R_hat = iw_map(Ψ, ν, R_new, total_weight, obs_dim)
    end

    R_hat .= 0.5 .* (R_hat .+ R_hat')
    lds.obs_model.R = R_hat
    return nothing
end

"""
    mstep!(lds, tfs, y, w)

Perform the M-step of the EM algorithm for a Linear Dynamical System with multi-trial data.
"""
function mstep!(
    lds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
    # Get initial parameters using new approach
    old_params = _get_all_params_vec(lds)

    # Update parameters
    update_initial_state_mean!(lds, tfs, w)
    update_initial_state_covariance!(lds, tfs, w)
    update_A_b!(lds, tfs, w)
    update_Q!(lds, tfs, w)
    update_C_d!(lds, tfs, y, w)
    update_R!(lds, tfs, y, w)

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
    max_iter::Int=100,
    tol::Float64=1e-6,
    progress=true,
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
    # Create a FilterSmooth object
    tfs = initialize_FilterSmooth(lds, size(y, 2), size(y, 3))

    # Initialize progress bar only if progress=true
    prog = if progress
        if O <: GaussianObservationModel
            Progress(max_iter; desc="Fitting LDS via EM...", barlen=50, showspeed=true)
        elseif O <: PoissonObservationModel
            Progress(
                max_iter;
                desc="Fitting Poisson LDS via LaPlaceEM...",
                barlen=50,
                showspeed=true,
            )
        else
            error("Unknown LDS model type")
        end
    else
        nothing
    end

    # Run EM
    for i in 1:max_iter
        # E-step
        elbo = estep!(lds, tfs, y)
        # M-step
        Δparams = mstep!(lds, tfs, y)
        # Update the log-likelihood vector and parameter difference
        push!(elbos, elbo)
        push!(param_diff, Δparams)

        # Update the progress bar only if it exists
        if progress && prog !== nothing
            next!(prog)
        end

        # Check convergence
        if abs(elbo - prev_elbo) < tol
            if progress && prog !== nothing
                finish!(prog)
            end
            return elbos, param_diff
        end

        prev_elbo = elbo
    end

    # Finish the progress bar if max_iter is reached
    if progress && prog !== nothing
        finish!(prog)
    end

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
- `ll::Vector{T}`: The log-likelihood value.

# Ref
- loglikelihood(
    x::AbstractArray{T,3},
    plds::LinearDynamicalSystem{T,S,O},
    y::AbstractArray{T,3}
)
"""
function loglikelihood(
    x::AbstractMatrix{U}, plds::LinearDynamicalSystem{T,S,O}, y::AbstractMatrix{T}
) where {U<:Real,T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}

    # Result type and setup
    R = promote_type(T, U)
    tsteps = size(y, 2)
    ll = zeros(R, tsteps)

    # Convert the log firing rate to firing rate
    d = exp.(plds.obs_model.log_d)

    # Pre-compute inverses
    inv_p0 = inv(plds.state_model.P0)
    inv_Q = inv(plds.state_model.Q)

    # Get dimensions
    C = plds.obs_model.C
    A = plds.state_model.A
    x0 = plds.state_model.x0
    obs_dim, latent_dim = size(C)
    obs_tmp = Vector{eltype(x)}(undef, obs_dim)

    @views for t in 1:tsteps
        obs_tmp .= C * x[:, t] .+ d
        ll[t] += (dot(y[:, t], obs_tmp) - sum(exp, obs_tmp))
    end

    # Prior term p(x₁) goes to t = 1
    dx1 = @view(x[:, 1]) .- plds.state_model.x0
    ll[1] += -R(0.5) * dot(dx1, inv_p0 * dx1)

    # Transition terms p(xₜ|xₜ₋₁) go to their respective t (t ≥ 2)
    A = plds.state_model.A
    b = plds.state_model.b
    trans_tmp = Vector{eltype(x)}(undef, latent_dim)

    @views for t in 2:tsteps
        trans_tmp .= x[:, t] .- (A * x[:, t - 1] .+ b)
        ll[t] += -R(0.5) * dot(trans_tmp, inv_Q * trans_tmp)
    end

    return ll
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
        ll[n] .= sum(loglikelihood(x[:, :, n], plds, y[:, :, n]))
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
            mul!(Ax_prev, A, x[:, t - 1])

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
            state_diff .= x[:, t + 1] .- Ax_t          # state_diff = x[:, t+1] - A * x[:, t]
            mul!(temp_grad, inv_Q, state_diff)       # temp_grad = inv_Q * state_diff
            mul!(grad[:, t], A', temp_grad)          # grad[:, t] = A' * temp_grad

            # Add common_term
            grad[:, t] .+= common_term

            # Second part: - inv_Q * (x[:, t] - A * x[:, t-1])
            mul!(Ax_prev, A, x[:, t - 1])              # Ax_prev = A * x[:, t-1]
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

    # Fill the super and sub diagonals
    H_sub = [H_sub_entry for _ in 1:(tsteps - 1)]
    H_super = [H_super_entry for _ in 1:(tsteps - 1)]

    λ = zeros(T, size(C, 1))
    z = similar(λ)
    poisson_tmp = Matrix{T}(undef, size(C, 2), size(C, 2))
    H_diag = [Matrix{T}(undef, size(x, 1), size(x, 1)) for _ in 1:tsteps]

    # minnimal allocation Hessian helper function
    function calculate_poisson_hess!(out::Matrix{T}, C::Matrix{T}, λ::Vector{T}) where {T}
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
    Q_observation_model(C, log_d, E_z, p_smooth, y, weights)

Calculate the Q-function for the observation model for a single trial with optional per-timestep weights.
"""
function Q_observation_model(
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    E_z::AbstractMatrix{T},
    p_smooth::AbstractArray{T,3},
    y::AbstractMatrix{T},
    weights::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real}
    obs_dim, state_dim = size(C)
    d = exp.(log_d)
    Q_val = zero(T)
    tsteps = size(y, 2)

    h = Vector{T}(undef, obs_dim)
    ρ = Vector{T}(undef, obs_dim)
    temp_vec = Vector{T}(undef, state_dim)

    @views for t in 1:tsteps
        wt = isnothing(weights) ? one(T) : weights[t]

        Ez_t = E_z[:, t]
        P_t = p_smooth[:, :, t]
        y_t = y[:, t]

        # h = C * Ez_t + d
        mul!(h, C, Ez_t)
        h .+= d

        # Compute ρ[i] = 0.5 * c_i' * P_t * c_i
        for i in 1:obs_dim
            c_i = view(C, i, :)
            mul!(temp_vec, P_t, c_i)
            ρ[i] = T(0.5) * dot(c_i, temp_vec)
        end

        # Compute ŷ = exp(h + ρ) in-place, reusing ρ as ŷ
        for i in 1:obs_dim
            ρ[i] = exp(h[i] + ρ[i])
        end

        # Compute weighted sum(y_t .* h .- ŷ)
        for i in 1:obs_dim
            Q_val += wt * (y_t[i] * h[i] - ρ[i])
        end
    end

    return Q_val
end

"""
    Q_observation_model(C, log_d, tfs, y, w)

Calculate the Q-function for the observation model across all trials using TrialFilterSmooth with optional weights.
"""
function Q_observation_model(
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real}
    trials = length(tfs.FilterSmooths)
    Q_vals = zeros(T, trials)

    @threads for k in 1:trials
        fs = tfs[k]
        weights = isnothing(w) ? nothing : w[k]
        Q_vals[k] = Q_observation_model(
            C, log_d, fs.E_z, fs.p_smooth, view(y,:,:,k), weights
        )
    end

    return sum(Q_vals)
end

"""
    Q_function(A, b, Q, C, log_d, x0, P0, E_z, E_zz, E_zz_prev, y)

Calculate the Q-function for a single trial of a Poisson Linear Dynamical System.
"""
function Q_function(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    Q::AbstractMatrix{T},
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    x0::AbstractVector{T},
    P0::AbstractMatrix{T},
    E_z::AbstractMatrix{T},
    E_zz::AbstractArray{T,3},
    E_zz_prev::AbstractArray{T,3},
    p_smooth::AbstractArray{T,3},
    y::AbstractMatrix{T},
) where {T<:Real}
    state_q = StateSpaceDynamics.Q_state(A, b, Q, P0, x0, E_z, E_zz, E_zz_prev)
    obs_q = Q_observation_model(C, log_d, E_z, p_smooth, y)
    return state_q + obs_q
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

Calculate the Evidence Lower Bound (ELBO) for a Poisson Linear Dynamical System (PLDS). Adds constant-free IW log-prior terms 
for `Q` and `P0` when priors are set, so the ELBO tracks the MAP objective.

# Note
Ensure that the dimensions of input arrays match the expected dimensions as described in the
arguments section.
"""
function calculate_elbo(
    plds::LinearDynamicalSystem{T,S,O}, tfs::TrialFilterSmooth{T}, y::AbstractArray{T,3}
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    # Set up parameters
    A, b, Q, x0, p0 = plds.state_model.A,
    plds.state_model.b, plds.state_model.Q, plds.state_model.x0,
    plds.state_model.P0
    C, log_d = plds.obs_model.C, plds.obs_model.log_d

    ntrials = length(tfs.FilterSmooths)
    Q_vals = zeros(T, ntrials)

    # Calculate total entropy from individual FilterSmooth objects
    total_entropy = sum(fs.entropy for fs in tfs.FilterSmooths)

    # Thread over trials (like Gaussian version)
    @threads for trial in 1:ntrials
        fs = tfs[trial]  # Get the FilterSmooth for this trial
        Q_vals[trial] = Q_function(
            A,
            b,
            Q,
            C,
            log_d,
            x0,
            p0,
            fs.E_z,
            fs.E_zz,
            fs.E_zz_prev,
            fs.p_smooth,
            view(y,:,:,trial),
        )
    end

    # IW priors on state covariances (if present)
    prior_term = zero(T)
    if plds.state_model.Q_prior !== nothing
        prior_term += iw_logprior_term(plds.state_model.Q, plds.state_model.Q_prior)
    end

    if plds.state_model.P0_prior !== nothing
        prior_term += iw_logprior_term(plds.state_model.P0, plds.state_model.P0_prior)
    end

    return sum(Q_vals) + prior_term - total_entropy
end

"""
    gradient_observation_model_single_trial!(grad, C, log_d, E_z, p_smooth, y, weights)

Compute the gradient for a single trial and add it to the accumulated gradient.
"""
function gradient_observation_model_single_trial!(
    grad::AbstractVector{T},
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    E_z::AbstractMatrix{T},
    p_smooth::AbstractArray{T,3},
    y::AbstractMatrix{T},
    weights::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:Real}
    d = exp.(log_d)
    obs_dim, latent_dim = size(C)
    tsteps = size(y, 2)

    # Pre-allocate temporary arrays
    h = Vector{T}(undef, obs_dim)
    ρ = Vector{T}(undef, obs_dim)
    λ = Vector{T}(undef, obs_dim)
    CP = Matrix{T}(undef, obs_dim, latent_dim)

    @views for t in 1:tsteps
        weight = isnothing(weights) ? one(T) : weights[t]

        E_z_t = E_z[:, t]
        P_smooth_t = p_smooth[:, :, t]
        y_t = y[:, t]

        # Compute h = C * z_t + d
        mul!(h, C, E_z_t)
        h .+= d

        # Pre-compute CP = C * P_smooth_t
        mul!(CP, C, P_smooth_t)

        # Compute ρ efficiently 
        for i in 1:obs_dim
            ρ[i] = T(0.5) * dot(C[i, :], CP[i, :])
        end

        # Compute λ = exp(h + ρ)
        for i in 1:obs_dim
            λ[i] = exp(h[i] + ρ[i])
        end

        # Gradient computation with weight
        for j in 1:latent_dim
            for i in 1:obs_dim
                idx = (j - 1) * obs_dim + i
                grad[idx] += weight * (y_t[i] * E_z_t[j] - λ[i] * (E_z_t[j] + CP[i, j]))
            end
        end

        # Update log_d gradient
        @views grad[(end - obs_dim + 1):end] .+= weight .* (y_t .- λ) .* d
    end
end

"""
    gradient_observation_model!(grad, C, log_d, tfs, y, w)

Compute the gradient of the Q-function with respect to the observation model parameters using TrialFilterSmooth.
"""
function gradient_observation_model!(
    grad::AbstractVector{T},
    C::AbstractMatrix{T},
    log_d::AbstractVector{T},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real}
    trials = length(tfs.FilterSmooths)

    fill!(grad, zero(T))

    # Accumulate gradients from all trials
    @threads for k in 1:trials
        fs = tfs[k]
        weights = isnothing(w) ? nothing : w[k]

        # Local gradient for this trial
        local_grad = zeros(T, length(grad))

        gradient_observation_model_single_trial!(
            local_grad, C, log_d, fs.E_z, fs.p_smooth, view(y,:,:,k), weights
        )

        # Thread-safe update of global gradient
        grad .+= local_grad
    end

    return grad .*= -1
end

"""
    update_observation_model!(plds, tfs, y, w)

Update the observation model parameters of a PLDS model.
"""
function update_observation_model!(
    plds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    if plds.fit_bool[5]
        params = vcat(vec(plds.obs_model.C), plds.obs_model.log_d)

        function f(params::Vector{T})
            C_size = plds.obs_dim * plds.latent_dim
            log_d = params[(end - plds.obs_dim + 1):end]
            C = reshape(params[1:C_size], plds.obs_dim, plds.latent_dim)
            return -Q_observation_model(C, log_d, tfs, y, w)
        end

        function g!(grad::Vector{T}, params::Vector{T})
            C_size = plds.obs_dim * plds.latent_dim
            log_d = params[(end - plds.obs_dim + 1):end]
            C = reshape(params[1:C_size], plds.obs_dim, plds.latent_dim)
            return gradient_observation_model!(grad, C, log_d, tfs, y, w)
        end

        opts = Optim.Options(;
            x_reltol=1e-12, x_abstol=1e-12, g_abstol=1e-12, f_reltol=1e-12, f_abstol=1e-12
        )

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
    mstep!(plds, tfs, y, w)

Perform the M-step of the EM algorithm for a Poisson Linear Dynamical System with multi-trial data.
"""
function mstep!(
    plds::LinearDynamicalSystem{T,S,O},
    tfs::TrialFilterSmooth{T},
    y::AbstractArray{T,3},
    w::Union{Nothing,AbstractVector{<:AbstractVector{T}}}=nothing,
) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
    # Get old params
    old_params = _get_all_params_vec(plds)

    # Update state parameters
    update_initial_state_mean!(plds, tfs, w)
    update_initial_state_covariance!(plds, tfs, w)
    update_A_b!(plds, tfs, w)
    update_Q!(plds, tfs, w)

    # Update observation parameters
    update_observation_model!(plds, tfs, y, w)

    # Return parameter delta
    new_params = _get_all_params_vec(plds)
    return norm(new_params - old_params)
end
