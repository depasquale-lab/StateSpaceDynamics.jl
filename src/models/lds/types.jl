export LinearDynamicalSystem
export GaussianStateModel, GaussianObservationModel, PoissonObservationModel
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
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    return GaussianStateModel{T,M,V}(;
        A=A, Q=Q, b=b, x0=x0, P0=P0, Q_prior=nothing, P0_prior=nothing
    )
end

function GaussianObservationModel(
    C::M, R::M, d::V
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
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
