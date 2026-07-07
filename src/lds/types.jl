"""
    AbstractStateModel{T<:Real}

Abstract supertype for latent-state models of a [`LinearDynamicalSystem`](@ref)
(e.g. [`GaussianStateModel`](@ref)). A state model defines how the latent state
evolves from one timestep to the next.
"""
abstract type AbstractStateModel{T<:Real} end

"""
    AbstractObservationModel{T<:Real}

Abstract supertype for observation (emission) models of a
[`LinearDynamicalSystem`](@ref) (e.g. [`GaussianObservationModel`](@ref) or
[`PoissonObservationModel`](@ref)). An observation model defines how observed
data are generated from the latent state.
"""
abstract type AbstractObservationModel{T<:Real} end

"""
    Data{T<:Real}

Container for the observed data passed to the Kalman path: observations `y`,
dynamics (latent) inputs `ux`, and observation inputs `uy`, each `(dim, tsteps, ntrials)`
(input fields may have zero rows when no controls are supplied).
"""
Base.@kwdef struct Data{T<:Real}
    y::Array{T,3}
    ux::Array{T,3}
    uy::Array{T,3}
end

"""
    GaussianStateModel{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}

Represents the state model of a Linear Dynamical System with Gaussian noise.

State evolution:
```math
x_1           ~ N(x_0, P_0)
x_{t+1} | x_t ~ N(A x_t + b + B ux_t, Q)
```
where `B·ux_t` is present only when `B` is supplied (i.e., has nonzero columns).

# Fields
- `A::M`: Transition matrix (size `latent_dim × latent_dim`).
- `Q::M`: Process noise covariance matrix.
- `b::V`: Bias vector (length `latent_dim`).
- `x0::V`: Initial state mean (length `latent_dim`).
- `P0::M`: Initial state covariance (size `latent_dim × latent_dim`).
- `B::M`: Optional dynamics input matrix (`latent_dim × ux_dim`).
    When supplied, inputs `ux` must be passed to `fit!`/`smooth!` via a keyword argument.
- `Q_prior::Union{Nothing,IWPrior{T}} = nothing`: Optional Inverse-Wishart prior on `Q`. If set, MAP updates use its mode.
- `P0_prior::Union{Nothing,IWPrior{T}} = nothing`: Optional Inverse-Wishart prior on `P0`. If set, MAP updates use its mode.
- `AB_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing`: Optional matrix-normal prior on
    the stacked dynamics matrix `[A B]`. Pair with `Q_prior` for a full MNIW prior on `(AB, Q)`.
    Prior matrices are stored as plain `Matrix{T}` (decoupled from `A`'s storage type `M`) so
    they match the internal `KalmanWorkspace` regardless of how `A` is stored.
"""
Base.@kwdef mutable struct GaussianStateModel{
    T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}
} <: AbstractStateModel{T}
    A::M
    Q::M
    b::V
    x0::V
    P0::M
    B::M = zeros(eltype(A), size(A, 1), 0)
    Q_prior::Union{Nothing,IWPrior{T}} = nothing
    P0_prior::Union{Nothing,IWPrior{T}} = nothing
    AB_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing
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
- `CD_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing`: Optional matrix-normal prior on
    the stacked emission matrix `[C D]`. Pair with `R_prior` for a full MNIW prior on `(CD, R)`.
    Prior matrices are stored as plain `Matrix{T}` (decoupled from `C`'s storage type `M`) so
    they match the internal `KalmanWorkspace` regardless of how `C` is stored.
"""
Base.@kwdef mutable struct GaussianObservationModel{
    T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}
} <: AbstractObservationModel{T}
    C::M
    R::M
    d::V
    D::M = zeros(eltype(C), size(C, 1), 0)  # eltype-preserving default
    R_prior::Union{Nothing,IWPrior{T}} = nothing
    CD_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing
end

# Convenience constructors (State)
function GaussianStateModel(
    A::M, Q::M, b::V, x0::V, P0::M
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    return GaussianStateModel{T,M,V}(;
        A=A,
        Q=Q,
        b=b,
        x0=x0,
        P0=P0,
        B=zeros(T, size(A, 1), 0),
        Q_prior=nothing,
        P0_prior=nothing,
        AB_prior=nothing,
    )
end

function GaussianStateModel(A::M, Q::M, B::M, P0::M) where {T<:Real,M<:AbstractMatrix{T}}
    return GaussianStateModel{T,M,Vector{T}}(;
        A=A,
        Q=Q,
        b=zeros(T, size(A, 1)),
        x0=zeros(T, size(A, 1)),
        P0=P0,
        B=B,
        Q_prior=nothing,
        P0_prior=nothing,
        AB_prior=nothing,
    )
end

# Convenience constructors (Observation)

function GaussianObservationModel(
    C::M, R::M, d::V
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    return GaussianObservationModel{T,M,V}(;
        C=C, R=R, d=d, D=zeros(eltype(C), size(C, 1), 0), R_prior=nothing, CD_prior=nothing
    )
end

function GaussianObservationModel(C::M, R::M, D::M) where {T<:Real,M<:AbstractMatrix{T}}
    return GaussianObservationModel{T,M,Vector{T}}(;
        C=C, R=R, d=zeros(T, size(C, 1)), D=D, R_prior=nothing, CD_prior=nothing
    )
end

"""
    PoissonObservationModel{
        T<:Real,
        M<:AbstractMatrix{T},
        V<:AbstractVector{T}
    } <: AbstractObservationModel{T}

Represents the observation model of a Linear Dynamical System with Poisson observations,
with canonical log-link:

```math
λ_t = exp(C x_t + d)
```

`d` is the standard Poisson-GLM intercept — the per-channel baseline log-rate,
unconstrained in ℝ; positivity of the rate `λ` is provided by the `exp`.

# Fields
- `C::AbstractMatrix{T}`: Observation matrix of size `(obs_dim × latent_dim)`. Maps latent
    states into observation space.
- `d::AbstractVector{T}`: Per-neuron baseline log-rate (length `obs_dim`). Free in ℝ.
- `CD_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing`: Optional matrix-normal prior on
    the stacked emission matrix `[C d]` (treated as a single regression of `log λ` on
    `[x; 1]`). Prior matrices are stored as plain `Matrix{T}`, decoupled from `C`'s storage
    type `M`. `M₀` and `Λ` have shapes `(obs_dim, latent_dim+1)` and
    `(latent_dim+1, latent_dim+1)` respectively. Unlike the Gaussian path there is no IW
    counterpart since Poisson has no observation-noise covariance — this is an MN-only
    prior contributing `½ tr(([C d] - M₀) Λ ([C d] - M₀)')` to the LBFGS objective.
"""
Base.@kwdef mutable struct PoissonObservationModel{
    T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}
} <: AbstractObservationModel{T}
    C::M
    d::V
    CD_prior::Union{Nothing,MNPrior{T,Matrix{T}}} = nothing
end

# 2-arg convenience constructor; matches the Gaussian path's positional form
# so callers don't have to spell out `CD_prior=nothing`.
function PoissonObservationModel(
    C::M, d::V
) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    return PoissonObservationModel{T,M,V}(; C=C, d=d, CD_prior=nothing)
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
- `fit_bool::Vector{Bool}`: Vector indicating which parameters to fit during optimization.
    Length 6 for the Gaussian path (`[x0, P0, A&b&B, Q, C&d&D, R]`); the M-step
    regression fits each row jointly. Length 5 for the Poisson path
    (`[x0, P0, A&b, Q, C&d]`).
"""
Base.@kwdef struct LinearDynamicalSystem{
    T<:Real,S<:AbstractStateModel{T},O<:AbstractObservationModel{T}
}
    state_model::S
    obs_model::O
    latent_dim::Int
    obs_dim::Int
    state_input_dim::Int = 0
    obs_input_dim::Int = 0
    fit_bool::Vector{Bool}
end

function LinearDynamicalSystem(
    state_model::S, obs_model::O; fit_bool::Union{Vector{Bool},Nothing}=nothing
) where {T<:Real,S<:AbstractStateModel{T},O<:AbstractObservationModel{T}}

    # Infer dimensions from matrices
    latent_dim = size(state_model.A, 1)
    obs_dim = size(obs_model.C, 1)
    state_input_dim = if hasproperty(state_model, :B) && !isnothing(state_model.B)
        size(state_model.B, 2)
    else
        0
    end
    obs_input_dim =
        hasproperty(obs_model, :D) && !isnothing(obs_model.D) ? size(obs_model.D, 2) : 0

    # Set default fit_bool based on observation model type. The M-step fits
    # [A b B] and [C d D] as joint regressions, so the Gaussian layout is length 6.
    if fit_bool === nothing
        if obs_model isa PoissonObservationModel
            # Poisson: [x0, P0, A&b, Q, C&d] (5 parameters)
            fit_bool = [true, true, true, true, true]
        else
            # Gaussian (BTD): [x0, P0, A&b&B, Q, C&d&D, R] (6 parameters)
            fit_bool = [true, true, true, true, true, true]
        end
    end

    # Create the LDS
    lds = LinearDynamicalSystem{T,S,O}(
        state_model,
        obs_model,
        latent_dim,
        obs_dim,
        state_input_dim,
        obs_input_dim,
        fit_bool,
    )

    # Validate the constructed LDS (throws on error)
    validate_LDS(lds)

    return lds
end

"""
    SLDS{T,S,O,TM,ISV}

A Switching Linear Dynamical System (SLDS). A hierarchical time-series model of the form:

```math
z_t | z_{t-1} ~ Categorical(A_{z_{t-1}, :})
x_t | x_{t-1}, z_t ~ N(A^{(z_t)} x_{t-1} + b^{(z_t)}, Q^{(z_t)})
y_t | x_t, z_t ~ N(C^{(z_t)} x_t + d^{(z_t)}, R^{(z_t)})
```

# Fields
- `A::TM`: Transition matrix for the discrete states (K x K)
- `πₖ::ISV`: Initial state distribution for the discrete states (K-dimensional vector)
- `LDSs::Vector{LinearDynamicalSystem{T,S,O}}`: Vector of K Linear Dynamical Systems, one for each discrete state
"""
@kwdef mutable struct SLDS{
    T<:Real,
    S<:AbstractStateModel,
    O<:AbstractObservationModel,
    TM<:AbstractMatrix{T},
    ISV<:AbstractVector{T},
}
    A::TM
    πₖ::ISV
    LDSs::Vector{LinearDynamicalSystem{T,S,O}}
end

"""
    SLDSDiscreteLayer{T,TM,TV}

Thin wrapper satisfying the `HiddenMarkovModels.AbstractHMM` interface for the discrete
switching layer of an SLDS.  The `logL` matrix (K×T) is pre-filled with per-state
log-likelihoods before each forward-backward call; `obs_seq` is then just `1:T` (timestep
indices) so that `obs_logdensities!` can look up the correct column.

Fields `A` and `πₖ` are kept as references to the parent SLDS matrices so that in-place
M-step updates are automatically reflected.
"""
mutable struct SLDSDiscreteLayer{T<:Real,TM<:AbstractMatrix{T},TV<:AbstractVector{T}} <:
               HMMs.AbstractHMM
    A::TM            # K×K row-stochastic transition matrix
    πₖ::TV           # K initial-state distribution
    logL::Matrix{T}  # K×T pre-computed log-likelihoods; mutated before each FB pass
end

HMMs.initialization(dl::SLDSDiscreteLayer) = dl.πₖ
HMMs.transition_matrix(dl::SLDSDiscreteLayer) = dl.A

# Override the log-density chokepoint so obs_distributions is never needed.
# obs is the timestep index t (an Int), supplied as obs_seq = 1:T.
function HMMs.obs_logdensities!(
    logb::AbstractVector, dl::SLDSDiscreteLayer, obs::Int, control; kwargs...
)
    logb .= view(dl.logL, :, obs)
    return nothing
end

# Provide eltype without going through obs_distributions
Base.eltype(::SLDSDiscreteLayer{T}, obs, control) where {T} = T

#=
Workaround for JET union-split false positive on views with unbound eltype
(remove when fixed upstream; see: https://github.com/depasquale-lab/StateSpaceDynamics.jl/issues/105)
=#
@inline tview(A::AbstractArray{T}, I...) where {T} = view(A, I...)::SubArray{T}
