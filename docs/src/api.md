# API Reference

This page collects the public API of StateSpaceDynamics.jl in one place. The model
pages ([Linear Dynamical Systems](LinearDynamicalSystems.md) and
[Switching Linear Dynamical Systems](SLDS.md)) intersperse the same docstrings with
theory and usage notes.

```@index
Pages = ["api.md"]
```

## Models

```@docs; canonical = false
LinearDynamicalSystem
SLDS
GaussianStateModel
GaussianObservationModel
PoissonObservationModel
```

```@docs
ProbabilisticPCA
Data
AbstractStateModel
AbstractObservationModel
```

## Priors

```@docs; canonical = false
IWPrior
```

```@docs
MNPrior
```

```@docs
x0_mean_prior
```

## Sampling

```@docs; canonical = false
Random.rand(rng::AbstractRNG, lds::LinearDynamicalSystem{T,S,O}, tsteps::Integer) where {T<:Real,S<:GaussianStateModel{T},O<:AbstractObservationModel{T}}
Random.rand(rng::AbstractRNG, slds::SLDS{T,S,O}, tsteps::Integer) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
```

```@docs
Random.rand(rng::AbstractRNG, ppca::ProbabilisticPCA, n::Int)
```

## Smoothing and fitting

```@docs; canonical = false
smooth
fit!(lds::LinearDynamicalSystem{T,S,O}, y::AbstractVector{<:AbstractMatrix{T}}; max_iter::Int=100, tol::Float64=1e-6, progress::Bool=true) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
fit!(slds::SLDS{T,S,O}, y::AbstractVector{<:AbstractMatrix{T}}; max_iter::Int=50, progress::Bool=true) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
```

```@docs
fit!(plds::LinearDynamicalSystem{T,S,O}, y::AbstractVector{<:AbstractMatrix{T}}) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
fit!(ppca::ProbabilisticPCA, X::AbstractMatrix{T}, max_iters::Int=100, tol::Float64=1e-6) where {T<:Real}
```

## Likelihoods and ELBO

```@docs
loglikelihood(lds::LinearDynamicalSystem{T,SM,OM}, y::AbstractVector{<:AbstractMatrix{T}}) where {T<:Real,SM<:GaussianStateModel{T},OM<:GaussianObservationModel{T}}
loglikelihood(plds::LinearDynamicalSystem{T,S,O}, y) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
loglikelihood(ppca::ProbabilisticPCA, X::AbstractMatrix{T}) where {T<:Real}
elbo!(lds::LinearDynamicalSystem{T,S,O}, suf::StateSpaceDynamics.SufficientStatistics{T}, sws::StateSpaceDynamics.SmoothWorkspace{T}, total_entropy::T) where {T<:Real,S<:GaussianStateModel{T},O<:GaussianObservationModel{T}}
elbo!(plds::LinearDynamicalSystem{T,S,O}, suf::StateSpaceDynamics.SufficientStatistics{T}, tfs::StateSpaceDynamics.TrialFilterSmooth{T}, y::AbstractVector{<:AbstractMatrix{T}}, sws_pool::Vector{StateSpaceDynamics.SmoothWorkspace{T}}) where {T<:Real,S<:GaussianStateModel{T},O<:PoissonObservationModel{T}}
elbo!(slds::SLDS{T,S,O}, tfs::StateSpaceDynamics.TrialFilterSmooth{T}, fb_storage::StateSpaceDynamics.HMMs.ForwardBackwardStorage, y::AbstractVector{<:AbstractMatrix{T}}, slds_ws::StateSpaceDynamics.SLDSSmoothWorkspace{T}) where {T<:Real,S<:AbstractStateModel,O<:AbstractObservationModel}
```

## Validation

```@docs
validate_LDS
validate_SLDS
validate_probvec
DimensionMismatchError
NotPositiveDefiniteError
NotSymmetricError
InvalidProbabilityVectorError
NumericalStabilityError
```

## Utilities

```@docs
random_rotation_matrix
gaussian_entropy
valid_Σ
block_tridgm
print_full
info_update!
CovUpdateCache
```

## Internals

Documented internal methods, listed here for completeness. These are not part
of the public API and may change between releases.

```@docs
fit!(dl::StateSpaceDynamics.SLDSDiscreteLayer{T}, fb_storage::StateSpaceDynamics.HMMs.ForwardBackwardStorage, obs_seq::AbstractVector) where {T<:Real}
```
