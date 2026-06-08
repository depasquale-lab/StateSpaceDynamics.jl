module StateSpaceDynamics

import HiddenMarkovModels as HMMs

using Distributions
using LinearAlgebra
using PDMats
using Random
using SparseArrays

using Optim: Optim, optimize, LBFGS, HagerZhang
using ProgressMeter: Progress, next!, finish!
using SpecialFunctions: loggamma
using Statistics: mean
using StatsAPI: StatsAPI
import StatsAPI: loglikelihood

using Base.Threads: @threads, @spawn
using Base.Iterators: partition
using Base: show

# Core types and utilities
include("core/GlobalTypes.jl")
include("core/priors.jl")
include("models/lds/types.jl")
include("models/lds/show.jl")
include("core/Workspaces.jl")
include("core/Utilities.jl")
include("core/block_tridiagonal.jl")

# Include optimization utilities
include("optimization/linesearch.jl")
include("optimization/newton.jl")

# Linear Dynamical Systems
# kalman.jl / cov_update.jl are retained for the Kalman filter + marginal
# likelihood (and future particle-filter use); the Kalman path is no longer a
# selectable E-step backend for `fit!`.
include("models/lds/cov_update.jl")
include("models/lds/kalman.jl")
include("models/lds/common.jl")
include("models/lds/latents.jl")
include("models/lds/sufficient_statistics.jl")
include("models/lds/gaussian.jl")
include("models/lds/simulate.jl")
include("models/lds/poisson.jl")
include("models/lds/SLDS.jl")

# Algorithms
include("algorithms/Preprocessing.jl")
include("algorithms/Valid.jl")

# Errors/Exceptions/Validations
export validate_SLDS, validate_LDS, validate_probvec
export DimensionMismatchError, NotPositiveDefiniteError, NotSymmetricError
export InvalidProbabilityVectorError, NumericalStabilityError

# Models and Types
export ProbabilisticPCA, SLDS, LinearDynamicalSystem, Data
export AbstractStateModel, AbstractObservationModel
export GaussianStateModel, GaussianObservationModel, PoissonObservationModel
export IWPrior, MNPrior
export CovUpdateCache

# Utilities
export fit!, block_tridgm
export valid_Σ, gaussian_entropy
export random_rotation_matrix
export print_full
export info_update!

# Common functions
export rand, smooth, fit!, loglikelihood, joint_loglikelihood

end
