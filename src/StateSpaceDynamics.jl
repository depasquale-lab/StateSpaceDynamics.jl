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
import StatsAPI: loglikelihood, fit!

using Base.Threads: @threads, @spawn
using Base.Iterators: partition
using Base: show

# Model-agnostic numerical kernels (no package types — reusable primitives).
include("numerics/linalg.jl")
include("numerics/optimization.jl")        # line search + Newton
include("numerics/block_tridiagonal.jl")   # BTD workspace + solver/inverse
include("numerics/cov_update.jl")          # info_update! + CovUpdateCache

# Model definitions + inference-state containers.
include("priors.jl")
include("types.jl")                         # abstract types, Data, model structs, SLDS
include("workspaces.jl")                    # FilterSmooth / SufficientStatistics / workspaces
include("show.jl")
include("validation.jl")

# Shared inference machinery.
# kalman.jl is retained for the Kalman filter + marginal likelihood (and future
# particle-filter use); the Kalman path is no longer a selectable E-step backend.
include("kalman.jl")
include("sufficient_statistics.jl")
include("dynamics.jl")                      # state-model Q-term + state M-step

# Observation models + composite / standalone models.
include("gaussian.jl")
include("poisson.jl")
include("simulate.jl")
include("slds.jl")
include("preprocessing.jl")                 # PPCA (standalone model)

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
export block_tridgm
export valid_Σ, gaussian_entropy
export random_rotation_matrix
export print_full
export info_update!

# Common functions
export rand, smooth, fit!, loglikelihood

end
