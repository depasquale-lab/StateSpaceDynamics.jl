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

# Conjugate priors — defined first because model structs reference IWPrior/MNPrior
# in their field type annotations.
include("stats/priors.jl")

# Model definitions + inference-state containers.
include("lds/types.jl")                             # abstract types, Data, model structs, SLDS
include("lds/workspaces.jl")                        # FilterSmooth / SufficientStatistics / workspaces
include("utils/show.jl")
include("utils/validation.jl")

# Shared latent inference machinery.
# kalman.jl is retained for the Kalman filter + marginal likelihood (and future
# particle-filter use); the Kalman path is no longer a selectable E-step backend.
include("stats/preprocessing.jl")           # PPCA (standalone model)
include("stats/kalman.jl")
include("stats/sufficient_statistics.jl")
include("stats/simulate.jl")

# latents models (LDS, PLDS, SLDS) + inference machinery (E-step).
include("lds/continuous_latents.jl")                # state-model Q-term + state M-step

# Observation models + composite / standalone models.
include("lds/gaussian_observations.jl")
include("lds/poisson_observations.jl")

# Fitting Functions
include("lds/fit_LDS.jl")
include("lds/fit_PLDS.jl")
include("lds/fit_SLDS.jl")

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
export rand, smooth, fit!, loglikelihood, elbo!

end
