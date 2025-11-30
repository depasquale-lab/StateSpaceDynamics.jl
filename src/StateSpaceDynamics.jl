module StateSpaceDynamics

using ArrayLayouts
using Distributions
using ForwardDiff
using LinearAlgebra
using LineSearches
using Optim
using ProgressMeter
using Random
using SparseArrays
using SpecialFunctions
using StaticArrays
using Statistics
using StatsBase
using StatsFuns

using Base.Threads: @threads
using Base: show

# Core types and utilities
include("core/GlobalTypes.jl")
include("core/Utilities.jl")

# Linear Dynamical Systems
include("models/lds/types.jl")
include("models/lds/gaussian.jl")
include("models/lds/poisson.jl")

# Other models
include("models/EmissionModels.jl")
include("models/HiddenMarkovModels.jl")
include("models/SLDS.jl")
include("models/MixtureModels.jl")

# Algorithms
include("algorithms/Preprocessing.jl")
include("algorithms/Valid.jl")

end
