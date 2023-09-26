module SSM

using Random
using Statistics
using Distributions
using LinearAlgebra
using StatsBase
using LogExpFunctions

include("Emissions.jl")
include("MixtureModels.jl")
include("HiddenMarkovModels.jl")
include("KalmanFilters.jl")

end