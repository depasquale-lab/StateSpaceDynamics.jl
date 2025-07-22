module SSD_Benchmark

# Include packages
using StateSpaceDynamics
using HiddenMarkovModels
using BenchmarkTools
using CSV
using CondaPkg
using DataFrames
using Distributions
using LinearAlgebra
using Optim
using Plots
using PythonCall
using StableRNGs
using StatsAPI
using Random
using Base.Threads: @threads

# Define type for organizing our implementations
abstract type Implementation end

export Implementation

# Include files
include("Instances.jl")
include("Params.jl")
include("hmm_benchmark_tools.jl")
include("lds_benchmark_tools.jl")

end