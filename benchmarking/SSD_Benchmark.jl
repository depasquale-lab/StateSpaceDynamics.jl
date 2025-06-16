module SSD_Benchmark

# Include packages
using StateSpaceDynamics
using HiddenMarkovModels
using BenchmarkTools
using CSV
using CondaPkg
using DataFrames
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

# Include files
include("Instances.jl")
include("Params.jl")
include("Controlled_BernoulliHMM.jl")
include("glmhmm_benchmarking_tools.jl")

end