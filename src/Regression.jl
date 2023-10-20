export RegressionModel, Link, GaussianRegression, IdentityLink,
       LogLink, LogitLink, ProbitLink, InverseLink, Loss, MSELoss, CrossEntropyLoss

# Abstract types
"""
    RegressionModel

Abstract type for GLM's, especially for SSM's.
"""
abstract type RegressionModel end

"""
    Link

Abstract type for Link functions.
"""
abstract type Link end

"""
    Loss

Abstract type for Loss functions.
"""
abstract type Loss end

# Loss Functions
"""
    MSELoss

Struct representing the mean squared error loss function.
"""
struct MSELoss <: Loss end

# Compute loss for MSE
function compute_loss(::MSELoss, y_pred::Vector{Float64}, y_true::Vector{Float64})::Float64
    return sum((y_pred .- y_true).^2) / length(y_true)
end

"""
    CrossEntropyLoss

Struct representing the cross-entropy loss function.
"""
struct CrossEntropyLoss <: Loss end

# Compute loss for CrossEntropy
function compute_loss(::CrossEntropyLoss, y_pred::Vector{Float64}, y_true::Vector{Float64})::Float64
    return -sum(y_true .* log.(y_pred) .+ (1 .- y_true) .* log.(1 .- y_pred)) / length(y_true)
end

# Link Functions
# Identity Link
"""
    IdentityLink

Struct representing the identity link function.
"""
struct IdentityLink <: Link end

# Functions for identity link
link(::IdentityLink, μ::Real)::Real = μ
invlink(::IdentityLink, η::Real)::Real = η
derivlink(::IdentityLink, μ::Real)::Real = one(μ)

# Log Link
"""
    LogLink

Struct representing the log link function.
"""
struct LogLink <: Link end

# Functions for log link
link(::LogLink, μ::Real)::Real = log(μ)
invlink(::LogLink, η::Real)::Real = exp(η)
derivlink(::LogLink, μ::Real)::Real = inv(μ)

# More link functions can be defined similarly...

# Define regression models here

# Gaussian Regression
"""
    GaussianRegression

Struct representing a Gaussian regression model.
"""
struct GaussianRegression <: RegressionModel
    X::Matrix{Real}
    y::Vector{Real}
    β::Vector{Real}
    link::Link
end

# Define a constructor that allows all fields to be set
function GaussianRegression(X::Matrix{Real}, y::Vector{Real}, β::Vector{Real}, link::Link)
    return GaussianRegression(X, y, β, link)
end

# Define a constructor that only requires X and y, and uses default values for β
function GaussianRegression(X::Matrix{Real}, y::Vector{Real}, link::Link=IdentityLink())
    n, p = size(X)
    β = zeros(p)  # initialize as zeros
    return GaussianRegression(X, y, β, link)
end

# Function to optimize the model
function fit!(model::GaussianRegression, loss::Loss=MSELoss(), max_iter::Int=1000)
    
end


