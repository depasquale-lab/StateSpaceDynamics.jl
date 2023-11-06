export RegressionModel, Link, GaussianRegression, BinomialRegression, PoissonRegression, IdentityLink,
       LogLink, LogitLink, ProbitLink, InverseLink, Loss, LSELoss, CrossEntropyLoss, PoissonLoss, fit!

EPSILON = 1e-15

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
Weighted Least Squares Loss

Struct representing the weighted least squares loss function. Mostly needed for Gaussian Markov regression EM.
"""
struct WLSLoss <: Loss
    weights::Vector{Float64}
end


# compute loss for WLS
function compute_loss(l::WLSLoss, y_pred::Vector{Float64}, y_true::Vector{Float64})::Float64
    return sum(l.weights .* (y_true .- y_pred).^2)
end


"""
    LSELoss

Struct representing the least squared error loss function i.e. OLS regression.
"""
struct LSELoss <: Loss end


# Compute loss for LSE
function compute_loss(::LSELoss, y_pred::Vector{Float64}, y_true::Vector{Float64})::Float64
    return sum((y_true .- y_pred).^2)
end

"""
    CrossEntropyLoss

Struct representing the cross-entropy loss function.
"""
struct CrossEntropyLoss <: Loss end

# Compute loss for CrossEntropy
function compute_loss(::CrossEntropyLoss, y_pred::Vector{Float64}, y_true::Vector{Float64})::Float64
    return -sum(y_true .* log.(y_pred) .+ (1 .- y_true) .* log.(1 .- y_pred))
end

"""
Poisson Loss

Struct representing the Poisson loss function.
"""

struct PoissonLoss <: Loss end

function compute_loss(::PoissonLoss, y_pred::Vector{Float64}, y_true::Vector{Float64})::Float64
    # Clamping y_pred because gettign extremely small negative values
    y_pred = clamp.(y_pred, EPSILON, Inf)
    return sum(y_pred .- (y_true .* log.(y_pred .+ EPSILON)))
end


# Link Functions
# Identity Link
"""
    IdentityLink

Struct representing the identity link function.
"""
struct IdentityLink <: Link end

# Functions for identity link with vectors
function link(::IdentityLink, μ::Vector{T})::Vector{T} where T <: Real
    return μ
end

function invlink(::IdentityLink, η::Vector{T})::Vector{T} where T <: Real
    return η
end

function derivlink(::IdentityLink, μ::Vector{T})::Vector{T} where T <: Real
    return ones(T, length(μ))
end

# Log Link
"""
    LogLink

Struct representing the log link function.
"""
struct LogLink <: Link end

# Functions for log link
function link(::LogLink, μ::Vector{T})::Vector{T} where T <: Real
    return log.(μ)
end

function invlink(::LogLink, η::Vector{T})::Vector{T} where T <: Real
    return exp.(η)
end

function derivlink(::LogLink, μ::Vector{T})::Vector{T} where T <: Real
    return inv.(μ)
end

"""
    LogitLink

Struct representing the logit link function.
"""
struct LogitLink <: Link end

# Functions for logit link
function link(::LogitLink, μ::Vector{T})::Vector{T} where T <: Real
    return log.(μ ./ (1 .- μ))
end

function invlink(::LogitLink, η::Vector{T})::Vector{T} where T <: Real
    return 1 ./ (1 .+ exp.(-η))
end

function derivlink(::LogitLink, μ::Vector{T})::Vector{T} where T <: Real
    return 1 ./ (μ .* (1 .- μ))
end


# More link functions can be defined similarly...

# Define regression models here

# Gaussian Regression
"""
GaussianRegression

Struct representing a Gaussian regression model.
"""
mutable struct GaussianRegression{T <: Real} <: RegressionModel
    X::Matrix{T}
    y::Vector{T}
    β::Vector{T}
    link::Link
end

# Define a constructor that only requires X and y, and uses default values for β
function GaussianRegression(X::Matrix{T}, y::Vector{T}, constant::Bool=true, link::Link=IdentityLink()) where T <: Real
    n, p = size(X)
    # add constant if specified, i.e. β₀ or the intercept term
    if constant
        X = hcat(ones(T, n), X)
        p += 1
    end
    β = zeros(p)  # initialize as zeros
    return GaussianRegression(X, y, β, link)
end

# loglikelihood function
function loglikelihood(model::GaussianRegression, β::Vector{T}) where T <: Real
    
end

# Poisson regression
"""
PoissonRegression

Struct representing a Poisson regression model.
"""

mutable struct PoissonRegression{T <: Real} <: RegressionModel
    X::Matrix{T}
    y::Vector{T}
    β::Vector{T}
    link::Link
end

# Define a constructor that only requires X and y, and uses default values for β
function PoissonRegression(X::Matrix{T}, y::Vector{T}, constant::Bool=true, link::Link=LogLink()) where {T<:Real}
    n, p = size(X)
    # add constant if specified, i.e. β₀ or the intercept term
    if constant
        X = hcat(ones(T, n), X)
        p += 1
    end
    β = zeros(p)  # initialize as zeros
    return PoissonRegression(X, y, β, link)
end

# logistic regression
"""
LogisticRegression

Struct representing a Logistic regression model.
"""
mutable struct BinomialRegression{T <: Real} <: RegressionModel
    X::Matrix{T}
    y::Vector{T}
    β::Vector{T}
    link::Link
end

# Define a constructor that only requires X and y, and uses default values for β
function BinomialRegression(X::Matrix{T}, y::Vector{T}, constant::Bool=true, link::Link=LogitLink()) where T <: Real
    n, p = size(X)
    # add constant if specified, i.e. β₀ or the intercept term
    if constant
        X = hcat(ones(T, n), X)
        p += 1
    end
    β = zeros(p)  # initialize as zeros
    return BinomialRegression(X, y, β, link)
end


# Fit function for any regression model
function fit!(model::RegressionModel, loss::Union{Loss, Nothing}=nothing, max_iter::Int=1000)
    # Auto-select loss if not provided
    if isnothing(loss)
        if model isa GaussianRegression
            loss = LSELoss()
        elseif model isa PoissonRegression
            loss = PoissonLoss()
        elseif model isa BinomialRegression
            loss = CrossEntropyLoss()
        else
            throw(ArgumentError("Automatic loss selection is not available for this model type"))
        end
    end
    function Objective(β)
        return compute_loss(loss, invlink(model.link, model.X * β), model.y)
    end
    result = optimize(Objective, model.β, LBFGS(), Optim.Options(iterations=max_iter))
    model.β = result.minimizer
end

