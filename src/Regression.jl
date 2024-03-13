"""Regression models i.e. GLM's, etc. This is not a regression package per se, but rather a collection of regression models that can be used for other models e.g. MarkovRegression.jl, etc."""
export GLM, Link, GaussianRegression, BinomialRegression, PoissonRegression, IdentityLink,
       LogLink, LogitLink, ProbitLink, InverseLink, Loss, LSELoss, CrossEntropyLoss, PoissonLoss, 
       loglikelihood, predict, residuals, fit!

# constants
EPSILON = 1e-15

# Abstract types
"""
GLM

Abstract type for GLM's, especially for SSM's.
"""
abstract type GLM end

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

# Functions for identity link with scalars
function link(::IdentityLink, μ::Float64)::Float64
    return μ
end

function invlink(::IdentityLink, η::Float64)::Float64
    return η
end

function derivlink(::IdentityLink, μ::Float64)::Float64
    return 1.0
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

# Gaussian Regression
"""
GaussianRegression

Struct representing a Gaussian regression model.
"""
mutable struct GaussianRegression{T <: Real} <: GLM
    X::Matrix{T}
    y::Vector{T}
    β::Vector{T}
    σ²::Float64
    link::Link
    loss::Loss
end

# Define a constructor that only requires X and y, and uses default values for β
function GaussianRegression(X::Matrix{T}, y::Vector{T}, constant::Bool=true, link::Link=IdentityLink(), loss::Loss=LSELoss()) where T <: Real
    n, p = size(X)
    # add constant if specified, i.e. β₀ or the intercept term
    if constant
        X = hcat(ones(T, n), X)
        p += 1
    end
    β = zeros(p)  # initialize as zeros
    # Initialize variance
    σ² = 1.0
    return GaussianRegression(X, y, β, σ², link, loss)
end

"""
PoissonRegression

Struct representing a Poisson regression model.
"""
mutable struct PoissonRegression{T <: Real} <: GLM
    X::Matrix{T}
    y::Vector{T}
    β::Vector{T}
    link::Link
    loss::Loss
end

# Define a constructor that only requires X and y, and uses default values for β
function PoissonRegression(X::Matrix{T}, y::Vector{T}, constant::Bool=true, link::Link=LogLink() ,loss::Loss=PoissonLoss()) where {T<:Real}
    n, p = size(X)
    # add constant if specified, i.e. β₀ or the intercept term
    if constant
        X = hcat(ones(T, n), X)
        p += 1
    end
    β = zeros(p)  # initialize as zeros
    return PoissonRegression(X, y, β, link, loss)
end

"""
BinomialRegression

Struct representing a Binomial regression model.
"""
mutable struct BinomialRegression{T <: Real} <: GLM
    X::Matrix{T}
    y::Vector{T}
    β::Vector{T}
    link::Link
    loss::Loss
end

# Define a constructor that only requires X and y, and uses default values for β
function BinomialRegression(X::Matrix{T}, y::Vector{T}, constant::Bool=true, link::Link=LogitLink(),loss::Loss=CrossEntropyLoss()) where T <: Real
    n, p = size(X)
    # add constant if specified, i.e. β₀ or the intercept term
    if constant
        X = hcat(ones(T, n), X)
        p += 1
    end
    β = zeros(p)  # initialize as zeros
    return BinomialRegression(X, y, β, link, loss)
end

# Predict function for any regression model where X is a matrix
function predict(model::GLM, X::AbstractMatrix)
    return invlink(model.link, vec(model.β' * X'))
end

# Predict function for any regression model where X is a vector
function predict(model::GLM, X::AbstractVector)
    return invlink(model.link, model.β' * X)
end

# Residuals function for any regression model
function residuals(model::GLM)
    return model.y - predict(model, model.X)
end

# Residuals For a specific point 
function residuals(model::GLM, X::AbstractVector, y::Real)
    return y - predict(model, X)
end

# dispatch loglikelihood
function loglikelihood(model::GaussianRegression{T}) where T <: Real
    return _loglikelihood(model, model.loss)
end

# dispatch loglikelihood single point
function loglikelihood(model::GaussianRegression{T}, X::AbstractVector, y::Real) where T <: Real
    return _loglikelihood(model, X, y, model.loss)
end

# loglikelihood function for a single point
function _loglikelihood(model::GaussianRegression, X::AbstractVector, y::Real, loss::LSELoss)
    residual = residuals(model, X, y)
    return -0.5 * log(2π) - 0.5 * log(model.σ²) - 0.5 * residual^2 / model.σ²
end

# loglikelihood function for a single point
function _loglikelihood(model::GaussianRegression, X::AbstractVector, y::Real, loss::WLSLoss)
    residual = residuals(model, X, y)
    return -0.5 * log(2π) - 0.5 * log(model.σ²) - 0.5 * residual^2 / model.σ²
end

# loglikelihood function with LSELoss
function _loglikelihood(model::GaussianRegression{T}, loss::LSELoss) where T <: Real
    n = size(model.X, 1)
    resid = residuals(model)
    LL = (-0.5 * n * log(2π)) - (0.5 * n * log(model.σ²)) - (0.5 * sum(resid.^2) / model.σ²)
    return LL
end

# log likelihood for regression with WLS loss function
function _loglikelihood(model::GaussianRegression{T}, loss::WLSLoss) where T <: Real
    n = size(model.X, 1)
    resid = residuals(model)
    LL = (-0.5 * n * log(2π)) - (0.5 * n * log(var(resid))) - (0.5 * sum(loss.weights .* resid.^2) / var(resid))
    return LL
end

# dispatch for variance update
function update_variance!(model::GaussianRegression)
    return _update_variance!(model, model.loss)
end

function _update_variance!(model::GaussianRegression, loss::WLSLoss)
    dof = size(model.X, 1) - size(model.X, 2)
    model.σ² = sum(loss.weights .* residuals(model).^2) / dof
end

# function to update σ² parameter of a gaussian regression model with lse loss
function _update_variance!(model::GaussianRegression, loss::LSELoss)
    dof = size(model.X, 1) - size(model.X, 2)
    model.σ² = sum(residuals(model).^2) / dof
end

# dispatch for regression model update
function update_model_params!(model::GLM)
    return _update_model_params!(model, model.loss)
end

# function to update regression model parameters for a gaussian regression model with lse loss
function _update_model_params!(model::GaussianRegression, loss::LSELoss)
    update_variance!(model)
end

# function to update regression model parameters for a gaussian regression model with wls loss
function _update_model_params!(model::GaussianRegression, loss::WLSLoss)
    update_variance!(model)
end

# Fit function for any regression model
function fit!(model::GLM, max_iter::Int=1000)
    function Objective(β)
        return compute_loss(model.loss, invlink(model.link, model.X * β), model.y)
    end
    result = optimize(Objective, model.β, LBFGS(), Optim.Options(iterations=max_iter))
    model.β = result.minimizer
    # update_model_params!(model)
end
