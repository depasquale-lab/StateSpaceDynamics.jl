"""
EmissionModels.jl

This module implements various emission models for state space modeling, including:
- Gaussian emissions
- Regression-based emissions (Gaussian, Bernoulli, Poisson)
- Composite emissions
"""

# Exports
export EmissionModel, RegressionEmission
export GaussianEmission, GaussianRegressionEmission, BernoulliRegressionEmission, PoissonRegressionEmission, AutoRegressionEmission
export CompositeModelEmission
export sample, loglikelihood, fit!

#=
Gaussian Emission Models
=#

"""
    mutable struct GaussianEmission <: EmissionModel

GaussianEmission model with mean and covariance.
"""
mutable struct GaussianEmission <: EmissionModel
    output_dim::Int # dimension of the data
    μ::Vector{<:Real}  # mean 
    Σ::Matrix{<:Real}  # covariance matrix
end

"""
    function GaussianEmission(; output_dim::Int, μ::Vector{<:Real}=zeros(output_dim), Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim))

Functon to create a GaussianEmission with given output dimension, mean, and covariance.

# Arguments
- `output_dim::Int`: The output dimension of the emission
- `μ::Vector{<:Real}=zeros(output_dim)`: The mean of the Gaussian
- `Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim))`: The covariance matrix of the Gaussian

# Returns
"""
function GaussianEmission(;
    output_dim::Int,
    μ::Vector{<:Real}=zeros(output_dim),
    Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim),
)
    return GaussianEmission(output_dim, μ, Σ)
end

"""
    loglikelihood(model::GaussianEmission, Y::Matrix{<:Real})

Calculate the log likelihood of the data `Y` given the Gaussian emission model.

# Arguments
- `model::GaussianEmission`: The Gaussian emission model for which to calculate the log likelihood.
- `Y::Matrix{<:Real}`: The data matrix, where each row represents an observation.

# Returns
- `Vector{Float64}`: A vector of log likelihoods, one for each observation in the data.
"""
function loglikelihood(model::GaussianEmission, Y::Matrix{<:Real})
    # Create MvNormal distribution with the model parameters
    dist = MvNormal(model.μ, model.Σ)

    # Calculate log likelihood for each observation
    return [logpdf(dist, Y[i, :]) for i in axes(Y, 1)]
end

"""
    sample(model::Gaussian; n::Int=1)

Generate `n` samples from a Gaussian model. Returns a matrix of size `(n, output_dim)`.
"""
function sample(model::GaussianEmission; n::Int=1)
    raw_samples = rand(MvNormal(model.μ, model.Σ), n)
    return Matrix(raw_samples')
end

"""
    fit!(model::GaussianEmission, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

Fit a GaussianEmission model to the data `Y`. 

# Arguments
- `model::GaussianEmission`: Gaussian model to fit.
- `Y::Matrix{<:Real}`: Data to fit the model to. Should be a matrix of size `(n, output_dim)`.
- `w::Vector{Float64}=ones(size(Y, 1))`: Weights for the data. Should be a vector of size `n`.
"""
function fit!(
    model::GaussianEmission, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1))
)
    weighted_sum = sum(Y .* w; dims=1)
    new_mean = weighted_sum[:] ./ sum(w)

    centered_data = Y .- new_mean'
    weighted_centered = centered_data .* sqrt.(w)
    new_covariance = (weighted_centered' * weighted_centered) ./ sum(w)
    new_covariance = stabilize_covariance_matrix(new_covariance)

    model.μ = new_mean
    model.Σ = new_covariance
    return model
end

#=
Regression Emission Models
=#
"""
    RegressionOptimization{T<:RegressionEmission}

Holds the optimization problem data for regression emissions.
"""
struct RegressionOptimization{T<:RegressionEmission}
    model::T
    X::Matrix{<:Real}
    y::Matrix{<:Real}
    w::Vector{Float64}
    β_shape::Tuple{Int,Int}  # Added to track original shape
end

# Unified interface for creating optimization problems
function create_optimization(
    model::RegressionEmission,
    X::Matrix{<:Real},
    y::Matrix{<:Real},
    w::Vector{Float64}=ones(size(y, 1)),
)
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end

    β_shape = size(model.β)
    return RegressionOptimization(model, X, y, w, β_shape)
end

# Helper functions for reshaping
vec_to_matrix(β_vec::Vector{<:Real}, shape::Tuple{Int,Int}) = reshape(β_vec, shape)
matrix_to_vec(β_mat::Matrix{<:Real}) = vec(β_mat)

# Default no-op post-optimization
post_optimization!(model::RegressionEmission, opt::RegressionOptimization) = nothing

"""
    calc_regularization(β::Matrix{<:Real}, λ::Float64, include_intercept::Bool)

Calculate L2 regularization term for regression coefficients.

# Arguments
- `β::Matrix{<:Real}`: Coefficient matrix
- `λ::Float64`: Regularization parameter
- `include_intercept::Bool`: Whether to exclude the intercept term from regularization

# Returns
- `Float64`: The regularization term value
"""
function calc_regularization(β::Matrix{<:Real}, λ::Float64, include_intercept::Bool=true)
    # calculate L2 penalty
    if include_intercept
        regularization = 0.5 * λ * sum(abs2, β[2:end, :])
    else
        regularization = 0.5 * λ * sum(abs2, β)
    end

    return regularization
end

"""
    calc_regularization_gradient(β::Matrix{<:Real}, λ::Float64, include_intercept::Bool)

Calculate gradient of L2 regularization term for regression coefficients.

# Arguments
- `β::Matrix{<:Real}`: Coefficient matrix
- `λ::Float64`: Regularization parameter
- `include_intercept::Bool`: Whether to exclude the intercept term from regularization
"""
function calc_regularization_gradient(
    β::Matrix{<:Real}, λ::Float64, include_intercept::Bool=true
)
    # calculate the gradient of the regularization component
    regularization = zeros(size(β))

    if include_intercept
        regularization[2:end, :] .= λ * β[2:end, :]
    else
        regularization .= λ * β
    end

    return regularization
end
#=
Gaussian Regression Functions
=#

"""
    GaussianRegressionEmission

A Gaussian regression Emission model.

# Fields
- `input_dim::Int`: Dimension of the input data.
- `output_dim::Int`: Dimension of the output data.
- `include_intercept::Bool = true`: Whether to include an intercept term; if true, the first column of β is assumed to be the intercept/bias.
- `β::Matrix{<:Real} = if include_intercept zeros(input_dim + 1, output_dim) else zeros(input_dim, output_dim) end`: Coefficient matrix of the model. Shape input_dim by output_dim. The first row are the intercept terms, if included.
- `Σ::Matrix{<:Real} = Matrix{Float64}(I, output_dim, output_dim)`: Covariance matrix of the model.
- `λ::Float64 = 0.0`: Regularization parameter.
"""
mutable struct GaussianRegressionEmission <: RegressionEmission
    input_dim::Int
    output_dim::Int
    β::Matrix{<:Real} # coefficient matrix of the model. Shape input_dim by output_dim. Column one is coefficients for target one, etc. The first row are the intercept terms, if included. 
    Σ::Matrix{<:Real} # covariance matrix of the model 
    include_intercept::Bool # whether to include an intercept term; if true, the first column of β is assumed to be the intercept/bias
    λ::Float64 # regularization parameter
end

function GaussianRegressionEmission(;
    input_dim::Int,
    output_dim::Int,
    include_intercept::Bool=true,
    β::Matrix{<:Real}=if include_intercept
        zeros(input_dim + 1, output_dim)
    else
        zeros(input_dim, output_dim)
    end,
    Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim),
    λ::Float64=0.0,
)
    return GaussianRegressionEmission(input_dim, output_dim, β, Σ, include_intercept, λ)
end

"""
    sample(model::GaussianRegressionEmission, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))

Generate `n` samples from a Gaussian regression model. Returns a matrix of size `(n, output_dim)`.

# Arguments
- `model::GaussianRegressionEmission`: Gaussian regression model.
- `Φ::Matrix{<:Real}`: Design matrix of shape `(n, input_dim)`.
- `n::Int=size(Φ, 1)`: Number of samples to generate.

# Returns
- `Y::Matrix{<:Real}`: Matrix of samples of shape `(n, output_dim)`.
"""
function sample(model::GaussianRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})
    # Ensure Φ is a 2D matrix even if it's a single sample
    Φ = size(Φ, 2) == 1 ? reshape(Φ, 1, :) : Φ

    # Add intercept column if specified
    if model.include_intercept
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    # Ensure the noise dimensions match the output dimension and sample size
    noise = rand(MvNormal(zeros(model.output_dim), model.Σ), size(Φ, 1))'
    return Φ * model.β + noise
end

"""
    loglikelihood(model::GaussianRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real})

Calculate the log likelihood of the data `Y` given the Gaussian regression emission model and the input features `Φ`.

# Arguments
- `model::GaussianRegressionEmission`: The Gaussian regression emission model for which to calculate the log likelihood.
- `Φ::Matrix{<:Real}`: The input features matrix (Observations x Features).
- `Y::Matrix{<:Real}`: The data matrix (Observations x Features).

# Returns
- `Vector{Float64}`: A vector of log likelihoods, one for each observation in the data.
"""
function loglikelihood(
    model::GaussianRegressionEmission,
    Φ::Matrix{<:Real},
    Y::Matrix{<:Real},
    w::Vector{Float64}=ones(size(Y, 1)),
)
    # Add intercept if specified
    Φ = model.include_intercept ? [ones(size(Φ, 1)) Φ] : Φ

    # residuals
    residuals = Y - Φ * model.β

    # Calculate weighted least squares
    weighted_residuals = residuals .^ 2 .* w

    return -0.5 .* weighted_residuals
end

function objective(
    opt::RegressionOptimization{GaussianRegressionEmission}, β_vec::Vector{T}
) where {T<:Real}
    β_mat = vec_to_matrix(β_vec, opt.β_shape)
    residuals = opt.y - opt.X * β_mat
    w_reshaped = reshape(opt.w, :, 1)

    # calculate regularization
    regularization = calc_regularization(β_mat, opt.model.λ, opt.model.include_intercept)

    # calculate pseudo log-likelihood
    pseudo_ll = 0.5 * sum(w_reshaped .* residuals .^ 2) + regularization
    return pseudo_ll
end

function objective_gradient!(
    G::Vector{Float64},
    opt::RegressionOptimization{GaussianRegressionEmission},
    β_vec::Vector{T},
) where {T<:Real}
    β_mat = vec_to_matrix(β_vec, opt.β_shape)
    residuals = opt.y - opt.X * β_mat

    # calc gradient of penalty
    regularization = calc_regularization_gradient(
        β_mat, opt.model.λ, opt.model.include_intercept
    )

    # calculate the gradient
    grad_mat = -(opt.X' * (Diagonal(opt.w) * residuals)) + regularization  # Fixed: Added negative sign
    return G .= vec(grad_mat)
end

# Special handling for Gaussian regression to update variance
function post_optimization!(model::GaussianRegressionEmission, opt::RegressionOptimization)
    residuals = opt.y - opt.X * model.β
    Σ = (residuals' * Diagonal(opt.w) * residuals) / size(opt.X, 1)
    return model.Σ = 0.5 * (Σ + Σ')  # Ensure symmetry
end

"""
    BernoulliRegressionEmission

A Bernoulli regression model.

# Fields
- `input_dim::Int`: Dimension of the input data.
- `include_intercept::Bool = true`: Whether to include an intercept term.
- `β::Vector{<:Real} = if include_intercept zeros(input_dim + 1) else zeros(input_dim) end`: Coefficients of the model. The first element is the intercept term, if included.
- `λ::Float64 = 0.0`: Regularization parameter.
```
"""
mutable struct BernoulliRegressionEmission <: RegressionEmission
    input_dim::Int
    output_dim::Int
    β::Matrix{<:Real} 
    include_intercept::Bool # whether to include an intercept term; if true, the first column of β is assumed to be the intercept/bias
    λ::Float64 # regularization parameter
end

function BernoulliRegressionEmission(;
    input_dim::Int,
    output_dim::Int,
    include_intercept::Bool=true,
    β::Matrix{<:Real}=if include_intercept
        zeros(input_dim + 1, output_dim)
    else
        zeros(input_dim, output_dim)
    end,
    λ::Float64=0.0,
)
    return BernoulliRegressionEmission(input_dim, output_dim, β, include_intercept, λ)
end

"""
    sample(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))

Generate `n` samples from a Bernoulli regression model. Returns a matrix of size `(n, 1)`.

# Arguments
- `model::BernoulliRegressionEmission`: Bernoulli regression model.
- `Φ::Matrix{<:Real}`: Design matrix of shape `(n, input_dim)`.
- `n::Int=size(Φ, 1)`: Number of samples to generate.

# Returns
- `Y::Matrix{<:Real}`: Matrix of samples of shape `(n, 1)`.
"""
function sample(model::BernoulliRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})
    # Ensure Φ is a 2D matrix even if it's a single sample
    Φ = size(Φ, 2) == 1 ? reshape(Φ, 1, :) : Φ

    # add intercept if specified
    if model.include_intercept && size(Φ, 2) == length(model.β) - 1
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    Y = rand.(Bernoulli.(logistic.(Φ * model.β)))

    return float.(reshape(Y, :, 1))
end

"""
    loglikelihood(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

Calculate the log likelihood of the data `Y` given the Bernoulli regression emission model and the input features `Φ`. Optionally, a vector of weights `w` can be provided.

# Arguments
- `model::BernoulliRegressionEmission`: The Bernoulli regression emission model for which to calculate the log likelihood.
- `Φ::Matrix{<:Real}`: The input features matrix (Observations x Features).
- `Y::Matrix{<:Real}`: The data matrix (Observations x Features).
- `w::Vector{Float64}`: A vector of weights corresponding to each observation (defaults to a vector of ones).

# Returns
- `Vector{Float64}`: A vector of log likelihoods, one for each observation in the data.
"""
function loglikelihood(
    model::BernoulliRegressionEmission,
    Φ::Matrix{<:Real},
    Y::Matrix{<:Real},
    w::Vector{Float64}=ones(size(Y, 1)),
)
    # add intercept if specified and not already included
    if model.include_intercept && size(Φ, 2) == length(model.β) - 1
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    # calculate log likelihood
    p = logistic.(Φ * model.β)

    obs_wise_loglikelihood = w .* (Y .* log.(p) .+ (1 .- Y) .* log.(1 .- p))

    # sum across independent feature log likelihoods if mulitple features
    if size(obs_wise_loglikelihood, 2) > 1
        obs_wise_loglikelihood = sum(obs_wise_loglikelihood, dims=2)
    end

    return obs_wise_loglikelihood
end

# Bernoulli Regression Implementation
function objective(
    opt::RegressionOptimization{BernoulliRegressionEmission}, β_vec::Vector{T}
) where {T<:Real}
    β_mat = vec_to_matrix(β_vec, opt.β_shape)
    p = logistic.(opt.X * β_mat)

    # calculate regularization
    regularization = calc_regularization(β_mat, opt.model.λ, opt.model.include_intercept)

    val = -sum(opt.w .* (opt.y .* log.(p) .+ (1 .- opt.y) .* log.(1 .- p))) + regularization

    return val
end

function objective_gradient!(
    G::Vector{Float64},
    opt::RegressionOptimization{BernoulliRegressionEmission},
    β_vec::Vector{T},
) where {T<:Real}
    β_mat = vec_to_matrix(β_vec, opt.β_shape)
    p = logistic.(opt.X * β_mat)

    # calc gradient of penalty
    regularization = calc_regularization_gradient(
        β_mat, opt.model.λ, opt.model.include_intercept
    )

    grad_mat = -(opt.X' * (opt.w .* (opt.y .- p))) + regularization
    return G .= vec(grad_mat)
end

"""
    PoissonRegressionEmission

A Poisson regression model.

# Fields
- `input_dim::Int`: Dimension of the input data.
- `include_intercept::Bool = true`: Whether to include an intercept term.
- `β::Vector{<:Real} = if include_intercept zeros(input_dim + 1) else zeros(input_dim) end`: Coefficients of the model. The first element is the intercept term, if included.
- `λ::Float64 = 0.0`: Regularization parameter.
"""
mutable struct PoissonRegressionEmission <: RegressionEmission
    input_dim::Int
    output_dim::Int
    β::Matrix{<:Real}
    include_intercept::Bool
    λ::Float64
end

function PoissonRegressionEmission(;
    input_dim::Int,
    output_dim::Int,
    include_intercept::Bool=true,
    β::Matrix{<:Real}=if include_intercept
        zeros(input_dim + 1, output_dim)
    else
        zeros(input_dim, output_dim)
    end,
    λ::Float64=0.0,
)
    return PoissonRegressionEmission(input_dim, output_dim, β, include_intercept, λ)
end

"""
    sample(model::PoissonRegressionEmission, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))

Generate `n` samples from a Poisson regression model. Returns a matrix of size `(n, 1)`.

# Arguments
- `model::PoissonRegressionEmission`: Poisson regression model.
- `Φ::Matrix{<:Real}`: Design matrix of shape `(n, input_dim)`.
- `n::Int=size(Φ, 1)`: Number of samples to generate.

# Returns
- `Y::Matrix{<:Real}`: Matrix of samples of shape `(n, 1)`.
"""
function sample(model::PoissonRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})
    # Ensure Φ is a 2D matrix even if it's a single sample
    Φ = size(Φ, 2) == 1 ? reshape(Φ, 1, :) : Φ

    # add intercept if specified
    if model.include_intercept && size(Φ, 2) == length(model.β) - 1
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    Y = rand.(Poisson.(exp.(Φ * model.β)))

    # convert Y 
    Y = reshape(Y, :, 1)
    Y = Float64.(Y)

    return Y
end

"""
    loglikelihood(model::PoissonRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

Calculate the log-likelihood of a Poisson regression model.

# Arguments
- `model::PoissonRegressionEmission`: Poisson regression model.
- `Φ::Matrix{<:Real}`: Design matrix of shape `(n, input_dim)`.
- `Y::Matrix{<:Real}`: Response matrix of shape `(n, 1)`.
- `w::Vector{Float64}`: Weights of the data points. Should be a vector of size `n`.

# Returns
- `loglikelihood::Float64`: Log-likelihood of the model.
"""
function loglikelihood(
    model::PoissonRegressionEmission,
    Φ::Matrix{<:Real},
    Y::Matrix{<:Real},
    w::Vector{Float64}=ones(size(Y, 1)),
)
    # add intercept if specified
    if model.include_intercept && size(Φ, 2) == length(model.β) - 1
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    # calculate log likelihood
    η = clamp.(Φ * model.β, -30, 30)
    rate = exp.(η)

    obs_wise_loglikelihood = w .* (Y .* log.(rate) .- rate .- loggamma.(Int.(Y) .+ 1))

    # sum across independent feature log likelihoods if mulitple features
    if size(obs_wise_loglikelihood, 2) > 1
        obs_wise_loglikelihood = sum(obs_wise_loglikelihood, dims=2)
    end

    return obs_wise_loglikelihood
end

# Poisson Regression Implementation
function objective(
    opt::RegressionOptimization{PoissonRegressionEmission}, β_vec::Vector{T}
) where {T<:Real}
    β_mat = vec_to_matrix(β_vec, opt.β_shape)

    η = clamp.(opt.X * β_mat, -30, 30)
    rate = exp.(η)

    # calculate regularization
    regularization = calc_regularization(β_mat, opt.model.λ, opt.model.include_intercept)

    val =
        -sum(opt.w .* (opt.y .* log.(rate) .- rate .- loggamma.(Int.(opt.y) .+ 1))) +
        regularization
    return val
end

function objective_gradient!(
    G::Vector{Float64},
    opt::RegressionOptimization{PoissonRegressionEmission},
    β_vec::Vector{T},
) where {T<:Real}
    β_mat = vec_to_matrix(β_vec, opt.β_shape)
    η = clamp.(opt.X * β_mat, -30, 30)
    rate = exp.(η)

    # calc gradient of penalty
    regularization = calc_regularization_gradient(
        β_mat, opt.model.λ, opt.model.include_intercept
    )

    grad_mat = (-opt.X' * (Diagonal(opt.w) * (opt.y .- rate))) + regularization
    return G .= vec(grad_mat)
end

# Unified fit! function for all regression emissions
function fit!(
    model::RegressionEmission,
    X::Matrix{<:Real},
    y::Matrix{<:Real},
    w::Vector{Float64}=ones(size(y, 1)),
)
    opt_problem = create_optimization(model, X, y, w)

    # Create closure functions for Optim.jl
    f(β) = objective(opt_problem, β)
    g!(G, β) = objective_gradient!(G, opt_problem, β)

    opts = Optim.Options(;
        x_abstol=1e-8,
        x_reltol=1e-8,
        f_abstol=1e-8,
        f_reltol=1e-8,
        g_abstol=1e-8,
        g_reltol=1e-8,
    )

    # Run optimization
    result = optimize(f, g!, vec(model.β), LBFGS(), opts)

    # Update model parameters
    model.β = vec_to_matrix(result.minimizer, opt_problem.β_shape)

    # Update additional parameters if needed (e.g., variance for Gaussian)
    post_optimization!(model, opt_problem)

    return model
end

# pass in inner model and original data
# reshape the data into gaussian regression data

"""
    AutoRegressionEmission <: EmissionModel

A mutable struct representing an autoregressive emission model, which wraps around an `AutoRegression` model.

# Fields
- `inner_model::AutoRegression`: The underlying autoregressive model used for the emissions.
"""
mutable struct AutoRegressionEmission <: AutoRegressiveEmission
    output_dim::Int
    order::Int
    innerGaussianRegression::GaussianRegressionEmission
end

# move these to utils at some point
# define getters for innerGaussianRegression fields
function Base.getproperty(model::AutoRegressiveEmission, sym::Symbol)
    if sym === :β
        return model.innerGaussianRegression.β
    elseif sym === :Σ
        return model.innerGaussianRegression.Σ
    elseif sym === :include_intercept
        return model.innerGaussianRegression.include_intercept
    elseif sym === :λ
        return model.innerGaussianRegression.λ
    else # fallback to getfield
        return getfield(model, sym)
    end
end

# define setters for innerGaussianRegression fields
function Base.setproperty!(model::AutoRegressiveEmission, sym::Symbol, value)
    if sym === :β
        model.innerGaussianRegression.β = value
    elseif sym === :Σ
        model.innerGaussianRegression.Σ = value
    elseif sym === :λ
        model.innerGaussianRegression.λ = value
    else # fallback to setfield!
        setfield!(model, sym, value)
    end
end


function AutoRegressionEmission(; 
    output_dim::Int, 
    order::Int, 
    include_intercept::Bool = true, 
    β::Matrix{<:Real} = if include_intercept zeros(output_dim * order + 1, output_dim) else zeros(output_dim * order, output_dim) end,
    Σ::Matrix{<:Real} = Matrix{Float64}(I, output_dim, output_dim),
    λ::Float64=0.0)

    innerGaussianRegression = GaussianRegressionEmission(
        input_dim=output_dim * order, 
        output_dim=output_dim, 
        β=β,
        Σ=Σ,
        include_intercept=include_intercept, 
        λ=λ)

    model = AutoRegressionEmission(output_dim, order, innerGaussianRegression)


    return model
end


function AR_to_Gaussian_data(Y_prev::Matrix{<:Real})
    # take each row of Y_prev and stack them horizontally to form the input row matrix Φ_gaussian
    Φ_gaussian = vcat([Y_prev[i, :] for i in 1:size(Y_prev, 1)]...)
    Φ_gaussian = reshape(Φ_gaussian, 1, :)

    return Φ_gaussian
end

function AR_to_Gaussian_data(Y_prev::Matrix{<:Real}, Y::Matrix{<:Real})
    order = size(Y_prev, 1)
    output_dim = size(Y_prev, 2)
    Φ_gaussian = zeros(size(Y, 1), output_dim * order)

    for i in 1:size(Y, 1)
        Φ_gaussian[i, :] = AR_to_Gaussian_data(Y_prev)


        old_part = Y_prev[2:end, :]
        new_part = Y[i, :]

        old_part = reshape(old_part, order - 1, output_dim)
        new_part = reshape(new_part, 1, output_dim)

        Y_prev = vcat(old_part, new_part)
    end
   

    return Φ_gaussian
end

"""
    sample(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}; observation_sequence::Matrix{<:Real}=Matrix{Float64}(undef, 0, model.output_dim))

Generate a sample from the given autoregressive emission model using the previous observations `Y_prev`, and append it to the provided observation sequence.

# Arguments
- `model::AutoRegressionEmission`: The autoregressive emission model to sample from.
- `Y_prev::Matrix{<:Real}`: The matrix of previous observations, where each row represents an observation.
- `observation_sequence::Matrix{<:Real}`: The sequence of observations to which the new sample will be appended (defaults to an empty matrix with appropriate dimensions).

# Returns
- `Matrix{Float64}`: The updated observation sequence with the new sample appended.
"""

function sample(model::AutoRegressiveEmission, Φ::Union{Matrix{<:Real}, Vector{<:Real}})
    # Create the design matrix Φ using past observations
    Φ = size(Φ, 2) == 1 ? reshape(Φ, 1, :) : Φ
    order = model.order
    output_dim = model.output_dim

    # Ensure Y_prev matches the AR order
    if size(Φ, 1) < order
        error("Y_prev must have at least as many rows as the AR order.")
    end

    # Ensure Φ is a 2D matrix even if it's a single sample
    Φ = size(Φ, 2) == 1 ? reshape(Φ, 1, :) : Φ

    # Flatten the last `order` rows of Y_prev to construct Φ
    Φ = reshape(Φ', 1, :)

    # Add intercept column if specified -> need to do this before reshape so that you include enough 1s in flattened vector
    if model.include_intercept
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    # Ensure the noise dimensions match the output dimension and sample size
    noise = rand(MvNormal(zeros(model.output_dim), model.Σ), size(Φ, 1))'

    return Φ * model.β + noise
end

"""
    loglikelihood(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real})

Calculate the log likelihood of the data `Y` given the autoregressive emission model and the previous observations `Y_prev`.

# Arguments
- `model::AutoRegressionEmission`: The autoregressive emission model for which to calculate the log likelihood.
- `Y_prev::Matrix{<:Real}`: The matrix of previous observations, where each row represents an observation.
- `Y::Matrix{<:Real}`: The data matrix, where each row represents an observation.

# Returns
- `Vector{Float64}`: A vector of log likelihoods, one for each observation in the data.
"""

function loglikelihood(model::AutoRegressiveEmission, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real})
    # confirm that the model has valid parameters
    Φ_gaussian = AR_to_Gaussian_data(Y_prev, Y)

    return loglikelihood(model.innerGaussianRegression, Φ_gaussian, Y)
end

"""
    SwitchingAutoRegression(; K::Int, output_dim::Int, order::Int, include_intercept::Bool=true, β::Matrix{<:Real}=if include_intercept zeros(output_dim * order + 1, output_dim) else zeros(output_dim * order, output_dim) end, Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim), λ::Float64=0.0, A::Matrix{<:Real}=initialize_transition_matrix(K), πₖ::Vector{Float64}=initialize_state_distribution(K))

Create a Switching AutoRegression Model

# Arguments
- `K::Int`: The number of hidden states.
- `output_dim::Int`: The dimensionality of the output data.
- `order::Int`: The order of the autoregressive model.
- `include_intercept::Bool=true`: Whether to include an intercept in the regression model.
- `β::Matrix{<:Real}`: The autoregressive coefficients (defaults to zeros).
- `Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim)`: The covariance matrix for the autoregressive model (defaults to an identity matrix).
- `λ::Float64=0.0`: Regularization parameter for the regression (defaults to zero).
- `A::Matrix{<:Real}=initialize_transition_matrix(K)`: The transition matrix of the HMM (Defaults to a random initialization). 
- `πₖ::Vector{Float64}=initialize_state_distribution(K)`: The initial state distribution of the HMM (Defaults to a random initialization).

# Returns
- `::HiddenMarkovModel`: A Switching AutoRegression Model
"""
function SwitchingAutoRegression(;
    K::Int,
    output_dim::Int,
    order::Int,
    include_intercept::Bool=true,
    β::Matrix{<:Real}=if include_intercept
        zeros(output_dim * order + 1, output_dim)
    else
        zeros(output_dim * order, output_dim)
    end,
    Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim),
    λ::Float64=0.0,
    A::Matrix{<:Real}=initialize_transition_matrix(K),
    πₖ::Vector{Float64}=initialize_state_distribution(K),
)
    # Create the emissions
    emissions = [
        AutoRegressionEmission(;
            output_dim=output_dim,
            order=order,
            include_intercept=include_intercept,
            β=β,
            Σ=Σ,
            λ=λ,
        ) for _ in 1:K
    ]
    # Return the HiddenMarkovModel
    return HiddenMarkovModel(; K=K, B=emissions, A=A, πₖ=πₖ)
end

"""
    CompositeModelEmission <: EmissionModel

A mutable struct representing a composite emission model that combines multiple emission models.

# Fields
- `inner_model::CompositeModel`: The underlying composite model used for the emissions.
"""
mutable struct CompositeModelEmission <: EmissionModel
    inner_model::CompositeModel
end

function sample(
    model::CompositeModelEmission,
    input_data::Vector{};
    observation_sequence::Vector{}=Vector(),
)
    if isempty(observation_sequence)
        for i in 1:length(model.components)
            push!(observation_sequence, (sample(model.components[i], input_data[i]...),))
        end
    else
        for i in 1:length(model.components)
            observation_sequence[i] = (
                sample(
                    model.components[i],
                    input_data[i]...;
                    observation_sequence=observation_sequence[i][1],
                ),
            )
        end
    end

    return observation_sequence
end

function loglikelihood(
    model::CompositeModelEmission, input_data::Vector{}, output_data::Vector{}
)
    loglikelihoods = Vector{}(undef, length(model.components))

    for i in 1:length(model.components)
        loglikelihoods[i] = loglikelihood(
            model.components[i], input_data[i]..., output_data[i]...
        )
    end
    return sum(loglikelihoods; dims=1)[1]
end

function fit!(
    model::CompositeModelEmission,
    input_data::Vector{},
    output_data::Vector{},
    w::Vector{Float64}=Vector{Float64}(),
)
    for i in 1:length(model.components)
        fit!!(model.components[i], input_data[i]..., output_data[i]..., w)
    end
end
