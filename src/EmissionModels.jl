"""
EmissionModels.jl

This module implements various emission models for state space modeling, including:
- Gaussian emissions
- Regression-based emissions (Gaussian, Bernoulli, Poisson)
"""

# Exports
export EmissionModel, RegressionEmission
export GaussianEmission, GaussianRegressionEmission, BernoulliRegressionEmission, PoissonRegressionEmission, AutoRegressionEmission
export sample, loglikelihood, fit!

#=
Gaussian Emission Models
=#

"""
    mutable struct GaussianEmission <: EmissionModel

GaussianEmission model with mean and covariance.
"""
mutable struct GaussianEmission{T<:Float64} <: EmissionModel
    output_dim::Int # dimension of the data
    μ::AbstractVector{T}  # mean 
    Σ::AbstractMatrix{T}  # covariance matrix
end

"""
    function GaussianEmission(; output_dim::Int, μ::Vector{<:Real}=zeros(output_dim), Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim))

Functon to create a GaussianEmission with given output dimension, mean, and covariance.

# Arguments
- `output_dim::Int`: The output dimension of the emission
- `μ::Vector{Float64}=zeros(output_dim)`: The mean of the Gaussian
- `Σ::Matrix{Float64}=Matrix{Float64}(I, output_dim, output_dim))`: The covariance matrix of the Gaussian

# Returns
"""
function GaussianEmission(;
    output_dim::Int,
    μ::Vector{T}=zeros(output_dim),
    Σ::AbstractMatrix{T}=Matrix{Float64}(I, output_dim, output_dim),
) where {T<:Float64}
    return GaussianEmission{T}(output_dim, μ, Σ)
end

"""
    loglikelihood(model::GaussianEmission, Y::Matrix{Float64})

Calculate the log likelihood of the data `Y` given the Gaussian emission model.

# Arguments
- `model::GaussianEmission`: The Gaussian emission model for which to calculate the log likelihood.
- `Y::Matrix{<:Real}`: The data matrix, where each row represents an observation.

# Returns
- `Vector{Float64}`: A vector of log likelihoods, one for each observation in the data.
"""
function loglikelihood(model::GaussianEmission, Y::Matrix{<:Real})
    loglikelihood(model, to_f64(Y))
end

function loglikelihood(model::GaussianEmission, Y::Matrix{Float64})
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
function fit!(model::GaussianEmission, Y::Matrix{<:Real}, w::Vector{Float64})
    fit!(model, to_f64(Y), w)
end 

function fit!(model::GaussianEmission, Y::Matrix{<:Real})
    fit!(model, to_f64(Y))  
end

function fit!(
    model::GaussianEmission, Y::Matrix{Float64}, w::Vector{Float64}=ones(size(Y, 1))
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
    X::Matrix{Float64}
    y::Matrix{Float64}
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
    return RegressionOptimization(model, to_f64(X), to_f64(y), w, β_shape)
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
- `β::Matrix{Float64}`: Coefficient matrix
- `λ::Float64`: Regularization parameter
- `include_intercept::Bool`: Whether to exclude the intercept term from regularization

# Returns
- `Float64`: The regularization term value
"""
function calc_regularization(β::AbstractMatrix{T}, λ::Float64, include_intercept::Bool=true) where {T<:Number}
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
    β::AbstractMatrix{T}, λ::Float64, include_intercept::Bool=true
) where {T<:Number}
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
mutable struct GaussianRegressionEmission{T<:Float64} <: RegressionEmission
    input_dim::Int
    output_dim::Int
    β::AbstractMatrix{T} # coefficient matrix of the model. Shape input_dim by output_dim. Column one is coefficients for target one, etc. The first row are the intercept terms, if included. 
    Σ::AbstractMatrix{T} # covariance matrix of the model 
    include_intercept::Bool # whether to include an intercept term; if true, the first column of β is assumed to be the intercept/bias
    λ::T # regularization parameter
end

function GaussianRegressionEmission(;
    input_dim::Int,
    output_dim::Int,
    include_intercept::Bool=true,
    β::AbstractMatrix{T}=if include_intercept
        zeros(input_dim + 1, output_dim)
    else
        zeros(input_dim, output_dim)
    end,
    Σ::AbstractMatrix{U}=Matrix{Float64}(I, output_dim, output_dim),
    λ::Float64=0.0,
) where {T<:Real, U<:Real}
    βf = to_f64(β)
    Σf = to_f64(Σ)

    GaussianRegressionEmission{Float64}(input_dim, output_dim, βf, Σf, include_intercept, λ)
end

"""
    sample(model::GaussianRegressionEmission, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))

Generate `n` samples from a Gaussian regression model. Returns a matrix of size `(n, output_dim)`.

# Arguments
- `model::GaussianRegressionEmission`: Gaussian regression model.
- `Φ::Matrix{Float64}`: Design matrix of shape `(n, input_dim)`.
- `n::Int=size(Φ, 1)`: Number of samples to generate.

# Returns
- `Y::Matrix{<:Real}`: Matrix of samples of shape `(n, output_dim)`.
"""
function sample(model::GaussianRegressionEmission{Float64}, Φ::AbstractVecOrMat{<:Real} )
    sample(model, to_f64(Φ))
end
  
function sample(model::GaussianRegressionEmission{Float64}, Φ::Union{Matrix{Float64},Vector{Float64}})
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
    model::GaussianRegressionEmission{Float64}, 
    Φ::Matrix{<:Real}, 
    Y::Matrix{<:Real},
)
    Φf = to_f64(Φ)
    Yf = to_f64(Y)
    loglikelihood(model, Φf, Yf)
end

function loglikelihood(
    model::GaussianRegressionEmission{Float64},
    Φ::Matrix{Float64},
    Y::Matrix{Float64},
)
    w = ones(size(Y,1))                   
    return loglikelihood(model, Φ, Y, w)  
end

function loglikelihood(
    model::GaussianRegressionEmission{Float64},
    Φ::Matrix{<:Real},
    Y::Matrix{<:Real},
    w::AbstractVector{<:Real},
)
    loglikelihood(model, to_f64(Φ), to_f64(Y), to_f64(w), )
end

function loglikelihood(
    model::GaussianRegressionEmission{Float64},
    Φ::Matrix{Float64},
    Y::Matrix{Float64},
    w::Vector{Float64},
)
    # Add intercept if specified
    Φ = model.include_intercept ? [ones(size(Φ, 1)) Φ] : Φ
    
    # residuals
    residuals = Y - Φ * model.β

    # Calculate weighted least squares
    weighted_residuals = residuals .^ 2 .* w

    if size(weighted_residuals,2) > 1
        weighted_residuals = vec(sum(weighted_residuals, dims=2))
    end

    return -0.5 .* weighted_residuals
end

"""
    AutoRegressionEmission <: EmissionModel

A mutable struct representing an autoregressive emission model, which wraps around an `AutoRegression` model.

# Fields
- `inner_model::AutoRegression`: The underlying autoregressive model used for the emissions.
"""
mutable struct AutoRegressionEmission{T<:Int} <: AutoRegressiveEmission
    output_dim::T
    order::T
    innerGaussianRegression::GaussianRegressionEmission{Float64}
end

function AutoRegressionEmission(; 
    output_dim::T, 
    order::T, 
    include_intercept::Bool = true, 
    β::Matrix{U} = if include_intercept zeros(output_dim * order + 1, output_dim) else zeros(output_dim * order, output_dim) end,
    Σ::AbstractMatrix{U} = Matrix{Float64}(I, output_dim, output_dim),
    λ::Float64=0.0) where {T<:Int, U<:Real}

    innerGaussianRegression = GaussianRegressionEmission(
        input_dim=output_dim * order, 
        output_dim=output_dim, 
        β=to_f64(β),
        Σ=to_f64(Σ),
        include_intercept=include_intercept, 
        λ=λ)

    model = AutoRegressionEmission{T}(output_dim, order, innerGaussianRegression)

    return model
end

"""
    construct_AR_feature_matrix(data::Matrix{Float64}, order::Int) -> Matrix{Float64}

Construct an autoregressive (AR) feature matrix from input time series data.

# Arguments
- `data::Matrix{Float64}`: A matrix of size `(num_feats, T)`, where `num_feats` is the number of features, and `T` is the number of timepoints.
- `order::Int`: The autoregressive order, determining how many past timepoints are included for each time step.

# Returns
- `Matrix{Float64}`: A transformed feature matrix of size `(num_feats * (order + 1), T - order)`, where each column contains stacked feature vectors from the current and past `order` timepoints.

# Example
```julia
data = rand(3, 10)  # 3 features, 10 timepoints
order = 2
AR_feats = construct_AR_feature_matrix(data, order)
size(AR_feats)  # (3 * (2 + 1), 10 - 2) => (9, 8)
"""
function construct_AR_feature_matrix(data::Matrix{Float64}, order::Int, include_intercept=false)
    # If intercept is needed, prepend a row of ones
    if include_intercept
        data = vcat(ones(1, size(data, 2)), data)
    end

    # Original data dimensions
    num_feats, T = size(data)

    # AR feature matrix initialization
    num_feats_AR = num_feats * (order+1)
    T_AR = T - order
    AR_feats_matrix = zeros(Float64, num_feats_AR, T_AR)

    # Fill in the AR_feats_matrix
    for iter = order+1:T
        AR_feats_matrix[:, iter-order] = reshape(data[:, iter-order:iter], :, 1)
    end

    return AR_feats_matrix

end

"""
    construct_AR_feature_matrix(data::Vector{Matrix{Float64}}, order::Int) -> Vector{Matrix{Float64}}

Constructs autoregressive (AR) feature matrices for multiple trials of time series data. Each trial is represented as a matrix, and the function applies the same AR transformation to each trial independently.

# Arguments
- `data::Vector{Matrix{Float64}}`: A vector of matrices, where each matrix represents a trial of time series data with dimensions `(num_feats, T)`, where `num_feats` is the number of features and `T` is the number of timepoints.
- `order::Int`: The autoregressive order, determining how many past timepoints are included for each time step.

# Returns
- `Vector{Matrix{Float64}}`: A vector of transformed feature matrices, where each matrix has dimensions `(num_feats * (order + 1), T - order)`, containing stacked feature vectors from the current and past `order` timepoints.

# Example
```julia
data = [rand(3, 10) for _ in 1:5]  # 5 trials, each with 3 features and 10 timepoints
order = 2
AR_feats_trials = construct_AR_feature_matrix(data, order)
size(AR_feats_trials[1])  # (9, 8), same transformation applied per trial
"""
function construct_AR_feature_matrix(data::Vector{Matrix{Float64}}, order::Int, include_intercept=false)
    # Initialize feature vector
    AR_feats_matrices = Vector{Matrix{Float64}}(undef, length(data))
    
    # Compute AR feature matrix for each trial
    for trial_idx in eachindex(data)
        print(data[trial_idx])
        AR_feats_matrices[trial_idx] = construct_AR_feature_matrix(data[trial_idx], order, include_intercept)
    end

    return AR_feats_matrices
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
function sample(model::AutoRegressionEmission{Int}, X::Matrix{Float64})
    # Extract the last column of X as input
    last_observation = X[:, end]

    # Sample new observation using the Gaussian regression emission function
    new_observation = sample(model.innerGaussianRegression, last_observation)
    new_observation = reshape(new_observation, :, 1)

    # Append the new sample as a new column
    X = hcat(X, new_observation)

    return X, new_observation
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
function loglikelihood(
    model::AutoRegressionEmission{Int},
    X::Matrix{<:Real},
    Y::Matrix{<:Real},
    w::Vector{Float64}=ones(size(Y, 1)),
)
    return loglikelihood(model.innerGaussianRegression, to_f64(X), to_f64(Y), to_f64(w))
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
- `A::Matrix{Float64}=initialize_transition_matrix(K)`: The transition matrix of the HMM (Defaults to a random initialization). 
- `πₖ::Vector{Float64}=initialize_state_distribution(K)`: The initial state distribution of the HMM (Defaults to a random initialization).

# Returns
- `::HiddenMarkovModel`: A Switching AutoRegression Model
"""
function SwitchingAutoRegression(;
    K::Int,
    output_dim::Int,
    order::Int,
    include_intercept::Bool=true,
    β::AbstractMatrix{T}=if include_intercept
        zeros(output_dim * order + 1, output_dim)
    else
        zeros(output_dim * order, output_dim)
    end,
    Σ::AbstractMatrix{T}=Matrix{Float64}(I, output_dim, output_dim),
    λ::Float64=0.0,
    A::AbstractMatrix{Float64}=initialize_transition_matrix(K),
    πₖ::AbstractVector{Float64}=initialize_state_distribution(K),
) where {T<:Real}
    # Create the emissions
    emissions = [
        AutoRegressionEmission(;
            output_dim=output_dim,
            order=order,
            include_intercept=include_intercept,
            β=to_f64(β),
            Σ=to_f64(Σ),
            λ=λ,
        ) for _ in 1:K
    ]
    # Return the HiddenMarkovModel
    return HiddenMarkovModel(; K=K, B=emissions, A=A, πₖ=πₖ)
end


function objective(
    opt::Union{RegressionOptimization{GaussianRegressionEmission{Float64}}, RegressionOptimization{AutoRegressionEmission{Int}}}, β_vec::AbstractVector{T}
) where {T<:Number}
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
    opt::Union{RegressionOptimization{GaussianRegressionEmission{Float64}}, RegressionOptimization{AutoRegressionEmission{Int}}},
    β_vec::AbstractVector{T},
) where {T<:Number}
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

# # Special handling for Gaussian regression to update variance
# function post_optimization!(model::GaussianRegressionEmission, opt::RegressionOptimization)
#     residuals = opt.y - opt.X * model.β
#     Σ = (residuals' * Diagonal(opt.w) * residuals) / size(opt.X, 1)
#     return model.Σ = 0.5 * (Σ + Σ')  # Ensure symmetry
# end

function post_optimization!(model::GaussianRegressionEmission{Float64}, opt::RegressionOptimization)
    residuals = opt.y - opt.X * model.β
    Σ = (residuals' * Diagonal(opt.w) * residuals) / size(opt.X, 1)
    model.Σ = 0.5 * (Σ + Σ')  # Ensure symmetry
    model.Σ = make_posdef!(model.Σ)
    return model.Σ
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
mutable struct BernoulliRegressionEmission{T<:Float64} <: RegressionEmission
    input_dim::Int
    output_dim::Int
    β::Matrix{T} 
    include_intercept::Bool # whether to include an intercept term; if true, the first column of β is assumed to be the intercept/bias
    λ::Float64 # regularization parameter
end

function BernoulliRegressionEmission(;
    input_dim::Int,
    output_dim::Int,
    include_intercept::Bool=true,
    β::AbstractMatrix{T}=if include_intercept
        zeros(input_dim + 1, output_dim)
    else
        zeros(input_dim, output_dim)
    end,
    λ::Float64=0.0,
) where {T<:Real}
    return BernoulliRegressionEmission{Float64}(input_dim, output_dim, to_f64(β), include_intercept, λ)
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
function sample(model::BernoulliRegressionEmission{Float64}, Φ::AbstractVecOrMat{<:Real})
    sample(model, to_f64(Φ))
end

function sample(model::BernoulliRegressionEmission{Float64}, Φ::Union{Matrix{Float64},Vector{Float64}})
    # Ensure Φ is a 2D matrix even if it's a single sample
    Φ = size(Φ, 2) == 1 ? reshape(Φ, 1, :) : Φ

    # add intercept if specified
    if model.include_intercept && size(Φ, 2) == size(model.β,1) - 1
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
    model::BernoulliRegressionEmission{Float64}, 
    Φ::Matrix{T}, 
    Y::Matrix{T}
) where {T<:Real}
    loglikelihood(model, to_f64(Φ), to_f64(Y))
end

function loglikelihood(
    model::BernoulliRegressionEmission{Float64}, 
    Φ::Matrix{T}, 
    Y::Matrix{T}
) where {T<:Float64}
    w = ones(size(Y,1))                   
    return loglikelihood(model, Φ, Y, w)  
end

function loglikelihood(
    model::BernoulliRegressionEmission{Float64},
    Φ::Matrix{T},
    Y::Matrix{T},
    w::AbstractVector{T},
) where {T<:Real}
    loglikelihood(model, to_f64(Φ), to_f64(Y), to_f64(w), )
end

function loglikelihood(
    model::BernoulliRegressionEmission{Float64},
    Φ::Matrix{T},
    Y::Matrix{T},
    w::Vector{T},
) where {T<:Float64}
    # add intercept if specified and not already included
    if model.include_intercept && size(Φ, 2) == size(model.β,1) - 1
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
    opt::RegressionOptimization{BernoulliRegressionEmission{Float64}},  β_vec::AbstractVector{T},
) where {T<:Number}
    β_mat = vec_to_matrix(β_vec, opt.β_shape)
    p = logistic.(opt.X * β_mat)

    # calculate regularization
    regularization = calc_regularization(β_mat, opt.model.λ, opt.model.include_intercept)

    val = -sum(opt.w .* (opt.y .* log.(p) .+ (1 .- opt.y) .* log.(1 .- p))) + regularization

    return val
end

function objective_gradient!(
    G::Vector{Float64},
    opt::RegressionOptimization{BernoulliRegressionEmission{Float64}},
    β_vec::AbstractVector{T},
) where {T<:Number}
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
mutable struct PoissonRegressionEmission{T<:Float64} <: RegressionEmission
    input_dim::Int
    output_dim::Int
    β::Matrix{T}
    include_intercept::Bool
    λ::T
end

function PoissonRegressionEmission(;
    input_dim::Int,
    output_dim::Int,
    include_intercept::Bool=true,
    β::Matrix{T}=if include_intercept
        zeros(input_dim + 1, output_dim)
    else
        zeros(input_dim, output_dim)
    end,
    λ::Float64=0.0,
) where {T<:Real}
    PoissonRegressionEmission{Float64}(input_dim, output_dim, to_f64(β), include_intercept, λ)
end

"""
    sample(model::PoissonRegressionEmission, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))

Generate `n` samples from a Poisson regression model. Returns a matrix of size `(n, 1)`.

# Arguments
- `model::PoissonRegressionEmission`: Poisson regression model.
- `Φ::Matrix{<:Real}`: Design matrix of shape `(n, input_dim)`.
- `n::Int=size(Φ, 1)`: Number of samples to generate.

# Returns
- `Y::Matrix{Float64}`: Matrix of samples of shape `(n, 1)`.
"""
function sample(model::PoissonRegressionEmission{Float64}, Φ::AbstractVecOrMat{<:Real} )
    sample(model, to_f64(Φ))
end

function sample(model::PoissonRegressionEmission{Float64}, 
    Φ::Union{Matrix{Float64},Vector{Float64}})   
    # Ensure Φ is a 2D matrix even if it's a single sample
    Φ = size(Φ, 2) == 1 ? reshape(Φ, 1, :) : Φ

    # add intercept if specified
    if model.include_intercept && size(Φ, 2) == size(model.β,1) - 1
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
    model::PoissonRegressionEmission{Float64},
    Φ::Matrix{<:Real},
    Y::Matrix{<:Real},
)
    loglikelihood(model, to_f64(Φ), to_f64(Y))
end

function loglikelihood(
    model::PoissonRegressionEmission{Float64}, 
    Φ::Matrix{Float64}, 
    Y::Matrix{Float64}
)
    w = ones(size(Y,1))                   
    return loglikelihood(model, Φ, Y, w)  
end

function loglikelihood(
    model::PoissonRegressionEmission{Float64},
    Φ::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    w::AbstractVector{<:Real},
)
    loglikelihood(model, to_f64(Φ), to_f64(Y), to_f64(w), )
end

function loglikelihood(
    model::PoissonRegressionEmission{Float64},
    Φ::Matrix{T},
    Y::Matrix{T},
    w::Vector{T},
) where {T<:Float64}
    # add intercept if specified
    if model.include_intercept && size(Φ, 2) == size(model.β,1) - 1
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
    opt::RegressionOptimization{PoissonRegressionEmission{Float64}}, β_vec::AbstractVector{T}
) where {T<:Number}
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
    opt::RegressionOptimization{PoissonRegressionEmission{Float64}},
    β_vec::AbstractVector{T},
) where {T<:Number}
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
)
    fit!(model, to_f64(X), to_f64(y))
end 

function fit!(
    model::RegressionEmission,
    X::Matrix{Float64},
    y::Matrix{Float64},
)
    w = ones(size(y,1))
    fit!(model, X, y, w)
end 

function fit!(
    model::RegressionEmission,
    X::Matrix{<:Real},
    y::Matrix{<:Real},
    w::AbstractVector{<:Real}
)
    fit!(model, to_f64(X), to_f64(y), to_f64(w))
end 

function fit!(
    model::RegressionEmission,
    X::Matrix{Float64},
    y::Matrix{Float64},
    w::AbstractVector{Float64},
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
    )

    # Run optimization
    result = optimize(f, g!, vec(model.β), LBFGS(), opts)

    # Update model parameters
    model.β = vec_to_matrix(result.minimizer, opt_problem.β_shape)

    # Update additional parameters if needed (e.g., variance for Gaussian)
    post_optimization!(model, opt_problem)

    return model
end