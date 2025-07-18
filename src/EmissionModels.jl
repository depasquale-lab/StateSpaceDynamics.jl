"""
EmissionModels.jl

This module implements various emission models for state space modeling, including:
- Gaussian emissions
- Regression-based emissions (Gaussian, Bernoulli, Poisson)
"""

# Exports
export EmissionModel, RegressionEmission
export GaussianEmission, GaussianRegressionEmission, BernoulliRegressionEmission, PoissonRegressionEmission, AutoRegressionEmission
export loglikelihood, fit!, create_optimization

#=
Gaussian Emission Models
=#

"""
    mutable struct GaussianEmission <: EmissionModel

GaussianEmission model with mean and covariance.
"""
mutable struct GaussianEmission{T<:Real, V<:AbstractVector{T}, M<:AbstractMatrix{T}} <: EmissionModel
    output_dim::Int # dimension of the data
    μ::V  # mean 
    Σ::M  # covariance matrix
end

"""
    function GaussianEmission(; output_dim::Int, μ::AbstractVector, Σ::AbstractMatrix)

Create a GaussianEmission model.

# Arguments
- `output_dim::Int`: The output dimension of the emission
- `μ::AbstractVector`: The mean of the Gaussian
- `Σ::AbstractMatrix`: The covariance matrix of the Gaussian

# Returns
- `GaussianEmission<:EmissionModel`: The Gaussian emission model
```
"""
function GaussianEmission(; 
    output_dim::Int,
    μ::AbstractVector,
    Σ::AbstractMatrix,
)
    if !check_same_type(μ[1], Σ[1])
        error("μ and Σ must be of the same element type. Got $(eltype(μ)) and $(eltype(Σ))")
    end
    return GaussianEmission(output_dim, μ, Σ)
end


"""
    loglikelihood(model::GaussianEmission, Y::AbstractMatrix{T}) where {T<:Real}

Calculate the log likelihood of the data `Y` given the Gaussian emission model.
"""
function loglikelihood(model::GaussianEmission, Y::AbstractMatrix{T}) where {T<:Real}
    # Create MvNormal distribution with the model parameters
    dist = MvNormal(model.μ, model.Σ)

    # Calculate log likelihood for each observation
    return [logpdf(dist, @view(Y[i, :])) for i in axes(Y, 1)]
end

"""
     Random.rand(rng::AbstractRNG, model::GaussianEmission; n::Int=1)

Generate `n` samples from a Gaussian emission model.

# Arguments
    - `rng::AbstractRNG`: Seed
    -`model::GaussianEmission`: The GaussianEmission model
    -`n::Int=1`: The number of samples to generate

# Returns
    -`samples::Matrix{<:Real}`: Matrix of samples (n, output_dim)
"""
function Random.rand(rng::AbstractRNG, model::GaussianEmission; n::Int=1)
    raw_samples = rand(rng, MvNormal(model.μ, model.Σ), n)
    return Matrix(raw_samples')
end

"""
    Random.rand(model::GaussianEmission; kwargs...)
    Random.rand(rng::AbstractRNG, model::GaussianEmission; n::Int=1)

Generate random samples from  a Gaussian emission model.
"""
function Random.rand(model::GaussianEmission; kwargs...)
    return rand(Random.default_rng(), model; kwargs...)
end

"""
    function fit!(model::GaussianEmission, 
            Y::AbstractMatrix{T}, 
            w::AbstractVector{T}=ones(size(Y, 1))) where {T<:Real}

Fit a GaussianEmission model to the data `Y` weighted by weights `w`.
"""
function fit!(
    model::GaussianEmission, Y::AbstractMatrix{T}, w::AbstractVector{T}=ones(size(Y, 1))
) where {T<:Real}
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

"""
    RegressionOptimization{T<:RegressionEmission}

Hold the optimization problem data for regression emissions.
"""
struct RegressionOptimization{R<:RegressionEmission, V<:AbstractVector{<:Real}, MX<:AbstractMatrix{<:Real}, MY<:AbstractMatrix{<:Real}}
    model::R
    X::MX
    y::MY
    w::V
    β_shape::Tuple{Int, Int}  # Added to track original shape
end


# Unified interface for creating optimization problems
"""
    create_optimization(
        model::RegressionEmission,
        X::AbstractMatrix{<:Real},
        y::AbstractMatrix{<:Real},
        w::V)

Create regression optimization problem.
"""
function create_optimization(
    model::RegressionEmission,
    X::AbstractMatrix{<:Real},
    y::AbstractMatrix{<:Real},
    w::AbstractVector{<:Real}=ones(size(y, 2)),
)
    if model.include_intercept
        X = vcat(ones(eltype(X), 1, size(X, 2)), X)
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
    calc_regularization(β::AbstractMatrix{T1}, λ::T2, include_intercept::Bool=true) where {T1<:Real, T2<:Real}

Calculate L2 regularization term for regression coefficients.
"""
function calc_regularization(β::AbstractMatrix{T1}, λ::T2, include_intercept::Bool=true) where {T1<:Real, T2<:Real}
    # Includes  T1 and T2 since autodiff passes in DualNumber which is not subtype float for β
    # calculate L2 penalty
    if include_intercept
        regularization = 0.5 * λ * sum(abs2, @view(β[2:end, :]))
    else
        regularization = 0.5 * λ * sum(abs2, β)
    end

    return regularization
end

"""
    function calc_regularization_gradient(β::AbstractMatrix{T1}, λ::T2, include_intercept::Bool=true) where {T1<:Real, T2<:Real}

Calculate gradient of L2 regularization term for regression coefficients.
"""
function calc_regularization_gradient(
    β::AbstractMatrix{T1}, λ::T2, include_intercept::Bool=true
) where {T1<:Real, T2<:Real}
    # calculate the gradient of the regularization component
    regularization = zeros(size(β))

    if include_intercept
        regularization[2:end, :] .= λ * @view(β[2:end, :])
    else
        regularization .= λ * β
    end

    return regularization
end

"""
    GaussianRegressionEmission

Store a Gaussian regression Emission model.

# Fields
- `input_dim::Int`: Dimension of the input data.
- `output_dim::Int`: Dimension of the output data.
- `include_intercept::Bool`: Whether to include an intercept term; if true, the first column of β is assumed to be the intercept/bias.
- `β::AbstractMatrix{<:Real} = if include_intercept zeros(input_dim + 1, output_dim) else zeros(input_dim, output_dim) end`: Coefficient matrix of the model. Shape input_dim by output_dim. The first row are the intercept terms, if included.
- `Σ::AbstractMatrix{<:Real}`: Covariance matrix of the model.
- `λ:<Real`: Regularization parameter.
"""
mutable struct GaussianRegressionEmission{T<:Real, M<:AbstractMatrix{T}} <: RegressionEmission
    input_dim::Int
    output_dim::Int
    β::M # coefficient matrix of the model. Shape input_dim by output_dim. Column one is coefficients for target one, etc. The first row are the intercept terms, if included. 
    Σ::M # covariance matrix of the model 
    include_intercept::Bool # whether to include an intercept term; if true, the first column of β is assumed to be the intercept/bias
    λ::T # regularization parameter
end

"""
    GaussianRegressionEmission(input_dim, output_dim, include_intercept, β, Σ, λ)

Create a Gaussian regression emission model.

# Arguments
    -`input_dim::Int`: Dimensionality of input data
    -`output_dim::Int`: Dimensionality of output data
    -`include_intercept:Bool`: Whether to include a regression intercept
    -`β::AbstractMatrix`: The regression coefficient matrix
    -`Σ::AbstractMatrix`: The regression covariance matrix
    -`λ<:Real`: The L2 regularization parameter

# Returns
    -`model::GaussianRegressionEmission`: The Gaussian regression emission model

"""
function GaussianRegressionEmission(;
    input_dim::Int,
    output_dim::Int,
    include_intercept::Bool,
    β::AbstractMatrix,
    Σ::AbstractMatrix,
    λ::Real
)

    if !check_same_type(β[1], Σ[1], λ)
        error("β, Σ, and λ must be of the same element type. Got $(eltype(β)), $(eltype(Σ)), and $(eltype(λ))")
    end

    return GaussianRegressionEmission(input_dim, output_dim, β, Σ, include_intercept, λ)
end

"""
     Random.rand(rng::AbstractRNG, model::GaussianRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})

Generate samples from a Gaussian regression emission model.

# Arguments
    - `rng::AbstractRNG`: Seed
    -`model::GaussianRegressionEmission`: The Gaussian regression model
    -`Φ::Union{Matrix{<:Real},Vector{<:Real}}`: The input data (defines number of samples to generate)

# Returns
    -`samples::Matrix{<:Real}`: Matrix of samples (n, output_dim)
"""
function Random.rand(rng::AbstractRNG, model::GaussianRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})
    Φ = Φ isa Vector ? reshape(Φ, :, 1) : Φ 
    n = size(Φ, 2)

    if model.include_intercept 
        Φ = vcat(ones(eltype(Φ), 1, n), Φ)
    end
   
    noise = rand(rng, MvNormal(zeros(model.output_dim), model.Σ), n)
    return model.β' * Φ + noise 
end

"""
    Random.rand(model::GaussianRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})
    Random.rand(rng::AbstractRNG, model::GaussianRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})

Generate samples from a Gaussian regression model.
"""
function Random.rand(model::GaussianRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})
    return rand(Random.default_rng(), model, Φ)
end

"""
    loglikelihood(model::GaussianRegressionEmission,
        Φ::AbstractMatrix{T},
        Y::AbstractMatrix{T},
        w::AbstractVector{T}=ones(size(Y, 1))) where {T<:Real}

Calculate the log likelihood of the data `Y` given the Gaussian regression emission model and the input features `Φ`.
"""
function loglikelihood(
    model::GaussianRegressionEmission,
    Φ::AbstractMatrix{T}, # f x n 
    Y::AbstractMatrix{T}, # d x n
    w::Union{Nothing,AbstractVector{T}} = nothing,
) where {T<:Real}

    n = size(Φ, 2)
    d = size(Y, 1)
    β = model.β

    if w === nothing
        w = ones(T, size(Y, 2))
    elseif eltype(w) !== eltype(Y)
        error("weights must be Vector{$(eltype(Y))}; Got Vector{$(eltype(w))}")
    end

    if model.include_intercept
        if size(Φ, 1) == size(β, 1) - 1
            Φ = vcat(ones(T, 1, n), Φ)
        elseif size(Φ, 1) != size(β, 1)
            error("Φ has incompatible size with model.β: got $(size(Φ)), expected $(size(β,1)) rows.")
        end
    end

    ll = Vector{T}(undef, n)
    for i in 1:n
        acc = zero(T)
        for j in 1:d
            acc += (Y[j, i] - dot(β[:, j], Φ[:, i]))^2 
        end
        ll[i] = -0.5 * w[i] * acc
    end

    return ll
end

"""
    AutoRegressionEmission <: EmissionModel

Store an autoregressive emission model, which wraps around a `GaussianRegressionEmission`.

# Fields
- `output_dim::Int`: The dimensionality of the output data
- `order::Int`: The order of the Autoregressive process
- `innerGaussianRegression::GaussianRegressionEmission`: The underlying Gaussian regression model used for the emissions.
"""
mutable struct AutoRegressionEmission <: AutoRegressiveEmission
    output_dim::Int
    order::Int
    innerGaussianRegression::GaussianRegressionEmission
end

"""
    AutoRegressionEmission(output_dim, order, include_intercept, β, Σ, λ)

Create an Autoregressive emission model.

# Arguments
    - `output_dim::Int`: The dimensionality of the output data.
    - `order::Int`: The order of the autoregressive process.
    - `include_intercept::Bool`: Whether to include a regression intercept.
    - `β::AbstractMatrix`: The regression coefficient matrix.
    - `Σ::AbstractMatrix`: The regression covariance matrix.
    - `λ<:Real`: The L2 regularization parameter.

# Returns
    - `model::AutoRegressionEmission`: The autoregressive emission model.
"""
function AutoRegressionEmission(; 
    output_dim::Int, 
    order::Int, 
    include_intercept::Bool, 
    β::AbstractMatrix,
    Σ::AbstractMatrix,
    λ::Real
)

    if !check_same_type(β[1], Σ[1], λ)
        error("β, Σ, and λ must be of the same element type. Got $(eltype(β)), $(eltype(Σ)), and $(eltype(λ))")
    end

    innerGaussianRegression = GaussianRegressionEmission(
        input_dim=output_dim, 
        output_dim=output_dim, 
        β=β,
        Σ=Σ,
        include_intercept=include_intercept, 
        λ=λ
    )

    return AutoRegressionEmission(output_dim, order, innerGaussianRegression)
end


"""
    construct_AR_feature_matrix(data::AbstractMatrix{T}, order::Int, include_intercept=false) where {T<:Real}

Construct an autoregressive (AR) feature matrix from input time series data.

# Arguments
- `data::AbstractMatrix{<:Real}`: A matrix of size `(num_feats, T)`, where `num_feats` is the number of features, and `T` is the number of timepoints.
- `order::Int`: The autoregressive order, determining how many past timepoints are included for each time step.
- `include_intercept::Bool=false`: Whether to include an intercept regression term.

# Returns
- `Matrix{Float64}`: A transformed feature matrix of size `(num_feats * (order + 1), T - order)`, where each column contains stacked feature vectors from the current and past `order` timepoints.
"""
function construct_AR_feature_matrix(data::AbstractMatrix{T}, order::Int, include_intercept=false) where {T<:Real}
    # If intercept is needed, prepend a row of ones
    if include_intercept
        data = vcat(ones(1, size(data, 2)), data)
    end

    # Original data dimensions
    num_feats, t = size(data)

    # AR feature matrix initialization
    num_feats_AR = num_feats * (order + 1)
    T_AR = t - order
    AR_feats_matrix = zeros(num_feats_AR, T_AR)

    # Fill in the AR_feats_matrix
    @views for iter = order+1:t
        AR_feats_matrix[:, iter - order] = reshape(data[:, iter - order:iter], :, 1)
    end

    return AR_feats_matrix
end

function construct_AR_feature_matrix(data::Vector{<:Matrix{<:Real}}, order::Int, include_intercept=false)
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
    Random.rand(rng::AbstractRNG, model::AutoRegressionEmission, X::Matrix{<:Real})

Generate samples from an autoregressive emission model.

# Arguments
- `rng::AbstractRNG`: The seed.
- `model::AutoRegressionEmission`: The autoregressive emission model.
- `X::Matrix{<:Real}: The current data from which to generate samples.`

# Returns
- `Matrix{Float64}`: The updated observation sequence with the new sample appended.
"""
function Random.rand(rng::AbstractRNG, model::AutoRegressionEmission, X::Matrix{<:Real})
    # Extract the last column of X as input
    last_observation = X[:, end]

    # Sample new observation using the inner Gaussian regression model
    new_observation = rand(rng, model.innerGaussianRegression, last_observation)
    new_observation = reshape(new_observation, :, 1)

    # Append the new sample as a new column
    X = hcat(X, new_observation)

    return X, new_observation
end

"""
    Random.rand(model::AutoRegressionEmission, X::Matrix{<:Real})
    Random.rand(rng::AbstractRNG, model::AutoRegressionEmission, X::Matrix{<:Real})

Generate samples from an autoregressive emission model.
"""
function Random.rand(model::AutoRegressionEmission, X::Matrix{<:Real})
    return rand(Random.default_rng(), model, X)
end

"""
    loglikelihood(
        model::AutoRegressionEmission,
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T},
        w::Vector{T}=ones(size(Y, 1))) where {T<:Real}

Calculate the log likelihood of the data `Y` given the autoregressive emission model and the previous observations `X`.
"""
function loglikelihood(
    model::AutoRegressionEmission,
    X::AbstractMatrix{T},
    Y::AbstractMatrix{T},
    w::Union{Nothing,AbstractVector{T}} = nothing,
) where {T<:Real}

    if w === nothing
        w = ones(eltype(Y), size(Y, 1))
    elseif eltype(w) !== eltype(Y)
        error("weights must be Vector{$(eltype(Y))}; Got Vector{$(eltype(w))}")
    end

    return loglikelihood(model.innerGaussianRegression, X, Y, w)
end

"""
    objective(
        opt::Union{RegressionOptimization{<:GaussianRegressionEmission}, 
        RegressionOptimization{<:AutoRegressionEmission}}, 
        β_vec::AbstractVector{T}) where {T<:Real}

Define the objective function for Gaussian/AR regression emission models.
"""
function objective(
    opt::Union{RegressionOptimization{<:GaussianRegressionEmission}, RegressionOptimization{<:AutoRegressionEmission}}, 
    β_vec::AbstractVector{T}
) where {T<:Real}
    β = reshape(β_vec, opt.β_shape)
    f, n = size(opt.X)
    d, _ = size(opt.y)
    total_ll = zero(T)

    # calculate residuals and w inside loop to avoid allocs 
    for i in 1:n
        for j in 1:d
            total_ll += opt.w[i] * (opt.y[j, i] - dot(β[:, j], opt.X[:, i]))^2
        end
    end

    # calculate regularization
    regularization = calc_regularization(β, opt.model.λ, opt.model.include_intercept)

    # calculate pseudo log-likelihood
    return 0.5 * total_ll + regularization
end

"""
    objective_gradient!(
        G::AbstractVector{T},
        opt::Union{RegressionOptimization{<:GaussianRegressionEmission}, RegressionOptimization{<:AutoRegressionEmission}},
        β_vec::AbstractVector{T}) where {T<:Real}

Define the gradient of the objective function for Gaussian/AR regression emission models
"""
function objective_gradient!(
    G::AbstractVector{T},
    opt::Union{RegressionOptimization{<:GaussianRegressionEmission}, RegressionOptimization{<:AutoRegressionEmission}},
    β_vec::AbstractVector{T},
) where {T<:Real}
    β_mat = vec_to_matrix(β_vec, opt.β_shape)
    f, n = size(opt.X)
    d, _ = size(opt.y)

    grad_matrix = zeros(T, size(β_mat))

    for t in 1:n
        w_t = opt.w[t]
        for o in 1:d

            pred = zero(T)
            for j in 1:f
                pred += opt.X[j, t] * β_mat[j, o]
            end
            r = opt.y[o, t] - pred

            for j in 1:f
                grad_matrix[j, o] -= opt.X[j, t] * w_t * r
            end
        end
    end

    # calc gradient of penalty
    reg_grad = calc_regularization_gradient(β_mat, opt.model.λ, opt.model.include_intercept)
    for j in 1:f
        for o in 1:d
            grad_matrix[j, o] += reg_grad[j, o]
        end
    end

    return G .= vec(grad_matrix)
end

# # Special handling for Gaussian regression to update variance
# function post_optimization!(model::GaussianRegressionEmission, opt::RegressionOptimization)
#     residuals = opt.y - opt.X * model.β
#     Σ = (residuals' * Diagonal(opt.w) * residuals) / size(opt.X, 1)
#     return model.Σ = 0.5 * (Σ + Σ')  # Ensure symmetry
# end

"""
    post_optimization!(model::GaussianRegressionEmission, opt::RegressionOptimization)

Stabilize the covariance matrix for GaussianRegressionEmissions.
"""
function post_optimization!(model::GaussianRegressionEmission, opt::RegressionOptimization)
    f, n = size(opt.X)
    d, _ = size(opt.y)

    residuals = zeros(eltype(opt.y), d, n)
    Σ = zeros(eltype(opt.X), d, d)

    # residuals 
    for i in 1:n
        for j in 1:d
            residuals[j, i] = opt.y[j, i] - dot(model.β[:, j], opt.X[:, i])
            Σ[j, j] += opt.w[i] * residuals[j, i]^2
        end 
    end

    Σ ./= n

    model.Σ = 0.5 * (Σ + Σ')  # Ensure symmetry
    model.Σ = make_posdef!(model.Σ)
    return model.Σ
end

"""
    BernoulliRegressionEmission

Store a Bernoulli regression model.

# Fields
- `input_dim::Int`: Dimensionality of the input data.
- `output_dim::Int`: Dimensionality of the outputd data.
- `include_intercept::Bool`: Whether to include an intercept term.
- `β::AbstractMatrix{<:Real}`: Bernoulli regression coefficients.
- `λ<:Real`: L2 Regularization parameter.
```
"""
mutable struct BernoulliRegressionEmission{T<:Real, M<:AbstractMatrix{T}} <: RegressionEmission
    input_dim::Int
    output_dim::Int
    β::M
    include_intercept::Bool # whether to include an intercept term; if true, the first column of β is assumed to be the intercept/bias
    λ::T # regularization parameter
end

"""
    BernoulliRegressionEmission(Args)

Create a Bernoulli regression emission model.

# Arguments
    - `input_dim::Int`: Dimensionality of the input dimension
    - `output_dim::Int`: Dimensionality of the output dimension
    - `include_intercept::Bool`: Whether to include a regression intercept.
    - `β::AbstractMatrix`: The regression coefficient matrix.
    - `Σ::AbstractMatrix`: The regression covariance matrix.
    - `λ<:Real`: The L2 regularization parameter.

# Returns
    - `model::BernoulliRegressionEmission`: The Bernoulli regression emission model.
"""
function BernoulliRegressionEmission(; 
    input_dim::Int,
    output_dim::Int,
    include_intercept::Bool,
    β::AbstractMatrix,
    λ::Real,
)

    if !check_same_type(β[1], λ)
        error("β and λ must be of the same element type. Got $(eltype(β)) and $(eltype(λ))")
    end

    return BernoulliRegressionEmission(input_dim, output_dim, β, include_intercept, λ)
end


"""
    Random.rand(rng::AbstractRNG, model::BernoulliRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})

Generate samples from a Bernoulli regression model.

# Arguments
    - `rng::AbstractRNG`: The seed.
    - `model::BernoulliRegressionEmission`: Bernoulli regression model.
    - `Φ::AbstractMatrix{<:Real}`: Design matrix of shape `(n, input_dim)`.

# Returns
- `Y::AbstractMatrix{<:Real}`: Matrix of samples of shape `(n, output_dim)`.
"""
function Random.rand(rng::AbstractRNG, model::BernoulliRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})
    Φ = size(Φ, 2) == 1 ? reshape(Φ, 1, :) : Φ
    if model.include_intercept && size(Φ, 2) == length(model.β) - 1
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end
    Y = rand.(rng, Bernoulli.(logistic.(Φ * model.β)))
    return float.(reshape(Y, :, 1))
end

"""
    Random.rand(model::BernoulliRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})
    Random.rand(rng::AbstractRNG, model::BernoulliRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})

Generate samples from a Bernoulli regression emission.
"""
function Random.rand(model::BernoulliRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})
    return rand(Random.default_rng(), model, Φ)
end

"""
    function loglikelihood(
        model::BernoulliRegressionEmission,
        Φ::AbstractMatrix{T1},
        Y::AbstractMatrix{T2},
        w::AbstractVector{T3}=ones(size(Y, 1))) where {T1<:Real, T2<:Real, T3<:Real}

Calculate the log likelihood of the data `Y` given the Bernoulli regression emission model and the input features `Φ`. Optionally, a vector of weights `w` can be provided.
"""
function loglikelihood(
    model::BernoulliRegressionEmission,
    Φ::AbstractMatrix{T}, 
    Y::AbstractMatrix{T},
    w::Union{Nothing,AbstractVector{T}} = nothing,
) where {T<:Real}
    f, n = size(Φ)
    d, _ = size(Y)
    β = model.β

    if w === nothing
        w = ones(T, n)
    elseif eltype(w) !== T
        error("weights must be Vector{$(eltype(Φ))}; Got Vector{$(eltype(w))}")
    end

    if model.include_intercept
        if size(Φ, 1) == size(β, 1) - 1
            Φ = vcat(ones(T, 1, n), Φ)
        elseif size(Φ, 1) != size(β, 1)
            error("Φ has incompatible size with model.β: got $(size(Φ)), expected $(size(β,1)) rows.")
        end
    end

    ll = Vector{T}(undef, n)
    for i in 1:n
        acc = zero(T)
        for j in 1:d
            p = inv(1 + exp(-dot(β[:, j], Φ[:, i])))
            acc += Y[j, i] * log(p + eps(T)) + (1 - Y[j, i]) * log(1 - p + eps(T))
        end 
        ll[i] = w[i] * acc
    end 

    return ll
end

"""
    objective(
        opt::RegressionOptimization{<:BernoulliRegressionEmission},
        β_vec::Vector{T}) where {T<:Real}

Define the objective function for a Bernoulli regression emission model.
"""
function objective(
    opt::RegressionOptimization{<:BernoulliRegressionEmission}, β_vec::Vector{T}
) where {T<:Real}
    β = reshape(β_vec, opt.β_shape)  
    f, n = size(opt.X)
    d, _ = size(opt.y)
    total = zero(T)

    for t in 1:n
        acc = zero(T)
        for o in 1:d
            p = inv(1 + exp(-dot(β[:, o], opt.X[:, t])))
            acc += opt.y[o, t] * log(p + eps()) + (1 - opt.y[o, t]) * log(1 - p + eps())
        end 
        
        total -= opt.w[t] * acc
    end 
    
    total += calc_regularization(β, opt.model.λ, opt.model.include_intercept)
    return total
end

"""
    objective_gradient!(
        G::AbstractVector{T},
        opt::RegressionOptimization{<:BernoulliRegressionEmission},
        β_vec::AbstractVector{T}) where {T<:Real}

Define the objective gradient for a Bernoulli regression emission model.
"""
function objective_gradient!(
    G::AbstractVector{T},
    opt::RegressionOptimization{<:BernoulliRegressionEmission},
    β_vec::AbstractVector{T},
) where {T<:Real}
    β_mat = vec_to_matrix(β_vec, opt.β_shape)
    f, n = size(opt.X)
    d, _ = size(opt.y)
    grad_matrix = zeros(T, f, d)

    for t in 1:n
        w_t = opt.w[t]
        for o in 1:d
            diff = opt.y[o, t] - inv(1 + exp(-dot(β_mat[:, o], opt.X[:, t])))        
            for j in 1:f
                grad_matrix[j, o] -= w_t * opt.X[j, t] * diff
            end
        end 
    end

    # calc gradient of penalty
    reg_gradient = calc_regularization_gradient(β_mat, opt.model.λ, opt.model.include_intercept)
    for j in 1:f, o in 1:d
        grad_matrix[j, o] += reg_gradient[j, o]
    end

    return G .= vec(grad_matrix)
end

"""
    PoissonRegressionEmission

A Poisson regression model.

# Fields
- `input_dim::Int`: Dimensionality of the input data.
- `output_dim::Int`: Dimensionality of the output data.
- `include_intercept::Bool`: Whether to include a regression intercept.
- `β::AbstractMatrix{<:Real}`: The regression coefficients matrix.
- `λ::Real;`: L2 Regularization parameter.
"""
mutable struct PoissonRegressionEmission{T<:Real, M<:AbstractMatrix{T}} <: RegressionEmission
    input_dim::Int
    output_dim::Int
    β::M
    include_intercept::Bool
    λ::T
end

"""
    PoissonRegressionEmission(Args)

Create a Poisson regression emission model.

# Arguments
    - `input_dim::Int`: Dimensionality of the input data.
    - `output_dim::Int`: Dimensionality of the output data.
    - `include_intercept::Bool`: Whether to include a regression intercept.
    - `β::AbstractMatrix{<:Real}`: The regression coefficients matrix.
    - `λ::Real;`: L2 Regularization parameter.

# Returns
    - `model::PoissonRegressionEmission`: The Poisson regression emission model.
"""
function PoissonRegressionEmission(; 
    input_dim::Int,
    output_dim::Int,
    include_intercept::Bool,
    β::AbstractMatrix,
    λ::Real,
)

    if !check_same_type(β[1], λ)
        error("β and λ must be of the same element type. Got $(eltype(β[1])) and $(eltype(λ))")
    end

    return PoissonRegressionEmission(input_dim, output_dim, β, include_intercept, λ)
end

"""
    Random.rand(rng::AbstractRNG, model::PoissonRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})

Generate samples from a Poisson regression model.

# Arguments
    - `rng::AbstractRNG`: The seed.
    - `model::PoissonRegressionEmission`: Poisson regression model.
    - `Φ::AbstractMatrix{<:Real}`: Design matrix of shape `(n, input_dim)`.

# Returns
- `Y::AbstractMatrix{<:Real}`: Matrix of samples of shape `(n, output_dim)`.
"""
function Random.rand(rng::AbstractRNG, model::PoissonRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})
    Φ = size(Φ, 2) == 1 ? reshape(Φ, 1, :) : Φ
    if model.include_intercept && size(Φ, 2) == length(model.β) - 1
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end
    Y = rand.(rng, Poisson.(exp.(Φ * model.β)))
    return Float64.(reshape(Y, :, 1))
end

"""
    Random.rand(model::PoissonRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})
    Random.rand(rng::AbstractRNG, model::PoissonRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})

Generate samples from a Poisson regression emission model.
"""
function Random.rand(model::PoissonRegressionEmission, Φ::Union{Matrix{<:Real},Vector{<:Real}})
    return rand(Random.default_rng(), model, Φ)
end

"""
    loglikelihood(
        model::PoissonRegressionEmission,
        Φ::AbstractMatrix{T1},
        Y::AbstractMatrix{T2},
        w::AbstractVector{T3}=ones(size(Y, 1))) where {T1<:Real, T2<:Real, T3<:Real}

Calculate the log-likelihood of a Poisson regression model.
"""
function loglikelihood(
    model::PoissonRegressionEmission,
    Φ::AbstractMatrix{T1},
    Y::AbstractMatrix{T2},
    w::Union{Nothing,AbstractVector{<:Real}} = nothing
) where {T1<:Real, T2<:Real}
    f, n = size(Φ)
    d, _ = size(Y)
    β = model.β
    T = promote_type(T1, T2)

    if w === nothing
        w = ones(T, n)
    elseif eltype(w) !== T
        error("weights must be Vector{$T}; Got Vector{$(eltype(w))}")
    end

     if model.include_intercept
        if size(Φ, 1) == size(β, 1) - 1
            Φ = vcat(ones(T, 1, n), Φ)
        elseif size(Φ, 1) != size(β, 1)
            error("Φ has incompatible size with model.β: got $(size(Φ)), expected $(size(β, 1)) rows.")
        end
    end 

    ll = Vector{T}(undef, n)
    for t in 1:n
        acc = zero(T)
        for o in 1:d   
            z = clamp(dot(β[:, o], Φ[:, t]), -20, 20)
            acc += Y[o, t] * z - exp(z) - loggamma(Y[o, t] + one(Int))
        end 
        ll[t] = w[t] * acc
    end 
    
    return ll
end

"""
    objective(
    opt::RegressionOptimization{<:PoissonRegressionEmission},
    β_vec::AbstractVector{T}) where {T<:Real}

Define the objective function for a Poisson regression emission.
"""
function objective(
    opt::RegressionOptimization{<:PoissonRegressionEmission}, 
    β_vec::AbstractVector{T}
) where {T<:Real}
    β = reshape(β_vec, opt.β_shape) 
    f, n = size(opt.X)
    d, _ = size(opt.y)
    total = zero(T)

    for t in 1:n
        for o in 1:d
            z = clamp(dot(β[:, o], opt.X[:, t]), -30, 30)
            total -= opt.w[t] * (opt.y[o, t] * z - exp(z) - loggamma(opt.y[o, t] + one(Int)))
        end 
    end 

    total += calc_regularization(β, opt.model.λ, opt.model.include_intercept)
    return total
end

"""
    objective_gradient!(
        G::AbstractVector{T},
        opt::RegressionOptimization{<:PoissonRegressionEmission},
        β_vec::AbstractVector{T}) where {T<:Real}

Define the objective gradient for a Poisson regression emission.
"""
function objective_gradient!(
    G::AbstractVector{T},
    opt::RegressionOptimization{<:PoissonRegressionEmission},
    β_vec::AbstractVector{T},
) where {T<:Real}
    β_mat = vec_to_matrix(β_vec, opt.β_shape)  # shape: f × d
    f, n = size(opt.X)
    d, _ = size(opt.y)
    gradient_matrix = zeros(T, f, d)

    for t in 1:n
        w_t = opt.w[t]
        for o in 1:d
            diff = opt.y[o, t] - exp(clamp(dot(β_mat[:, o], opt.X[:, t]), -30, 30))
            for j in 1:f
                gradient_matrix[j, o] -= w_t * opt.X[j, t] * diff
            end
        end
    end

    # calc gradient of penalty
    reg_gradient = calc_regularization_gradient(β_mat, opt.model.λ, opt.model.include_intercept)
    for j in 1:f, o in 1:d
        gradient_matrix[j, o] += reg_gradient[j, o]
    end

    return G .= vec(gradient_matrix)
end

# Unified fit! function for all regression emissions
# We use T1, T2, and T3 because in some emission models (ie Poisson, Bernoulli) X and y could be different types.

"""
    fit!(
        model::RegressionEmission,
        X::AbstractMatrix{T1},
        y::AbstractMatrix{T2},
        w::AbstractVector{T3}=ones(size(y, 1))) where {T1<:Real, T2<:Real, T3<:Real}

Fit a regression emission model give input data `X`, output data `y`, and weights `w`.

# Arguments
    - `model::RegressionEmission`: A regression emission model.
    - `X::AbstractMatrix{<:Real}:`: Input data.
    - `y::AbstractMatrix{<:Real}`: Output data.
    - `w::AbstractVector{<:Real}`: Weights to define each point's contribution to the fit.

# Returns
    - `model::RegressionEmission`: The regression model with the newly updated parameters.

"""
function fit!(
    model::RegressionEmission,
    X::AbstractMatrix{T1},
    y::AbstractMatrix{T2},
    w::AbstractVector{T3}=ones(size(y, 2)),
) where {T1<:Real, T2<:Real, T3<:Real}

    # must transpose x and y to go with column wise iteration instead of row 
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