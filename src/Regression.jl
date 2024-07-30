export GaussianRegression, BernoulliRegression, PoissonRegression, fit!, loglikelihood, least_squares, update_variance!, predict

# below used in notebooks and unit tests
export surrogate_loglikelihood, surrogate_loglikelihood_gradient!


# abstract regression type
abstract type Regression end

"""
    GaussianRegression(β::Matrix{<:Real}, Σ::Matrix{<:Real}, num_features::Int, num_targets::Int, include_intercept::Bool, λ::Float64)

A struct representing a Gaussian regression model.

# Fields
- `β::Matrix{<:Real}`: Coefficients of the regression model. Has shape (num_features, num_targets). Column one is coefficients for target one, etc. The first row is the intercept term, if included.
- `Σ::Matrix{<:Real}`: Covariance matrix of the model.
- `num_features::Int`: Number of features in the model.
- `num_targets::Int`: Number of targets in the model.
- `include_intercept::Bool`: Whether to include an intercept term in the model.
- `λ::Float64`: Regularization parameter for the model.

# Constructors
- `GaussianRegression(; num_features::Int, num_targets::Int, include_intercept::Bool=true, λ::Float64=0.0)`
- `GaussianRegression(β::Matrix{<:Real}, Σ::Matrix{<:Real}; num_features::Int, num_targets::Int, include_intercept::Bool=true, λ::Float64=0.0)`

# Examples
```julia
model = GaussianRegression(num_features=2, num_targets=1)
model = GaussianRegression(ones(3, 1), ones(1, 1), num_features=2, num_targets=1)
model = GaussianRegression(ones(2,1), ones(1, 1), num_features=2, num_targets=1, include_intercept=false, λ=0.1)
```
"""
mutable struct GaussianRegression <: Regression
    num_features::Int
    num_targets::Int
    β::Matrix{<:Real} # coefficient matrix of the model. Shape num_features by num_targets. Column one is coefficients for target one, etc. The first row are the intercept terms, if included. 
    Σ::Matrix{<:Real} # covariance matrix of the model 
    include_intercept::Bool # whether to include an intercept term; if true, the first column of β is assumed to be the intercept/bias
    λ::Float64 # regularization parameter
  
    function GaussianRegression(; num_features::Int, num_targets::Int, include_intercept::Bool = true, λ::Float64=0.0)
        if include_intercept
            input_dim = num_features + 1
        else
            input_dim = num_features
        end

        new(num_features, num_targets, ones(input_dim, num_targets), Matrix{Float64}(I, num_targets, num_targets), include_intercept, λ)
    end
    
    function GaussianRegression(β::Matrix{<:Real}, Σ::Matrix{<:Real}; num_features::Int, num_targets::Int, include_intercept::Bool = true, λ::Float64=0.0)
        if include_intercept
            input_dim = num_features + 1
        else
            input_dim = num_features
        end

        @assert size(β) == (input_dim, num_targets)
        @assert size(Σ) == (num_targets, num_targets)

        new(num_features, num_targets, β, Σ, include_intercept, λ)
    end
end

"""
    sample(model::GaussianRegression, X::Matrix{<:Real})

Sample from a Gaussian regression model.

# Arguments
- `model::GaussianRegression`: Gaussian regression model.
- `X::Matrix{<:Real}`: Design matrix. Each row is an observation.

# Returns
- `y::Matrix{<:Real}`: Sampled response matrix. Each row is a sample.

# Examples
```julia
model = GaussianRegression(num_features=2, num_targets=1)
X = rand(100, 2)
y = sample(model, X)
```
"""
function sample(model::GaussianRegression, X::Matrix{<:Real})
    # confirm that the model has been fit
    @assert !all(model.β .== 0) "Coefficient matrix is all zeros. Did you forget to initialize?"
    @assert isposdef(model.Σ) "Covariance matrix is not positive definite. Did you forget to initialize?"
    # add intercept if specified
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end
    return X * model.β + rand(MvNormal(zeros(model.num_targets), model.Σ), size(X, 1))'
end


"""
    loglikelihood(model::GaussianRegression, X::Matrix{<:Real}, y::Matrix{<:Real})

Calculate the log-likelihood of a Gaussian regression model.

# Arguments
- `model::GaussianRegression`: Gaussian regression model.
- `X::Matrix{<:Real}`: Design matrix. Each row is an observation.
- `y::Matrix{<:Real}`: Response matrix. Each row is a response vector.

# Examples
```julia
model = GaussianRegression(num_features=2, num_targets=1)
X = rand(100, 2)
y = X * [0.1, 0.2] + 0.1 * randn(100)
y = reshape(y, 100, 1)
loglikelihood(model, X, y)
```
"""
function loglikelihood(model::GaussianRegression, X::Matrix{<:Real}, y::Matrix{<:Real})
    # confirm dimensions of X and y are correct
    @assert size(X, 1) == size(y, 1) "Number of rows (number of observations) in X and y must be equal."
    @assert size(y, 2) == model.num_targets "Number of columns in y must be equal to the number of targets in the model."
    @assert size(X, 2) == model.num_features "Number of columns in X must be equal to the number of features in the model."

    # confirm that the model has been fit
    @assert !all(model.β .== 0) "Coefficient matrix is all zeros. Did you forget to initialize?"
    
    @assert isposdef(model.Σ) "Covariance matrix is not positive definite. Did you forget to initialize?"

    # add intercept if specified
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end

    # calculate inverse of covariance matrix
    Σ_inv = inv(model.Σ)

    # calculate log likelihood
    residuals = y - X * model.β

    log_likelihood = -0.5 * size(X, 1) * size(X, 2) * log(2π) - 0.5 * size(X, 1) * logdet(model.Σ) - 0.5 * sum(residuals .* (Σ_inv * residuals')')

    return log_likelihood
end

   
"""
    surrogate_loglikelihood(model::GaussianRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))

Calculate the (weighted) least squares objective function for a Gaussian regression model, with an L2 penalty on the coefficients.

# Arguments
- `model::GaussianRegression`: Gaussian regression model.
- `X::Matrix{<:Real}`: Design matrix. Each row is an observation.
- `y::Matrix{<:Real}`: Response matrix. Each row is a response vector.
- `w::Vector{Float64}`: Weights for the observations.

# Examples
```julia
model = GaussianRegression(num_features=2, num_targets=1)
X = rand(100, 2)
y = X * [0.1, 0.2] + 0.1 * randn(100)
y = reshape(y, 100, 1)
least_squares(model, X, y)

model = GaussianRegression(num_features=2, num_targets=1)
X = rand(100, 2)
y = X * [0.1, 0.2] + 0.1 * randn(100)
y = reshape(y, 100, 1)
w = rand(100)
least_squares(model, X, y, w)
```
"""
function surrogate_loglikelihood(model::GaussianRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))
    # assume covariance is the identity, so the log likelihood is just the negative squared error. Ignore loglikelihood terms that don't depend on β.

    # confirm dimensions of X and y are correct
    @assert size(X, 1) == size(y, 1) "Number of rows (number of observations) in X and y must be equal."
    @assert size(y, 2) == model.num_targets "Number of columns in y must be equal to the number of targets in the model."
    @assert size(X, 2) == model.num_features "Number of columns in X must be equal to the number of features in the model."

    # confirm that the model has been fit
    @assert !all(model.β .== 0) "Coefficient matrix is all zeros. Did you forget to initialize?"
    @assert isposdef(model.Σ) "Covariance matrix is not positive definite. Did you forget to initialize?"

    # add intercept if specified
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end


    # calculate log likelihood
    residuals = y - X * model.β


        
    # reshape w for broadcasting
    w = reshape(w, (length(w), 1))

    log_likelihood = -0.5 * sum(broadcast(*, w, residuals.^2)) - (model.λ * sum(model.β.^2))

    return log_likelihood
end

"""
    surrogate_loglikelihood_gradient!(G::Matrix{<:Real}, model::GaussianRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))

Calculate the gradient of the (weighted) least squares objective function for a Gaussian regression model, with an L2 penalty on the coefficients.

# Examples
```julia
model = GaussianRegression(num_features=2, num_targets=1)
X = rand(100, 2)
y = X * [0.1, 0.2] + 0.1 * randn(100)
y = reshape(y, 100, 1)
G = zeros(3, 1)
surrogate_loglikelihood_gradient!(G, model, X, y)
```
"""
function surrogate_loglikelihood_gradient!(G::Matrix{<:Real}, model::GaussianRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))
    # WARNING: asserts may slow down computation. Remove later?

    # confirm dimensions of X and y are correct
    @assert size(X, 1) == size(y, 1) "Number of rows (number of observations) in X and y must be equal."
    @assert size(y, 2) == model.num_targets "Number of columns in y must be equal to the number of targets in the model."
    @assert size(X, 2) == model.num_features "Number of columns in X must be equal to the number of features in the model."

    # confirm the size of w is correct
    @assert length(w) == size(y, 1) "Length of w must be equal to the number of observations in y."

    # confirm that the model has been fit
    @assert !all(model.β .== 0) "Coefficient matrix is all zeros. Did you forget to initialize?"
    @assert isposdef(model.Σ) "Covariance matrix is not positive definite. Did you forget to initialize?"

    # add intercept if specified
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end


    # calculate log likelihood
    residuals = y - X * model.β

    
    G .=  X' * Diagonal(w) * residuals - (2*model.λ*model.β)
    
    

end

"""
    update_variance!(model::GaussianRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))

Update the (weighted) variance of a Gaussian regression model. Uses the biased estimator.

# Arguments
- `model::GaussianRegression`: Gaussian regression model.
- `X::Matrix{<:Real}`: Design matrix. Each row is an observation.
- `y::Vector{Float64}`: Response vector. Each row is a response vector.
- `w::Vector{Float64}`: Weights for the observations.

# Examples
```julia
model = GaussianRegression(num_features=2, num_targets=1)
X = rand(100, 2)
y = X * [0.1, 0.2] + 0.1 * randn(100)
y = reshape(y, 100, 1)
update_variance!(model, X, y)
```
"""
function update_variance!(model::GaussianRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))
    # WARNING: asserts may slow down computation. Remove later?

    # confirm dimensions of X and y are correct
    @assert size(X, 1) == size(y, 1) "Number of rows (number of observations) in X and y must be equal."
    @assert size(y, 2) == model.num_targets "Number of columns in y must be equal to the number of targets in the model."
    @assert size(X, 2) == model.num_features "Number of columns in X must be equal to the number of features in the model."

    # confirm the size of w is correct
    @assert length(w) == size(y, 1) "Length of w must be equal to the number of observations in y."

    # confirm that the model has been fit
    @assert !all(model.β .== 0) "Coefficient matrix is all zeros. Did you forget to initialize?"
    @assert isposdef(model.Σ) "Covariance matrix is not positive definite. Did you forget to initialize?"

    # add intercept if specified
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end


    residuals = y - X * model.β
    
    
    model.Σ = (residuals' * Diagonal(w) * residuals) / size(X, 1)

    # ensure rounding errors are not causing the covariance matrix to be non-positive definite
    model.Σ = stabilize_covariance_matrix(model.Σ)

   
    
end

"""
    fit!(model::GaussianRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))

Fit a Gaussian regression model using maximum likelihood estimation.

# Arguments
- `model::GaussianRegression`: Gaussian regression model.
- `X::Matrix{<:Real}`: Design matrix. Each row is an observation.
- `y::Matrix{<:Real}`: Response matrix. Each row is a response vector.
- `w::Vector{Float64}`: Weights for the observations.

# Examples
```julia
model = GaussianRegression(num_features=2, num_targets=1)
X = rand(100, 2)
y = rand(100, 1)
fit!(model, X, y)
```
"""
function fit!(model::GaussianRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))
    # confirm dimensions of X and y are correct
    @assert size(X, 1) == size(y, 1) "Number of rows (number of observations) in X and y must be equal."
    @assert size(y, 2) == model.num_targets "Number of columns in y must be equal to the number of targets in the model."
    @assert size(X, 2) == model.num_features "Number of columns in X must be equal to the number of features in the model."

    # confirm the size of w is correct
    @assert length(w) == size(y, 1) "Length of w must be equal to the number of observations in y."

    # confirm that the model has been fit
    @assert !all(model.β .== 0) "Coefficient matrix is all zeros. Did you forget to initialize?"
    @assert isposdef(model.Σ) "Covariance matrix is not positive definite. Did you forget to initialize?"
    

    
    # minimize objective
    function objective(β)
        log_likelihood = surrogate_loglikelihood(
            GaussianRegression(
                β, 
                model.Σ, 
                num_features=model.num_features, 
                num_targets=model.num_targets, 
                include_intercept=model.include_intercept), 
            X, y, w)
        return -log_likelihood / size(X, 1)
    end

    function objective_grad!(G, β)
        surrogate_loglikelihood_gradient!(
            G, 
            GaussianRegression(
                β, 
                model.Σ, 
                num_features=model.num_features, 
                num_targets=model.num_targets, 
                include_intercept=model.include_intercept), 
            X, y, w)
        # make it the gradient of the negative log likelihood
        G .= -G / size(X, 1)
    end


    result = optimize(objective, objective_grad!, model.β, LBFGS())
    # update parameters
    model.β = result.minimizer

    
    update_variance!(model, X, y, w)
end



"""
    BernoulliRegression(β::Vector{Float64}, include_intercept::Bool, λ::Float64=0.0)

# Fields
- `β::Vector{Float64}`: Coefficients of the regression model.
- `include_intercept::Bool`: Whether to include an intercept term in the model.
- `λ::Float64`: Regularization parameter for the model.

# Constructors
- `BernoulliRegression(; include_intercept::Bool = true, λ::Float64=0.0)`
- `BernoulliRegression(β::Vector{Float64}, include_intercept::Bool, λ::Float64=0.0)`

# Examples
```julia
model = BernoulliRegression()
model = BernoulliRegression(include_intercept=false, λ=0.1)
model = BernoulliRegression([0.1, 0.2], true, 0.1)
```
"""
mutable struct BernoulliRegression <: Regression
    β::Vector{Float64}
    include_intercept::Bool
    λ::Float64
    # Empty constructor
    function BernoulliRegression(; include_intercept::Bool = true, λ::Float64=0.0) 
        @assert λ >= 0.0 "Regularization parameter must be non-negative."
        new(Vector{Float64}(), include_intercept, λ)
    end
    # Parametric Constructor
    function BernoulliRegression(β::Vector{Float64}, include_intercept::Bool, λ::Float64=0.0)
        @assert λ >= 0.0 "Regularization parameter must be non-negative."
        new(β, include_intercept, λ)
    end
end

"""
    loglikelihood(model::BernoulliRegression, X::Matrix{<:Real}, y::Union{Vector{Float64}, BitVector}, w::Vector{Float64}=ones(length(y))

Calculate the log-likelihood of a Bernoulli regression model.

# Arguments
- `model::BernoulliRegression`: Bernoulli regression model.
- `X::Matrix{<:Real}`: Design matrix.
- `y::Union{Vector{Float64}, BitVector}`: Response vector.
- `w::Vector{Float64}`: Weights for the observations.

# Examples
```julia
model = BernoulliRegression()
X = rand(100, 2)
y = rand(Bool, 100)
loglikelihood(model, X, y)
```
"""
function loglikelihood(model::BernoulliRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified and not already included
    if model.include_intercept && size(X, 2) == length(model.β) - 1 
        X = hcat(ones(size(X, 1)), X)
    end
    # calculate log likelihood
    p = logistic.(X * model.β)
    
    return sum(w .* (y .* log.(p) .+ (1 .- y) .* log.(1 .- p)))
end

"""
    loglikelihood(model::BernoulliRegression, X::Vector{Float64}, y::Union{Float64, Bool, Int64}, w::Float64=1.0)

Calculate the log-likelihood of a single observation of a Bernoulli regression model.

# Arguments
- `model::BernoulliRegression`: Bernoulli regression model.
- `X::Vector{Float64}`: Design vector.
- `y::Union{Float64, Bool, Int64}`: Response value.
- `w::Float64`: Weight for the observation.

# Examples
```julia
model = BernoulliRegression()
X = rand(2)
y = rand(Bool)
loglikelihood(model, X, y)
```
"""
function loglikelihood(model::BernoulliRegression, X::Vector{Float64}, y::Union{Float64, Bool, Int64}, w::Float64=1.0)
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified
    if model.include_intercept && length(X) == length(model.β) - 1
        X = vcat(1.0, X)
    end
    # calculate log likelihood
    p = logistic.(X' * model.β) # use stats fun for this
    # convert y if neccesary
    y = convert(Float64, y)
    return sum(w .* (y .* log.(p) .+ (1 .- y) .* log.(1 .- p))) 
end

"""
    gradient!(grad::Vector{Float64}, model::BernoulliRegression, X::Matrix{<:Real}, y::Union{Vector{Float64}, BitVector}, w::Vector{Float64}=ones(length(y))

Calculate the gradient of the negative log-likelihood function for a Bernoulli regression model. 

# Arguments
- `grad::Vector{Float64}`: Gradient of the negative log-likelihood function.
- `model::BernoulliRegression`: Bernoulli regression model.
- `X::Matrix{<:Real}`: Design matrix.
- `y::Union{Vector{Float64}, BitVector}`: Response vector.
- `w::Vector{Float64}`: Weights for the observations.
"""
function gradient!(grad::Vector{Float64}, model::BernoulliRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))
    # confirm the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified
    if model.include_intercept && size(X, 2) == length(model.β) - 1 
        X = hcat(ones(size(X, 1)), X)
    end
    # calculate probs 
    p = logistic.(X * model.β)
    # convert y if necessary
    # y = convert(Vector{Float64}, y)
    # calculate gradient
    grad .= -(X' * (w .* (y .- p))) + 2 * model.λ * model.β
end

"""
    fit!(model::BernoulliRegression, X::Matrix{<:Real}, y::Union{Vector{Float64}, BitVector}, w::Vector{Float64}=ones(length(y))

Fit a Bernoulli regression model using maximum likelihood estimation.

# Arguments
- `model::BernoulliRegression`: Bernoulli regression model.
- `X::Matrix{<:Real}`: Design matrix.
- `y::Union{Vector{Float64}, BitVector}`: Response vector.
- `w::Vector{Float64}`: Weights for the observations.

# Examples
```julia
model = BernoulliRegression()
X = rand(100, 2)
y = rand(Bool, 100)
fit!(model, X, y)

model = BernoulliRegression()
X = rand(100, 2)
y = rand(Bool, 100)
w = rand(100)
fit!(model, X, y, w)
```
"""
function fit!(model::BernoulliRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))
    # add intercept if specified
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end
    # get number of parameters
    p = size(X, 2)
    # initialize parameters
    model.β = rand(p)
    # convert y if necessary
    # y = convert(Vector{Float64}, y)
    # minimize objective
    objective(β) = -loglikelihood(BernoulliRegression(β, true, model.λ), X, y, w) + (model.λ * sum(β.^2))
    objective_grad!(β, g) = gradient!(g, BernoulliRegression(β, true, model.λ), X, y, w) # troubleshoot this
    result = optimize(objective, model.β, LBFGS())
    # update parameters
    model.β = result.minimizer
end
    
"""
    mutable struct PoissonRegression <: Regression

# Fields
- `β::Vector{Float64}`: Coefficients of the regression model
- `include_intercept::Bool`: Whether to include an intercept term in the model

# Constructors
- `PoissonRegression(; include_intercept::Bool = true, λ::Float64=0.0)`
- `PoissonRegression(β::Vector{Float64}, include_intercept::Bool, λ::Float64=0.0)`

# Examples
```julia
model = PoissonRegression()
model = PoissonRegression(include_intercept=false, λ=0.1)
model = PoissonRegression([0.1, 0.2], true, 0.1)
```
"""
mutable struct PoissonRegression <: Regression
    β::Vector{Float64}
    include_intercept::Bool
    λ::Float64
    # Empty constructor
    function PoissonRegression(; include_intercept::Bool = true, λ::Float64=0.0) 
        @assert λ >= 0.0 "Regularization parameter must be non-negative."
        new(Vector{Float64}(), include_intercept, λ)
    end
    # Parametric Constructor
    function PoissonRegression(β::Vector{Float64}, include_intercept::Bool, λ::Float64=0.0)
        @assert λ >= 0.0 "Regularization parameter must be non-negative."
        new(β, include_intercept, λ)
    end
end

"""
    loglikelihood(model::PoissonRegression, X::Matrix{<:Real}, y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(y)))

Calculate the log-likelihood of a Poisson regression model.

# Arguments
- `model::PoissonRegression`: Poisson regression model
- `X::Matrix{<:Real}`: Design matrix
- `y::Union{Vector{Float64}, Vector{Int64}}`: Response vector
- `w::Vector{Float64}`: Weights for the observations

# Examples
```julia
model = PoissonRegression()
X = rand(100, 2)
y = rand(Poisson(1), 100)
loglikelihood(model, X, y)
```
"""
function loglikelihood(model::PoissonRegression, X::Matrix{<:Real}, y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(y)))
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified
    if model.include_intercept && size(X, 2) == length(model.β) - 1
        X = hcat(ones(size(X, 1)), X)
    end
    # calculate log likelihood
    λ = exp.(X * model.β)
    # convert y if necessary
    y = convert(Vector{Float64}, y)
    return sum(w .* (y .* log.(λ) .- λ .- loggamma.(Int.(y) .+ 1)))
end

"""
    loglikelihood(model::PoissonRegression, X::Vector{Float64}, y::Union{Float64, Int64}, w::Float64=1.0)

Calculate the log-likelihood of a single observation of a Poisson regression model.

# Arguments
- `model::PoissonRegression`: Poisson regression model
- `X::Vector{Float64}`: Design vector
- `y::Union{Float64, Int64}`: Response value
- `w::Float64`: Weight for the observation

# Examples
```julia
model = PoissonRegression()
X = rand(2)
y = rand(Poisson(1))
loglikelihood(model, X, y)
```
"""
function loglikelihood(model::PoissonRegression, X::Vector{Float64}, y::Union{Float64, Int64}, w::Float64=1.0)
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified
    if model.include_intercept && size(X, 2) == length(model.β) - 1
        X = vcat(1.0, X)
    end
    # calculate log likelihood
    λ = exp.(X' * model.β)
    # convert y if necessary
    y = convert(Float64, y)
    return sum(w .* (y .* log.(λ) .- λ .- log.(factorial.(Int.(y))))) 
end

"""
    gradient!(grad::Vector{Float64}, model::PoissonRegression, X::Matrix{<:Real}, y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(y)))

Calculate the gradient of the negative log-likelihood function for a Poisson regression model.

# Arguments
- `grad::Vector{Float64}`: Gradient of the negative log-likelihood function
- `model::PoissonRegression`: Poisson regression model
- `X::Matrix{<:Real}`: Design matrix
- `y::Union{Vector{Float64}, Vector{Int64}}`: Response vector
- `w::Vector{Float64}`: Weights for the observations

# Examples
```julia
model = PoissonRegression()
X = rand(100, 2)
y = rand(Poisson(1), 100)
G = zeros(2)
gradient!(G, model, X, y)
```
"""
function gradient!(grad::Vector{Float64}, model::PoissonRegression, X::Matrix{<:Real}, y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(y)))
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified
    if model.include_intercept && size(X, 2) == length(model.β) - 1
        X = hcat(ones(size(X, 1)), X)
    end
    # calculate the rate
    rate = exp.(X * model.β)
    # convert y if necessary
    y = convert(Vector{Float64}, y)
    # calculate gradient
    grad .= -X' * (Diagonal(w) * (y .- rate)) + (model.λ * 2 * model.β)
end

"""
    fit!(model::PoissonRegression, X::Matrix{<:Real}, y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(y)))

Fit a Poisson regression model using maximum likelihood estimation.

# Arguments
- `model::PoissonRegression`: Poisson regression model
- `X::Matrix{<:Real}`: Design matrix
- `y::Union{Vector{Float64}, Vector{Int64}}`: Response vector
- `w::Vector{Float64}`: Weights for the observations

# Examples
```julia
model = PoissonRegression()
X = rand(100, 2)
y = rand(Poisson(1), 100)
fit!(model, X, y)

model = PoissonRegression()
X = rand(100, 2)
y = rand(Poisson(1), 100)
w = rand(100)
fit!(model, X, y, w)
```
"""
function fit!(model::PoissonRegression, X::Matrix{<:Real}, y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(y)))
    # add intercept if specified
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end
    # get number of parameters
    p = size(X, 2)
    # initialize parameters
    model.β = rand(p)
    # convert y if necessary
    y = convert(Vector{Float64}, y)
    # minimize objective
    objective(β) = -loglikelihood(PoissonRegression(β, true, model.λ), X, y, w) + (model.λ * sum(β.^2))
    objective_grad!(β, g) = gradient!(g, PoissonRegression(β, true, model.λ), X, y, w)
    result = optimize(objective, model.β, LBFGS())
    # update parameters
    model.β = result.minimizer
end

