export GaussianRegression, BernoulliRegression, PoissonRegression, fit!, loglikelihood, least_squares, update_variance!, predict

# below used in notebooks and unit tests
export surrogate_loglikelihood, surrogate_loglikelihood_gradient!


# abstract regression type
abstract type Regression end

"""

"""
mutable struct GaussianRegression <: Regression
    num_features::Int
    num_targets::Int
    β::Matrix{Float64} # coefficient matrix of the model
    Σ::Matrix{Float64} # covariance matrix of the model 
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
    
    function GaussianRegression(β::Matrix{Float64}, Σ::Matrix{Float64}; num_features::Int, num_targets::Int, include_intercept::Bool = true, λ::Float64=0.0)
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
    predict(model::GaussianRegression, X::Matrix{Float64})

Predict the response variable using a Gaussian regression model. X should be 'observations' by 'features'.

# Returns
- A matrix of predictions (with shape 'observations' by 'predicted features'). 

# Examples
```julia

```
"""
function predict(model::GaussianRegression, X::Matrix{Float64})
    # confirm that the model has been fit
    @assert !all(model.β .== 0) "Coefficient matrix is all zeros. Did you forget to initialize?"
    @assert isposdef(model.Σ) "Covariance matrix is not positive definite. Did you forget to initialize?"
    # add intercept if specified
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end
    return X * model.β
end



"""
    loglikelihood(model::GaussianRegression, X::Matrix{Float64}, y::Vector{Float64})

Calculate the log-likelihood of a Gaussian regression model.

Args:
- `model::GaussianRegression`: Gaussian regression model
- `X::Matrix{Float64}`: Design matrix
- `y::Matrix{Float64}`: Target data (shape 'observations' by 'target features')

Example:
```julia
model = GaussianRegression()
X = rand(100, 2)
y = X * [0.1, 0.2] + 0.1 * randn(100)
loglikelihood(model, X, y)
```
"""
function loglikelihood(model::GaussianRegression, X::Matrix{Float64}, y::Matrix{Float64})
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

   

function surrogate_loglikelihood(model::GaussianRegression, X::Matrix{Float64}, y::Matrix{Float64}, w::Vector{Float64}=ones(size(y, 1)))
    # assume covariance is the identity, so the log likelihood is just the negative squared error. Ignore loglikelihood terms that don't depend on β.

    # WARNING: asserts may slow down computation. Remove later?

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

"""
function surrogate_loglikelihood_gradient!(G::Matrix{Float64}, model::GaussianRegression, X::Matrix{Float64}, y::Matrix{Float64}, w::Vector{Float64}=ones(size(y, 1)))
    # WARNING: asserts may slow down computation. Remove later?

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

    
    G .=  X' * Diagonal(w) * residuals - (2*model.λ*model.β)
    
    

end

"""
    update_variance!(model::GaussianRegression, X::Matrix{Float64}, y::Vector{Float64}, w::Vector{Float64}=ones(length(y)))

Update the (weighted) variance of a Gaussian regression model. Uses the biased estimator.

Args:
- `model::GaussianRegression`: Gaussian regression model
- `X::Matrix{Float64}`: Design matrix
- `y::Vector{Float64}`: Response vector
- `w::Vector{Float64}`: Weights for the observations

Example:
```julia
model = GaussianRegression()
X = rand(100, 2)
y = X * [0.1, 0.2] + 0.1 * randn(100)
update_variance!(model, X, y)

model = GaussianRegression()
X = rand(100, 2)
y = X * [0.1, 0.2] + 0.1 * randn(100)
w = rand(100)
update_variance!(model, X, y, w)
```
"""
function update_variance!(model::GaussianRegression, X::Matrix{Float64}, y::Matrix{Float64}, w::Vector{Float64}=ones(size(y), 1))
    # WARNING: asserts may slow down computation. Remove later?

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


    residuals = y - X * model.β
    
    
    model.Σ = (residuals' * Diagonal(w) * residuals) / size(X, 1)
    
end


function fit!(model::GaussianRegression, X::Matrix{Float64}, y::Matrix{Float64}, w::Vector{Float64}=ones(size(y, 1)))
    # confirm dimensions of X and y are correct
    @assert size(X, 1) == size(y, 1) "Number of rows (number of observations) in X and y must be equal."
    @assert size(y, 2) == model.num_targets "Number of columns in y must be equal to the number of targets in the model."
    @assert size(X, 2) == model.num_features "Number of columns in X must be equal to the number of features in the model."

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
        G .= -1 * G / size(X, 1)
    end

    # learning_rate = 0.01
    # max_iterations = 1000
    # tolerance = 1e-6

    # for i in 1:max_iterations
    #     # compute gradient
    #     G = similar(model.β)
    #     objective_grad!(G, model.β)


    #     println("Iteration: ", i, " Gradient: ", G)

    #     # update parameters
    #     model.β -= learning_rate * G

    #     # check convergence
    #     if norm(G) < tolerance
    #         break
    #     end
    # end

    # println("Optimized weights are: ", model.β)

    result = optimize(objective, objective_grad!, model.β, LBFGS())
    # update parameters
    model.β = result.minimizer

    
    update_variance!(model, X, y, w)
end

function fit_old!(model::GaussianRegression, X::Matrix{Float64}, y::Matrix{Float64}, w::Vector{Float64}=ones(length(y)))
    # add intercept if specified
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end
    # get number of parameters
    p = size(X, 2)
    # initialize parameters
    model.β = rand(p)
    model.σ² = 1.0
    # minimize objective
    objective(β) = -surrogate_loglikelihood(GaussianRegression(β, model.σ², true, model.λ), X, y, w)
    objective_grad!(G, β) = gradient!(G, GaussianRegression(β, model.σ², true, model.λ), X, y, w)

    result = optimize(objective, objective_grad!, model.β, LBFGS())
    # update parameters
    model.β = result.minimizer
    update_variance!(model, X, y, w)
end

"""
    BernoulliRegression(β::Vector{Float64}, include_intercept::Bool, λ::Float64=0.0)

Args:
- `β::Vector{Float64}`: Coefficients of the regression model
- `include_intercept::Bool`: Whether to include an intercept term in the model
- `λ::Float64`: Regularization parameter for the model

Constructors:
- `BernoulliRegression(; include_intercept::Bool = true, λ::Float64=0.0)`
- `BernoulliRegression(β::Vector{Float64}, include_intercept::Bool, λ::Float64=0.0)`

Example:
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
    loglikelihood(model::BernoulliRegression, X::Matrix{Float64}, y::Union{Vector{Float64}, BitVector}, w::Vector{Float64}=ones(length(y))

Calculate the log-likelihood of a Bernoulli regression model.

Args:
- `model::BernoulliRegression`: Bernoulli regression model
- `X::Matrix{Float64}`: Design matrix
- `y::Union{Vector{Float64}, BitVector}`: Response vector
- `w::Vector{Float64}`: Weights for the observations

Example:
```julia
model = BernoulliRegression()
X = rand(100, 2)
y = rand(Bool, 100)
loglikelihood(model, X, y)
```
"""
function loglikelihood(model::BernoulliRegression, X::Matrix{Float64}, y::Union{Vector{Float64}, Vector{Int64}, BitVector}, w::Vector{Float64}=ones(length(y)))
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified and not already included
    if model.include_intercept && size(X, 2) == length(model.β) - 1 
        X = hcat(ones(size(X, 1)), X)
    end
    # calculate log likelihood
    p = logistic.(X * model.β)
    # convert y if neccesary
    y = convert(Vector{Float64}, y)
    return sum(w .* (y .* log.(p) .+ (1 .- y) .* log.(1 .- p)))
end

"""
    loglikelihood(model::BernoulliRegression, X::Vector{Float64}, y::Union{Float64, Bool, Int64}, w::Float64=1.0)

Calculate the log-likelihood of a single observation of a Bernoulli regression model.

Args:
- `model::BernoulliRegression`: Bernoulli regression model
- `X::Vector{Float64}`: Design vector
- `y::Union{Float64, Bool, Int64}`: Response value
- `w::Float64`: Weight for the observation

Example:
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
    gradient!(grad::Vector{Float64}, model::BernoulliRegression, X::Matrix{Float64}, y::Union{Vector{Float64}, BitVector}, w::Vector{Float64}=ones(length(y))

Calculate the gradient of the negative log-likelihood function for a Bernoulli regression model. 

Args:
- `grad::Vector{Float64}`: Gradient of the negative log-likelihood function
- `model::BernoulliRegression`: Bernoulli regression model
- `X::Matrix{Float64}`: Design matrix
- `y::Union{Vector{Float64}, BitVector}`: Response vector
- `w::Vector{Float64}`: Weights for the observations
"""
function gradient!(grad::Vector{Float64}, model::BernoulliRegression, X::Matrix{Float64}, y::Union{Vector{Float64}, Vector{Int64}, BitVector}, w::Vector{Float64}=ones(length(y)))
    # confirm the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified
    if model.include_intercept && size(X, 2) == length(model.β) - 1 
        X = hcat(ones(size(X, 1)), X)
    end
    # calculate probs 
    p = logistic.(X * model.β)
    # convert y if necessary
    y = convert(Vector{Float64}, y)
    # calculate gradient
    grad .= -(X' * (w .* (y .- p))) + 2 * model.λ * model.β
end

"""
    fit!(model::BernoulliRegression, X::Matrix{Float64}, y::Union{Vector{Float64}, BitVector}, w::Vector{Float64}=ones(length(y))

Fit a Bernoulli regression model using maximum likelihood estimation.

Args:
- `model::BernoulliRegression`: Bernoulli regression model
- `X::Matrix{Float64}`: Design matrix
- `y::Union{Vector{Float64}, BitVector}`: Response vector
- `w::Vector{Float64}`: Weights for the observations

Example:
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
function fit!(model::BernoulliRegression, X::Matrix{Float64}, y::Union{Vector{Float64}, Vector{Int64}, BitVector}, w::Vector{Float64}=ones(length(y)))
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
    objective(β) = -loglikelihood(BernoulliRegression(β, true, model.λ), X, y, w) + (model.λ * sum(β.^2))
    objective_grad!(β, g) = gradient!(g, BernoulliRegression(β, true, model.λ), X, y, w) # troubleshoot this
    result = optimize(objective, model.β, LBFGS())
    # update parameters
    model.β = result.minimizer
end
    
"""
    mutable struct PoissonRegression <: Regression

Args:
- `β::Vector{Float64}`: Coefficients of the regression model
- `include_intercept::Bool`: Whether to include an intercept term in the model

Constructors:
- `PoissonRegression(; include_intercept::Bool = true, λ::Float64=0.0)`
- `PoissonRegression(β::Vector{Float64}, include_intercept::Bool, λ::Float64=0.0)`

Example:
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
    loglikelihood(model::PoissonRegression, X::Matrix{Float64}, y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(y)))

Calculate the log-likelihood of a Poisson regression model.

Args:
- `model::PoissonRegression`: Poisson regression model
- `X::Matrix{Float64}`: Design matrix
- `y::Union{Vector{Float64}, Vector{Int64}}`: Response vector
- `w::Vector{Float64}`: Weights for the observations

Example:
```julia
model = PoissonRegression()
X = rand(100, 2)
y = rand(Poisson(1), 100)
loglikelihood(model, X, y)
```
"""
function loglikelihood(model::PoissonRegression, X::Matrix{Float64}, y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(y)))
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

Args:
- `model::PoissonRegression`: Poisson regression model
- `X::Vector{Float64}`: Design vector
- `y::Union{Float64, Int64}`: Response value
- `w::Float64`: Weight for the observation

Example:
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
    gradient!(grad::Vector{Float64}, model::PoissonRegression, X::Matrix{Float64}, y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(y)))

Calculate the gradient of the negative log-likelihood function for a Poisson regression model.

Args:
- `grad::Vector{Float64}`: Gradient of the negative log-likelihood function
- `model::PoissonRegression`: Poisson regression model
- `X::Matrix{Float64}`: Design matrix
- `y::Union{Vector{Float64}, Vector{Int64}}`: Response vector
- `w::Vector{Float64}`: Weights for the observations

Example:
```julia
model = PoissonRegression()
X = rand(100, 2)
y = rand(Poisson(1), 100)
G = zeros(2)
gradient!(G, model, X, y)
```
"""
function gradient!(grad::Vector{Float64}, model::PoissonRegression, X::Matrix{Float64}, y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(y)))
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
    fit!(model::PoissonRegression, X::Matrix{Float64}, y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(y)))

Fit a Poisson regression model using maximum likelihood estimation.

Args:
- `model::PoissonRegression`: Poisson regression model
- `X::Matrix{Float64}`: Design matrix
- `y::Union{Vector{Float64}, Vector{Int64}}`: Response vector
- `w::Vector{Float64}`: Weights for the observations

Example:
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
function fit!(model::PoissonRegression, X::Matrix{Float64}, y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(y)))
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

