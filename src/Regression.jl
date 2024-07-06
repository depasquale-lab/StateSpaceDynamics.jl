export GaussianRegression, BernoulliRegression, PoissonRegression, fit!, loglikelihood, least_squares, update_variance!

# abstract regression type
abstract type Regression end

"""
    mutable struct GaussianRegression <: Regression

Args:
- `β::Vector{Float64}`: Coefficients of the regression model
- `σ²::Float64`: Variance of the regression model
- `include_intercept::Bool`: Whether to include an intercept term in the model

Constructors:
- `GaussianRegression(; include_intercept::Bool = true, λ::Float64=0.0)`
- `GaussianRegression(β::Vector{Float64}, σ²::Float64, include_intercept::Bool, λ::Float64=0.0)`

Example:
```julia
model = GaussianRegression()
model = GaussianRegression(include_intercept=false, λ=0.1)
model = GaussianRegression([0.1, 0.2], 0.1, true, 0.1)
```
"""
mutable struct GaussianRegression <: Regression
    β::Vector{Float64} # coefficients of the model
    σ²::Float64 # variance of the model
    include_intercept::Bool # whether to include an intercept term
    λ::Float64 # regularization parameter
  
    function GaussianRegression(; include_intercept::Bool = true, λ::Float64=0.0)
        @assert λ >= 0.0 "Regularization parameter must be non-negative."
        new(Vector{Float64}(), 0.0, include_intercept, λ)
    end
    
    function GaussianRegression(β::Vector{Float64}, σ²::Float64, include_intercept::Bool, λ::Float64=0.0)
        @assert λ >= 0.0 "Regularization parameter must be non-negative."
        new(β, σ², include_intercept, λ)
    end
end


"""
    loglikelihood(model::GaussianRegression, X::Matrix{Float64}, y::Vector{Float64})

Calculate the log-likelihood of a Gaussian regression model.

Args:
- `model::GaussianRegression`: Gaussian regression model
- `X::Matrix{Float64}`: Design matrix
- `y::Vector{Float64}`: Response vector

Example:
```julia
model = GaussianRegression()
X = rand(100, 2)
y = X * [0.1, 0.2] + 0.1 * randn(100)
loglikelihood(model, X, y)
```
"""
function loglikelihood(model::GaussianRegression, X::Matrix{Float64}, y::Vector{Float64})
    # confirm that the model has been fit
    @assert !isempty(model.β) && model.σ² != 0.0 "Model parameters not initialized, please call fit! first."
    # add intercept if specified
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end
    # calculate log likelihood
    residuals = y - X * model.β
    n = length(y)
    -0.5 * n * log(2π * model.σ²) - (0.5 / model.σ²) * sum(residuals.^2)
end

"""
    loglikelihood(model::GaussianRegression, X::Vector{Float64}, y::Float64)

Calculate the log-likelihood of a single observation of Gaussian regression model.

Args:
- `model::GaussianRegression`: Gaussian regression model
- `X::Matrix{Float64}`: Design matrix
- `y::Vector{Float64}`: Response vector

Example:
```julia
model = GaussianRegression()
X = rand(2)
y = X * [0.1, 0.2] + 0.1 * randn()
loglikelihood(model, X, y)
```
"""
function loglikelihood(model::GaussianRegression, X::Vector{Float64}, y::Float64)
    # confirm that the model has been fit
    @assert !isempty(model.β) && model.σ² != 0.0 "Model parameters not initialized, please call fit! first."
    # add intercept if specified
    if model.include_intercept
        X = vcat(1.0, X)
    end
    # calculate log likelihood
    residuals = y - (X' * model.β)
    n = length(y)
    -0.5 * n * log(2π * model.σ²) - (0.5 / model.σ²) * sum(residuals.^2)
end

"""
    least_squares(model::GaussianRegression, X::Matrix{Float64}, y::Vector{Float64}, w::Vector{Float64}=ones(length(y)))

Calculate the (weighted) least squares objective function for a Gaussian regression model.

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
least_squares(model, X, y)

model = GaussianRegression()
X = rand(100, 2)
y = X * [0.1, 0.2] + 0.1 * randn(100)
w = rand(100)
least_squares(model, X, y, w)
```
"""
function least_squares(model::GaussianRegression, X::Matrix{Float64}, y::Vector{Float64}, w::Vector{Float64}=ones(length(y)))
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    residuals =  y - (X * model.β)
    return sum(w.*(residuals.^2)) + (model.λ * sum(model.β.^2))
end

"""
    gradient!(G::Vector{Float64}, model::GaussianRegression, X::Matrix{Float64}, y::Vector{Float64}, w::Vector{Float64}=ones(length(y)))

Calculate the gradient of the least squares objective function for a Gaussian regression model.

    
Args:
- `G::Vector{Float64}`: Gradient of the objective function
- `model::GaussianRegression`: Gaussian regression model
- `X::Matrix{Float64}`: Design matrix
- `y::Vector{Float64}`: Response vector
- `w::Vector{Float64}`: Weights for the observations

Example:
```julia
model = GaussianRegression()
X = rand(100, 2)
y = X * [0.1, 0.2] + 0.1 * randn(100)
G = zeros(2)
gradient!(G, model, X, y)
```
"""
function gradient!(G::Vector{Float64}, model::GaussianRegression, X::Matrix{Float64}, y::Vector{Float64}, w::Vector{Float64}=ones(length(y)))
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # calculate gradient
    residuals = y - X * model.β
    G .= (-2 * X' * Diagonal(w) * residuals) + (2*model.λ*model.β)
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
function update_variance!(model::GaussianRegression, X::Matrix{Float64}, y::Vector{Float64}, w::Vector{Float64}=ones(length(y)))
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # get number of parameters
    p = length(model.β)
    residuals = y - X * model.β
    model.σ² = sum(w.*(residuals.^2)) / sum(w) # biased estimate, could use n-1
end

"""
    fit!(model::GaussianRegression, X::Matrix{Float64}, y::Vector{Float64}, w::Vector{Float64}=ones(length(y)))

Fit a Gaussian regression model using weighted least squares.

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
fit!(model, X, y)

model = GaussianRegression()
X = rand(100, 2)
y = X * [0.1, 0.2] + 0.1 * randn(100)
w = rand(100)
fit!(model, X, y, w)
``` 
"""
function fit!(model::GaussianRegression, X::Matrix{Float64}, y::Vector{Float64}, w::Vector{Float64}=ones(length(y)))
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
    objective(β) = least_squares(GaussianRegression(β, model.σ², true, model.λ), X, y, w)
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
    # Clamp probabilities to avoid log(0) and log(1)
    p = clamp.(p, 1e-16, 1-1e-16)
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
    # Clamp probabilities to avoid log(0) and log(1)
    p = clamp(p, 1e-16, 1-1e-16)
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

