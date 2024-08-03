export GaussianRegression, BernoulliRegression, PoissonRegression, AutoRegression, fit!, loglikelihood, least_squares, update_variance!, sample, set_params!

# below used in notebooks and unit tests
export define_objective, define_objective_gradient

# temporary exports
export Regression


# abstract regression type
abstract type Regression end

"""
    GaussianRegression(β::Matrix{<:Real}, Σ::Matrix{<:Real}, input_dim::Int, output_dim::Int, include_intercept::Bool, λ::Float64)

A struct representing a Gaussian regression model.

# Fields
- `β::Matrix{<:Real}`: Coefficients of the regression model. Has shape (input_dim, output_dim). Column one is coefficients for target one, etc. The first row is the intercept term, if included.
- `Σ::Matrix{<:Real}`: Covariance matrix of the model.
- `input_dim::Int`: Number of features in the model.
- `output_dim::Int`: Number of targets in the model.
- `include_intercept::Bool`: Whether to include an intercept term in the model.
- `λ::Float64`: Regularization parameter for the model.

# Constructors
- `GaussianRegression(; input_dim::Int, output_dim::Int, include_intercept::Bool=true, λ::Float64=0.0)`
- `GaussianRegression(β::Matrix{<:Real}, Σ::Matrix{<:Real}; input_dim::Int, output_dim::Int, include_intercept::Bool=true, λ::Float64=0.0)`

# Examples
```julia
model = GaussianRegression(input_dim=2, output_dim=1)
model = GaussianRegression(ones(3, 1), ones(1, 1), input_dim=2, output_dim=1)
model = GaussianRegression(ones(2,1), ones(1, 1), input_dim=2, output_dim=1, include_intercept=false, λ=0.1)
```
"""
mutable struct GaussianRegression <: Regression
    input_dim::Int
    output_dim::Int
    β::Matrix{<:Real} # coefficient matrix of the model. Shape input_dim by output_dim. Column one is coefficients for target one, etc. The first row are the intercept terms, if included. 
    Σ::Matrix{<:Real} # covariance matrix of the model 
    include_intercept::Bool # whether to include an intercept term; if true, the first column of β is assumed to be the intercept/bias
    λ::Float64 # regularization parameter
  
    function GaussianRegression(; input_dim::Int, output_dim::Int, include_intercept::Bool = true, λ::Float64=0.0)
        if include_intercept
            new(input_dim, output_dim, zeros(input_dim + 1, output_dim), Matrix{Float64}(I, output_dim, output_dim), include_intercept, λ)
        else
            new(input_dim, output_dim, zeros(input_dim, output_dim), Matrix{Float64}(I, output_dim, output_dim), include_intercept, λ)
        end

    end
    
    function GaussianRegression(β::Matrix{<:Real}, Σ::Matrix{<:Real}; input_dim::Int, output_dim::Int, include_intercept::Bool = true, λ::Float64=0.0)
        if include_intercept
            @assert size(β) == (input_dim + 1, output_dim)
        else
            @assert size(β) == (input_dim, output_dim)
        end

        
        @assert size(Σ) == (output_dim, output_dim)

        new(input_dim, output_dim, β, Σ, include_intercept, λ)
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
model = GaussianRegression(input_dim=2, output_dim=1)
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
    return X * model.β + rand(MvNormal(zeros(model.output_dim), model.Σ), size(X, 1))'
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
model = GaussianRegression(input_dim=2, output_dim=1)
X = rand(100, 2)
y = X * [0.1, 0.2] + 0.1 * randn(100)
y = reshape(y, 100, 1)
loglikelihood(model, X, y)
```
"""
function loglikelihood(model::GaussianRegression, X::Matrix{<:Real}, y::Matrix{<:Real})
    # confirm dimensions of X and y are correct
    @assert size(X, 1) == size(y, 1) "Number of rows (number of observations) in X and y must be equal."
    @assert size(y, 2) == model.output_dim "Number of columns in y must be equal to the number of targets in the model."
    @assert size(X, 2) == model.input_dim "Number of columns in X must be equal to the number of features in the model."



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



function define_objective(model::GaussianRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))
    # assume covariance is the identity, so the log likelihood is just the negative squared error. Ignore loglikelihood terms that don't depend on β.

    # confirm dimensions of X and y are correct
    @assert size(X, 1) == size(y, 1) "Number of rows (number of observations) in X and y must be equal."
    @assert size(y, 2) == model.output_dim "Number of columns in y must be equal to the number of targets in the model."
    @assert size(X, 2) == model.input_dim "Number of columns in X must be equal to the number of features in the model."

    # add intercept if specified
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end

    function objective(β::Matrix{<:Real})
        # calculate log likelihood
        residuals = y - X * β

        # reshape w for broadcasting
        w = reshape(w, (length(w), 1))
        pseudo_loglikelihood = -0.5 * sum(broadcast(*, w, residuals.^2)) - (model.λ * sum(β.^2))

        return -pseudo_loglikelihood / size(X, 1)
    end

    return objective
end


function define_objective_gradient(model::GaussianRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))
    # assume covariance is the identity, so the log likelihood is just the negative squared error. Ignore loglikelihood terms that don't depend on β.

    # confirm dimensions of X and y are correct
    @assert size(X, 1) == size(y, 1) "Number of rows (number of observations) in X and y must be equal."
    @assert size(y, 2) == model.output_dim "Number of columns in y must be equal to the number of targets in the model."
    @assert size(X, 2) == model.input_dim "Number of columns in X must be equal to the number of features in the model."

    # confirm the size of w is correct
    @assert length(w) == size(y, 1) "Length of w must be equal to the number of observations in y."

    # add intercept if specified
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end

    function objective_gradient!(G::Matrix{<:Real}, β::Matrix{<:Real})
        # calculate log likelihood
        residuals = y - X * β

        G .= -(X' * Diagonal(w) * residuals - (2*model.λ*β)) / size(X, 1)
    end
    
    return objective_gradient!
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
model = GaussianRegression(input_dim=2, output_dim=1)
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
    @assert size(y, 2) == model.output_dim "Number of columns in y must be equal to the number of targets in the model."
    @assert size(X, 2) == model.input_dim "Number of columns in X must be equal to the number of features in the model."

    # confirm the size of w is correct
    @assert length(w) == size(y, 1) "Length of w must be equal to the number of observations in y."


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
model = GaussianRegression(input_dim=2, output_dim=1)
X = rand(100, 2)
y = rand(100, 1)
fit!(model, X, y)
```
"""
function fit!(model::GaussianRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))
    # confirm dimensions of X and y are correct
    @assert size(X, 1) == size(y, 1) "Number of rows (number of observations) in X and y must be equal."
    @assert size(y, 2) == model.output_dim "Number of columns in y must be equal to the number of targets in the model."
    @assert size(X, 2) == model.input_dim "Number of columns in X must be equal to the number of features in the model."

    # confirm the size of w is correct
    @assert length(w) == size(y, 1) "Length of w must be equal to the number of observations in y."
    
    # minimize objective
    objective = define_objective(model, X, y, w)
    objective_grad! = define_objective_gradient(model, X, y, w)


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
    input_dim::Int
    β::Vector{<:Real}
    include_intercept::Bool
    λ::Float64
    # Empty constructor
    function BernoulliRegression(; input_dim::Int, include_intercept::Bool = true, λ::Float64=0.0) 
        @assert λ >= 0.0 "Regularization parameter must be non-negative."
        if include_intercept
            new(input_dim, zeros(input_dim + 1), include_intercept, λ)
        else
            new(input_dim, zeros(input_dim), include_intercept, λ)
        end
    end
    # Parametric Constructor
    function BernoulliRegression(β::Vector{<:Real}; input_dim::Int, include_intercept::Bool = true, λ::Float64=0.0)
        @assert λ >= 0.0 "Regularization parameter must be non-negative."
        if include_intercept
            @assert size(β, 1) == input_dim + 1
        else
            @assert size(β, 1) == input_dim
        end

        new(input_dim, β, include_intercept, λ)
    end
end


function sample(model::BernoulliRegression, X::Matrix{<:Real})
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified
    if model.include_intercept && size(X, 2) == length(model.β) - 1
        X = hcat(ones(size(X, 1)), X)
    end

    y = rand.(Bernoulli.(logistic.(X * model.β)))

    # convert y 
    y = reshape(y, :, 1)
    y = Float64.(y)

    return y
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


function define_objective(model::BernoulliRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))
    # assume covariance is the identity, so the log likelihood is just the negative squared error. Ignore loglikelihood terms that don't depend on β.

    # confirm dimensions of X and y are correct
    @assert size(X, 1) == size(y, 1) "Number of rows (number of observations) in X and y must be equal."
    @assert size(y, 2) == 1 "BernoulliRegression Y data should be a single column."
    @assert size(X, 2) == model.input_dim "Number of columns in X must be equal to the number of features in the model."

    # add intercept if specified
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end

    function objective(β)
        # calculate log likelihood
        p = logistic.(X * β)
        
        val = -sum(w .* (y .* log.(p) .+ (1 .- y) .* log.(1 .- p))) + (model.λ * sum(β.^2))

        return val / size(X, 1)
    end

    return objective
end


function define_objective_gradient(model::BernoulliRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))
    # assume covariance is the identity, so the log likelihood is just the negative squared error. Ignore loglikelihood terms that don't depend on β.

    # confirm dimensions of X and y are correct
    @assert size(X, 1) == size(y, 1) "Number of rows (number of observations) in X and y must be equal."
    @assert size(X, 2) == model.input_dim "Number of columns in X must be equal to the number of features in the model."

    # confirm the size of w is correct
    @assert length(w) == size(y, 1) "Length of w must be equal to the number of observations in y."

    # add intercept if specified
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end

    function objective_gradient!(G, β)
        # calculate log likelihood
        p = logistic.(X * β)

        G .= (-(X' * (w .* (y .- p))) + 2 * model.λ * β) / size(X, 1)
    end
    
    return objective_gradient!
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
    # minimize objective
    objective = define_objective(model, X, y, w)
    objective_grad! = define_objective_gradient(model, X, y, w)

    result = optimize(objective, objective_grad!, model.β, LBFGS())

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
    input_dim::Int
    β::Vector{<:Real}
    include_intercept::Bool
    λ::Float64
    # Empty constructor
    function PoissonRegression(; input_dim::Int, include_intercept::Bool = true, λ::Float64=0.0) 
        @assert λ >= 0.0 "Regularization parameter must be non-negative."
        if include_intercept
            new(input_dim, zeros(input_dim + 1), include_intercept, λ)
        else
            new(input_dim, zeros(input_dim), include_intercept, λ)
        end
    end
    # Parametric Constructor
    function PoissonRegression(β::Vector{<:Real}; input_dim::Int, include_intercept::Bool = true, λ::Float64=0.0)
        @assert λ >= 0.0 "Regularization parameter must be non-negative."
        if include_intercept
            @assert size(β, 1) == input_dim + 1
        else
            @assert size(β, 1) == input_dim
        end

        new(input_dim, β, include_intercept, λ)
    end
end


function sample(model::PoissonRegression, X::Matrix{<:Real})
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified
    if model.include_intercept && size(X, 2) == length(model.β) - 1
        X = hcat(ones(size(X, 1)), X)
    end

    y = rand.(Poisson.(exp.(X * model.β)))

    # convert y 
    y = reshape(y, :, 1)
    y = Float64.(y)

    return y
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
function loglikelihood(model::PoissonRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified
    if model.include_intercept && size(X, 2) == length(model.β) - 1
        X = hcat(ones(size(X, 1)), X)
    end

    # calculate log likelihood
    λ = exp.(X * model.β)

    return sum(w .* (y .* log.(λ) .- λ .- loggamma.(Int.(y) .+ 1)))
end


function define_objective(model::PoissonRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))
    # assume covariance is the identity, so the log likelihood is just the negative squared error. Ignore loglikelihood terms that don't depend on β.

    # confirm dimensions of X and y are correct
    @assert size(X, 1) == size(y, 1) "Number of rows (number of observations) in X and y must be equal."
    @assert size(y, 2) == 1 "PoissonRegression Y data should be a single column."
    @assert size(X, 2) == model.input_dim "Number of columns in X must be equal to the number of features in the model."

    # add intercept if specified
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end

    function objective(β)
        # calculate the rate
        rate = exp.(X * β)

        val = -sum(w .* (y .* log.(rate) .- rate .- loggamma.(Int.(y) .+ 1))) + (model.λ * sum(β.^2))

        return val / size(X, 1)
    end

    return objective
end

function define_objective_gradient(model::PoissonRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))
    # assume covariance is the identity, so the log likelihood is just the negative squared error. Ignore loglikelihood terms that don't depend on β.

    # confirm dimensions of X and y are correct
    @assert size(X, 1) == size(y, 1) "Number of rows (number of observations) in X and y must be equal."
    @assert size(X, 2) == model.input_dim "Number of columns in X must be equal to the number of features in the model."

    # confirm the size of w is correct
    @assert length(w) == size(y, 1) "Length of w must be equal to the number of observations in y."

    # add intercept if specified
    if model.include_intercept
        X = hcat(ones(size(X, 1)), X)
    end

    function objective_gradient!(G, β)
        # calculate the rate
        rate = exp.(X * β)

        G .= (-X' * (Diagonal(w) * (y .- rate)) + (model.λ * 2 * β)) / size(X, 1)
    end
    
    return objective_gradient!
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
function gradient!(grad::Vector{Float64}, model::PoissonRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y,1)))
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified
    if model.include_intercept && size(X, 2) == length(model.β) - 1
        X = hcat(ones(size(X, 1)), X)
    end
    # calculate the rate
    rate = exp.(X * model.β)
    # convert y if necessary
    # y = convert(Vector{Float64}, y)
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
function fit!(model::PoissonRegression, X::Matrix{<:Real}, y::Matrix{<:Real}, w::Vector{Float64}=ones(size(y, 1)))
    # minimize objective
    objective = define_objective(model, X, y, w)
    objective_grad! = define_objective_gradient(model, X, y, w)

    result = optimize(objective, objective_grad!, model.β, LBFGS())

    # update parameters
    model.β = result.minimizer
end





mutable struct AutoRegression <: Regression
    data_dim::Int
    order::Int
    innerGaussianRegression::GaussianRegression

    function AutoRegression(; data_dim::Int, order::Int, include_intercept::Bool = true, λ::Float64=0.0)
        innerGaussianRegression = GaussianRegression(input_dim = data_dim * order, output_dim = data_dim, include_intercept = include_intercept, λ = λ)
        new(data_dim, order, innerGaussianRegression)
    end

    function AutoRegression(β::Matrix{<:Real}, Σ::Matrix{<:Real}; data_dim::Int, order::Int, include_intercept::Bool = true, λ::Float64=0.0)
        innerGaussianRegression = GaussianRegression(β, Σ, input_dim = data_dim * order, output_dim = data_dim, include_intercept = include_intercept, λ = λ)
        new(data_dim, order, innerGaussianRegression)
    end
end

function AR_to_Gaussian_data(y_prev::Matrix{<:Real})
    # take each row of y_prev and stack them horizontally to form the input row matrix X_gaussian
    X_gaussian = vcat([y_prev[i, :] for i in 1:size(y_prev, 1)]...)
    X_gaussian = reshape(X_gaussian, 1, :)

    return X_gaussian
end

function AR_to_Gaussian_data(y_prev::Matrix{<:Real}, y::Matrix{<:Real})
    order = size(y_prev, 1)
    data_dim = size(y_prev, 2)
    X_gaussian = zeros(size(y, 1), data_dim * order)

    for i in 1:size(y, 1)
        X_gaussian[i, :] = AR_to_Gaussian_data(y_prev)


        old_part = y_prev[2:end, :]
        new_part = y[i, :]

        old_part = reshape(old_part, order - 1, data_dim)
        new_part = reshape(new_part, 1, data_dim)

        y_prev = vcat(old_part, new_part)
    end
   

    return X_gaussian
end

# setting Vector{Matrix{Float64}} to Vector{Matrix{<:Real}} throws an error for some reason...
function set_params!(model::AutoRegression; βs::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(), intercept::Vector{<:Real} = Vector{Float64}(), Σ::Matrix{<:Real} = Matrix{Float64}())
    if length(βs) != 0
        @assert length(βs) == model.order

        if model.innerGaussianRegression.include_intercept
            model.innerGaussianRegression.β[2:end, :] = vcat([βs[i] for i in 1:length(βs)]...)
        else
            model.innerGaussianRegression.β = vcat([βs[i] for i in 1:length(βs)]...)
        end
    end

    # check if intercept is empty
    if length(intercept) != 0
        @assert length(intercept) == model.data_dim
        @assert model.innerGaussianRegression.include_intercept
        model.innerGaussianRegression.β[1, :] = intercept
    end

    if size(Σ) != (0, 0)
        @assert size(Σ) == (model.data_dim, model.data_dim)
        model.innerGaussianRegression.Σ = Σ
    end
end

function sample(model::AutoRegression, y_prev::Matrix{<:Real})
    X_gaussian = AR_to_Gaussian_data(y_prev)
    return sample(model.innerGaussianRegression, X_gaussian)
end

function sample(model::AutoRegression, y_prev::Matrix{<:Real}, n::Int)
    y = zeros(n, model.data_dim)

    for i in 1:n
        y[i, :] = sample(model, y_prev)

        old_part = y_prev[2:end, :]
        new_part = y[i, :]

        old_part = reshape(old_part, model.order - 1, model.data_dim)
        new_part = reshape(new_part, 1, model.data_dim)

        y_prev = vcat(old_part, new_part)
    end
    
    return y
end

function loglikelihood(model::AutoRegression, y_prev::Matrix{<:Real}, y::Matrix{<:Real})
    X_gaussian = AR_to_Gaussian_data(y_prev, y)

    return loglikelihood(model.innerGaussianRegression, X_gaussian, y)
end


function fit!(model::AutoRegression, y_prev::Matrix{<:Real}, y::Matrix{<:Real})
    X_gaussian = AR_to_Gaussian_data(y_prev, y)

    return fit!(model.innerGaussianRegression, X_gaussian, y)
end