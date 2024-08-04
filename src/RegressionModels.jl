export GaussianRegression, BernoulliRegression, PoissonRegression, AutoRegression, fit!, loglikelihood, least_squares, update_variance!, sample, set_params!

# below used in notebooks and unit tests
export define_objective, define_objective_gradient

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
mutable struct GaussianRegression <: RegressionModel
    input_dim::Int
    output_dim::Int
    β::Matrix{<:Real} # coefficient matrix of the model. Shape input_dim by output_dim. Column one is coefficients for target one, etc. The first row are the intercept terms, if included. 
    Σ::Matrix{<:Real} # covariance matrix of the model 
    include_intercept::Bool # whether to include an intercept term; if true, the first column of β is assumed to be the intercept/bias
    λ::Float64 # regularization parameter
end


function validate_model(model::GaussianRegression)
    if model.include_intercept
        @assert size(model.β) == (model.input_dim + 1, model.output_dim) "β must be of size (input_dim + 1, output_dim) if an intercept/bias is included."
    else
        @assert size(model.β) == (model.input_dim, model.output_dim)
    end

    @assert size(model.Σ) == (model.output_dim, model.output_dim)
    @assert valid_Σ(model.Σ)
    @assert model.λ >= 0.0
end


function GaussianRegression(; 
    input_dim::Int, 
    output_dim::Int, 
    include_intercept::Bool = true, 
    β::Matrix{<:Real} = if include_intercept zeros(input_dim + 1, output_dim) else zeros(input_dim, output_dim) end,
    Σ::Matrix{<:Real} = Matrix{Float64}(I, output_dim, output_dim),
    λ::Float64 = 0.0)

    new_model = GaussianRegression(input_dim, output_dim, β, Σ, include_intercept, λ)

    validate_model(new_model)
    
    return new_model
end

"""
    sample(model::GaussianRegression, Φ::Matrix{<:Real}) 

Sample from a Gaussian regression model.

# Arguments
- `model::GaussianRegression`: Gaussian regression model.
- `Φ::Matrix{<:Real}`: Design matrix. Each row is an observation.

# Returns
- `Y::Matrix{<:Real}`: Sampled response matrix. Each row is a sample.

# Examples
```julia
model = GaussianRegression(input_dim=2, output_dim=1)
Φ = rand(100, 2)
Y = sample(model, Φ)
```
"""
function sample(model::GaussianRegression, Φ::Matrix{<:Real})
    # confirm that the model has been fit
    @assert !all(model.β .== 0) "Coefficient matrix is all zeros. Did you forget to initialize?"
    @assert isposdef(model.Σ) "Covariance matrix is not positive definite. Did you forget to initialize?"
    # add intercept if specified
    if model.include_intercept
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end
    return Φ * model.β + rand(MvNormal(zeros(model.output_dim), model.Σ), size(Φ, 1))'
end


"""
    loglikelihood(model::GaussianRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real})

Calculate the log-likelihood of a Gaussian regression model.

# Arguments
- `model::GaussianRegression`: Gaussian regression model.
- `Φ::Matrix{<:Real}`: Design matrix. Each row is an observation.
- `Y::Matrix{<:Real}`: Response matrix. Each row is a response vector.

# Examples
```julia
model = GaussianRegression(input_dim=2, output_dim=1)
Φ = rand(100, 2)
Y = Φ * [0.1, 0.2] + 0.1 * randn(100)
Y = reshape(Y, 100, 1)
loglikelihood(model, Φ, Y)
```
"""
function loglikelihood(model::GaussianRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real})
    # confirm dimensions of Φ and Y are correct
    @assert size(Φ, 1) == size(Y, 1) "Number of rows (number of observations) in Φ and Y must be equal."
    @assert size(Y, 2) == model.output_dim "Number of columns in Y must be equal to the number of targets in the model."
    @assert size(Φ, 2) == model.input_dim "Number of columns in Φ must be equal to the number of features in the model."



    # add intercept if specified
    if model.include_intercept
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    # calculate inverse of covariance matrix
    Σ_inv = inv(model.Σ)

    # calculate log likelihood
    residuals = Y - Φ * model.β

    log_likelihood = -0.5 * size(Φ, 1) * size(Φ, 2) * log(2π) - 0.5 * size(Φ, 1) * logdet(model.Σ) - 0.5 * sum(residuals .* (Σ_inv * residuals')')

    return log_likelihood
end



function define_objective(model::GaussianRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    # assume covariance is the identity, so the log likelihood is just the negative squared error. Ignore loglikelihood terms that don't depend on β.

    # confirm dimensions of Φ and Y are correct
    @assert size(Φ, 1) == size(Y, 1) "Number of rows (number of observations) in Φ and Y must be equal."
    @assert size(Y, 2) == model.output_dim "Number of columns in Y must be equal to the number of targets in the model."
    @assert size(Φ, 2) == model.input_dim "Number of columns in Φ must be equal to the number of features in the model."

    # add intercept if specified
    if model.include_intercept
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    function objective(β::Matrix{<:Real})
        # calculate log likelihood
        residuals = Y - Φ * β

        # reshape w for broadcasting
        w = reshape(w, (length(w), 1))
        pseudo_loglikelihood = -0.5 * sum(broadcast(*, w, residuals.^2)) - (model.λ * sum(β.^2))

        return -pseudo_loglikelihood / size(Φ, 1)
    end

    return objective
end


function define_objective_gradient(model::GaussianRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    # assume covariance is the identity, so the log likelihood is just the negative squared error. Ignore loglikelihood terms that don't depend on β.

    # confirm dimensions of Φ and Y are correct
    @assert size(Φ, 1) == size(Y, 1) "Number of rows (number of observations) in Φ and Y must be equal."
    @assert size(Y, 2) == model.output_dim "Number of columns in Y must be equal to the number of targets in the model."
    @assert size(Φ, 2) == model.input_dim "Number of columns in Φ must be equal to the number of features in the model."

    # confirm the size of w is correct
    @assert length(w) == size(Y, 1) "Length of w must be equal to the number of observations in Y."

    # add intercept if specified
    if model.include_intercept
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    function objective_gradient!(G::Matrix{<:Real}, β::Matrix{<:Real})
        # calculate log likelihood
        residuals = Y - Φ * β

        G .= -(Φ' * Diagonal(w) * residuals - (2*model.λ*β)) / size(Φ, 1)
    end
    
    return objective_gradient!
end

"""
    update_variance!(model::GaussianRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

Update the (weighted) variance of a Gaussian regression model. Uses the biased estimator.

# Arguments
- `model::GaussianRegression`: Gaussian regression model.
- `Φ::Matrix{<:Real}`: Design matrix. Each row is an observation.
- `Y::Vector{Float64}`: Response vector. Each row is a response vector.
- `w::Vector{Float64}`: Weights for the observations.

# Examples
```julia
model = GaussianRegression(input_dim=2, output_dim=1)
Φ = rand(100, 2)
Y = Φ * [0.1, 0.2] + 0.1 * randn(100)
Y = reshape(Y, 100, 1)
update_variance!(model, Φ, Y)
```
"""
function update_variance!(model::GaussianRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    # WARNING: asserts may slow down computation. Remove later?

    # confirm dimensions of Φ and Y are correct
    @assert size(Φ, 1) == size(Y, 1) "Number of rows (number of observations) in Φ and Y must be equal."
    @assert size(Y, 2) == model.output_dim "Number of columns in Y must be equal to the number of targets in the model."
    @assert size(Φ, 2) == model.input_dim "Number of columns in Φ must be equal to the number of features in the model."

    # confirm the size of w is correct
    @assert length(w) == size(Y, 1) "Length of w must be equal to the number of observations in Y."


    # add intercept if specified
    if model.include_intercept
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end


    residuals = Y - Φ * model.β
    
    
    model.Σ = (residuals' * Diagonal(w) * residuals) / size(Φ, 1)

    # ensure rounding errors are not causing the covariance matrix to be non-positive definite
    model.Σ = stabilize_covariance_matrix(model.Σ)

   
    
end

"""
    fit!(model::GaussianRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

Fit a Gaussian regression model using maximum likelihood estimation.

# Arguments
- `model::GaussianRegression`: Gaussian regression model.
- `Φ::Matrix{<:Real}`: Design matrix. Each row is an observation.
- `Y::Matrix{<:Real}`: Response matrix. Each row is a response vector.
- `w::Vector{Float64}`: Weights for the observations.

# Examples
```julia
model = GaussianRegression(input_dim=2, output_dim=1)
Φ = rand(100, 2)
Y = rand(100, 1)
fit!(model, Φ, Y)
```
"""
function fit!(model::GaussianRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    # confirm dimensions of Φ and Y are correct
    @assert size(Φ, 1) == size(Y, 1) "Number of rows (number of observations) in Φ and Y must be equal."
    @assert size(Y, 2) == model.output_dim "Number of columns in Y must be equal to the number of targets in the model."
    @assert size(Φ, 2) == model.input_dim "Number of columns in Φ must be equal to the number of features in the model."

    # confirm the size of w is correct
    @assert length(w) == size(Y, 1) "Length of w must be equal to the number of observations in Y."
    
    # minimize objective
    objective = define_objective(model, Φ, Y, w)
    objective_grad! = define_objective_gradient(model, Φ, Y, w)


    result = optimize(objective, objective_grad!, model.β, LBFGS())

    # update parameters
    model.β = result.minimizer
    update_variance!(model, Φ, Y, w)
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
mutable struct BernoulliRegression <: RegressionModel
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


function sample(model::BernoulliRegression, Φ::Matrix{<:Real})
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified
    if model.include_intercept && size(Φ, 2) == length(model.β) - 1
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    Y = rand.(Bernoulli.(logistic.(Φ * model.β)))

    # convert Y 
    Y = reshape(Y, :, 1)
    Y = Float64.(Y)

    return Y
end

"""
    loglikelihood(model::BernoulliRegression, Φ::Matrix{<:Real}, Y::Union{Vector{Float64}, BitVector}, w::Vector{Float64}=ones(length(Y))

Calculate the log-likelihood of a Bernoulli regression model.

# Arguments
- `model::BernoulliRegression`: Bernoulli regression model.
- `Φ::Matrix{<:Real}`: Design matrix.
- `Y::Union{Vector{Float64}, BitVector}`: Response vector.
- `w::Vector{Float64}`: Weights for the observations.

# Examples
```julia
model = BernoulliRegression()
Φ = rand(100, 2)
Y = rand(Bool, 100)
loglikelihood(model, Φ, Y)
```
"""
function loglikelihood(model::BernoulliRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified and not already included
    if model.include_intercept && size(Φ, 2) == length(model.β) - 1 
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end
    # calculate log likelihood
    p = logistic.(Φ * model.β)
    
    return sum(w .* (Y .* log.(p) .+ (1 .- Y) .* log.(1 .- p)))
end


function define_objective(model::BernoulliRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    # assume covariance is the identitY, so the log likelihood is just the negative squared error. Ignore loglikelihood terms that don't depend on β.

    # confirm dimensions of Φ and Y are correct
    @assert size(Φ, 1) == size(Y, 1) "Number of rows (number of observations) in Φ and Y must be equal."
    @assert size(Y, 2) == 1 "BernoulliRegression Y data should be a single column."
    @assert size(Φ, 2) == model.input_dim "Number of columns in Φ must be equal to the number of features in the model."

    # add intercept if specified
    if model.include_intercept
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    function objective(β)
        # calculate log likelihood
        p = logistic.(Φ * β)
        
        val = -sum(w .* (Y .* log.(p) .+ (1 .- Y) .* log.(1 .- p))) + (model.λ * sum(β.^2))

        return val / size(Φ, 1)
    end

    return objective
end


function define_objective_gradient(model::BernoulliRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    # assume covariance is the identity, so the log likelihood is just the negative squared error. Ignore loglikelihood terms that don't depend on β.

    # confirm dimensions of Φ and Y are correct
    @assert size(Φ, 1) == size(Y, 1) "Number of rows (number of observations) in Φ and Y must be equal."
    @assert size(Φ, 2) == model.input_dim "Number of columns in Φ must be equal to the number of features in the model."

    # confirm the size of w is correct
    @assert length(w) == size(Y, 1) "Length of w must be equal to the number of observations in Y."

    # add intercept if specified
    if model.include_intercept
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    function objective_gradient!(G, β)
        # calculate log likelihood
        p = logistic.(Φ * β)

        G .= (-(Φ' * (w .* (Y .- p))) + 2 * model.λ * β) / size(Φ, 1)
    end
    
    return objective_gradient!
end




"""
    fit!(model::BernoulliRegression, Φ::Matrix{<:Real}, Y::Union{Vector{Float64}, BitVector}, w::Vector{Float64}=ones(length(Y))

Fit a Bernoulli regression model using maximum likelihood estimation.

# Arguments
- `model::BernoulliRegression`: Bernoulli regression model.
- `Φ::Matrix{<:Real}`: Design matrix.
- `Y::Union{Vector{Float64}, BitVector}`: Response vector.
- `w::Vector{Float64}`: Weights for the observations.

# Examples
```julia
model = BernoulliRegression()
Φ = rand(100, 2)
Y = rand(Bool, 100)
fit!(model, Φ, Y)

model = BernoulliRegression()
Φ = rand(100, 2)
Y = rand(Bool, 100)
w = rand(100)
fit!(model, Φ, Y, w)
```
"""
function fit!(model::BernoulliRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    # minimize objective
    objective = define_objective(model, Φ, Y, w)
    objective_grad! = define_objective_gradient(model, Φ, Y, w)

    result = optimize(objective, objective_grad!, model.β, LBFGS())

    # update parameters
    model.β = result.minimizer
end
    
"""
    mutable struct PoissonRegression <: RegressionModel

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
mutable struct PoissonRegression <: RegressionModel
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


function sample(model::PoissonRegression, Φ::Matrix{<:Real})
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
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
    loglikelihood(model::PoissonRegression, Φ::Matrix{<:Real}, Y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(Y)))

Calculate the log-likelihood of a Poisson regression model.

# Arguments
- `model::PoissonRegression`: Poisson regression model
- `Φ::Matrix{<:Real}`: Design matrix
- `Y::Union{Vector{Float64}, Vector{Int64}}`: Response vector
- `w::Vector{Float64}`: Weights for the observations

# Examples
```julia
model = PoissonRegression()
Φ = rand(100, 2)
Y = rand(Poisson(1), 100)
loglikelihood(model, Φ, Y)
```
"""
function loglikelihood(model::PoissonRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified
    if model.include_intercept && size(Φ, 2) == length(model.β) - 1
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    # calculate log likelihood
    λ = exp.(Φ * model.β)

    return sum(w .* (Y .* log.(λ) .- λ .- loggamma.(Int.(Y) .+ 1)))
end


function define_objective(model::PoissonRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    # assume covariance is the identity, so the log likelihood is just the negative squared error. Ignore loglikelihood terms that don't depend on β.

    # confirm dimensions of Φ and Y are correct
    @assert size(Φ, 1) == size(Y, 1) "Number of rows (number of observations) in Φ and Y must be equal."
    @assert size(Y, 2) == 1 "PoissonRegression Y data should be a single column."
    @assert size(Φ, 2) == model.input_dim "Number of columns in Φ must be equal to the number of features in the model."

    # add intercept if specified
    if model.include_intercept
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    function objective(β)
        # calculate the rate
        rate = exp.(Φ * β)

        val = -sum(w .* (Y .* log.(rate) .- rate .- loggamma.(Int.(Y) .+ 1))) + (model.λ * sum(β.^2))

        return val / size(Φ, 1)
    end

    return objective
end

function define_objective_gradient(model::PoissonRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    # assume covariance is the identity, so the log likelihood is just the negative squared error. Ignore loglikelihood terms that don't depend on β.

    # confirm dimensions of Φ and Y are correct
    @assert size(Φ, 1) == size(Y, 1) "Number of rows (number of observations) in Φ and Y must be equal."
    @assert size(Φ, 2) == model.input_dim "Number of columns in Φ must be equal to the number of features in the model."

    # confirm the size of w is correct
    @assert length(w) == size(Y, 1) "Length of w must be equal to the number of observations in Y."

    # add intercept if specified
    if model.include_intercept
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    function objective_gradient!(G, β)
        # calculate the rate
        rate = exp.(Φ * β)

        G .= (-Φ' * (Diagonal(w) * (Y .- rate)) + (model.λ * 2 * β)) / size(Φ, 1)
    end
    
    return objective_gradient!
end



"""
    gradient!(grad::Vector{Float64}, model::PoissonRegression, Φ::Matrix{<:Real}, Y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(Y)))

Calculate the gradient of the negative log-likelihood function for a Poisson regression model.

# Arguments
- `grad::Vector{Float64}`: Gradient of the negative log-likelihood function
- `model::PoissonRegression`: Poisson regression model
- `Φ::Matrix{<:Real}`: Design matrix
- `Y::Union{Vector{Float64}, Vector{Int64}}`: Response vector
- `w::Vector{Float64}`: Weights for the observations

# Examples
```julia
model = PoissonRegression()
Φ = rand(100, 2)
Y = rand(Poisson(1), 100)
G = zeros(2)
gradient!(G, model, Φ, Y)
```
"""
function gradient!(grad::Vector{Float64}, model::PoissonRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y,1)))
    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified
    if model.include_intercept && size(Φ, 2) == length(model.β) - 1
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end
    # calculate the rate
    rate = exp.(Φ * model.β)
    # convert Y if necessary
    # Y = convert(Vector{Float64}, Y)
    # calculate gradient
    grad .= -Φ' * (Diagonal(w) * (Y .- rate)) + (model.λ * 2 * model.β)
end

"""
    fit!(model::PoissonRegression, Φ::Matrix{<:Real}, Y::Union{Vector{Float64}, Vector{Int64}}, w::Vector{Float64}=ones(length(Y)))

Fit a Poisson regression model using maximum likelihood estimation.

# Arguments
- `model::PoissonRegression`: Poisson regression model
- `Φ::Matrix{<:Real}`: Design matrix
- `Y::Union{Vector{Float64}, Vector{Int64}}`: Response vector
- `w::Vector{Float64}`: Weights for the observations

# Examples
```julia
model = PoissonRegression()
Φ = rand(100, 2)
Y = rand(Poisson(1), 100)
fit!(model, Φ, Y)

model = PoissonRegression()
Φ = rand(100, 2)
Y = rand(Poisson(1), 100)
w = rand(100)
fit!(model, Φ, Y, w)
```
"""
function fit!(model::PoissonRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    # minimize objective
    objective = define_objective(model, Φ, Y, w)
    objective_grad! = define_objective_gradient(model, Φ, Y, w)

    result = optimize(objective, objective_grad!, model.β, LBFGS())

    # update parameters
    model.β = result.minimizer
end





mutable struct AutoRegression <: RegressionModel
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

function AR_to_Gaussian_data(Y_prev::Matrix{<:Real})
    # take each row of Y_prev and stack them horizontally to form the input row matrix Φ_gaussian
    Φ_gaussian = vcat([Y_prev[i, :] for i in 1:size(Y_prev, 1)]...)
    Φ_gaussian = reshape(Φ_gaussian, 1, :)

    return Φ_gaussian
end

function AR_to_Gaussian_data(Y_prev::Matrix{<:Real}, Y::Matrix{<:Real})
    order = size(Y_prev, 1)
    data_dim = size(Y_prev, 2)
    Φ_gaussian = zeros(size(Y, 1), data_dim * order)

    for i in 1:size(Y, 1)
        Φ_gaussian[i, :] = AR_to_Gaussian_data(Y_prev)


        old_part = Y_prev[2:end, :]
        new_part = Y[i, :]

        old_part = reshape(old_part, order - 1, data_dim)
        new_part = reshape(new_part, 1, data_dim)

        Y_prev = vcat(old_part, new_part)
    end
   

    return Φ_gaussian
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

function sample(model::AutoRegression, Y_prev::Matrix{<:Real})
    Φ_gaussian = AR_to_Gaussian_data(Y_prev)
    return sample(model.innerGaussianRegression, Φ_gaussian)
end

function sample(model::AutoRegression, Y_prev::Matrix{<:Real}, n::Int)
    Y = zeros(n, model.data_dim)

    for i in 1:n
        Y[i, :] = sample(model, Y_prev)

        old_part = Y_prev[2:end, :]
        new_part = Y[i, :]

        old_part = reshape(old_part, model.order - 1, model.data_dim)
        new_part = reshape(new_part, 1, model.data_dim)

        Y_prev = vcat(old_part, new_part)
    end
    
    return Y
end

function loglikelihood(model::AutoRegression, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real})
    Φ_gaussian = AR_to_Gaussian_data(Y_prev, Y)

    return loglikelihood(model.innerGaussianRegression, Φ_gaussian, Y)
end


function fit!(model::AutoRegression, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real})
    Φ_gaussian = AR_to_Gaussian_data(Y_prev, Y)

    return fit!(model.innerGaussianRegression, Φ_gaussian, Y)
end