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

function validate_data(model::GaussianRegression, Φ=nothing, Y=nothing, w=nothing)
    if !isnothing(Φ)
        @assert size(Φ, 2) == model.input_dim "Number of columns in Φ must be equal to the input dimension of the model."
    end
    if !isnothing(Y)
        @assert size(Y, 2) == model.output_dim "Number of columns in Y must be equal to the output dimension of the model."
    end
    if !isnothing(w)
        @assert length(w) == size(Y, 1) "Length of w must be equal to the number of observations in Y."
    end
    if !isnothing(Φ) && !isnothing(Y)
        @assert size(Φ, 1) == size(Y, 1) "Number of rows (number of observations) in Φ and Y must be equal."
    end
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
function sample(model::GaussianRegression, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))
    @assert n <= size(Φ, 1) "n must be less than or equal to the number of observations in Φ."
    # cut the length of Φ to n
    Φ = Φ[1:n, :]

    # confirm that the model has valid parameters
    validate_model(model)

    validate_data(model, Φ)

    # add intercept if specified
    if model.include_intercept
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end
    return Φ * model.β + rand(MvNormal(zeros(model.output_dim), model.Σ), size(Φ, 1))'
end

# custom sampling function for the HMM. Returns observation_sequence with new observation appended to bottom.
function sample(model::GaussianRegression, observation_sequence::Matrix{<:Real}, Φ::Matrix{<:Real})
    # find the number of observations in the observation sequence
    t = size(observation_sequence, 1) + 1
    # get the n+1th observation
    new_observation = sample(model, Φ[t:t, :], n=1)

    return vcat(observation_sequence, new_observation)
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
function loglikelihood(model::GaussianRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}; observation_wise::Bool=false)
    # confirm that the model has valid parameters
    validate_model(model)

    validate_data(model, Φ, Y)



    # add intercept if specified
    if model.include_intercept
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    # calculate inverse of covariance matrix
    Σ_inv = inv(model.Σ)

    # calculate log likelihood
    residuals = Y - Φ * model.β

    if !observation_wise
        loglikelihood = -0.5 * size(Φ, 1) * size(Φ, 2) * log(2π) - 0.5 * size(Φ, 1) * logdet(model.Σ) - 0.5 * sum(residuals .* (Σ_inv * residuals')')
        return loglikelihood
    else 
        obs_wise_loglikelihood = zeros(size(Φ, 1))

        # calculate observation wise loglikelihood (a vector of loglikelihoods for each observation)
        @threads for i in 1:size(Φ, 1)
            obs_wise_loglikelihood[i] = -0.5 * size(Φ, 2) * log(2π) - 0.5 * logdet(model.Σ) - 0.5 * sum(residuals[i, :] .* (Σ_inv * residuals[i, :]))
        end

        return obs_wise_loglikelihood
    end
end


# assume covariance is the identity, so the log likelihood is just the negative squared error. Ignore loglikelihood terms that don't depend on β.
function define_objective(model::GaussianRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

    validate_data(model, Φ, Y, w)

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
    validate_data(model, Φ, Y, w)

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
    # confirm that the model has valid parameters
    validate_model(model)

    validate_data(model, Φ, Y, w)


    # add intercept if specified
    if model.include_intercept
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end


    residuals = Y - Φ * model.β
    
    
    model.Σ = (residuals' * Diagonal(w) * residuals) / size(Φ, 1)

    # ensure rounding errors are not causing the covariance matrix to be non-positive definite
    model.Σ = stabilize_covariance_matrix(model.Σ)

    # print the covariance matrix
    println("Covariance matrix: ", model.Σ)

   
    
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
    # confirm that the model has valid parameters
    validate_model(model)
    validate_data(model, Φ, Y, w)

    # confirm the size of w is correct
    @assert length(w) == size(Y, 1) "Length of w must be equal to the number of observations in Y."
    
    # minimize objective
    objective = define_objective(model, Φ, Y, w)
    objective_grad! = define_objective_gradient(model, Φ, Y, w)


    result = optimize(objective, objective_grad!, model.β, LBFGS())

    # update parameters
    model.β = result.minimizer
    update_variance!(model, Φ, Y, w)

    # confirm that the model has valid parameters
    validate_model(model)
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
end


function validate_model(model::BernoulliRegression)
    if model.include_intercept
        @assert size(model.β, 1) == model.input_dim + 1 "β must be of size (input_dim + 1) if an intercept/bias is included."
    else
        @assert size(model.β, 1) == model.input_dim
    end

    @assert model.λ >= 0.0
end

function validate_data(model::BernoulliRegression, Φ=nothing, Y=nothing, w=nothing)
    if !isnothing(Φ)
        @assert size(Φ, 2) == model.input_dim "Number of columns in Φ must be equal to the input dimension of the model."
    end
    if !isnothing(Y)
        @assert size(Y, 2) == 1 "BernoulliRegression Y data should be a single column."
    end
    if !isnothing(w)
        @assert length(w) == size(Y, 1) "Length of w must be equal to the number of observations in Y."
    end
    if !isnothing(Φ) && !isnothing(Y)
        @assert size(Φ, 1) == size(Y, 1) "Number of rows (number of observations) in Φ and Y must be equal."
    end
end

function BernoulliRegression(; 
    input_dim::Int, 
    include_intercept::Bool = true, 
    β::Vector{<:Real} = if include_intercept zeros(input_dim + 1) else zeros(input_dim) end,
    λ::Float64 = 0.0)

    new_model = BernoulliRegression(input_dim, β, include_intercept, λ)

    validate_model(new_model)
    
    return new_model
end





function sample(model::BernoulliRegression, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))
    @assert n <= size(Φ, 1) "n must be less than or equal to the number of observations in Φ."
    # cut the length of Φ to n
    Φ = Φ[1:n, :]

    # confirm that the model has valid parameters
    validate_model(model)
    validate_data(model, Φ)

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

# custom sampling function for the HMM. Returns observation_sequence with new observation appended to bottom.
function sample(model::BernoulliRegression, observation_sequence::Matrix{<:Real}, Φ::Matrix{<:Real})
    # find the number of observations in the observation sequence
    t = size(observation_sequence, 1) + 1
    # get the n+1th observation
    new_observation = sample(model, Φ[t:t, :], n=1)

    return vcat(observation_sequence, new_observation)
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
function loglikelihood(model::BernoulliRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)); observation_wise::Bool=false)
    # confirm that the model has valid parameters
    validate_model(model)
    validate_data(model, Φ, Y, w)

    # add intercept if specified and not already included
    if model.include_intercept && size(Φ, 2) == length(model.β) - 1 
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    # calculate log likelihood
    p = logistic.(Φ * model.β)

    obs_wise_loglikelihood = w .* (Y .* log.(p) .+ (1 .- Y) .* log.(1 .- p))

    if !observation_wise
        
        return sum(obs_wise_loglikelihood)
    else
        return obs_wise_loglikelihood
    end
end


function define_objective(model::BernoulliRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    validate_model(model)
    validate_data(model, Φ, Y, w)

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
    validate_model(model)
    validate_data(model, Φ, Y, w)

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
    # confirm that the model has valid parameters
    validate_model(model)
    validate_data(model, Φ, Y, w)

    # minimize objective
    objective = define_objective(model, Φ, Y, w)
    objective_grad! = define_objective_gradient(model, Φ, Y, w)

    result = optimize(objective, objective_grad!, model.β, LBFGS())

    # update parameters
    model.β = result.minimizer

    # confirm that the model has valid parameters
    validate_model(model)
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
end

function validate_model(model::PoissonRegression)
    if model.include_intercept
        @assert size(model.β, 1) == model.input_dim + 1 "β must be of size (input_dim + 1) if an intercept/bias is included."
    else
        @assert size(model.β, 1) == model.input_dim
    end

    @assert model.λ >= 0.0
end

function validate_data(model::PoissonRegression, Φ=nothing, Y=nothing, w=nothing)
    if !isnothing(Φ)
        @assert size(Φ, 2) == model.input_dim "Number of columns in Φ must be equal to the input dimension of the model."
    end
    if !isnothing(Y)
        @assert size(Y, 2) == 1 "PoissonRegression Y data should be a single column."
    end
    if !isnothing(w)
        @assert length(w) == size(Y, 1) "Length of w must be equal to the number of observations in Y."
    end
    if !isnothing(Φ) && !isnothing(Y)
        @assert size(Φ, 1) == size(Y, 1) "Number of rows (number of observations) in Φ and Y must be equal."
    end
end

function PoissonRegression(; 
    input_dim::Int, 
    include_intercept::Bool = true, 
    β::Vector{<:Real} = if include_intercept zeros(input_dim + 1) else zeros(input_dim) end,
    λ::Float64 = 0.0)

    new_model = PoissonRegression(input_dim, β, include_intercept, λ)

    validate_model(new_model)
    
    return new_model
end


function sample(model::PoissonRegression, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))
    @assert n <= size(Φ, 1) "n must be less than or equal to the number of observations in Φ."
    # cut the length of Φ to n
    Φ = Φ[1:n, :]

    # confirm that the model has valid parameters
    validate_model(model)
    validate_data(model, Φ)

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

# custom sampling function for the HMM. Returns observation_sequence with new observation appended to bottom.
function sample(model::PoissonRegression, observation_sequence::Matrix{<:Real}, Φ::Matrix{<:Real})
    # find the number of observations in the observation sequence
    t = size(observation_sequence, 1) + 1
    # get the n+1th observation
    new_observation = sample(model, Φ[t:t, :], n=1)

    return vcat(observation_sequence, new_observation)
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
function loglikelihood(model::PoissonRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)); observation_wise::Bool=false)
    # confirm that the model has valid parameters
    validate_model(model)
    validate_data(model, Φ, Y, w)

    # confirm that the model has been fit
    @assert !isempty(model.β) "Model parameters not initialized, please call fit! first."
    # add intercept if specified
    if model.include_intercept && size(Φ, 2) == length(model.β) - 1
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    # calculate log likelihood
    λ = exp.(Φ * model.β)

    obs_wise_loglikelihood = w .* (Y .* log.(λ) .- λ .- loggamma.(Int.(Y) .+ 1))

    if !observation_wise
        return sum(obs_wise_loglikelihood)
    else
        return obs_wise_loglikelihood
    end
end


function define_objective(model::PoissonRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    validate_model(model)
    validate_data(model, Φ, Y, w)

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
    validate_model(model)
    validate_data(model, Φ, Y, w)

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
    # confirm that the model has valid parameters
    validate_model(model)
    validate_data(model, Φ, Y, w)

    # minimize objective
    objective = define_objective(model, Φ, Y, w)
    objective_grad! = define_objective_gradient(model, Φ, Y, w)

    result = optimize(objective, objective_grad!, model.β, LBFGS())

    # update parameters
    model.β = result.minimizer

    # confirm that the model has valid parameters
    validate_model(model)
end





mutable struct AutoRegression <: RegressionModel
    data_dim::Int
    order::Int
    innerGaussianRegression::GaussianRegression
end

# define getters for innerGaussianRegression fields
function Base.getproperty(model::AutoRegression, sym::Symbol)
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
function Base.setproperty!(model::AutoRegression, sym::Symbol, value)
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

function validate_model(model::AutoRegression)
    @assert model.innerGaussianRegression.input_dim == model.data_dim * model.order
    @assert model.innerGaussianRegression.output_dim == model.data_dim

    validate_model(model.innerGaussianRegression)
end

function validate_data(model::AutoRegression, Y_prev=nothing, Y=nothing, w=nothing)
    if !isnothing(Y_prev)
        @assert size(Y_prev, 2) == model.data_dim "Number of columns in Y_prev must be equal to the data dimension of the model."
        @assert size(Y_prev, 1) == model.order "Number of rows in Y_prev must be equal to the order of the model."
    end
    if !isnothing(Y)
        @assert size(Y, 2) == model.data_dim "Number of columns in Y must be equal to the data dimension of the model."
    end
    if !isnothing(w)
        @assert length(w) == size(Y, 1) "Length of w must be equal to the number of observations in Y."
    end
end

function AutoRegression(; 
    data_dim::Int, 
    order::Int, 
    include_intercept::Bool = true, 
    β::Matrix{<:Real} = if include_intercept zeros(data_dim * order + 1, data_dim) else zeros(data_dim * order, data_dim) end,
    Σ::Matrix{<:Real} = Matrix{Float64}(I, data_dim, data_dim),
    λ::Float64=0.0)

    innerGaussianRegression = GaussianRegression(
        input_dim=data_dim * order, 
        output_dim=data_dim, 
        β=β,
        Σ=Σ,
        include_intercept=include_intercept, 
        λ=λ)

    model = AutoRegression(data_dim, order, innerGaussianRegression)

    validate_model(model)

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


function _sample(model::AutoRegression, Y_prev::Matrix{<:Real})
    Φ_gaussian = AR_to_Gaussian_data(Y_prev)
    return sample(model.innerGaussianRegression, Φ_gaussian)
end

function sample(model::AutoRegression, Y_prev::Matrix{<:Real}; n::Int=1)
    # confirm that the model has valid parameters
    validate_model(model)
    validate_data(model, Y_prev)

    Y = zeros(n, model.data_dim)

    for i in 1:n
        Y[i, :] = _sample(model, Y_prev)

        old_part = Y_prev[2:end, :]
        new_part = Y[i, :]

        old_part = reshape(old_part, model.order - 1, model.data_dim)
        new_part = reshape(new_part, 1, model.data_dim)

        Y_prev = vcat(old_part, new_part)
    end
    
    return Y
end

# custom sampling function for the HMM. Returns observation_sequence with new observation appended to bottom.
function sample(model::AutoRegression, observation_sequence::Matrix{<:Real}, Y_prev::Matrix{<:Real})

    full_sequence = vcat(Y_prev, observation_sequence)

    # get the n+1th observation
    new_observation = sample(model, full_sequence[end-model.order+1:end, :], n=1)

    return vcat(observation_sequence, new_observation)
end


function loglikelihood(model::AutoRegression, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real}; observation_wise::Bool=false)
    # confirm that the model has valid parameters
    validate_model(model)
    validate_data(model, Y_prev, Y)

    Φ_gaussian = AR_to_Gaussian_data(Y_prev, Y)

    return loglikelihood(model.innerGaussianRegression, Φ_gaussian, Y; observation_wise=observation_wise)
end


function fit!(model::AutoRegression, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    # confirm that the model has valid parameters
    validate_model(model)
    validate_data(model, Y_prev, Y, w)

    Φ_gaussian = AR_to_Gaussian_data(Y_prev, Y)

    fit!(model.innerGaussianRegression, Φ_gaussian, Y, w)

    # confirm that the model has valid parameters
    validate_model(model)
end