export BernoulliRegressionEmission, PoissonRegressionEmission, AutoRegressionEmission, fit!, loglikelihood, least_squares, update_variance!, sample

# below used in notebooks and unit tests
export define_objective, define_objective_gradient
export getproperty, setproperty!


"""
    BernoulliRegression

A Bernoulli regression model.

# Fields
- `input_dim::Int`: Dimension of the input data.
- `include_intercept::Bool = true`: Whether to include an intercept term.
- `β::Vector{<:Real} = if include_intercept zeros(input_dim + 1) else zeros(input_dim) end`: Coefficients of the model. The first element is the intercept term, if included.
- `λ::Float64 = 0.0`: Regularization parameter.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = BernoulliRegression(input_dim=2)
# output
```
"""
mutable struct BernoulliRegressionEmission <: EmissionModel
    input_dim::Int
    β::Vector{<:Real}
    include_intercept::Bool
    λ::Float64
end


function validate_model(model::BernoulliRegressionEmission)
    if model.include_intercept
        @assert size(model.β, 1) == model.input_dim + 1 "β must be of size (input_dim + 1) if an intercept/bias is included."
    else
        @assert size(model.β, 1) == model.input_dim
    end

    @assert model.λ >= 0.0
end

function validate_data(model::BernoulliRegressionEmission, Φ=nothing, Y=nothing, w=nothing)
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

function BernoulliRegressionEmission(; 
    input_dim::Int, 
    include_intercept::Bool = true, 
    β::Vector{<:Real} = if include_intercept zeros(input_dim + 1) else zeros(input_dim) end,
    λ::Float64 = 0.0)

    new_model = BernoulliRegressionEmission(input_dim, β, include_intercept, λ)

    validate_model(new_model)
    
    return new_model
end


"""
    sample(model::BernoulliRegression, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))

Generate `n` samples from a Bernoulli regression model. Returns a matrix of size `(n, 1)`.

# Arguments
- `model::BernoulliRegression`: Bernoulli regression model.
- `Φ::Matrix{<:Real}`: Design matrix of shape `(n, input_dim)`.
- `n::Int=size(Φ, 1)`: Number of samples to generate.

# Returns
- `Y::Matrix{<:Real}`: Matrix of samples of shape `(n, 1)`.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = BernoulliRegression(input_dim=2)
Φ = rand(100, 2)
Y = sample(model, Φ)
# output
```
"""
function sample(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))
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


"""
    loglikelihood(model::BernoulliRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

Calculate the log-likelihood of a Bernoulli regression model.

# Arguments
- `model::BernoulliRegression`: Bernoulli regression model.
- `Φ::Matrix{<:Real}`: Design matrix of shape `(n, input_dim)`.
- `Y::Matrix{<:Real}`: Response matrix of shape `(n, 1)`.
- `w::Vector{Float64}`: Weights of the data points. Should be a vector of size `n`.

# Returns
- `loglikelihood::Float64`: Log-likelihood of the model.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = BernoulliRegression(input_dim=2)
Φ = rand(100, 2)
Y = sample(model, Φ)
loglikelihood(model, Φ, Y)
# output
```
"""
function loglikelihood(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)); observation_wise::Bool=false)
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

        
    return sum(obs_wise_loglikelihood)
end


function define_objective(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
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


function define_objective_gradient(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
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
    fit!(model::BernoulliRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

Fit a Bernoulli regression model using maximum likelihood estimation.

# Arguments
- `model::BernoulliRegression`: Bernoulli regression model.
- `Φ::Matrix{<:Real}`: Design matrix of shape `(n, input_dim)`.
- `Y::Matrix{<:Real}`: Response matrix of shape `(n, 1)`.
- `w::Vector{Float64}`: Weights of the data points. Should be a vector of size `n`.

# Examples
```jldoctest; output = true
true_model = BernoulliRegression(input_dim=2)
Φ = rand(100, 2)
Y = sample(true_model, Φ)

est_model = BernoulliRegression(input_dim=2)
fit!(est_model, Φ, Y)

loglikelihood(est_model, Φ, Y) > loglikelihood(true_model, Φ, Y)

# output
true
```
"""
function fit!(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
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
    PoissonRegression

A Poisson regression model.

# Fields
- `input_dim::Int`: Dimension of the input data.
- `include_intercept::Bool = true`: Whether to include an intercept term.
- `β::Vector{<:Real} = if include_intercept zeros(input_dim + 1) else zeros(input_dim) end`: Coefficients of the model. The first element is the intercept term, if included.
- `λ::Float64 = 0.0`: Regularization parameter.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = PoissonRegression(input_dim=2)
# output
```
"""
mutable struct PoissonRegressionEmission <: EmissionModel
    input_dim::Int
    β::Vector{<:Real}
    include_intercept::Bool
    λ::Float64
end

function validate_model(model::PoissonRegressionEmission)
    if model.include_intercept
        @assert size(model.β, 1) == model.input_dim + 1 "β must be of size (input_dim + 1) if an intercept/bias is included."
    else
        @assert size(model.β, 1) == model.input_dim
    end

    @assert model.λ >= 0.0
end

function validate_data(model::PoissonRegressionEmission, Φ=nothing, Y=nothing, w=nothing)
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

function PoissonRegressionEmission(; 
    input_dim::Int, 
    include_intercept::Bool = true, 
    β::Vector{<:Real} = if include_intercept zeros(input_dim + 1) else zeros(input_dim) end,
    λ::Float64 = 0.0)

    new_model = PoissonRegressionEmission(input_dim, β, include_intercept, λ)

    validate_model(new_model)
    
    return new_model
end

"""
    sample(model::PoissonRegression, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))

Generate `n` samples from a Poisson regression model. Returns a matrix of size `(n, 1)`.

# Arguments
- `model::PoissonRegression`: Poisson regression model.
- `Φ::Matrix{<:Real}`: Design matrix of shape `(n, input_dim)`.
- `n::Int=size(Φ, 1)`: Number of samples to generate.

# Returns
- `Y::Matrix{<:Real}`: Matrix of samples of shape `(n, 1)`.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = PoissonRegression(input_dim=2)
Φ = rand(100, 2)
Y = sample(model, Φ)
# output
```
"""
function sample(model::PoissonRegressionEmission, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))
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
function hmm_sample(model::PoissonRegressionEmission, observation_sequence::Matrix{<:Real}, Φ::Matrix{<:Real})
    # find the number of observations in the observation sequence
    t = size(observation_sequence, 1) + 1
    # get the n+1th observation
    new_observation = sample(model, Φ[t:t, :], n=1)

    return vcat(observation_sequence, new_observation)
end


"""
    loglikelihood(model::PoissonRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

Calculate the log-likelihood of a Poisson regression model.

# Arguments
- `model::PoissonRegression`: Poisson regression model.
- `Φ::Matrix{<:Real}`: Design matrix of shape `(n, input_dim)`.
- `Y::Matrix{<:Real}`: Response matrix of shape `(n, 1)`.
- `w::Vector{Float64}`: Weights of the data points. Should be a vector of size `n`.

# Returns
- `loglikelihood::Float64`: Log-likelihood of the model.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = PoissonRegression(input_dim=2)
Φ = rand(100, 2)
Y = sample(model, Φ)
loglikelihood(model, Φ, Y)
# output
```
"""
function loglikelihood(model::PoissonRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)); observation_wise::Bool=false)
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


function define_objective(model::PoissonRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
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

function define_objective_gradient(model::PoissonRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
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




function gradient!(grad::Vector{Float64}, model::PoissonRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y,1)))
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
    fit!(model::PoissonRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

Fit a Poisson regression model using maximum likelihood estimation.

# Arguments
- `model::PoissonRegression`: Poisson regression model.
- `Φ::Matrix{<:Real}`: Design matrix of shape `(n, input_dim)`.
- `Y::Matrix{<:Real}`: Response matrix of shape `(n, 1)`.
- `w::Vector{Float64}`: Weights of the data points. Should be a vector of size `n`.

# Examples
```jldoctest; output = true
true_model = PoissonRegression(input_dim=2)
Φ = rand(100, 2)
Y = sample(true_model, Φ)

est_model = PoissonRegression(input_dim=2)
fit!(est_model, Φ, Y)

loglikelihood(est_model, Φ, Y) > loglikelihood(true_model, Φ, Y)

# output
true
```
"""
function fit!(model::PoissonRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
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


# """
#     AutoRegression

# An autoregressive model.

# # Fields
# - `output_dim::Int`: Dimension of the output data.
# - `order::Int`: Order of the autoregressive model.
# - `include_intercept::Bool = true`: Whether to include an intercept term. If true, the row of `β` is the intercept terms.
# - `β::Matrix{<:Real} = if include_intercept zeros(output_dim * order + 1, output_dim) else zeros(output_dim * order, output_dim) end`: Coefficients of the model. The top `output_dim`x`output_dim` block is coefficients for the first order, the next `output_dim`x`output_dim` block is coefficients for the second order, and so on.
# - `Σ::Matrix{<:Real} = Matrix{Float64}(I, output_dim, output_dim)`: Covariance matrix of the model.
# - `λ::Float64 = 0.0`: Regularization parameter.

# # Examples
# ```jldoctest; output = false, filter = r"(?s).*" => s""
# β = [0 0; 1.0 0.0; 0.0 1.0; 1.0 0.0; 0.0 1.0]
# model = AutoRegression(output_dim=2, order=2, β=β)
# # output
# ```
# """
mutable struct AutoRegressionEmission <: EmissionModel
    output_dim::Int
    order::Int
    #innerGaussianRegression::GaussianRegressionEmission
end

# # define getters for innerGaussianRegression fields
# function Base.getproperty(model::AutoRegressionEmission, sym::Symbol)
#     if sym === :β
#         return model.innerGaussianRegression.β
#     elseif sym === :Σ
#         return model.innerGaussianRegression.Σ
#     elseif sym === :include_intercept
#         return model.innerGaussianRegression.include_intercept
#     elseif sym === :λ
#         return model.innerGaussianRegression.λ
#     else # fallback to getfield
#         return getfield(model, sym)
#     end
# end

# # define setters for innerGaussianRegression fields
# function Base.setproperty!(model::AutoRegressionEmission, sym::Symbol, value)
#     if sym === :β
#         model.innerGaussianRegression.β = value
#     elseif sym === :Σ
#         model.innerGaussianRegression.Σ = value
#     elseif sym === :λ
#         model.innerGaussianRegression.λ = value
#     else # fallback to setfield!
#         setfield!(model, sym, value)
#     end
# end

# function validate_model(model::AutoRegressionEmission)
#     @assert model.innerGaussianRegression.input_dim == model.output_dim * model.order
#     @assert model.innerGaussianRegression.output_dim == model.output_dim

#     validate_model(model.innerGaussianRegression)
# end

# function validate_data(model::AutoRegressionEmission, Y_prev=nothing, Y=nothing, w=nothing)
#     if !isnothing(Y_prev)
#         @assert size(Y_prev, 2) == model.output_dim "Number of columns in Y_prev must be equal to the data dimension of the model."
#         @assert size(Y_prev, 1) == model.order "Number of rows in Y_prev must be equal to the order of the model. Got: rows=$(size(Y_prev, 1)) and order=$(model.order)"
#     end
#     if !isnothing(Y)
#         @assert size(Y, 2) == model.output_dim "Number of columns in Y must be equal to the data dimension of the model."
#     end
#     if !isnothing(w)
#         @assert length(w) == size(Y, 1) "Length of w must be equal to the number of observations in Y."
#     end
# end

# function AutoRegressionEmission(; 
#     output_dim::Int, 
#     order::Int, 
#     include_intercept::Bool = true, 
#     β::Matrix{<:Real} = if include_intercept zeros(output_dim * order + 1, output_dim) else zeros(output_dim * order, output_dim) end,
#     Σ::Matrix{<:Real} = Matrix{Float64}(I, output_dim, output_dim),
#     λ::Float64=0.0)

#     innerGaussianRegression = GaussianRegressionEmission(
#         input_dim=output_dim * order, 
#         output_dim=output_dim, 
#         β=β,
#         Σ=Σ,
#         include_intercept=include_intercept, 
#         λ=λ)

#     model = AutoRegressionEmission(output_dim, order, innerGaussianRegression)

#     validate_model(model)

#     return model
# end

# function AR_to_Gaussian_data(Y_prev::Matrix{<:Real})
#     # take each row of Y_prev and stack them horizontally to form the input row matrix Φ_gaussian
#     Φ_gaussian = vcat([Y_prev[i, :] for i in 1:size(Y_prev, 1)]...)
#     Φ_gaussian = reshape(Φ_gaussian, 1, :)

#     return Φ_gaussian
# end

# function AR_to_Gaussian_data(Y_prev::Matrix{<:Real}, Y::Matrix{<:Real})
#     order = size(Y_prev, 1)
#     output_dim = size(Y_prev, 2)
#     Φ_gaussian = zeros(size(Y, 1), output_dim * order)

#     for i in 1:size(Y, 1)
#         Φ_gaussian[i, :] = AR_to_Gaussian_data(Y_prev)


#         old_part = Y_prev[2:end, :]
#         new_part = Y[i, :]

#         old_part = reshape(old_part, order - 1, output_dim)
#         new_part = reshape(new_part, 1, output_dim)

#         Y_prev = vcat(old_part, new_part)
#     end
   

#     return Φ_gaussian
# end


# function _sample(model::AutoRegressionEmission, Y_prev::Matrix{<:Real})
#     Φ_gaussian = AR_to_Gaussian_data(Y_prev)
#     return sample(model.innerGaussianRegression, Φ_gaussian)
# end


# """ 
#     sample(model::AutoRegression, Y_prev::Matrix{<:Real}; n::Int=1)

# Generate `n` samples from an autoregressive model. Returns a matrix of size `(n, output_dim)`.

# # Arguments
# - `model::AutoRegression`: Autoregressive model.
# - `Y_prev::Matrix{<:Real}`: Matrix of shape `(order, output_dim)` containing the previous samples.
# - `n::Int=1`: Number of samples to generate.

# # Returns
# - `Y::Matrix{<:Real}`: Matrix of samples of shape `(n, output_dim)`.

# # Examples
# ```jldoctest; output = false, filter = r"(?s).*" => s""
# model = AutoRegression(output_dim=2, order=2)
# Y_prev = rand(2, 2)
# Y = sample(model, Y_prev, n=10)
# # output
# ```
# """
# function sample(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}; n::Int=1)
#     # confirm that the model has valid parameters
#     validate_model(model)
#     validate_data(model, Y_prev)

#     Y = zeros(n, model.output_dim)

#     for i in 1:n
#         Y[i, :] = _sample(model, Y_prev)

#         old_part = Y_prev[2:end, :]
#         new_part = Y[i, :]

#         old_part = reshape(old_part, model.order - 1, model.output_dim)
#         new_part = reshape(new_part, 1, model.output_dim)

#         Y_prev = vcat(old_part, new_part)
#     end
    
#     return Y
# end

# # custom sampling function for the HMM. Returns observation_sequence with new observation appended to bottom.
# # not used. emission_sample() has replaced this.
# function hmm_sample(model::AutoRegressionEmission, observation_sequence::Matrix{<:Real}, Y_prev::Matrix{<:Real})

#     full_sequence = vcat(Y_prev, observation_sequence)

#     # get the n+1th observation
#     new_observation = sample(model, full_sequence[end-model.order+1:end, :], n=1)

#     return vcat(observation_sequence, new_observation)
# end

# """
#     loglikelihood(model::AutoRegression, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real})

# Calculate the log-likelihood of an autoregressive model.

# # Arguments
# - `model::AutoRegression`: Autoregressive model.
# - `Y_prev::Matrix{<:Real}`: Matrix of shape `(order, output_dim)` containing the previous samples.
# - `Y::Matrix{<:Real}`: Matrix of shape `(n, output_dim)` containing the current samples.

# # Returns
# - `loglikelihood::Float64`: Log-likelihood of the model.

# # Examples
# ```jldoctest; output = false, filter = r"(?s).*" => s""
# model = AutoRegression(output_dim=2, order=2)
# Y_prev = rand(2, 2)
# Y = sample(model, Y_prev, n=10)

# loglikelihood(model, Y_prev, Y)
# # output
# ```
# """
# function loglikelihood(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real})
#     # confirm that the model has valid parameters
#     validate_model(model)
#     validate_data(model, Y_prev, Y)

#     Φ_gaussian = AR_to_Gaussian_data(Y_prev, Y)

#     return loglikelihood(model.innerGaussianRegression, Φ_gaussian, Y)
# end

# """
#     fit!(model::AutoRegression, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

# Fit an autoregressive model using maximum likelihood estimation.

# # Arguments
# - `model::AutoRegression`: Autoregressive model.
# - `Y_prev::Matrix{<:Real}`: Matrix of shape `(order, output_dim)` containing the previous samples.
# - `Y::Matrix{<:Real}`: Matrix of shape `(n, output_dim)` containing the current samples.
# - `w::Vector{Float64}`: Weights of the data points. Should be a vector of size `n`.

# # Examples
# ```jldoctest; output = true
# true_model = AutoRegression(output_dim=2, order=2)
# Y_prev = rand(2, 2)
# Y = sample(true_model, Y_prev, n=10)

# est_model = AutoRegression(output_dim=2, order=2)
# fit!(est_model, Y_prev, Y)

# loglikelihood(est_model, Y_prev, Y) > loglikelihood(true_model, Y_prev, Y)

# # output
# true
# ```
# """
# function fit!(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
#     # confirm that the model has valid parameters
#     validate_model(model)
#     validate_data(model, Y_prev, Y, w)

#     Φ_gaussian = AR_to_Gaussian_data(Y_prev, Y)

#     fit!(model.innerGaussianRegression, Φ_gaussian, Y, w)

#     # confirm that the model has valid parameters
#     validate_model(model)
# end

