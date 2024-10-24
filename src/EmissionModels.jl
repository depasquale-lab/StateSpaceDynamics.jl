export Emission, getproperty, setproperty!
export GaussianEmission, validate_data, emission_sample, emission_loglikelihood, emission_fit!
export SwitchingGaussianRegression, SwitchingBernoulliRegressionm, SwitchingAutoRegression, GaussianHMM

"""
Every emission model must implement the following functions:

- emission_sample(model::EmissionModel, data...; observation_sequence)
    The point of this function is to iteratively sample from the emission model through repeated calls of the form:
    `
    observation_sequence = emission_sample(model, data...)
    observation_sequence = emission_sample(model, data..., observation_sequence=observation_sequence)
    observation_sequence = emission_sample(model, data..., observation_sequence=observation_sequence)
    observation_sequence = emission_sample(model, data..., observation_sequence=observation_sequence)
    `
    Et cetera.

    NOTE: The observation_sequence argument is optional, and when it is not passed in, the function should return a NEW observation sequence with one observation.
- emission_loglikelihood(model::EmissionModel, data...)
    This function should return a vector of loglikelihoods for each observation in the data.
- emission_fit!(model::EmissionModel, data..., weights)
    This function should fit the model to the data, with weights for each observation.

Make sure to add any new emission models to the Emission function at the end of this file!!
"""


"""
    GaussianEmission <: EmissionModel

A mutable struct representing a Gaussian emission model, which wraps around a `Gaussian` model.

# Fields
- `inner_model::Gaussian`: The underlying Gaussian model used for the emissions.
"""
# mutable struct GaussianEmission <: EmissionModel
#     inner_model:: Gaussian
# end


"""
    emission_sample(model::GaussianEmission; observation_sequence::Matrix{<:Real}=Matrix{Float64}(undef, 0, model.output_dim))

Generate a sample from the given Gaussian emission model and append it to the provided observation sequence.

# Arguments
- `model::GaussianEmission`: The Gaussian emission model to sample from.
- `observation_sequence::Matrix{<:Real}`: The sequence of observations to which the new sample will be appended (defaults to an empty matrix with the same output dimension as the model).

# Returns
- `Matrix{Float64}`: The updated observation sequence with the new sample appended.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = GaussianEmission(Gaussian(output_dim=2))
sequence = emission_sample(model)
sequence = emission_sample(model, observation_sequence=sequence)
# output
"""
function emission_sample(model::GaussianEmission; observation_sequence::Matrix{<:Real}=Matrix{Float64}(undef, 0, model.output_dim))
    validate_model(model)

    raw_samples = rand(MvNormal(model.μ, model.Σ), 1)    

    return vcat(observation_sequence, Matrix(raw_samples'))
end


"""
    emission_loglikelihood(model::GaussianEmission, Y::Matrix{<:Real})

Calculate the log likelihood of the data `Y` given the Gaussian emission model.

# Arguments
- `model::GaussianEmission`: The Gaussian emission model for which to calculate the log likelihood.
- `Y::Matrix{<:Real}`: The data matrix, where each row represents an observation.

# Returns
- `Vector{Float64}`: A vector of log likelihoods, one for each observation in the data.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = GaussianEmission(Gaussian(output_dim=2))
Y = randn(10, 2)  # Observations x Features
loglikelihoods = SSD.emission_loglikelihood(model, Y)
# output
"""
function emission_loglikelihood(model::GaussianEmission, Y::Matrix{<:Real})
    validate_model(model)
    validate_data(model, Y)

    # calculate inverse of covariance matrix
    Σ_inv = inv(model.Σ)

    # calculate log likelihood
    residuals = broadcast(-, Y, model.μ')
    observation_wise_loglikelihood = zeros(size(Y, 1))

    # calculate observation wise loglikelihood (a vector of loglikelihoods for each observation)
    @threads for i in 1:size(Y, 1)
        observation_wise_loglikelihood[i] = -0.5 * size(Y, 2) * log(2π) - 0.5 * logdet(model.Σ) - 0.5 * sum(residuals[i, :] .* (Σ_inv * residuals[i, :]))
    end

    return observation_wise_loglikelihood
end


"""
    emission_fit!(model::GaussianEmission, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

Call the fit!() function in BasicModels.jl to fit the Gaussian emission model to the data `Y` using the provided weights `w`.

# Arguments
- `model::GaussianEmission`: The Gaussian emission model to be fit.
- `Y::Matrix{<:Real}`: The data matrix (Observations x Features)
- `w::Vector{Float64}`: A vector of weights corresponding to each observation (defaults to a vector of ones).

# Returns
- `Nothing`: The function modifies the model in place.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = GaussianEmission(Gaussian(output_dim=2))
Y = randn(10, 2)
emission_fit!(model, Y)
# output
"""
# function emission_fit!(model::GaussianEmission, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
#     fit!(model.inner_model, Y, w)
# end

function emission_fit!(model::GaussianEmission, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    fit!(model, Y, w)
end


"""
    GaussianHMM(; K::Int, output_dim::Int, A::Matrix{<:Real}=initialize_transition_matrix(K), πₖ::Vector{Float64}=initialize_state_distribution(K))

Create a Hidden Markov Model with Gaussian Emissions

# Arguments
- `K::Int`: The number of hidden states
- `output_dim::Int`: The dimensionality of the observation
- `A::Matrix{<:Real}=initialize_transition_matrix(K)`: The transition matrix of the HMM (defaults to random initialization)
- `πₖ::Vector{Float64}=initialize_state_distribution(K)`: The initial state distribution of the HMM (defaults to random initialization)

# Returns
- `::HiddenMarkovModel`: Hidden Markov Model Object with Gaussian Emissions

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = GaussianHMM(K=2, output_dim=5)
# output
```
"""
function GaussianHMM(; K::Int, output_dim::Int, A::Matrix{<:Real}=initialize_transition_matrix(K), πₖ::Vector{Float64}=initialize_state_distribution(K))
    # Create emission models
    emissions = [GaussianEmission(output_dim=output_dim) for _ in 1:K]
    # Return constructed GaussianHMM
    return HiddenMarkovModel(K=K, B=emissions, A=A, πₖ=πₖ)
end


"""
    GaussianRegressionEmission <: EmissionModel

A mutable struct representing a Gaussian regression emission model, which wraps around a `GaussianRegression` model.

# Fields
- `inner_model::GaussianRegression`: The underlying Gaussian regression model used for the emissions.
"""
# mutable struct GaussianRegressionEmission <: EmissionModel
#     inner_model:: GaussianRegression
# end


"""
    emission_sample(model::GaussianRegressionEmission, Φ::Matrix{<:Real}; observation_sequence::Matrix{<:Real}=Matrix{Float64}(undef, 0, model.output_dim))

Generate a sample from the given Gaussian regression emission model using the input features `Φ`, and append it to the provided observation sequence.

# Arguments
- `model::GaussianRegressionEmission`: The Gaussian regression emission model to sample from.
- `Φ::Matrix{<:Real}`: The input features matrix (Observations x Features).
- `observation_sequence::Matrix{<:Real}`: The sequence of observations to which the new sample will be appended (defaults to an empty matrix with the same output dimension as the model).

# Returns
- `Matrix{Float64}`: The updated observation sequence with the new sample appended.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = GaussianRegressionEmission(GaussianRegression(input_dim=3, output_dim=2))
Φ = randn(10, 3)
sequence = emission_sample(model, Φ)
sequence = emission_sample(model, Φ, observation_sequence=sequence)
# output
"""
function emission_sample(model::GaussianRegressionEmission, Φ::Matrix{<:Real}; observation_sequence::Matrix{<:Real}=Matrix{Float64}(undef, 0, model.output_dim))
    validate_model(model)
    validate_data(model, Φ)

    # find the number of observations in the observation sequence
    t = size(observation_sequence, 1) + 1
    # get the n+1th observation
    new_observation = sample(model, Φ[t:t, :], n=1)

    return vcat(observation_sequence, new_observation)
end


"""
    emission_loglikelihood(model::GaussianRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real})

Calculate the log likelihood of the data `Y` given the Gaussian regression emission model and the input features `Φ`.

# Arguments
- `model::GaussianRegressionEmission`: The Gaussian regression emission model for which to calculate the log likelihood.
- `Φ::Matrix{<:Real}`: The input features matrix (Observations x Features).
- `Y::Matrix{<:Real}`: The data matrix (Observations x Features).

# Returns
- `Vector{Float64}`: A vector of log likelihoods, one for each observation in the data.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = GaussianRegressionEmission(GaussianRegression(input_dim=3, output_dim=2))
Φ = randn(10, 3)
Y = randn(10, 2)
loglikelihoods = emission_loglikelihood(model, Φ, Y)
# output
"""
function emission_loglikelihood(model::GaussianRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real})
    validate_model(model)
    validate_data(model, Φ, Y)

    # calculate observation wise likelihoods for all states
    observation_wise_loglikelihood = zeros(size(Y, 1))

    # calculate observation wise loglikelihood (a vector of loglikelihoods for each observation)
    @threads for i in 1:size(Y, 1)
        observation_wise_loglikelihood[i] = loglikelihood(model, Φ[i:i, :], Y[i:i, :])
    end

    return observation_wise_loglikelihood
end


"""
    emission_fit!(model::GaussianRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

Call the fit!() function in RegressionModels.jl to fit the Gaussian regression emission model to the data `Y` using the input features `Φ` and the provided weights `w`.

# Arguments
- `model::GaussianRegressionEmission`: The Gaussian regression emission model to be fitted.
- `Φ::Matrix{<:Real}`: The input features matrix (Observations x Features).
- `Y::Matrix{<:Real}`: The data matrix (Observations x Features).
- `w::Vector{Float64}`: A vector of weights corresponding to each observation (defaults to a vector of ones).

# Returns
- `Nothing`: The function modifies the model in place.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = GaussianRegressionEmission(GaussianRegression(input_dim=3, output_dim=2))
Φ = randn(10, 3)
Y = randn(10, 2)
emission_fit!(model, Φ, Y)
# output
"""
# function emission_fit!(model::GaussianRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
#     fit!(model.inner_model, Φ, Y, w)
# end

function emission_fit!(model::GaussianRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    fit!(model, Φ, Y, w)
end


"""
    SwitchingGaussianRegression(; 
        K::Int,
        input_dim::Int,
        output_dim::Int,
        include_intercept::Bool = true,
        β::Matrix{<:Real} = if include_intercept
            zeros(input_dim + 1, output_dim)
        else
            zeros(input_dim, output_dim)
        end,
        Σ::Matrix{<:Real} = Matrix{Float64}(I, output_dim, output_dim),
        λ::Float64 = 0.0,
        A::Matrix{<:Real} = initialize_transition_matrix(K),
        πₖ::Vector{Float64} = initialize_state_distribution(K)
    )

Create a Switching Gaussian Regression Model

# Arguments
- `K::Int`: The number of hidden states.
- `input_dim::Int`: The dimensionality of the input features.
- `output_dim::Int`: The dimensionality of the output predictions.
- `include_intercept::Bool`: Whether to include an intercept in the regression model (default is `true`).
- `β::Matrix{<:Real}`: The regression coefficients (defaults to zeros based on `input_dim` and `output_dim`).
- `Σ::Matrix{<:Real}`: The covariance matrix of the Gaussian emissions (defaults to an identity matrix).
- `λ::Float64`: The regularization parameter for the regression (default is `0.0`).
- `A::Matrix{<:Real}`: The transition matrix of the Hidden Markov Model (defaults to random initialization).
- `πₖ::Vector{Float64}`: The initial state distribution of the Hidden Markov Model (defaults to random initialization).

# Returns
- `::HiddenMarkovModel`: A Switching Gaussian Regression Model
# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = SwitchingGaussianRegression(K=2, input_dim=5, output_dim=10)
# output
"""
function SwitchingGaussianRegression(; 
    K::Int,
    input_dim::Int,
    output_dim::Int,
    include_intercept::Bool = true,
    β::Matrix{<:Real} = if include_intercept
        zeros(input_dim + 1, output_dim)
    else
        zeros(input_dim, output_dim)
    end,
    Σ::Matrix{<:Real} = Matrix{Float64}(I, output_dim, output_dim),
    λ::Float64 = 0.0,
    A::Matrix{<:Real} = initialize_transition_matrix(K),
    πₖ::Vector{Float64} = initialize_state_distribution(K)
)
    # Create emission models
    emissions = [GaussianRegressionEmission(input_dim=input_dim, output_dim=output_dim, include_intercept=include_intercept, β=β, Σ=Σ, λ=λ) for _ in 1:K]

    # Return the HiddenMarkovModel
    return HiddenMarkovModel(K=K, B=emissions, A=A, πₖ=πₖ)
end


"""
    BernoulliRegressionEmission <: EmissionModel

A mutable struct representing a Bernoulli regression emission model, which wraps around a `BernoulliRegression` model.

# Fields
- `inner_model::BernoulliRegression`: The underlying Bernoulli regression model used for the emissions.
"""
mutable struct BernoulliRegressionEmission <: EmissionModel
    inner_model:: BernoulliRegression
end


"""
    emission_sample(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}; observation_sequence::Matrix{<:Real}=Matrix{Float64}(undef, 0, 1))

Generate a sample from the given Bernoulli regression emission model using the input features `Φ`, and append it to the provided observation sequence.

# Arguments
- `model::BernoulliRegressionEmission`: The Bernoulli regression emission model to sample from.
- `Φ::Matrix{<:Real}`: The input features matrix (Observations x Features).
- `observation_sequence::Matrix{<:Real}`: The sequence of observations to which the new sample will be appended (defaults to an empty matrix).

# Returns
- `Matrix{Float64}`: The updated observation sequence with the new sample appended.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = BernoulliRegressionEmission(BernoulliRegression(input_dim=3))
Φ = randn(10, 3)
sequence = emission_sample(model, Φ)
sequence = emission_sample(model, Φ, observation_sequence=sequence)
# output
"""
function emission_sample(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}; observation_sequence::Matrix{<:Real} = Matrix{Float64}(undef, 0, 1))
    # find the number of observations in the observation sequence
    t = size(observation_sequence, 1) + 1
    # get the n+1th observation
    new_observation = sample(model.inner_model, Φ[t:t, :], n=1)

    return vcat(observation_sequence, new_observation)
end


"""
    emission_loglikelihood(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

Calculate the log likelihood of the data `Y` given the Bernoulli regression emission model and the input features `Φ`. Optionally, a vector of weights `w` can be provided.

# Arguments
- `model::BernoulliRegressionEmission`: The Bernoulli regression emission model for which to calculate the log likelihood.
- `Φ::Matrix{<:Real}`: The input features matrix (Observations x Features).
- `Y::Matrix{<:Real}`: The data matrix (Observations x Features).
- `w::Vector{Float64}`: A vector of weights corresponding to each observation (defaults to a vector of ones).

# Returns
- `Vector{Float64}`: A vector of log likelihoods, one for each observation in the data.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = BernoulliRegressionEmission(BernoulliRegression(input_dim=3))
Φ = randn(10, 3)
Y = rand(Bool, 10, 1)
loglikelihoods = emission_loglikelihood(model, Φ, Y)
# output
"""
function emission_loglikelihood(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
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


    return obs_wise_loglikelihood
end


"""
    emission_fit!(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

Calls fit!() function in RegressionModels.jl to fit the Bernoulli regression emission model to the data `Y` using the input features `Φ` and the provided weights `w`.

# Arguments
- `model::BernoulliRegressionEmission`: The Bernoulli regression emission model to be fitted.
- `Φ::Matrix{<:Real}`: The input features matrix (Observations x Features).
- `Y::Matrix{<:Real}`: The data matrix (Observations x Features).
- `w::Vector{Float64}`: A vector of weights corresponding to each observation (defaults to a vector of ones).

# Returns
- `Nothing`: The function modifies the model in place.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = BernoulliRegressionEmission(BernoulliRegression(input_dim=3))
Φ = randn(10, 3)
Y = rand(Bool, 10, 1)
emission_fit!(model, Φ, Y)
# output
"""
function emission_fit!(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    fit!(model.inner_model, Φ, Y, w)
end


"""
    SwitchingBernoulliRegression(; K::Int, input_dim::Int, include_intercept::Bool=true, β::Vector{<:Real}=if include_intercept zeros(input_dim + 1) else zeros(input_dim) end, λ::Float64=0.0, A::Matrix{<:Real}=initialize_transition_matrix(K), πₖ::Vector{Float64}=initialize_state_distribution(K))

Create a Switching Bernoulli Regression Model

# Arguments
- `K::Int`: The number of hidden states.
- `input_dim::Int`: The dimensionality of the input data.
- `include_intercept::Bool=true`: Whether to include an intercept in the regression model (defaults to true).
- `β::Vector{<:Real}`: The regression coefficients (defaults to zeros). 
- `λ::Float64=0.0`: Regularization parameter for the regression (defaults to zero).
- `A::Matrix{<:Real}=initialize_transition_matrix(K)`: The transition matrix of the HMM (defaults to random initialization).
- `πₖ::Vector{Float64}=initialize_state_distribution(K)`: The initial state distribution of the HMM (defaults to random initialization).

# Returns
- `::HiddenMarkovModel`: A Switching Bernoulli Regression Model

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = SwitchingBernoulliRegression(K=2, input_dim=5)
# output
"""
function SwitchingBernoulliRegression(; 
    K::Int,
    input_dim::Int,
    include_intercept::Bool=true,
    β::Vector{<:Real} = if include_intercept zeros(input_dim + 1) else zeros(input_dim) end,
    λ::Float64 = 0.0,
    A::Matrix{<:Real} = initialize_transition_matrix(K),
    πₖ::Vector{Float64} = initialize_state_distribution(K)
)
    # Create emission models
    emissions = [BernoulliRegression(input_dim=input_dim, include_intercept=include_intercept, β=β, λ=λ) for _ in 1:K]
    # Return the HiddenMarkovModel
    return HiddenMarkovModel(K=K, B=emissions, A=A, πₖ=πₖ)
end


"""
    AutoRegressionEmission <: EmissionModel

A mutable struct representing an autoregressive emission model, which wraps around an `AutoRegression` model.

# Fields
- `inner_model::AutoRegression`: The underlying autoregressive model used for the emissions.
"""
mutable struct AutoRegressionEmission <: EmissionModel
    inner_model:: AutoRegression
end


"""
    emission_sample(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}; observation_sequence::Matrix{<:Real}=Matrix{Float64}(undef, 0, model.output_dim))

Generate a sample from the given autoregressive emission model using the previous observations `Y_prev`, and append it to the provided observation sequence.

# Arguments
- `model::AutoRegressionEmission`: The autoregressive emission model to sample from.
- `Y_prev::Matrix{<:Real}`: The matrix of previous observations, where each row represents an observation.
- `observation_sequence::Matrix{<:Real}`: The sequence of observations to which the new sample will be appended (defaults to an empty matrix with appropriate dimensions).

# Returns
- `Matrix{Float64}`: The updated observation sequence with the new sample appended.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = AutoRegressionEmission(AutoRegression(output_dim=2, order=3))
Y_prev = randn(10, 2)
sequence = emission_sample(model, Y_prev)
sequence = emission_sample(model, Y_prev, observation_sequence=sequence)
# output
"""
function emission_sample(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}; observation_sequence::Matrix{<:Real}=Matrix{Float64}(undef, 0, model.output_dim))

    full_sequence = vcat(Y_prev, observation_sequence)

    # get the n+1th observation
    new_observation = sample(model.inner_model, full_sequence[end-model.order+1:end, :], n=1)

    return vcat(observation_sequence, new_observation)
end


"""
    emission_loglikelihood(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real})

Calculate the log likelihood of the data `Y` given the autoregressive emission model and the previous observations `Y_prev`.

# Arguments
- `model::AutoRegressionEmission`: The autoregressive emission model for which to calculate the log likelihood.
- `Y_prev::Matrix{<:Real}`: The matrix of previous observations, where each row represents an observation.
- `Y::Matrix{<:Real}`: The data matrix, where each row represents an observation.

# Returns
- `Vector{Float64}`: A vector of log likelihoods, one for each observation in the data.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = AutoRegressionEmission(AutoRegression(output_dim=2, order=10))
Y_prev = randn(10, 2)
Y = randn(10, 2)
loglikelihoods = emission_loglikelihood(model, Y_prev, Y)
# output
"""
function emission_loglikelihood(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real})
    # confirm that the model has valid parameters
    validate_model(model)
    validate_data(model, Y_prev, Y)

    Φ_gaussian = AR_to_Gaussian_data(Y_prev, Y)

    # extract inner gaussian regression and wrap it with a GaussianEmission
    innerGaussianRegression_emission = GaussianRegressionEmission(model.inner_model.innerGaussianRegression)

    return emission_loglikelihood(innerGaussianRegression_emission, Φ_gaussian, Y)
end


"""
    emission_fit!(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

Calls to fit!() function in RegressionModels.jl to fit the autoregressive emission model to the data `Y` using the previous observations `Y_prev` and the provided weights `w`.

# Arguments
- `model::AutoRegressionEmission`: The autoregressive emission model to be fitted.
- `Y_prev::Matrix{<:Real}`: The matrix of previous observations, where each row represents an observation.
- `Y::Matrix{<:Real}`: The data matrix, where each row represents an observation.
- `w::Vector{Float64}`: A vector of weights corresponding to each observation (defaults to a vector of ones).

# Returns
- `Nothing`: The function modifies the model in place.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = AutoRegressionEmission(AutoRegression(output_dim=2, order=10))
Y_prev = randn(10, 2)
Y = randn(10, 2)
emission_fit!(model, Y_prev, Y)
# output
"""
function emission_fit!(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    fit!(model.inner_model, Y_prev, Y, w)
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

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = SwitchingAutoRegression(K=3, output_dim=2, order=2)
# output
"""
function SwitchingAutoRegression(; 
    K::Int,
    output_dim::Int, 
    order::Int, 
    include_intercept::Bool = true, 
    β::Matrix{<:Real} = if include_intercept zeros(output_dim * order + 1, output_dim) else zeros(output_dim * order, output_dim) end,
    Σ::Matrix{<:Real} = Matrix{Float64}(I, output_dim, output_dim),
    λ::Float64=0.0,
    A::Matrix{<:Real} = initialize_transition_matrix(K),
    πₖ::Vector{Float64} = initialize_state_distribution(K)
)
    # Create the emissions
    emissions = [AutoRegression(output_dim=output_dim, order=order, include_intercept=include_intercept, β=β, Σ=Σ, λ=λ) for _ in 1:K]
    # Return the HiddenMarkovModel
    return HiddenMarkovModel(K=K, B=emissions, A=A, πₖ=πₖ)
end


"""
    CompositeModelEmission <: EmissionModel

A mutable struct representing a composite emission model that combines multiple emission models.

# Fields
- `inner_model::CompositeModel`: The underlying composite model used for the emissions.
"""
mutable struct CompositeModelEmission <: EmissionModel
    inner_model:: CompositeModel
end


"""
    validate_model(model::CompositeModelEmission)

Validate that the `CompositeModelEmission` and all its component emission models are correctly set up.

# Arguments
- `model::CompositeModelEmission`: The composite emission model to validate.

# Returns
- `Nothing`: This function does not return a value but throws an error if any component is not a valid emission model.
"""
function validate_model(model::CompositeModelEmission)
    validate_model(model.inner_model)

    for component in model.components
        if !(component isa EmissionModel)
            throw(ArgumentError("The model $(typeof(component)) is not a valid emission model."))
        end
    end
end

function emission_sample(model::CompositeModelEmission, input_data::Vector{}; observation_sequence::Vector{}=Vector())
    validate_model(model)

    if isempty(observation_sequence)
        for i in 1:length(model.components)
            push!(observation_sequence, (emission_sample(model.components[i], input_data[i]...),))
        end 
    else
        for i in 1:length(model.components)
            observation_sequence[i] = (emission_sample(model.components[i], input_data[i]...; observation_sequence=observation_sequence[i][1]),)
        end 
    end

    return observation_sequence
end

function emission_loglikelihood(model::CompositeModelEmission, input_data::Vector{}, output_data::Vector{})
    validate_model(model)
    validate_data(model, input_data, output_data)

    loglikelihoods = Vector{}(undef, length(model.components))


    for i in 1:length(model.components)
        loglikelihoods[i] = emission_loglikelihood(model.components[i], input_data[i]..., output_data[i]...)
    end
    return sum(loglikelihoods, dims=1)[1]
end

function emission_fit!(model::CompositeModelEmission, input_data::Vector{}, output_data::Vector{}, w::Vector{Float64}=Vector{Float64}())
    for i in 1:length(model.components)
        emission_fit!(model.components[i], input_data[i]..., output_data[i]..., w)
    end
end

"""
Validation Functions
"""

function validate_model(model::EmissionModel)
    validate_model(model.inner_model)
end


function validate_data(model::EmissionModel, data...)
    validate_data(model.inner_model, data...)
end


"""
Emission handler
"""
# function Emission(model::Model)
#     if model isa Gaussian
#         return GaussianEmission(model)
#     elseif model isa GaussianRegression
#         return GaussianRegressionEmission(model)
#     elseif model isa BernoulliRegression
#         return BernoulliRegressionEmission(model)
#     elseif model isa AutoRegression
#         return AutoRegressionEmission(model)
#     elseif model isa CompositeModel
#         emission_components = Emission.(model.components)
#         new_composite = CompositeModel(emission_components)
#         return CompositeModelEmission(new_composite)
#     else
#         # throw an error if the model is not a valid emission model
#         throw(ArgumentError("The model is not a valid emission model."))
#     end
# end 

