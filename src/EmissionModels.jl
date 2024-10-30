export Emission, getproperty, setproperty!
export GaussianEmission, sample, loglikelihood, fit!
export SwitchingGaussianRegression, SwitchingBernoulliRegression, SwitchingAutoRegression, GaussianHMM
export GaussianRegressionEmission
"""
Each emission model should have:
    1) Mutable struct definition
    2) Constructor Function
    3) Log_likelihood Function
    4) Sampling Function
    5) Fit Function (and associated gradient functions for regression models)
"""

"""
*** GAUSSIAN EMISSION FUNCTIONS ***
"""

"""
    mutable struct GaussianEmission <: EmissionModel

GaussianEmission model with mean and covariance.
"""
mutable struct GaussianEmission <: EmissionModel
    output_dim::Int # dimension of the data
    μ::Vector{<:Real}  # mean 
    Σ::Matrix{<:Real}  # covariance matrix
end


"""
    function GaussianEmission(; output_dim::Int, μ::Vector{<:Real}=zeros(output_dim), Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim))

Functon to create a GaussianEmission with given output dimension, mean, and covariance.

# Arguments
- `output_dim::Int`: The output dimension of the emission
- `μ::Vector{<:Real}=zeros(output_dim)`: The mean of the Gaussian
- `Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim))`: The covariance matrix of the Gaussian

# Returns
"""
function GaussianEmission(; 
    output_dim::Int, 
    μ::Vector{<:Real}=zeros(output_dim), 
    Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim))

    return GaussianEmission(output_dim, μ, Σ)
end


"""
    loglikelihood(model::GaussianEmission, Y::Matrix{<:Real})

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
loglikelihoods = SSD.loglikelihood(model, Y)
# output
"""
function loglikelihood(model::GaussianEmission, Y::Matrix{<:Real})
    Σ_inv = inv(model.Σ)
    residuals = broadcast(-, Y, model.μ')
    observation_wise_loglikelihood = zeros(size(Y, 1))

    for i in axes(Y, 1)
        observation_wise_loglikelihood[i] = -0.5 * size(Y, 2) * log(2π) - 
                                          0.5 * logdet(model.Σ) - 
                                          0.5 * sum(residuals[i, :] .* (Σ_inv * residuals[i, :]))
    end

    return observation_wise_loglikelihood
end


"""
    sample(model::Gaussian; n::Int=1)

Generate `n` samples from a Gaussian model. Returns a matrix of size `(n, output_dim)`.

# Examples
```jldoctest; output = false
model = Gaussian(output_dim=2)
samples = sample(model, n=3)

println(size(samples))

# output
(3, 2)
```
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

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
true_model = Gaussian(output_dim=2)
Y = sample(true_model, n=3)

est_model = Gaussian(output_dim=2)
fit!(est_model, Y)

# output
```
"""
function fit!(model::GaussianEmission, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    weighted_sum = sum(Y .* w, dims=1)
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

10/29/2024 3:02 PM refactoring checkpoint (Above functions are complete)

"""



"""
*** Gaussian Regression Functions ***
"""

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

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
β = rand(3, 1)
model = GaussianRegression(input_dim=2, output_dim=1, β=β)
# output
```
"""
mutable struct GaussianRegressionEmission <: EmissionModel
    input_dim::Int
    output_dim::Int
    β::Matrix{<:Real} # coefficient matrix of the model. Shape input_dim by output_dim. Column one is coefficients for target one, etc. The first row are the intercept terms, if included. 
    Σ::Matrix{<:Real} # covariance matrix of the model 
    include_intercept::Bool # whether to include an intercept term; if true, the first column of β is assumed to be the intercept/bias
    λ::Float64 # regularization parameter
end


function GaussianRegressionEmission(; 
    input_dim::Int, 
    output_dim::Int, 
    include_intercept::Bool = true, 
    β::Matrix{<:Real} = if include_intercept zeros(input_dim + 1, output_dim) else zeros(input_dim, output_dim) end,
    Σ::Matrix{<:Real} = Matrix{Float64}(I, output_dim, output_dim),
    λ::Float64 = 0.0)
    
    return GaussianRegressionEmission(input_dim, output_dim, β, Σ, include_intercept, λ)
end


"""
    sample(model::GaussianRegression, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))

Generate `n` samples from a Gaussian regression model. Returns a matrix of size `(n, output_dim)`.

# Arguments
- `model::GaussianRegression`: Gaussian regression model.
- `Φ::Matrix{<:Real}`: Design matrix of shape `(n, input_dim)`.
- `n::Int=size(Φ, 1)`: Number of samples to generate.

# Returns
- `Y::Matrix{<:Real}`: Matrix of samples of shape `(n, output_dim)`.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = GaussianRegression(input_dim=2, output_dim=1)
Φ = rand(100, 2)
Y = sample(model, Φ)
# output
"""
function sample(model::GaussianRegressionEmission, Φ::Matrix{<:Real}; n::Int=size(Φ, 1))
    @assert n <= size(Φ, 1) "n must be less than or equal to the number of observations in Φ."
    # cut the length of Φ to n
    Φ = Φ[1:n, :]

    # add intercept if specified
    if model.include_intercept
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end

    return Φ * model.β + rand(MvNormal(zeros(model.output_dim), model.Σ), size(Φ, 1))'
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

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = GaussianRegressionEmission(GaussianRegression(input_dim=3, output_dim=2))
Φ = randn(10, 3)
Y = randn(10, 2)
loglikelihoods = loglikelihood(model, Φ, Y)
# output
"""

# function loglikelihood(model::GaussianRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real})

#     # calculate observation wise likelihoods for all states
#     observation_wise_loglikelihood = zeros(size(Y, 1))

#     # calculate observation wise loglikelihood (a vector of loglikelihoods for each observation)
#     @threads for i in 1:size(Y, 1)
#         observation_wise_loglikelihood[i] = loglikelihood(model, Φ[i:i, :], Y[i:i, :])
#     end

#     return observation_wise_loglikelihood
# end

# function loglikelihood(model::GaussianRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real})
#     # add intercept if specified
#     if model.include_intercept
#         Φ = hcat(ones(size(Φ, 1)), Φ)
#     end

#     # calculate inverse of covariance matrix
#     Σ_inv = inv(model.Σ)

#     # calculate log likelihood
#     residuals = Y - Φ * model.β


#     loglikelihood = -0.5 * size(Φ, 1) * size(Φ, 2) * log(2π) - 0.5 * size(Φ, 1) * logdet(model.Σ) - 0.5 * sum(residuals .* (Σ_inv * residuals')')
#     return loglikelihood
# end

function loglikelihood(model::GaussianRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real})
    # Add intercept if specified
    X = model.include_intercept ? [ones(size(Φ, 1)) Φ] : Φ
    
    # Create MvNormal distribution for each observation
    μ = X * model.β
    dist = MvNormal(model.Σ)
    
    # Calculate log likelihood for each observation
    return [logpdf(dist, Y[i,:] .- μ[i]) for i in axes(Y, 1)]
end



"""
    loglikelihood(model::GaussianRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real})

Calculate the log-likelihood of a Gaussian regression model.

# Arguments
- `model::GaussianRegression`: Gaussian regression model.
- `Φ::Matrix{<:Real}`: Design matrix of shape `(n, input_dim)`. 
- `Y::Matrix{<:Real}`: Response matrix of shape `(n, output_dim)`.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = GaussianRegression(input_dim=2, output_dim=1)
Φ = rand(100, 2)
Y = sample(model, Φ)
loglikelihood(model, Φ, Y)
# output
```
"""



# assume covariance is the identity, so the log likelihood is just the negative squared error. Ignore loglikelihood terms that don't depend on β.
function define_objective(model::GaussianRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

    # add intercept if specified
    if model.include_intercept
        Φ = [ones(size(Φ, 1)) Φ]
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


function define_objective_gradient(model::GaussianRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
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


function update_variance!(model::GaussianRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

    # add intercept if specified
    if model.include_intercept
        Φ = [ones(size(Φ, 1)) Φ]
    end

    residuals = Y - Φ * model.β
    
    Σ = (residuals' * Diagonal(w) * residuals) / size(Φ, 1)

    # ensure rounding errors are not causing the covariance matrix to be non-positive definite
    model.Σ .= 0.5 * (Σ * Σ') 
end
"""
    fit!!(model::GaussianRegression, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

Fit a Gaussian regression emission model using maximum likelihood estimation and OLS.

# Arguments
- `model::GaussianRegression`: Gaussian regression model.
- `Φ::Matrix{<:Real}`: Design matrix of shape `(n, input_dim)`.
- `Y::Matrix{<:Real}`: Response matrix of shape `(n, output_dim)`.
- `w::Vector{Float64}`: Weights of the data points. Should be a vector of size `n`.

# Examples
```jldoctest; output = true
true_model = GaussianRegression(input_dim=2, output_dim=1)
Φ = rand(100, 2)
Y = sample(true_model, Φ)

est_model = GaussianRegression(input_dim=2, output_dim=1)
fit!(est_model, Φ, Y)

loglikelihood(est_model, Φ, Y) > loglikelihood(true_model, Φ, Y)

# output
true
```
"""
function fit!(model::GaussianRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
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
# mutable struct BernoulliRegressionEmission <: EmissionModel
#     inner_model:: BernoulliRegression
# end


"""
    sample(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}; observation_sequence::Matrix{<:Real}=Matrix{Float64}(undef, 0, 1))

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
sequence = sample(model, Φ)
sequence = sample(model, Φ, observation_sequence=sequence)
# output
"""
function sample(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}; observation_sequence::Matrix{<:Real} = Matrix{Float64}(undef, 0, 1))
    # find the number of observations in the observation sequence
    t = size(observation_sequence, 1) + 1
    # get the n+1th observation
    new_observation = sample(model, Φ[t:t, :], n=1)

    return vcat(observation_sequence, new_observation)
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

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = BernoulliRegressionEmission(BernoulliRegression(input_dim=3))
Φ = randn(10, 3)
Y = rand(Bool, 10, 1)
loglikelihoods = loglikelihood(model, Φ, Y)
# output
"""
function loglikelihood(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

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
    fit!(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

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
fit!(model, Φ, Y)
# output
"""
# function fit!(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
#     fit!(model.inner_model, Φ, Y, w)
# end

function fit!(model::BernoulliRegressionEmission, Φ::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    fit!(model, Φ, Y, w)
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
    emissions = [BernoulliRegressionEmission(input_dim=input_dim, include_intercept=include_intercept, β=β, λ=λ) for _ in 1:K]
    # Return the HiddenMarkovModel
    return HiddenMarkovModel(K=K, B=emissions, A=A, πₖ=πₖ)
end


"""
    AutoRegressionEmission <: EmissionModel

A mutable struct representing an autoregressive emission model, which wraps around an `AutoRegression` model.

# Fields
- `inner_model::AutoRegression`: The underlying autoregressive model used for the emissions.
"""
# mutable struct AutoRegressionEmission <: EmissionModel
#     inner_model:: AutoRegression
# end


"""
    sample(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}; observation_sequence::Matrix{<:Real}=Matrix{Float64}(undef, 0, model.output_dim))

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
sequence = sample(model, Y_prev)
sequence = sample(model, Y_prev, observation_sequence=sequence)
# output
"""
function sample(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}; observation_sequence::Matrix{<:Real}=Matrix{Float64}(undef, 0, model.output_dim))

    full_sequence = vcat(Y_prev, observation_sequence)

    # get the n+1th observation
    new_observation = sample(model, full_sequence[end-model.order+1:end, :], n=1)

    return vcat(observation_sequence, new_observation)
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

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = AutoRegressionEmission(AutoRegression(output_dim=2, order=10))
Y_prev = randn(10, 2)
Y = randn(10, 2)
loglikelihoods = loglikelihood(model, Y_prev, Y)
# output
"""
function loglikelihood(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real})

    Φ_gaussian = AR_to_Gaussian_data(Y_prev, Y)

    # extract inner gaussian regression and wrap it with a GaussianEmission <- old comment from when we had inner_models
    innerGaussianRegression_emission = model.innerGaussianRegression

    return loglikelihood(innerGaussianRegression_emission, Φ_gaussian, Y)
end


"""
    fit!(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

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
fit!(model, Y_prev, Y)
# output
"""
# function fit!(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
#     fit!(model.inner_model, Y_prev, Y, w)
# end

function fit!(model::AutoRegressionEmission, Y_prev::Matrix{<:Real}, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    fit!(model, Y_prev, Y, w)
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
    emissions = [AutoRegressionEmission(output_dim=output_dim, order=order, include_intercept=include_intercept, β=β, Σ=Σ, λ=λ) for _ in 1:K]
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


function sample(model::CompositeModelEmission, input_data::Vector{}; observation_sequence::Vector{}=Vector())

    if isempty(observation_sequence)
        for i in 1:length(model.components)
            push!(observation_sequence, (sample(model.components[i], input_data[i]...),))
        end 
    else
        for i in 1:length(model.components)
            observation_sequence[i] = (sample(model.components[i], input_data[i]...; observation_sequence=observation_sequence[i][1]),)
        end 
    end

    return observation_sequence
end

function loglikelihood(model::CompositeModelEmission, input_data::Vector{}, output_data::Vector{})
    loglikelihoods = Vector{}(undef, length(model.components))

    for i in 1:length(model.components)
        loglikelihoods[i] = loglikelihood(model.components[i], input_data[i]..., output_data[i]...)
    end
    return sum(loglikelihoods, dims=1)[1]
end

function fit!(model::CompositeModelEmission, input_data::Vector{}, output_data::Vector{}, w::Vector{Float64}=Vector{Float64}())
    for i in 1:length(model.components)
        fit!!(model.components[i], input_data[i]..., output_data[i]..., w)
    end
end

