export AutoRegressionEmission, fit!, loglikelihood, least_squares, update_variance!, sample

# below used in notebooks and unit tests
export define_objective, define_objective_gradient
export getproperty, setproperty!


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

