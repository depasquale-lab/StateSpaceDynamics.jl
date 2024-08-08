export GaussianEmission, getproperty, setproperty!

# define getters for inner_model fields
function Base.getproperty(model::EmissionModel, sym::Symbol)
    if sym === :inner_model
        return getfield(model, sym)
    else # fallback to getfield
        return getfield(model.inner_model, sym)
    end
end

# define setters for inner_model fields
function Base.setproperty!(model::EmissionModel, sym::Symbol, value)
    if sym === :inner_model
        setfield!(model, sym, value)
    else # fallback to setfield!
        setfield!(model.inner_model, sym, value)
    end
end




mutable struct GaussianEmission <: EmissionModel
    inner_model:: Gaussian
end

function sample(model::GaussianEmission, observation_sequence::Matrix{<:Real}=Matrix{Float64}(undef, 0, model.output_dim))
    validate_model(model)

    raw_samples = rand(MvNormal(model.μ, model.Σ), 1)    

    return vcat(observation_sequence, Matrix(raw_samples'))
end

function loglikelihoods(model::GaussianEmission, Y::Matrix{<:Real})
    validate_model(model.inner_model)
    validate_data(model.inner_model, Y)

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

function fit!(model::GaussianEmission, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    fit!(model.inner_model, Y, w)
end