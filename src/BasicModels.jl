export Gaussian, fit!, sample, loglikelihood

"""
Gaussian: Struct representing a basic Gaussian model.
"""
mutable struct Gaussian <: BasicModel
    data_dim::Int # dimension of the data
    μ::Vector{<:Real}  # mean 
    Σ::Matrix{<:Real}  # covariance matrix
end

function validate_model(model::Gaussian)
    @assert size(model.μ, 1) == model.data_dim

    @assert size(model.Σ) == (model.data_dim, model.data_dim)
    @assert valid_Σ(model.Σ)
end

function validate_data(model::Gaussian, Y=nothing, w=nothing)
    if !isnothing(Y)
        @assert size(Y, 2) == model.data_dim
    end
    if !isnothing(Y) && !isnothing(w)
        @assert length(w) == size(Y, 1)
    end
end

function Gaussian(; 
    data_dim::Int, 
    μ::Vector{<:Real}=zeros(data_dim), 
    Σ::Matrix{<:Real}=Matrix{Float64}(I, data_dim, data_dim))
    
    model = Gaussian(data_dim, μ, Σ)
    
    validate_model(model)

    return model
end

function sample(model::Gaussian; n::Int=1)
    validate_model(model)

    # confirm that Σ is valid
    @assert valid_Σ(model.Σ) "Σ must be positive definite and hermitian"

    raw_samples = rand(MvNormal(model.μ, model.Σ), n)    

    return Matrix(raw_samples')
end

function TimeSeries(model::Gaussian, samples::Matrix{<:Real})
    return TimeSeries([samples[i, :] for i in 1:size(samples, 1)])
end

function revert_TimeSeries(model::Gaussian, time_series::TimeSeries)
    return permutedims(hcat(time_series.data...), (2,1))
end

function loglikelihood(model::Gaussian, Y::Matrix{<:Real}; observation_wise::Bool=false)
    validate_model(model)
    validate_data(model, Y)

    # calculate inverse of covariance matrix
    Σ_inv = inv(model.Σ)

    # calculate log likelihood
    residuals = broadcast(-, Y, model.μ')

    if !observation_wise
        loglikelihood = -0.5 * size(Y, 1) * size(Y, 2) * log(2π) - 0.5 * size(Y, 1) * logdet(model.Σ) - 0.5 * sum(residuals .* (Σ_inv * residuals')')
        return loglikelihood
    else 
        obs_wise_loglikelihood = zeros(size(Y, 1))

        # calculate observation wise loglikelihood (a vector of loglikelihoods for each observation)
        @threads for i in 1:size(Y, 1)
            obs_wise_loglikelihood[i] = -0.5 * size(Y, 2) * log(2π) - 0.5 * logdet(model.Σ) - 0.5 * sum(residuals[i, :] .* (Σ_inv * residuals[i, :]))
        end

        return obs_wise_loglikelihood
    end
end

function fit!(model::Gaussian, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
    validate_model(model)
    validate_data(model, Y, w)

    weighted_sum = sum(Y .* w, dims=1)
    new_mean = weighted_sum[:] ./ sum(w)

    centered_data = Y .- new_mean'
    weighted_centered = centered_data .* sqrt.(w)
    new_covariance = (weighted_centered' * weighted_centered) ./ sum(w)

    new_covariance = stabilize_covariance_matrix(new_covariance)

    model.μ = new_mean
    model.Σ = new_covariance

    validate_model(model)

    return model
end