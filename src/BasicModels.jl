export Gaussian, fit!, sample, loglikelihood

"""
Gaussian: Struct representing a basic Gaussian model.
"""
mutable struct Gaussian <: BasicModel
    output_dim::Int # dimension of the data
    μ::Vector{<:Real}  # mean 
    Σ::Matrix{<:Real}  # covariance matrix
end

function validate_model(model::Gaussian)
    @assert size(model.μ, 1) == model.output_dim

    @assert size(model.Σ) == (model.output_dim, model.output_dim)
    @assert valid_Σ(model.Σ)
end

function validate_data(model::Gaussian, Y=nothing, w=nothing)
    if !isnothing(Y)
        @assert size(Y, 2) == model.output_dim
    end
    if !isnothing(Y) && !isnothing(w)
        @assert length(w) == size(Y, 1)
    end
end

function Gaussian(; 
    output_dim::Int, 
    μ::Vector{<:Real}=zeros(output_dim), 
    Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim))
    
    model = Gaussian(output_dim, μ, Σ)
    
    validate_model(model)

    return model
end

function sample(model::Gaussian; n::Int=1)
    validate_model(model)

    raw_samples = rand(MvNormal(model.μ, model.Σ), n)    

    return Matrix(raw_samples')
end

function loglikelihood(model::Gaussian, Y::Matrix{<:Real})
    validate_model(model)
    validate_data(model, Y)

    # calculate inverse of covariance matrix
    Σ_inv = inv(model.Σ)

    # calculate log likelihood
    residuals = broadcast(-, Y, model.μ')


    loglikelihood = -0.5 * size(Y, 1) * size(Y, 2) * log(2π) - 0.5 * size(Y, 1) * logdet(model.Σ) - 0.5 * sum(residuals .* (Σ_inv * residuals')')
    return loglikelihood
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