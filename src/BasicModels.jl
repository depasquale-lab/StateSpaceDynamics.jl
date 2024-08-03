export Gaussian

"""
Gaussian: Struct representing a basic Gaussian model.
"""
mutable struct Gaussian <: BasicModel
    data_dim::Int # dimension of the data
    μ::Vector{<:Real}  # mean 
    Σ::Matrix{<:Real}  # covariance matrix

    function Gaussian(; data_dim::Int, μ::Vector{<:Real}=zeros(data_dim), Σ::Matrix{<:Real}=Matrix{Float64}(I, data_dim, data_dim))
        @assert size(Σ) == (data_dim, data_dim) "Σ must be of size (data_dim, data_dim)"
        @assert length(μ) == data_dim "μ vector must be of length data_dim"
        new(data_dim, μ, Σ)
    end
end

function sample(model::Gaussian, n::Int)
    # confirm that Σ is valid
    @assert valid_Σ(model.Σ) "Σ must be positive definite and hermitian"

    raw_samples = rand(MvNormal(model.μ, model.Σ), n)    

    return Matrix(raw_samples')
end

function loglikelihood(model::Gaussian, Y::Matrix{<:Real})
    ll = 0.0
    for y in eachrow(Y)
        ll += logpdf(MvNormal(model.μ, model.Σ), y)
    end
    return ll
end

function fit!(model::Gaussian, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))
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