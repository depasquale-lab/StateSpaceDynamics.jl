export Gaussian, fit!, sample, loglikelihood

"""
    Gaussian

A multidimensional Gaussian model with a mean and covariance matrix.

# Fields
- `output_dim::Int`: Dimension of the data.
- `μ::Vector{<:Real}=zeros(output_dim)`: Mean of the Gaussian.
- `Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim)`: Covariance matrix of the Gaussian.

```jldoctest; output = false
Gaussian(output_dim=2)

# output

Gaussian(2, [0.0, 0.0], [1.0 0.0; 0.0 1.0])
```
"""
mutable struct GaussianEmission <: EmissionModel
    output_dim::Int # dimension of the data
    μ::Vector{<:Real}  # mean 
    Σ::Matrix{<:Real}  # covariance matrix
end

function validate_model(model::GaussianEmission)
    @assert size(model.μ, 1) == model.output_dim

    @assert size(model.Σ) == (model.output_dim, model.output_dim)
    @assert valid_Σ(model.Σ)
end

function validate_data(model::GaussianEmission, Y=nothing, w=nothing)
    if !isnothing(Y)
        @assert size(Y, 2) == model.output_dim
    end
    if !isnothing(Y) && !isnothing(w)
        @assert length(w) == size(Y, 1)
    end
end

function GaussianEmission(; 
    output_dim::Int, 
    μ::Vector{<:Real}=zeros(output_dim), 
    Σ::Matrix{<:Real}=Matrix{Float64}(I, output_dim, output_dim))
    
    model = GaussianEmission(output_dim, μ, Σ)
    
    validate_model(model)

    return model
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
    validate_model(model)

    raw_samples = rand(MvNormal(model.μ, model.Σ), n)    

    return Matrix(raw_samples')
end


"""
    loglikelihood(model::Gaussian, Y::Matrix{<:Real})

Calculate the log likelihood of the data `Y` given the Gaussian model. The data `Y` is assumed to be a matrix of size `(n, output_dim)`.

# Examples
```jldoctest; output = false, filter = r"(?s).*" => s""
model = Gaussian(output_dim=2)
Y = sample(model, n=3)
loglikelihood(model, Y)

# output
```
"""
function loglikelihood(model::GaussianEmission, Y::Matrix{<:Real})
    validate_model(model)
    validate_data(model, Y)

    # calculate inverse of covariance matrix
    Σ_inv = inv(model.Σ)

    # calculate log likelihood
    residuals = broadcast(-, Y, model.μ')


    loglikelihood = -0.5 * size(Y, 1) * size(Y, 2) * log(2π) - 0.5 * size(Y, 1) * logdet(model.Σ) - 0.5 * sum(residuals .* (Σ_inv * residuals')')
    return loglikelihood
end

"""
    fit!(model::Gaussian, Y::Matrix{<:Real}, w::Vector{Float64}=ones(size(Y, 1)))

Fit a Gaussian model to the data `Y`. 

# Arguments
- `model::Gaussian`: Gaussian model to fit.
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