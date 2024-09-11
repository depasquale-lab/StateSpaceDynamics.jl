export GaussianEmission, RegressionEmissions, loglikelihood

"""
GaussianEmission: Struct representing a Gaussian emission model.
"""
mutable struct GaussianEmission <: EmissionsModel
    μ::Vector{Float64}  # State-dependent mean
    Σ::Matrix{<:Real}  # State-dependent covariance
end

# Loglikelihood function
function loglikelihood(emission::GaussianEmission, observation::Vector{Float64})
    return logpdf(MvNormal(emission.μ, emission.Σ), observation)
end

# Likelihood function
function likelihood(emission::GaussianEmission, observation::Vector{Float64})
    return pdf(MvNormal(emission.μ, emission.Σ), observation)
end

# Sampling function 
function sample_emission(emission::GaussianEmission)
    return rand(MvNormal(emission.μ, emission.Σ))
end

# Update emissions model for Gaussian model
function updateEmissionModel!(
    emission::GaussianEmission, data::Matrix{<:Real}, γ::Vector{Float64}
)
    # Assuming data is of size (T, D) where T is the number of observations and D is the observation dimension
    T, D = size(data)
    # Update mean
    weighted_sum = sum(data .* γ; dims=1)
    new_mean = weighted_sum[:] ./ sum(γ)
    # Update covariance
    centered_data = data .- new_mean'
    weighted_centered = centered_data .* sqrt.(γ)
    new_covariance = (weighted_centered' * weighted_centered) ./ sum(γ)
    # check if the covariance is symmetric
    if !ishermitian(new_covariance)
        new_covariance = (new_covariance + new_covariance') * 0.5
    end
    # check if matrix is posdef
    if !isposdef(new_covariance)
        new_covariance = new_covariance + 1e-12 * I
    end
    # update the emission model
    emission.μ = new_mean
    emission.Σ = new_covariance
    return emission
end

"""
PoissonEmissions: Struct representing a Poisson emission model. This assumes a Poisson distribution for each 
dimension of the observation. This is referred to as a compound Poisson distribution. This is used in HMM-Poisson models,
though, Multivariate Poisson distributions have been derived to capture correlations between dimensions.
"""
mutable struct PoissonEmissions <: EmissionsModel
    λ::Vector{Float64} # rate of events per unit time 
end

# loglikelihood of the poisson model.
function loglikelihood(emission::PoissonEmissions, observation::Vector{Float64})
    D = length(emission.λ)
    ll = 0.0
    for d in 1:D
        ll += logpdf(Poisson(emission.λ[d]), observation[:, d])
    end
    return ll
end

function updateEmissionModel!(emission::PoissonEmissions, data::Matrix{<:Real})
    #TODO:  Implement updateEmissionModel! for poisson model
end

"""
MultinomialEmissions: Struct representing a Multinomial emission model.
"""
mutable struct MultinomialEmissions <: EmissionsModel
    n::Int64 # number of trials
    p::Matrix{<:Real} # probability of each category
end

# loglikelihood of the multinomial model.
function loglikelihood(emission::MultinomialEmissions, observation::Vector{Float64})
    return logpdf(Multinomial(emission.n, emission.p), observation)
end

function updateEmissionModel!(emission::MultinomialEmissions, data::Matrix{<:Real})
    #TODO: Implement updateEmissionModel! for multinomial model
end

"""
RegressionEmissions: A struct representing a regression model for emissions. This is used in HMM-GLM models.
"""
mutable struct RegressionEmissions <: EmissionsModel
    regression::Regression
end

# loglikelihood of the regression model.
function loglikelihood(emission::RegressionEmissions, X::Vector{Float64}, y::Float64)
    return loglikelihood(emission.regression, X, y)
end

# loglikelihood of the regression model.
function loglikelihood(emission::RegressionEmissions, X::Matrix{<:Real}, y::Matrix{<:Real})
    return loglikelihood(emission.regression, X, y)
end

# Update the parameters of the regression model, e.g. the betas.
function update_emissions_model!(
    emission::RegressionEmissions,
    X::Matrix{<:Real},
    y::Vector{Float64},
    w::Vector{Float64}=ones(length(y)),
)
    return fit!(emission.regression, X, y, w)
end

# Update the parameters of the regression model, e.g. the betas.
function update_emissions_model!(
    emission::RegressionEmissions,
    X::Matrix{<:Real},
    y::Matrix{<:Real},
    w::Vector{Float64}=ones(size(y, 1)),
)

    # confirm dimensions of X and y are correct
    @assert size(X, 1) == size(y, 1) "Number of rows (number of observations) in X and y must be equal."

    # confirm the size of w is correct
    @assert length(w) == size(y, 1) "Length of w must be equal to the number of observations in y."

    return fit!(emission.regression, X, y, w)
end
