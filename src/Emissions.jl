export GaussianEmission, RegressionEmissions

"""
Gaussian Emission: Struct representing a Gaussian emission model.
"""
mutable struct GaussianEmission <: EmissionsModel
    μ::Vector{Float64}  # State-dependent mean
    Σ::Matrix{Float64}  # State-dependent covariance
end

# Loglikelihood function
function loglikelihood(emission::GaussianEmission, observation::Vector{Float64})
    return logpdf(MvNormal(emission.μ, emission.Σ), observation)
end

# Sampling function 
function sample_emission(emission::GaussianEmission)
    return rand(MvNormal(emission.μ, emission.Σ))
end

# Update emissions model for Gaussian model
function updateEmissionModel!(emission::GaussianEmission, data::Matrix{Float64}, γ::Vector{Float64})
    # Assuming data is of size (T, D) where T is the number of observations and D is the observation dimension
    T, D = size(data)
    # Update mean
    weighted_sum = sum(data .* γ, dims=1)
    new_mean = weighted_sum[:] ./ sum(γ)
    
    # Update covariance
    centered_data = data .- new_mean'
    weighted_centered = centered_data .* sqrt.(γ)
    new_covariance = (weighted_centered' * weighted_centered) ./ sum(γ)
    if !ishermitian(new_covariance)
        new_covariance = (new_covariance + new_covariance') * 0.5
    end
    emission.μ = new_mean
    emission.Σ = new_covariance
    return emission
end

mutable struct PoissonEmissions <: EmissionsModel
    λ::Vector{Float64} # rate of events per unit time 
end

# loglikelihood of the poisson model.
function loglikelihood(emission::PoissonEmissions, observation::Vector{Float64})
    #TODO Implement loglikelihood for poisson model
end

function updateEmissionModel!(emission::PoissonEmissions, data::Matrix{Float64}, γ::Vector{Float64})
    #TODO:  Implement updateEmissionModel! for poisson model
end

struct GaussianOrthogonalEmissions <: EmissionsModel
    #TODO: Implement Gaussian Orthogonal Emissions
end

"""
RegressionEmissions: A struct representing a regression model for emissions. This is used in HMM-GLM models.
"""
mutable struct RegressionEmissions <: EmissionsModel
    regression_model::GLM
end

# loglikelihood of the regression model.
function loglikelihood(emission::RegressionEmissions, observation::Vector{Float64})
    loglikelihood(emission.regression_model)
end

# Update the parameters of the regression model, e.g. the betas.
function updateEmissionModel!(emission::RegressionEmissions)
    fit!(emission.regression_model)
end

"""
AutoRegressiveEmissions: AutoRegression Emissions Model
"""
mutable struct AutoRegressiveEmissions <: EmissionsModel
    autoregression_model::Autoregression
end

# loglikelihood of the autoregressive model.
function loglikelihood(emission::AutoRegressiveEmissions, observation::Vector{Float64})
    loglikelihood(emission.autoregression_model, observation)
end

function updateEmissionModel!(emission::AutoRegressiveEmissions)
    #TODO: Implement updateEmissionModel! for autoregressive model
end

"""
LDSEmissions: Linear Dynamical System Emissions Model
"""

mutable struct LDSEmissions<: EmissionsModel
    lds_model::LDS
end

# loglikelihood of the linear dynamical system model.
function loglikelihood(emission::LDSEmissions, observation::Vector{Float64})
    loglikelihood(emission.lds_model, observation)
end

function updateEmissionModel!(emission::LDSEmissions)
    #TODO: Implement updateEmissionModel! for linear dynamical system model
end

