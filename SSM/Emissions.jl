export EmissionsModel, GaussianEmission

# Create emission models here 
abstract type EmissionsModel end

# Gaussian Emission
struct GaussianEmission <: EmissionsModel
    μ::Vector{Float64}  # State-dependent mean
    Σ::Matrix{Float64}  # State-dependent covariance
end

# Likelihood function
function likelihood(emission::GaussianEmission, observation::Vector{Float64})
    return pdf(MvNormal(emission.μ, emission.Σ), observation)
end

# Sampling function 
function sample_emission(emission::GaussianEmission)
    return rand(MvNormal(emission.μ, emission.Σ))
end

struct PoissonEmissions <: EmissionsModel
    λ::Float64 # rate of events per unit time 
end

