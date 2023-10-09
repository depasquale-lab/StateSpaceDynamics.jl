abstract type DynamicalSystem end

# LDS Definition
struct LDS <: DynamicalSystem
    #TODO: Implement LDS
    A::Matrix{Float64} # Transition Matrix
    C::Matrix{Float64} # Observation Matrix
    B::Matrix{Float64} # Control Matrix
    Q::Matrix{Float64} # Process Noise Covariance
    R::Matrix{Float64} # Observation Noise Covariance
    x0::Vector{Float64} # Initial State
    P0::Matrix{Float64} # Initial Covariance
    D::Int # Observation Dimension
    emissions::String # Emission Model
end

# SLDS Definition
struct SLDS <: DynamicalSystem
    #TODO: Implement SLDS
end

#rSLDS Definition
struct rSLDS <: DynamicalSystem
    #TODO: Implement rSLDS
end