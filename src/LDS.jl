abstract type DynamicalSystem end

# LDS Definition
mutable struct LDS <: DynamicalSystem
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

"""LDS Constructor"""
function LDS(A::Union{Matrix{Float64}, Nothing}=nothing, 
             C::Union{Matrix{Float64}, Nothing}=nothing,
             B::Union{Matrix{Float64}, Nothing}=nothing,
             Q::Union{Matrix{Float64}, Nothing}=nothing,
             R::Union{Matrix{Float64}, Nothing}=nothing,
             x0::Union{Vector{Float64}, Nothing}=nothing,
             P0::Union{Matrix{Float64}, Nothing}=nothing,
             D::Union{Int, Nothing}=nothing,
             emissions::String="Gaussian")
    if any(isnothing, [A, C, B, Q, R, x0, P0, D])
        throw(ErrorException("You must specify all parameters for the LDS model."))
    end
end


# SLDS Definition
mutable struct SLDS <: DynamicalSystem
    #TODO: Implement SLDS
end

#rSLDS Definition
mutable struct rSLDS <: DynamicalSystem
    #TODO: Implement rSLDS
end