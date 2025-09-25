export SLDS, valid_SLDS
@kwdef struct SLDS{T<:Real,
                   S<:AbstractStateModel,
                   O<:AbstractObservationModel,
                   TM<:AbstractMatrix{T},
                   ISV<:AbstractVector{T}}
    A::TM                                  
    Z₀::ISV                                 
    LDSs::Vector{LinearDynamicalSystem{T,S,O}} 
end
