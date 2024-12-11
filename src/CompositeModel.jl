export CompositeModel

"""
    CompositeModel

A model that combines multiple sub-models into a single composite model.
Each component model can process its own input and output data independently.

# Fields
- `components::Vector{<:EmissionModel}`: Vector of component models

# Example
```julia
# Create a composite model for tracking both position and velocity
position_model = GaussianEmission(output_dim=3)  # 3D position
velocity_model = GaussianEmission(output_dim=3)  # 3D velocity
model = CompositeModel([position_model, velocity_model])

# Sample from the model
position_data = Matrix{Float64}[]
velocity_data = Matrix{Float64}[]
samples = sample(model, [position_data, velocity_data], n=100)

# Fit the model
fit!(model, [position_data, velocity_data])
```
"""
mutable struct CompositeModel <: EmissionModel
    components::Vector{<:EmissionModel}

    function CompositeModel(components::Vector{<:EmissionModel})
        @assert !isempty(components) "CompositeModel must have at least one component"
        return new(components)
    end
end

"""
    sample(model::CompositeModel, input_data::Vector{T}=[() for _ in 1:length(model.components)]; n::Int=1) where T

Sample from each component model in the composite model.

# Arguments
- `model::CompositeModel`: The composite model to sample from
- `input_data::Vector`: Vector of input data for each component model. Default is empty tuples.
- `n::Int=1`: Number of samples to generate

# Returns
- Vector of sample outputs from each component model

# Notes
- Each element in input_data should match the input requirements of its corresponding component model
- Returns a vector where each element is a tuple containing the samples for that component
"""
function sample(
    model::CompositeModel,
    input_data::Vector{T}=[() for _ in 1:length(model.components)];
    n::Int=1,
) where {T}
    @assert length(input_data) == length(model.components) "Input data length must match number of components"
    @assert n > 0 "Number of samples must be positive"

    observations = Vector{Any}(undef, length(model.components))

    try
        for i in eachindex(model.components)
            observations[i] = (sample(model.components[i], input_data[i]...; n=n),)
        end
    catch e
        throw(ErrorException("Error sampling from component $i: $(e.msg)"))
    end

    return observations
end

"""
    loglikelihood(model::CompositeModel, input_data::Vector, output_data::Vector; 
                 observation_wise::Bool=false)

Calculate the log likelihood of the data under the composite model.

# Arguments
- `model::CompositeModel`: The composite model
- `input_data::Vector`: Vector of input data for each component
- `output_data::Vector`: Vector of output data for each component
- `observation_wise::Bool=false`: If true, return observation-wise log likelihoods

# Returns
- If observation_wise=false: Total log likelihood summed across all components
- If observation_wise=true: Vector of observation-wise log likelihoods

# Notes
- For observation-wise computation, all components must have the same number of observations
"""
function loglikelihood(
    model::CompositeModel,
    input_data::Vector,
    output_data::Vector;
    observation_wise::Bool=false,
)
    @assert length(input_data) == length(model.components) "Input data length must match number of components"
    @assert length(output_data) == length(model.components) "Output data length must match number of components"

    loglikelihoods = Vector{Any}(undef, length(model.components))

    try
        for i in eachindex(model.components)
            loglikelihoods[i] = loglikelihood(
                model.components[i],
                input_data[i]...,
                output_data[i]...;
                observation_wise=observation_wise,
            )
        end

        # Sum log likelihoods across components
        return sum(loglikelihoods; dims=1)[1]
    catch e
        throw(ErrorException("Error computing log likelihood for component $i: $(e.msg)"))
    end
end

"""
    fit!(model::CompositeModel, input_data::Vector, output_data::Vector, 
         w::AbstractVector{Float64}=Float64[])

Fit the composite model to the provided data.

# Arguments
- `model::CompositeModel`: The composite model to fit
- `input_data::Vector`: Vector of input data for each component
- `output_data::Vector`: Vector of output data for each component
- `w::AbstractVector{Float64}=Float64[]`: Optional weights for the observations

# Notes
- The same weights are applied to all components
- Each component's fit! method must handle the case of empty weights
"""
function fit!(
    model::CompositeModel,
    input_data::Vector,
    output_data::Vector,
    w::AbstractVector{Float64}=Float64[],
)
    @assert length(input_data) == length(model.components) "Input data length must match number of components"
    @assert length(output_data) == length(model.components) "Output data length must match number of components"

    try
        for i in eachindex(model.components)
            fit!(model.components[i], input_data[i]..., output_data[i]..., w)
        end
    catch e
        throw(ErrorException("Error fitting component $i: $(e.msg)"))
    end
end

"""
    fit!(model::CompositeModel, output_data::Vector, w::AbstractVector{Float64}=Float64[])

Alternative fit method for models that don't require input data.

# Arguments
- `model::CompositeModel`: The composite model to fit
- `output_data::Vector`: Vector of output data for each component
- `w::AbstractVector{Float64}=Float64[]`: Optional weights for the observations
"""
function fit!(
    model::CompositeModel, output_data::Vector, w::AbstractVector{Float64}=Float64[]
)
    @assert length(output_data) == length(model.components) "Output data length must match number of components"

    try
        for i in eachindex(model.components)
            fit!(model.components[i], output_data[i]..., w)
        end
    catch e
        throw(ErrorException("Error fitting component $i: $(e.msg)"))
    end
end
