export CompositeModel, sample, fit!, loglikelihood


mutable struct CompositeModel <: Model
    components::Vector{<:Model}

    function CompositeModel(components::Vector{<:Model})
        model = new(components)
        return model
    end
end


function sample(model::CompositeModel, input_data::Vector{}=[() for i in 1:length(model.components)]; n::Int=1)

    observations_vector = Vector{}(undef, length(model.components))

    for i in 1:length(model.components)
        observations_vector[i] = (sample(model.components[i], input_data[i]...; n=n), )
    end 

    return observations_vector
end

function loglikelihood(model::CompositeModel, input_data::Vector{}, output_data::Vector{}; observation_wise::Bool=false)

    loglikelihoods = Vector{}(undef, length(model.components))

    if observation_wise == true
        for i in 1:length(model.components)
            loglikelihoods[i] = loglikelihood(model.components[i], input_data[i]..., output_data[i]..., observation_wise=true)
        end
        return sum(loglikelihoods, dims=1)[1]
    else
        for i in 1:length(model.components)
            loglikelihoods[i] = loglikelihood(model.components[i], input_data[i]..., output_data[i]..., data[i]...)
        end
        return sum(loglikelihoods, dims=1)[1]
    end

    
end

function fit!(model::CompositeModel, input_data::Vector{}, output_data::Vector{}, w::Vector{Float64}=Vector{Float64}())
    # the other fit! functions will validate model and data

    for i in 1:length(model.components)
        fit!(model.components[i], input_data[i]..., output_data[i]..., w)
    end
end

function fit!(model::CompositeModel, output_data::Vector{}, w::Vector{Float64}=Vector{Float64}())
    # the other fit! functions will validate model and data

    for i in 1:length(model.components)
        fit!(model.components[i], output_data[i]..., w)
    end
end