mutable struct HiddenMarkovModel <: Model
    A::Matrix{<:Real} # transition matrix
    B::Vector{Model} # Vector of emission Models
    πₖ::Vector{Float64} # initial state distribution
    K::Int # number of states
end

function validate_model(model::HiddenMarkovModel)
    # check that the transition matrix is the proper shape
    @assert size(model.A) == (model.K, model.K)
    # check that the initial state distribution is the same length as the number of states
    @assert model.K == length(model.πₖ)
    # check that the number of states is equal to the number of emission models
    @assert model.K == length(model.B)
    # check that all emission model are the same type
    @assert all([model.B[1] isa typeof(model.B[i]) for i in 2:length(model.B)])

    # check that all emission models are valid
    for i in 1:length(model.B)
        validate_model(model.B[i])
    end
end

function validate_data(model::HiddenMarkovModel, data...)
    # check that the data is the correct length
    validate_data(model.B[1], data...)
end



function update_emissions!(model::HiddenMarkovModel, data, w::Matrix{<:Real})
    # update regression models 
 
     @threads for k in 1:model.K
         fit!(model.B[k], data, weights=w[:, k])
     end
 
 end

function E_step(model::HiddenMarkovModel, data)
    # run forward-backward algorithm
    α = forward(model, data)
    β = backward(model, data)
    γ = calculate_γ(model, α, β)
    ξ = calculate_ξ(model, α, β, data)
    return γ, ξ, α, β
end

function M_step!(model::HiddenMarkovModel, γ::Matrix{<:Real}, ξ::Array{Float64, 3}, data)
    # update initial state distribution
    update_initial_state_distribution!(model, γ)   
    # update transition matrix
    update_transition_matrix!(model, γ, ξ)
    # update regression models
    update_regression!(model, data, exp.(γ)) 
end


function fit!(model::HiddenMarkovModel, data...; max_iters::Int=100, tol::Float64=1e-6)
    # confirm model is valid
    validate_model(model)

    # confirm data is in the correct format
    validate_data(model, data...)

    T = length(loglikelihood(model.B[1], data..., observation_wise=true))

    K = size(model.A, 1)
    log_likelihood = -Inf
    # Initialize progress bar
    p = Progress(max_iters; dt=1, desc="Computing Baum-Welch...",)
    for iter in 1:max_iters
        # Update the progress bar
        next!(p; showvalues = [(:iteration, iter), (:log_likelihood, log_likelihood)])
        # E-Step
        γ, ξ, α, β = E_step(model, data)
        # Compute and update the log-likelihood
        log_likelihood_current = logsumexp(α[T, :])
        println(log_likelihood_current)
        if abs(log_likelihood_current - log_likelihood) < tol
            finish!(p)
            break
        else
            log_likelihood = log_likelihood_current
        end
        # M-Step
        M_step!(model, γ, ξ, data)
    end


    # confirm model is valid
    validate_model(model)
end