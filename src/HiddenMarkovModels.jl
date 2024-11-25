export HiddenMarkovModel, fit!, sample, loglikelihood, viterbi
export kmeans_init!

# for unit tests
export estep
export class_probabilities

"""
    HiddenMarkovModel

A Hidden Markov Model (HMM) with custom emissions.

# Fields
- `K::Int`: Number of states.
- `B::Vector=Vector()`: Vector of emission models.
- `emission=nothing`: If B is missing emissions, clones of this model will be used to fill in the rest.
- `A::Matrix{<:Real}`: Transition matrix.
- `πₖ::Vector{Float64}`: Initial state distribution.
"""
mutable struct HiddenMarkovModel <: AbstractHMM
    A::Matrix{<:Real} # transition matrix
    B::Vector{EmissionModel} # Vector of emission Models
    πₖ::Vector{Float64} # initial state distribution
    K::Int # number of states
end

function initialize_transition_matrix(K::Int)
    # initialize a transition matrix
    A = zeros(Float64, K, K)
    @threads for i in 1:K
        A[i, :] = rand(Dirichlet(ones(K)))
    end
    return A
end

function initialize_state_distribution(K::Int)
    # initialize a state distribution
    return rand(Dirichlet(ones(K)))
end

function HiddenMarkovModel(;
    K::Int,
    B::Vector=Vector(),
    emission=nothing,
    A::Matrix{<:Real}=initialize_transition_matrix(K),
    πₖ::Vector{Float64}=initialize_state_distribution(K),
)

    # if B does not have all K emission models, then fill in the rest with deep copies of "emission"
    if !isnothing(emission) && length(B) < K
        for i in (length(B) + 1):K
            push!(B, deepcopy(emission))
        end
    end

    emission_models = B
    #emission_models = Emission.(B)

    model = HiddenMarkovModel(A, emission_models, πₖ, K)

    # check that the transition matrix is the proper shape
    @assert size(model.A) == (model.K, model.K)
    @assert isapprox(sum(model.A; dims=2), ones(model.K))
    # check that the initial state distribution is the same length as the number of states
    @assert model.K == length(model.πₖ)
    @assert sum(model.πₖ) ≈ 1.0
    # check that the number of states is equal to the number of emission models
    @assert model.K == length(model.B)
    # check that all emission model are the same type
    @assert all([model.B[i] isa EmissionModel for i in 1:length(model.B)])

    return model
end

function kmeans_init!(model::HiddenMarkovModel, data::Matrix{T}) where {T<:Real}
    num_states = model.K
    # run k-means 
    means, labels = kmeans_clustering(permutedims(data), num_states) # permute dims to interface with kmenas clustering function 
    covs = [cov(permutedims(data[:, labels .== i])) for i in 1:num_states]
    # initialize the emission models
    for k in 1:num_states
        model.B[k].μ = means[:, k]
        model.B[k].Σ = covs[k]
    end
end

"""
    sample(model::HiddenMarkovModel, data...; n::Int)

Generate `n` samples from a Hidden Markov Model. Returns a tuple of the state sequence and the observation sequence.

# Arguments
- `model::HiddenMarkovModel`: The Hidden Markov Model to sample from.
- `data...`: The data to fit the Hidden Markov Model. Requires the same format as the emission model.
- `n::Int`: The number of samples to generate.

# Returns
- `state_sequence::Vector{Int}`: The state sequence, where each element is an integer 1:K.
- `observation_sequence::Matrix{Float64}`: The observation sequence. This takes the form of the emission model's output.
"""
function sample(model::HiddenMarkovModel, X::Union{Matrix{<:Real},Nothing}=nothing; n::Int)
    state_sequence = Vector{Int}(undef, n)
    # Change to (dimension, time) ordering
    observation_sequence = Matrix{Float64}(undef, model.B[1].output_dim, n)

    # Initialize the first state and observation
    state_sequence[1] = rand(Categorical(model.πₖ))
    observation_sequence[:, 1] = if isnothing(X)
        sample(model.B[state_sequence[1]])
    else
        sample(model.B[state_sequence[1]], X[:, 1])
    end

    # Sample the state paths and observations
    for t in 2:n  # t represents time steps
        state_sequence[t] = rand(Categorical(model.A[state_sequence[t - 1], :]))
        observation_sequence[:, t] = if isnothing(X)
            sample(model.B[state_sequence[t]])
        else
            sample(model.B[state_sequence[t]], X[:, t])
        end
    end

    return state_sequence, observation_sequence
end

function emission_loglikelihoods(model::HiddenMarkovModel, data...)
    # Pre-allocate the loglikelihood matrix, should be states x observations
    loglikelihoods = zeros(model.K, size(data[1], 1))  # data is being passed in as time x states from the estep function

    # Calculate observation wise likelihoods for all states
    @threads for k in 1:(model.K)
        loglikelihoods[k, :] .= loglikelihood(model.B[k], data...)
    end

    return loglikelihoods
end

"""
    loglikelihood(model::HiddenMarkovModel, data...)

Calculate the log likelihood of the data given the Hidden Markov Model.

# Arguments
- `model::HiddenMarkovModel`: The Hidden Markov Model to calculate the log likelihood for.
- `data...`: The data to calculate the log likelihood for. Requires the same format as the emission model.

# Returns
- `loglikelihood::Float64`: The log likelihood of the data given the Hidden Markov Model.
"""
function loglikelihood(model::HiddenMarkovModel, data...)
    transposed_data = permutedims.(data) # Transpose the data to get the correct shape in EmissionModels.jl

    lls = emission_loglikelihoods(model, transposed_data...)

    # Run forward algorithm
    α = forward(model, lls)
    return logsumexp(α[:, end])
end

function forward(model::HiddenMarkovModel, loglikelihoods::Matrix{<:Real})
    time_steps = size(loglikelihoods, 2)

    # Initialize an α-matrix 
    α = zeros(model.K, time_steps)

    # Calculate α₁
    @threads for k in 1:(model.K)
        α[k, 1] = log(model.πₖ[k]) + loglikelihoods[k, 1]
    end
    # Now perform the rest of the forward algorithm for t=2 to time_steps
    for t in 2:time_steps
        @threads for k in 1:(model.K)
            values_to_sum = Float64[]
            for i in 1:(model.K)
                push!(values_to_sum, log(model.A[i, k]) + α[i, t - 1])
            end
            log_sum_alpha_a = logsumexp(values_to_sum)
            α[k, t] = log_sum_alpha_a + loglikelihoods[k, t]
        end
    end
    return α
end

function backward(model::HiddenMarkovModel, loglikelihoods::Matrix{<:Real})
    time_steps = size(loglikelihoods, 2)

    # Initialize a β matrix
    β = zeros(Float64, model.K, time_steps)

    # Set last β values. In log-space, 0 corresponds to a value of 1 in the original space.
    β[:, end] .= 0  # log(1) = 0

    # Calculate β, starting from time_steps-1 and going backward to 1
    for t in (time_steps - 1):-1:1
        @threads for i in 1:(model.K)
            values_to_sum = Float64[]
            for j in 1:(model.K)
                push!(
                    values_to_sum,
                    log(model.A[i, j]) + loglikelihoods[j, t + 1] + β[j, t + 1],
                )
            end
            β[i, t] = logsumexp(values_to_sum)
        end
    end
    return β
end

function calculate_γ(model::HiddenMarkovModel, α::Matrix{<:Real}, β::Matrix{<:Real})
    time_steps = size(α, 2)
    γ = α .+ β
    @threads for t in 1:time_steps
        γ[:, t] .-= logsumexp(γ[:, t])
    end
    return γ
end

function calculate_ξ(
    model::HiddenMarkovModel,
    α::Matrix{<:Real},
    β::Matrix{<:Real},
    loglikelihoods::Matrix{<:Real},
)
    time_steps = size(α, 2)
    ξ = zeros(Float64, model.K, model.K, time_steps - 1)
    for t in 1:(time_steps - 1)
        # Array to store the unnormalized ξ values
        log_ξ_unnormalized = zeros(Float64, model.K, model.K)
        @threads for i in 1:(model.K)
            for j in 1:(model.K)
                log_ξ_unnormalized[i, j] =
                    α[i, t] + log(model.A[i, j]) + loglikelihoods[j, t + 1] + β[j, t + 1]
            end
        end
        # Normalize the ξ values using log-sum-exp operation
        ξ[:, :, t] .= log_ξ_unnormalized .- logsumexp(log_ξ_unnormalized)
    end
    return ξ
end

function estep(model::HiddenMarkovModel, data)
    # compute lls of the observations
    loglikelihoods = emission_loglikelihoods(model, data...)

    # run forward-backward algorithm
    α = forward(model, loglikelihoods)
    β = backward(model, loglikelihoods)
    γ = calculate_γ(model, α, β)
    ξ = calculate_ξ(model, α, β, loglikelihoods)
    return γ, ξ, α, β
end

function update_initial_state_distribution!(model::HiddenMarkovModel, γ::Matrix{<:Real})
    # Update initial state probabilities
    return model.πₖ .= exp.(γ[:, 1])
end

function update_transition_matrix!(
    model::HiddenMarkovModel, γ::Matrix{<:Real}, ξ::Array{Float64,3}
)
    # Update transition probabilities
    @threads for i in 1:(model.K)
        for j in 1:(model.K)
            model.A[i, j] = exp(logsumexp(ξ[i, j, :]) - logsumexp(γ[i, 1:(end - 1)]))
        end
    end
end

function update_emissions!(model::HiddenMarkovModel, data, w::Matrix{<:Real})
    # update regression models 
    @threads for k in 1:(model.K)
        fit!(model.B[k], data..., w[:, k])
    end
end

function mstep!(model::HiddenMarkovModel, γ::Matrix{<:Real}, ξ::Array{Float64,3}, data)
    # update initial state distribution
    update_initial_state_distribution!(model, γ)
    # update transition matrix
    update_transition_matrix!(model, γ, ξ)
    # update regression models
    return update_emissions!(model, data, exp.(permutedims(γ)))
end

# Trialized versions of functions
function mstep!(
    model::HiddenMarkovModel, γ::Vector{Matrix{Float64}}, ξ::Vector{Array{Float64,3}}, data
)
    # update initial state distribution
    update_initial_state_distribution!(model, γ)
    # update transition matrix
    update_transition_matrix!(model, γ, ξ)
    # update regression models
    γ_exp = [exp.(γ_trial) for γ_trial in γ]
    return update_emissions!(model, data, vcat(permutedims.(γ_exp)...))
end

function update_initial_state_distribution!(
    model::HiddenMarkovModel, γ::Vector{Matrix{Float64}}
)
    # Update initial state probabilities for trialized data
    num_trials = length(γ)
    return model.πₖ = mean([exp.(γ[i][:, 1]) for i in 1:num_trials])
end

function update_transition_matrix!(
    model::HiddenMarkovModel, γ::Vector{Matrix{Float64}}, ξ::Vector{Array{Float64,3}}
)
    # Update transition matrix for trialized data
    K = size(model.A, 1)
    num_trials = length(γ)

    E = cat(ξ...; dims=3)
    G = hcat([γ[i][:, 1:(size(γ[i], 2) - 1)] for i in 1:num_trials]...)

    @threads for i in 1:K
        for j in 1:K
            model.A[i, j] = exp(logsumexp(E[i, j, :]) - logsumexp(G[i, :]))
        end
    end
end

"""
    fit!(model::HiddenMarkovModel, Y::Matrix{<:Real}, X::Union{Matrix{<:Real}, Nothing}=nothing; max_iters::Int=100, tol::Float64=1e-6)

Fit the Hidden Markov Model using the EM algorithm.

# Arguments
- `model::HiddenMarkovModel`: The Hidden Markov Model to fit.
- `Y::Matrix{<:Real}`: The emission data.
- `X::Union{Matrix{<:Real}, Nothing}=nothing`: Optional input data for fitting Switching Regression Models
- `max_iters::Int=100`: The maximum number of iterations to run the EM algorithm.
- `tol::Float64=1e-6`: When the log likelihood is improving by less than this value, the algorithm will stop.
"""
function fit!(
    model::HiddenMarkovModel,
    Y::Matrix{<:Real},
    X::Union{Matrix{<:Real},Nothing}=nothing;
    max_iters::Int=100,
    tol::Float64=1e-6,
)
    lls = [-Inf]

    data = X === nothing ? (Y,) : (X, Y)

    # transpose data so that correct dimensions are passed to EmissionModels.jl, a bit hacky but works for now.
    transpose_data = Matrix.(transpose.(data))

    log_likelihood = -Inf
    # Initialize progress bar
    p = Progress(max_iters; desc="Running EM algorithm...", barlen=50, showspeed=true)
    for iter in 1:max_iters
        next!(p)
        # E-Step
        γ, ξ, α, β = estep(model, transpose_data)
        # Compute and update the log-likelihood
        log_likelihood_current = logsumexp(α[:, end])
        push!(lls, log_likelihood_current)
        #println("iter $(iter) loglikelihood: ", log_likelihood_current)
        if abs(log_likelihood_current - log_likelihood) < tol
            finish!(p)
            return lls
        else
            log_likelihood = log_likelihood_current
        end
        # M-Step
        mstep!(model, γ, ξ, transpose_data)
    end
    return lls
end

"""
    fit!(model::HiddenMarkovModel, Y::Matrix{<:Real}, X::Union{Matrix{<:Real}, Nothing}=nothing; max_iters::Int=100, tol::Float64=1e-6)

Fit the Hidden Markov Model to multiple trials of data using the EM algorithm.

# Arguments
- `model::HiddenMarkovModel`: The Hidden Markov Model to fit.
- `Y::Vector{<:Matrix{<:Real}}`: The trialized emission data.
- `X::Union{Vector{<:Matrix{<:Real}}, Nothing}=nothing`: Optional input data for fitting Switching Regression Models
- `max_iters::Int=100`: The maximum number of iterations to run the EM algorithm.
- `tol::Float64=1e-6`: When the log likelihood is improving by less than this value, the algorithm will stop.
"""
function fit!(
    model::HiddenMarkovModel,
    Y::Vector{<:Matrix{<:Real}},
    X::Union{Vector{<:Matrix{<:Real}},Nothing}=nothing;
    max_iters::Int=100,
    tol::Float64=1e-6,
)
    lls = [-Inf]
    data = X === nothing ? (Y,) : (X, Y)

    # Initialize log_likelihood
    log_likelihood = -Inf

    # Transform each matrix in each tuple to the correct orientation
    transposed_matrices = map(data_tuple -> Matrix.(transpose.(data_tuple)), data)
    zipped_matrices = collect(zip(transposed_matrices...))

    p = Progress(max_iters; desc="Running EM algorithm...", barlen=50, showspeed=true)
    for iter in 1:max_iters
        # estep
        output = estep.(Ref(model), zipped_matrices)
        γ, ξ, α, β = map(x -> x[1], output),
        map(x -> x[2], output), map(x -> x[3], output),
        map(x -> x[4], output)

        # Calculate log_likelihood
        log_likelihood_current = sum(map(α -> logsumexp(α[:, end]), α))
        push!(lls, log_likelihood_current)

        next!(p)

        # Check for convergence
        if abs(log_likelihood_current - log_likelihood) < tol
            finish!(p)
            break
        else
            log_likelihood = log_likelihood_current
        end

        # Get data trial tuples stacked for mstep!()
        stacked_data = stack_tuples(zipped_matrices)

        # M_step
        mstep!(model, γ, ξ, stacked_data)
    end

    return lls
end

"""
    class_probabilities(model::HiddenMarkovModel, data...)

Calculate the class probabilities for each observation. Returns a matrix of size `(T, K)` where `K` is the number of states and `T` is the number of observations.

# Arguments
- `model::HiddenMarkovModel`: The Hidden Markov Model to calculate the class probabilities for.
- `data...`: The data to calculate the class probabilities for. Requires the same format as the emission model loglikelihood() function.

# Returns
- `class_probabilities::Matrix{Float64}`: The class probabilities for each observation. Of shape `(T, K)`. Each row of the Matrix sums to 1.
"""
function class_probabilities(model::HiddenMarkovModel, data...)
    γ, ξ, α, β = estep(model, data)
    return exp.(γ)
end

"""
    viterbi(model::HiddenMarkovModel, data...)

Calculate the most likely sequence of states given the data.

# Arguments
- `model::HiddenMarkovModel`: The Hidden Markov Model to calculate the most likely sequence of states for.
- `data...`: The data to calculate the most likely sequence of states for. Requires the same format as the emission model's loglikelihood() function.

# Returns
- `best_path::Vector{Int}`: The most likely sequence of states.
"""
function viterbi(model::HiddenMarkovModel, data...)
    # Calculate observation wise likelihoods for all states
    loglikelihoods_state_1 = loglikelihood(model.B[1], data...)
    loglikelihoods = zeros(model.K, length(loglikelihoods_state_1))
    loglikelihoods[:, 1] = loglikelihoods_state_1

    @threads for k in 2:(model.K)
        loglikelihoods[k, :] = loglikelihood(model.B[k], data...)
    end

    T = length(loglikelihoods_state_1)
    K = size(model.A, 1)  # Number of states

    # Step 1: Initialization
    viterbi = zeros(Float64, K, T)
    backpointer = zeros(Int, K, T)
    for i in 1:K
        viterbi[i, 1] = log(model.πₖ[i]) + loglikelihoods[i, 1]
        backpointer[i, 1] = 0
    end

    # Step 2: Recursion
    for t in 2:T
        for j in 1:K
            max_prob, max_state = -Inf, 0
            for i in 1:K
                prob = viterbi[i, t - 1] + log(model.A[i, j]) + loglikelihoods[j, t]
                if prob > max_prob
                    max_prob = prob
                    max_state = i
                end
            end
            viterbi[j, t] = max_prob
            backpointer[j, t] = max_state
        end
    end

    # Step 3: Termination
    best_path_prob, best_last_state = findmax(viterbi[:, T])
    best_path = [best_last_state]
    for t in T:-1:2
        push!(best_path, backpointer[best_path[end], t])
    end
    return reverse(best_path)
end
