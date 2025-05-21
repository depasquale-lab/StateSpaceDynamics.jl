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
    A:: Matrix{Float64} # transition matrix
    B::Vector{EmissionModel} # Vector of emission Models
    πₖ::Vector{Float64} # initial state distribution
    K::Int # number of states
end

function initialize_forward_backward(model::AbstractHMM, num_obs::Int)
    num_states = model.K
    ForwardBackward(
        zeros(num_states, num_obs),
        zeros(num_states, num_obs),
        zeros(num_states, num_obs),
        zeros(num_states, num_obs),
        zeros(Float64, num_states, num_states, num_obs - 1)
    )
end

function aggregate_forward_backward!(
    aggregated_FB::ForwardBackward{T}, 
    FB_storages::Vector{ForwardBackward{T}}
) where {T<:Real}
    # Concatenate each field into the respective field in the aggregated struct
    aggregated_FB.loglikelihoods .= hcat([fb.loglikelihoods for fb in FB_storages]...)
    aggregated_FB.α .= hcat([fb.α for fb in FB_storages]...)
    aggregated_FB.β .= hcat([fb.β for fb in FB_storages]...)
    aggregated_FB.γ .= hcat([fb.γ for fb in FB_storages]...)
    aggregated_FB.ξ = cat([fb.ξ for fb in FB_storages]..., dims=3)
    
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
    A::AbstractMatrix{<:Real}=initialize_transition_matrix(K),
    πₖ::Vector{Float64}=initialize_state_distribution(K),
)

    # if B does not have all K emission models, then fill in the rest with deep copies of "emission"
    if !isnothing(emission) && length(B) < K
        @warn """
        User did not provide as many emmission models as the function expects (K=$K).
        Filling in the rest with deep copies of the provided emission model.
        If this is not the desired behavior, specify all the emission models in the 'B=Vector{EmissionModel}' argument 
        rather than 'emission'
        """
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
function sample(model::HiddenMarkovModel, X::AbstractMatrix{<:Real}; n::Int, autoregressive::Bool=false)
    sample(model, to_f64(X); n=n, autoregressive=autoregressive)
end

function sample(model::HiddenMarkovModel; n::Int, autoregressive::Bool=false)
    sample(model, nothing; n=n, autoregressive=autoregressive)
end

function sample(model::HiddenMarkovModel, X::Union{Matrix{<:Float64},Nothing}=nothing; n::Int, autoregressive::Bool=false)
    
    if autoregressive ==false
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
    else
        state_sequence = Vector{Int}(undef, n)
        # Change to (dimension, time) ordering
        observation_sequence = Matrix{Float64}(undef, model.B[1].output_dim, n)

        # Initialize the first state and observation
        state_sequence[1] = rand(Categorical(model.πₖ))
        X, observation_sequence[:,1] = sample(model.B[state_sequence[1]], X)

        # Sample the state paths and observations
        for t in 2:n  # t represents time steps
            state_sequence[t] = rand(Categorical(model.A[state_sequence[t - 1], :]))
            X, observation_sequence[:, t] = sample(model.B[state_sequence[t]], X)
        end

        return state_sequence, observation_sequence

    end
end

# New function for FB storage
function emission_loglikelihoods!(model::HiddenMarkovModel, FB_storage::ForwardBackward, data...)
    log_likelihoods = FB_storage.loglikelihoods

    # Calculate observation wise likelihoods for all states
    @threads for k in 1:model.K
        log_likelihoods[k, :] .= loglikelihood(model.B[k], data...)
    end
end

function forward!(model::AbstractHMM, FB_storage::ForwardBackward)
    # Reference storage
    α = FB_storage.α
    loglikelihoods = FB_storage.loglikelihoods
    A = model.A
    πₖ = model.πₖ
    K = model.K
    time_steps = size(loglikelihoods, 2)

    # Preallocate reusable arrays
    values_to_sum = zeros(K)
    log_A = log.(A)  # Precompute log of transition probabilities
    log_πₖ = log.(πₖ)  # Precompute log of initial state probabilities

    # Calculate α₁
    @inbounds for k in 1:K
        α[k, 1] = log_πₖ[k] + loglikelihoods[k, 1]
    end

    # Compute α for all time steps
    @inbounds for t in 2:time_steps
        for k in 1:K
            for i in 1:K
                values_to_sum[i] = log_A[i, k] + α[i, t - 1]
            end
            α[k, t] = logsumexp(values_to_sum) + loglikelihoods[k, t]
        end
    end
end

function backward!(model::AbstractHMM, FB_storage::ForwardBackward)
    # Reference storage
    β = FB_storage.β
    loglikelihoods = FB_storage.loglikelihoods
    A = model.A
    K = model.K
    time_steps = size(loglikelihoods, 2)
    
    # Preallocate reusable arrays
    values_to_sum = zeros(K)
    log_A = log.(A)
    
    # Initialize last column of β
    @inbounds β[:, end] .= 0

    # Compute β for all time steps
    @inbounds for t in (time_steps - 1):-1:1
        for i in 1:K
            for j in 1:K
                values_to_sum[j] = log_A[i, j] + loglikelihoods[j, t + 1] + β[j, t + 1]
            end
            β[i, t] = logsumexp(values_to_sum)
        end
    end
end

function calculate_γ!(model::AbstractHMM, FB_storage::ForwardBackward)
    α = FB_storage.α
    β = FB_storage.β

    time_steps = size(α, 2)
    FB_storage.γ = α .+ β
    γ = FB_storage.γ

    @inbounds for t in 1:time_steps
        γ[:, t] .-= logsumexp(view(γ,:,t))
    end
end

function calculate_ξ!(
    model::AbstractHMM,
    FB_storage::ForwardBackward
)
    α = FB_storage.α
    β = FB_storage.β
    loglikelihoods = FB_storage.loglikelihoods
    ξ = FB_storage.ξ
    A = model.A
    log_A = log.(A)  # Precompute log transition probabilities
    K = model.K
    time_steps = size(α, 2)

    # Preallocate reusable arrays
    log_ξ_unnormalized = zeros(K, K)

    @inbounds for t in 1:(time_steps - 1)
        for i in 1:K
            α_t = α[i, t]  # Cache α[i, t] for reuse
            for j in 1:K
                log_ξ_unnormalized[i, j] = α_t + log_A[i, j] + loglikelihoods[j, t + 1] + β[j, t + 1]
            end
        end
        # Normalize the ξ values in log-space using log-sum-exp
        log_norm_factor = logsumexp(log_ξ_unnormalized)
        ξ[:, :, t] .= log_ξ_unnormalized .- log_norm_factor
    end
end

function estep!(model::HiddenMarkovModel, data, FB_storage)
    # compute lls of the observations
    emission_loglikelihoods!(model, FB_storage, data...)

    # run forward-backward algorithm
    forward!(model, FB_storage)
    backward!(model, FB_storage)
    calculate_γ!(model, FB_storage)
    calculate_ξ!(model, FB_storage)
end

function update_initial_state_distribution!(model::AbstractHMM, FB_storage::ForwardBackward)
    # Update initial state probabilities
    γ = FB_storage.γ
    return model.πₖ .= exp.(γ[:, 1])
end

function update_transition_matrix!(
    model::AbstractHMM, FB_storage::ForwardBackward
)
    γ = FB_storage.γ
    ξ = FB_storage.ξ
    # Update transition probabilities -> @threading good here?
    for i in 1:(model.K)
        for j in 1:(model.K)
            model.A[i, j] = exp(logsumexp(ξ[i, j, :]) - logsumexp(γ[i, 1:(end - 1)]))
        end
    end
end

function update_transition_matrix!(
    model::AbstractHMM, FB_storage_vec::Vector{ForwardBackward{Float64}}
)
    for j in 1:(model.K)
        for k in 1:(model.K)
            num = exp(logsumexp(vcat([FB_trial.ξ[j, k, :] for FB_trial in FB_storage_vec]...)))
            denom = exp.(logsumexp(vcat([FB_trial.ξ[j, :, :]' for FB_trial in FB_storage_vec]...)))  # this logsumexp takes care of both sums in denom
            model.A[j,k] = num / denom
        end
    end
end

function update_emissions!(model::AbstractHMM, FB_storage::ForwardBackward, data)
    # update regression models
    w = exp.(permutedims(FB_storage.γ))
    # check threading speed here
    @threads for k in 1:(model.K)
        fit!(model.B[k], data..., w[:, k])
    end
end

function mstep!(model::AbstractHMM, FB_storage::ForwardBackward, data)
    # update initial state distribution
    update_initial_state_distribution!(model, FB_storage)
    # update transition matrix
    update_transition_matrix!(model, FB_storage)
    # update regression models
    update_emissions!(model, FB_storage, data)
end

function mstep!(model::AbstractHMM, FB_storage_vec::Vector{ForwardBackward{Float64}}, Aggregate_FB_storage::ForwardBackward, data)
    # update initial state distribution
    update_initial_state_distribution!(model, FB_storage_vec)
    # update transition matrix
    update_transition_matrix!(model, FB_storage_vec)
    # update regression models
    update_emissions!(model, Aggregate_FB_storage, data)
end

function update_initial_state_distribution!(model::AbstractHMM, FB_storage_vec::Vector{ForwardBackward{Float64}})
    num_trials = length(FB_storage_vec)
    return model.πₖ = mean([exp.(FB_storage_vec[i].γ[:, 1]) for i in 1:num_trials])
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
function fit!(model::HiddenMarkovModel, Y::Matrix{<:Real}, X::Matrix{<:Real}; max_iters::Int=100, tol::Float64=1e-6,)
    fit!(model, to_f64(Y), to_f64(X); max_iters=max_iters, tol=tol)
end 

function fit!(model::HiddenMarkovModel, Y::Matrix{<:Real}; max_iters::Int=100, tol::Float64=1e-6,)
    fit!(model, to_f64(Y), nothing; max_iters=max_iters, tol=tol)
end

function fit!(
    model::HiddenMarkovModel,
    Y::Matrix{Float64},
    X::Union{Matrix{Float64},Nothing}=nothing;
    max_iters::Int=100,
    tol::Float64=1e-6,
)
    lls = [-Inf]

    data = X === nothing ? (Y,) : (X, Y)

    # transpose data so that correct dimensions are passed to EmissionModels.jl, a bit hacky but works for now.
    transpose_data = Matrix.(transpose.(data))

    num_obs = size(transpose_data[1], 1)
    # initialize forward backward storage
    FB_storage = initialize_forward_backward(model, num_obs)

    log_likelihood = -Inf
    # Initialize progress bar
    # p = Progress(max_iters; desc="Running EM algorithm...", barlen=50, showspeed=true)
    for iter in 1:max_iters
        # next!(p)
        # E-Step
        estep!(model, transpose_data, FB_storage)

        # Compute and update the log-likelihood
        log_likelihood_current = logsumexp(FB_storage.α[:, end])
        push!(lls, log_likelihood_current)
        if abs(log_likelihood_current - log_likelihood) < tol
            # finish!(p)
            return lls
        else
            log_likelihood = log_likelihood_current
        end
        # M-Step
        mstep!(model, FB_storage, transpose_data)

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
function fit!(model::HiddenMarkovModel, Y::Vector{<:Matrix{<:Real}}, X::Vector{<:Matrix{<:Real}}; max_iters::Int=100, tol::Float64=1e-6,)
    fit!(model, to_f64(Y), to_f64(X); max_iters=max_iters, tol=tol)
end 

function fit!(model::HiddenMarkovModel, Y::Vector{<:Matrix{<:Real}}; max_iters::Int=100, tol::Float64=1e-6)
    fit!(model, to_f64(Y), nothing; max_iters=max_iters, tol=tol)
end 

function fit!(
    model::HiddenMarkovModel,
    Y::Vector{<:Matrix{Float64}},
    X::Union{Vector{<:Matrix{Float64}},Nothing}=nothing;
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
    total_obs = sum(size(trial_mat[1], 1) for trial_mat in zipped_matrices)

    # initialize a vector of ForwardBackward storage and an aggregate storage
    FB_storage_vec = [initialize_forward_backward(model, size(trial_tuple[1],1)) for trial_tuple in zipped_matrices]
    Aggregate_FB_storage = initialize_forward_backward(model, total_obs)
    
    p = Progress(max_iters; desc="Running EM algorithm...", barlen=50, showspeed=true)
    for iter in 1:max_iters
        # broadcast estep!() to all storage structs
        output = estep!.(Ref(model), zipped_matrices, FB_storage_vec)

        # collect storage stucts into one struct for m step
        aggregate_forward_backward!(Aggregate_FB_storage, FB_storage_vec)

        # Calculate log_likelihood
        log_likelihood_current = sum([logsumexp(FB_vec.α[:, end]) for FB_vec in FB_storage_vec])
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
        mstep!(model, FB_storage_vec, Aggregate_FB_storage, stacked_data)
    end

    return lls
end


"""
    function class_probabilities(model::HiddenMarkovModel, Y::Matrix{<:Real}, X::Union{Matrix{<:Real},Nothing}=nothing;)

Calculate the class probabilities at each time point using forward backward algorithm

# Arguments
- `model::HiddenMarkovModel`: The Hidden Markov Model to fit.
- `Y::Matrix{<:Real}`: The emission data
- `X::Union{Matrix{<:Real},Nothing}=nothing`: Optional input data for fitting Switching Regression Models

# Returns
- `class_probabilities::Matrix{Float64}`: The class probabilities at each timepoint
"""
function class_probabilities(model::HiddenMarkovModel, Y::Matrix{Float64}, X::Union{Matrix{Float64},Nothing}=nothing;)
    data = X === nothing ? (Y,) : (X, Y)
    # transpose data so that correct dimensions are passed to EmissionModels.jl, a bit hacky but works for now.
    transpose_data = Matrix.(transpose.(data))
    num_obs = size(transpose_data[1], 1)
    # initialize forward backward storage
    FB_storage = initialize_forward_backward(model, num_obs)

    # Get class probabilities using Estep
    estep!(model, transpose_data, FB_storage)

    return exp.(FB_storage.γ)
end

"""
    function class_probabilities(model::HiddenMarkovModel, Y::Vector{<:Matrix{<:Real}}, X::Union{Vector{<:Matrix{<:Real}},Nothing}=nothing;)

Calculate the class probabilities at each time point using forward backward algorithm on multiple trials of data

# Arguments
- `model::HiddenMarkovModel`: The Hidden Markov Model to fit.
- `Y::Vectpr{<:Matrix{<:Real}}`: The trials of emission data
- `X::Union{Vector{<:Matrix{<:Real}},Nothing}=nothing`: Optional trials of input data for fitting Switching Regression Models

# Returns
- `class_probabilities::Vector{<:Matrix{Float64}}`: Each trial's class probabilities at each timepoint
"""
function class_probabilities(
    model::HiddenMarkovModel,
    Y_trials::Vector{<:Matrix{Float64}},
    X_trials::Union{Vector{<:Matrix{Float64}}, Nothing} = nothing
)
    n_trials = length(Y_trials)
    # Preallocate storage for class probabilities
    all_class_probs = Vector{Matrix{Float64}}(undef, n_trials)
    # Loop through each trial and compute class probabilities
    for i in 1:n_trials
        Y = Y_trials[i]
        X = X_trials === nothing ? nothing : X_trials[i]
        all_class_probs[i] = class_probabilities(model, Y, X)
    end

    return all_class_probs
end


"""
    viterbi(model::HiddenMarkovModel, Y::Matrix{<:Real}, X::Union{Matrix{<:Real},Nothing}=nothing;)

Get most likely class labels using the Viterbi algorithm

# Arguments
- `model::HiddenMarkovModel`: The Hidden Markov Model to fit.
- `Y::Matrix{<:Real}`: The emission data
- `X::Union{Matrix{<:Real},Nothing}=nothing`: Optional input data for fitting Switching Regression Models

# Returns
- `best_path::Vector{Float64}`: The most likely state label at each timepoint
"""
function viterbi(model::HiddenMarkovModel, Y::Matrix{<:Real}, X::Matrix{<:Real};)
    viterbi(model, to_f64(Y), to_f64(X);)
end 

function viterbi(model::HiddenMarkovModel, Y::Matrix{<:Real};)
    viterbi(model, to_f64(Y))
end 

function viterbi(model::HiddenMarkovModel, Y::Matrix{Float64}, X::Union{Matrix{Float64},Nothing}=nothing;)
    data = X === nothing ? (Y,) : (X, Y)

    # transpose data so that correct dimensions are passed to EmissionModels.jl, a bit hacky but works for now.
    transpose_data = Matrix.(transpose.(data))
    num_obs = size(transpose_data[1], 1)
    FB_storage = initialize_forward_backward(model, num_obs)

    # Estep to get loglikelihoods for each emission model at each timepoint
    estep!(model, transpose_data, FB_storage)

    # Number of states and timepoints
    num_states = length(model.πₖ)
    num_timepoints = size(FB_storage.loglikelihoods, 2)

    # Initialize viterbi_storage with log probabilities and backpointer matrix
    viterbi_storage = zeros(Float64, num_states, num_timepoints)
    backpointers = zeros(Int, num_states, num_timepoints)

    # Initialization step (t = 1)
    for s in 1:num_states
        viterbi_storage[s, 1] = log(model.πₖ[s]) + FB_storage.loglikelihoods[s, 1]
    end

    # Recursion step
    for t in 2:num_timepoints
        for s in 1:num_states
            # Calculate the log-probability for each previous state transitioning to current state
            log_probs = viterbi_storage[:, t-1] .+ log.(model.A[:, s])  # Previous state's prob + transition
            # Find the maximum log-probability and the corresponding state
            max_prob, max_state = findmax(log_probs)
            viterbi_storage[s, t] = max_prob + FB_storage.loglikelihoods[s, t]
            backpointers[s, t] = max_state
        end
    end

    # Backtracking step
    best_path = zeros(Int, num_timepoints)
    # Start with the state that has the highest probability at the last timepoint
    _, best_last_state = findmax(viterbi_storage[:, end])
    best_path[end] = best_last_state

    # Trace back the best path
    for t in num_timepoints-1:-1:1
        best_path[t] = backpointers[best_path[t+1], t+1]
    end

    return best_path
end

"""
    viterbi(model::HiddenMarkovModel, Y::Vector{<:Matrix{<:Real}}, X::Union{Vector{<:Matrix{<:Real}},Nothing}=nothing;)

Get most likely class labels using the Viterbi algorithm for multiple trials of data

# Arguments
- `model::HiddenMarkovModel`: The Hidden Markov Model to fit.
- `Y::Vectpr{<:Matrix{<:Real}}`: The trials of emission data
- `X::Union{Vector{<:Matrix{<:Real}},Nothing}=nothing`: Optional trials of input data for fitting Switching Regression Models

# Returns
- `best_path::Vector{<:Vector{Float64}}`: Each trial's best state path
"""
function viterbi(model::HiddenMarkovModel, Y::Vector{<:Matrix{<:Real}}, X::Vector{<:Matrix{<:Real}};)
    viterbi(model, to_f64(Y), to_f64(X);)
end 

function viterbi(model::HiddenMarkovModel, Y::Vector{<:Matrix{<:Real}};)
    viterbi(model, to_f64(Y), nothing)
end 

function viterbi(
    model::HiddenMarkovModel,
    Y::Vector{<:Matrix{Float64}},
    X::Union{Vector{<:Matrix{Float64}},Nothing}=nothing
)
    # Storage for each trials viterbi path
    viterbi_paths = Vector{Vector{Int}}(undef, length(Y))

    # Run session viterbi on each trial
    for i in 1:length(Y)
        Xi = X === nothing ? nothing : X[i]
        viterbi_paths[i] = viterbi(model, Y[i], Xi)
    end

    return viterbi_paths
end