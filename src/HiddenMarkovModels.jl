# Public API
export HiddenMarkovModel, fit!, rand, loglikelihood, viterbi, class_probabilities
export kmeans_init!

"""
    HiddenMarkovModel

Store a Hidden Markov Model (HMM) with custom emissions.

# Fields
- `A::AbstractMatrix{<:Real}`: Transition matrix.
- `B::AbstractVector{<:EmissionModel}`: State-dependent emission models.
- `πₖ::AbstractVector{<:Real}`: Initial state distribution.
- `K::Int`: Number of states.
"""
mutable struct HiddenMarkovModel{T<:Real, V<:AbstractVector{T}, M<:AbstractMatrix{T}, VE<:AbstractVector{<:EmissionModel}} <: AbstractHMM
    A::M # transition matrix
    B::VE # Vector of emission Models
    πₖ::V # initial state distribution
    K::Int # number of states
end

"""
    initialize_forward_backward(model::AbstractHMM, num_obs::Int)

Initialize the forward backward storage struct.
"""
function initialize_forward_backward(model::AbstractHMM, num_obs::Int, ::Type{T}) where {T<:Real}
    num_states = model.K
    ForwardBackward(
        zeros(T, num_states, num_obs),
        zeros(T, num_states, num_obs),
        zeros(T, num_states, num_obs),
        zeros(T, num_states, num_obs),
        zeros(T, num_states, num_states, num_obs - 1)
    )
end


"""
    ForwardBackward(
        loglikelihoods::Matrix{T},
        α::Matrix{T},
        β::Matrix{T},
        γ::Matrix{T},
        ξ::Array{T,3}) where {T<:Real}

Initialize the forward backward storage struct.
"""
function ForwardBackward(
    loglikelihoods::Matrix{T},
    α::Matrix{T},
    β::Matrix{T},
    γ::Matrix{T},
    ξ::Array{T,3}
) where {T<:Real}
    ForwardBackward{T, Vector{T}, Matrix{T}, Array{T,3}}(loglikelihoods, α, β, γ, ξ)
end

"""
    aggregate_forward_backward!(aggregated_FB::ForwardBackward, FB_storages::Vector{<:ForwardBackward})

Aggregate single trial ForwardBackward structs to one session-wide struct.
"""
function aggregate_forward_backward!(
    aggregated_FB::ForwardBackward, 
    FB_storages::Vector{<:ForwardBackward}
)
    # Concatenate each field into the respective field in the aggregated struct
    aggregated_FB.loglikelihoods .= hcat([fb.loglikelihoods for fb in FB_storages]...)
    aggregated_FB.α .= hcat([fb.α for fb in FB_storages]...)
    aggregated_FB.β .= hcat([fb.β for fb in FB_storages]...)
    aggregated_FB.γ .= hcat([fb.γ for fb in FB_storages]...)
    aggregated_FB.ξ = cat([fb.ξ for fb in FB_storages]..., dims=3)
    
end

"""
    initialize_transition_matrix(K::Int)

Initialize the HMM transition matrix.
"""
function initialize_transition_matrix(K::Int)
    # initialize a transition matrix
    A = zeros(Float64, K, K)
    @threads for i in 1:K
        A[i, :] = rand(Dirichlet(ones(K)))
    end
    return A
end

"""
    initialize_state_distribution(K::Int)

Initialize the HMM initial state distribution
"""
function initialize_state_distribution(K::Int)
    # initialize a state distribution
    return rand(Dirichlet(ones(K)))
end

"""
    HiddenMarkovModel(;
        K::Int,
        B::AbstractVector{<:EmissionModel},
        emission=nothing,
        A::AbstractMatrix{T},
        πₖ::AbstractVector{T}) where {T<:Real}

Create a hidden Markov model.

# Arguments
    - `K::Int`: Number of latent states.
    - `B::AbstractVector{<:EmissionModel}`: State-dependent emission models.
    - ``
"""
function HiddenMarkovModel(;
    K::Int,
    B::AbstractVector{<:EmissionModel},
    A::AbstractMatrix{T},
    πₖ::AbstractVector{T},
) where {T<:Real}

    model = HiddenMarkovModel(A, B, πₖ, K)

    # check that each state has an emission model
    @assert length(B) == model.K
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

"""
    kmeans_init!(model::HiddenMarkovModel, data::Matrix{T}) where {T<:Real}

Initialize HMM emission models using K-means clustering.

# Arguments
    - `model::HiddenMarkovModel`: The HMM
    - `data::Matrix{T}`: The data
"""
function kmeans_init!(model::HiddenMarkovModel, data::Matrix{T}) where {T<:Real}
    num_states = model.K
    # run k-means 
    means, labels = kmeans_clustering(data, num_states)
    covs = [cov(permutedims(data[:, labels .== i])) for i in 1:num_states]
    # initialize the emission models
    for k in 1:num_states
        model.B[k].μ = means[:, k]
        model.B[k].Σ = covs[k]
    end
end

"""
    Random.rand(
        rng::AbstractRNG,
        model::HiddenMarkovModel,
        X::Union{Matrix{<:Real}, Nothing}=nothing;
        n::Int,
        autoregressive::Bool=false)

Generate `n` samples from a Hidden Markov Model. Returns a tuple of the state sequence and the observation sequence.

# Arguments
- `rng::AbstractRNG`: The seed.
- `model::HiddenMarkovModel`: The Hidden Markov Model to sample from.
- `X`: The input data for switching regression models.
- `n::Int`: The number of samples to generate.

# Returns
- `state_sequence::Vector{Int}`: The state sequence, where each element is an integer 1:K.
- `observation_sequence::Matrix{Float64}`: The observation sequence. This takes the form of the emission model's output.
"""
function Random.rand(
    rng::AbstractRNG,
    model::HiddenMarkovModel,
    X::Union{Matrix{<:Real}, Nothing}=nothing;
    n::Int,
    autoregressive::Bool=false,
)
    T = typeof(model.A[1])
    state_sequence = Vector{Int}(undef, n)
    observation_sequence = Matrix{T}(undef, model.B[1].output_dim, n)

    if !autoregressive
        # Sample initial state
        state_sequence[1] = rand(rng, Categorical(model.πₖ))
        observation_sequence[:, 1] = isnothing(X) ?
            rand(rng, model.B[state_sequence[1]]) :
            rand(rng, model.B[state_sequence[1]], X[:, 1])

        # Sample remaining steps
        for t in 2:n
            state_sequence[t] = rand(rng, Categorical(model.A[state_sequence[t - 1], :]))
            observation_sequence[:, t] = isnothing(X) ?
                rand(rng, model.B[state_sequence[t]]) :
                rand(rng, model.B[state_sequence[t]], X[:, t])
        end
    else
        # Autoregressive case
        state_sequence[1] = rand(rng, Categorical(model.πₖ))
        X, observation_sequence[:, 1] = rand(rng, model.B[state_sequence[1]], X)

        for t in 2:n
            state_sequence[t] = rand(rng, Categorical(model.A[state_sequence[t - 1], :]))
            X, observation_sequence[:, t] = rand(rng, model.B[state_sequence[t]], X)
        end
    end

    return state_sequence, observation_sequence
end

"""
    Random.rand(
        model::HiddenMarkovModel,
        X::Union{Matrix{<:Real}, Nothing}=nothing;
        n::Int,
        autoregressive::Bool=false)

Generate `n` sammples from an HMM.
"""
function Random.rand(
    model::HiddenMarkovModel,
    X::Union{Matrix{<:Real}, Nothing}=nothing;
    n::Int,
    autoregressive::Bool=false,
)
    return rand(Random.default_rng(), model, X; n=n, autoregressive=autoregressive)
end


"""
    emission_loglikelihoods!(model::HiddenMarkovModel, FB_storage::ForwardBackward, data...)

Calculate observation likelihoods for each state.
"""
function emission_loglikelihoods!(model::HiddenMarkovModel, FB_storage::ForwardBackward, data...)
    log_likelihoods = FB_storage.loglikelihoods

    @threads for k in 1:model.K
        log_likelihoods[k, :] .= loglikelihood(model.B[k], data...)
    end
end

"""
    forward!(model::AbstractHMM, FB_storage::ForwardBackward)

Run the forward algorithm given an `AbstractHMM` and `ForwardBackd` storage struct.
"""
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
    for k in 1:K
        α[k, 1] = log_πₖ[k] + loglikelihoods[k, 1]
    end

    # Compute α for all time steps
    for t in 2:time_steps
        for k in 1:K
            for i in 1:K
                values_to_sum[i] = log_A[i, k] + α[i, t - 1]
            end
            α[k, t] = logsumexp(values_to_sum) + loglikelihoods[k, t]
        end
    end
end

"""
    backward!(model::AbstractHMM, FB_storage::ForwardBackward)

Run the backward algorithm given an `AbstractHMM` and `ForwardBackd` storage struct.
"""
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
    β[:, end] .= 0

    # Compute β for all time steps
    for t in (time_steps - 1):-1:1
        for i in 1:K
            for j in 1:K
                values_to_sum[j] = log_A[i, j] + loglikelihoods[j, t + 1] + β[j, t + 1]
            end
            β[i, t] = logsumexp(values_to_sum)
        end
    end
end

"""
    calculate_γ!(model::AbstractHMM, FB_storage::ForwardBackward)

Calculate the marginal posterior distribution for each state.
"""
function calculate_γ!(model::AbstractHMM, FB_storage::ForwardBackward)
    α = FB_storage.α
    β = FB_storage.β

    time_steps = size(α, 2)
    FB_storage.γ = α .+ β
    γ = FB_storage.γ

    for t in 1:time_steps
        γ[:, t] .-= logsumexp(view(γ,:,t))
    end
end

"""
    calculate_ξ!(model::AbstractHMM, FB_storage::ForwardBackward)

Calculate the joint posterior distribution for state transitions.
"""
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

    for t in 1:(time_steps - 1)
        @views begin
            for i in 1:K
                α_t = α[i, t]  # scalar, reuse as-is
                for j in 1:K
                    log_ξ_unnormalized[i, j] = α_t + log_A[i, j] + loglikelihoods[j, t + 1] + β[j, t + 1]
                end
            end

            # Normalize log-space ξ with logsumexp over a view
            log_norm_factor = logsumexp(log_ξ_unnormalized)

            ξ[:, :, t] .= log_ξ_unnormalized .- log_norm_factor
        end
    end
end

"""
    estep!(model::HiddenMarkovModel, data, FB_storage)

Run the E-step of the Expecation-Maximiation algorithm for HMMs.
"""
function estep!(model::HiddenMarkovModel, data, FB_storage)
    # compute lls of the observations
    emission_loglikelihoods!(model, FB_storage, data...)

    # run forward-backward algorithm
    forward!(model, FB_storage)
    backward!(model, FB_storage)
    calculate_γ!(model, FB_storage)
    calculate_ξ!(model, FB_storage)
end

"""
    update_initial_state_distribution!(model::AbstractHMM, FB_storage::ForwardBackward)

Update the initial state distribution of an HMM.
"""
function update_initial_state_distribution!(model::AbstractHMM, FB_storage::ForwardBackward)
    # Update initial state probabilities
    γ = FB_storage.γ
    return model.πₖ .= exp.(γ[:, 1])
end

function update_initial_state_distribution!(model::AbstractHMM, FB_storage_vec::Vector{<:ForwardBackward})
    num_trials = length(FB_storage_vec)
    return model.πₖ = mean([exp.(FB_storage_vec[i].γ[:, 1]) for i in 1:num_trials])
end

"""
    update_transition_matrix!(model::AbstractHMM, FB_storage::ForwardBackward)

Update the transition matrix of an HMM.
"""
function update_transition_matrix!(
    model::AbstractHMM, FB_storage::ForwardBackward
)
    γ = FB_storage.γ
    ξ = FB_storage.ξ
    for i in 1:model.K
        for j in 1:model.K
            model.A[i, j] = exp(
                logsumexp(@view ξ[i, j, :]) - logsumexp(@view γ[i, 1:end-1])
            )
        end
    end
end

function update_transition_matrix!(
    model::AbstractHMM, FB_storage_vec::Vector{<:ForwardBackward}
)
    for j in 1:model.K
        for k in 1:model.K
            # Numerator: aggregated ξ[j, k, :]
            num = exp(logsumexp(vcat([@view FB_trial.ξ[j, k, :] for FB_trial in FB_storage_vec]...)))

            # Denominator: aggregated ξ[j, :, :] over all t
            denom = exp(logsumexp(vcat([
                vec(@view FB_trial.γ[j, 1:end-1]) for FB_trial in FB_storage_vec
            ]...)))

            model.A[j, k] = num / denom
        end
    end
end

"""
    update_emissions!(model::AbstractHMM, FB_storage::ForwardBackward, data)

Update the emission models of an HMM.
"""
function update_emissions!(model::AbstractHMM, FB_storage::ForwardBackward, data)
    w = exp.(permutedims(FB_storage.γ))  # size (T, K)
    @threads for k in 1:model.K
        fit!(model.B[k], data..., @view w[:, k])
    end
end

"""
    mstep!(model::AbstractHMM, FB_storage::ForwardBackward, data)

Carry out the M-step of Expectation-Maximization.
"""
function mstep!(model::AbstractHMM, FB_storage::ForwardBackward, data)
    # update initial state distribution
    update_initial_state_distribution!(model, FB_storage)
    # update transition matrix
    update_transition_matrix!(model, FB_storage)
    # update regression models
    update_emissions!(model, FB_storage, data)
end

function mstep!(model::AbstractHMM, FB_storage_vec::Vector{<:ForwardBackward}, Aggregate_FB_storage::ForwardBackward, data)
    # update initial state distribution
    update_initial_state_distribution!(model, FB_storage_vec)
    # update transition matrix
    update_transition_matrix!(model, FB_storage_vec)
    # update regression models
    update_emissions!(model, Aggregate_FB_storage, data)
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
    Y::AbstractMatrix{T},
    X::Union{AbstractMatrix{<:Real},Nothing}=nothing;
    max_iters::Int=100,
    tol::Float64=1e-6,
) where {T<:Real}
    lls = Vector{T}()

    data = X === nothing ? (Y,) : (X, Y)

    # transpose data so that correct dimensions are passed to EmissionModels.jl, a bit hacky but works for now.
    transpose_data = Matrix.(transpose.(data))

    num_obs = size(transpose_data[1], 1)
    # initialize forward backward storage
    FB_storage = initialize_forward_backward(model, num_obs, T)

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
function fit!(
    model::HiddenMarkovModel,
    Y::Vector{<:AbstractMatrix{T}},
    X::Union{Vector{<:AbstractMatrix{<:Real}},Nothing}=nothing;
    max_iters::Int=100,
    tol::Float64=1e-6,
) where {T<:Real}
    lls = Vector{T}()
    data = X === nothing ? (Y,) : (X, Y)

    # Initialize log_likelihood
    log_likelihood = -Inf

    # Transform each matrix in each tuple to the correct orientation
    transposed_matrices = map(data_tuple -> Matrix.(transpose.(data_tuple)), data)
    zipped_matrices = collect(zip(transposed_matrices...))
    total_obs = sum(size(trial_mat[1], 1) for trial_mat in zipped_matrices)

    # initialize a vector of ForwardBackward storage and an aggregate storage
    FB_storage_vec = [initialize_forward_backward(model, size(trial_tuple[1],1), T) for trial_tuple in zipped_matrices]
    Aggregate_FB_storage = initialize_forward_backward(model, total_obs, T)
    
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
function class_probabilities(model::HiddenMarkovModel, Y::AbstractMatrix{T}, 
    X::Union{AbstractMatrix{<:Real},Nothing}=nothing;) where {T<:Real}
    data = X === nothing ? (Y,) : (X, Y)
    # transpose data so that correct dimensions are passed to EmissionModels.jl, a bit hacky but works for now.
    transpose_data = Matrix.(transpose.(data))
    num_obs = size(transpose_data[1], 1)
    # initialize forward backward storage
    FB_storage = initialize_forward_backward(model, num_obs, T)

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
    Y_trials::Vector{<:Matrix{<:Real}},
    X_trials::Union{Vector{<:Matrix{<:Real}}, Nothing} = nothing
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
function viterbi(model::HiddenMarkovModel, Y::AbstractMatrix{T}, 
    X::Union{AbstractMatrix{<:Real},Nothing}=nothing;) where {T<:Real}
    data = X === nothing ? (Y,) : (X, Y)

    # transpose data so that correct dimensions are passed to EmissionModels.jl
    transpose_data = Matrix.(transpose.(data))  # You could consider a view here too if you're memory-sensitive

    num_obs = size(transpose_data[1], 1)
    FB_storage = initialize_forward_backward(model, num_obs, T)

    estep!(model, transpose_data, FB_storage)

    num_states = length(model.πₖ)
    num_timepoints = size(FB_storage.loglikelihoods, 2)

    viterbi_storage = zeros(Float64, num_states, num_timepoints)
    backpointers = zeros(Int, num_states, num_timepoints)

    # Initialization
    for s in 1:num_states
        viterbi_storage[s, 1] = log(model.πₖ[s]) + FB_storage.loglikelihoods[s, 1]
    end

    # Recursion
    for t in 2:num_timepoints
        for s in 1:num_states
            v_prev = @view viterbi_storage[:, t - 1]
            A_col = @view model.A[:, s]
    
            log_probs = v_prev .+ log.(A_col)
            max_prob, max_state = findmax(log_probs)
    
            viterbi_storage[s, t] = max_prob + FB_storage.loglikelihoods[s, t]
            backpointers[s, t] = max_state
        end
    end

    # Backtracking
    best_path = zeros(Int, num_timepoints)
    _, best_last_state = findmax(@view viterbi_storage[:, end])
    best_path[end] = best_last_state

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
function viterbi(
    model::HiddenMarkovModel,
    Y::Vector{<:Matrix{<:Real}},
    X::Union{Vector{<:Matrix{<:Real}},Nothing}=nothing
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