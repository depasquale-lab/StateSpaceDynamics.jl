# Public API
export HiddenMarkovModel
export kmeans_init!, fit!, rand, loglikelihood, viterbi, class_probabilities

"""
    HiddenMarkovModel

Store a Hidden Markov Model (HMM) with custom emissions.

# Fields
- `A::AbstractMatrix{<:Real}`: Transition matrix.
- `B::AbstractVector{<:EmissionModel}`: State-dependent emission models.
- `πₖ::AbstractVector{<:Real}`: Initial state distribution.
- `K::Int`: Number of states.
"""
mutable struct HiddenMarkovModel{
    T<:Real,V<:AbstractVector{T},M<:AbstractMatrix{T},VE<:AbstractVector{<:EmissionModel}
} <: AbstractHMM
    A::M # transition matrix
    B::VE # Vector of emission Models
    πₖ::V # initial state distribution
    K::Int # number of states
end

function Base.show(io::IO, hmm::HiddenMarkovModel; gap="")
    println(io, gap, "Hidden Markov Model:")
    println(io, gap, "--------------------")

    if hmm.K > 3
        println(io, gap, " size(A)  = ($(size(hmm.A,1)), $(size(hmm.A,2)))")
        println(io, gap, " size(πₖ) = ($(size(hmm.πₖ,1)),)")
    else
        println(io, gap, " A  = $(round.(hmm.A, sigdigits=3))")
        println(io, gap, " πₖ = $(round.(hmm.πₖ, sigdigits=3))")
    end

    println(io, gap, " Emission Models:")
    println(io, gap, " ----------------")

    show_all = get(io, :limit, true) == false

    if hmm.K > 4 && !show_all
        # only show 3
        for b in hmm.B[1:3]
            Base.show(io, b; gap=gap * "  ")
            println(io, gap, "  ----------------")
        end
        println(io, gap * "  $(hmm.K-3) more ..., see `print_full()`")
    else
        for (i, b) in enumerate(hmm.B)
            Base.show(io, b; gap=gap * "  ")
            if i < hmm.K
                println(io, gap, "  ----------------")
            end
        end
    end

    return nothing
end

"""
    initialize_forward_backward(model::AbstractHMM, num_obs::Int)

Initialize the forward backward storage struct.
"""
function initialize_forward_backward(
    model::AbstractHMM, num_obs::Int, ::Type{T}
) where {T<:Real}
    num_states = model.K

    return ForwardBackward(
        zeros(T, num_states, num_obs),
        zeros(T, num_states, num_obs),
        zeros(T, num_states, num_obs),
        zeros(T, num_states, num_obs),
        zeros(T, num_states, num_states),
    )
end

"""
    ForwardBackward(
        loglikelihoods::Matrix{T},
        α::Matrix{T},
        β::Matrix{T},
        γ::Matrix{T},
        ξ::Matrix{T} where {T<:Real}

Initialize the forward backward storage struct.
"""
function ForwardBackward(
    loglikelihoods::Matrix{T}, α::Matrix{T}, β::Matrix{T}, γ::Matrix{T}, ξ::Matrix{T}
) where {T<:Real}
    return ForwardBackward{T,Vector{T},Matrix{T},Matrix{T}}(loglikelihoods, α, β, γ, ξ)
end

"""
    aggregate_forward_backward!(
        aggregated_FB::ForwardBackward,
        FB_storages::Vector{<:ForwardBackward}
    )

Aggregate single trial ForwardBackward structs to one session-wide struct.
"""
function aggregate_forward_backward!(
    aggregated_FB::ForwardBackward, FB_storages::Vector{<:ForwardBackward}
)
    # Concatenate each field into the respective field in the aggregated struct
    aggregated_FB.loglikelihoods .= hcat([fb.loglikelihoods for fb in FB_storages]...)
    aggregated_FB.α .= hcat([fb.α for fb in FB_storages]...)
    aggregated_FB.β .= hcat([fb.β for fb in FB_storages]...)
    aggregated_FB.γ .= hcat([fb.γ for fb in FB_storages]...)
    return aggregated_FB.ξ = hcat([fb.ξ for fb in FB_storages]...)
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
        πₖ::AbstractVector{T}
    ) where {T<:Real}

Create a hidden Markov model.

# Arguments
    - `K::Int`: Number of latent states.
    - `B::AbstractVector{<:EmissionModel}`: State-dependent emission models.
    - ``
"""
function HiddenMarkovModel(;
    K::Int, B::AbstractVector{<:EmissionModel}, A::AbstractMatrix{T}, πₖ::AbstractVector{T}
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

    return nothing
end

"""
    Random.rand(
        rng::AbstractRNG,
        model::HiddenMarkovModel,
        X::Union{Matrix{<:Real}, Nothing}=nothing;
        n::Int,
        autoregressive::Bool=false
    )

Generate `n` samples from a Hidden Markov Model. Returns a tuple of the state sequence and
the observation sequence.

# Arguments
- `rng::AbstractRNG`: The seed.
- `model::HiddenMarkovModel`: The Hidden Markov Model to sample from.
- `X`: The input data for switching regression models.
- `n::Int`: The number of samples to generate.

# Returns
- `state_sequence::Vector{Int}`: The state sequence, where each element is an integer 1:K.
- `observation_sequence::Matrix{Float64}`: The observation sequence. This takes the form of
    the emission model's output.
"""
function Random.rand(
    rng::AbstractRNG,
    model::HiddenMarkovModel,
    X::Union{Matrix{<:Real},Nothing}=nothing;
    n::Int,
    autoregressive::Bool=false,
)
    T = typeof(model.A[1])
    state_sequence = Vector{Int}(undef, n)
    observation_sequence = Matrix{T}(undef, model.B[1].output_dim, n)

    if !autoregressive
        # Sample initial state
        state_sequence[1] = rand(rng, Categorical(model.πₖ))
        observation_sequence[:, 1] = if isnothing(X)
            rand(rng, model.B[state_sequence[1]])
        else
            rand(rng, model.B[state_sequence[1]], X[:, 1])
        end

        # Sample remaining steps
        for t in 2:n
            state_sequence[t] = rand(rng, Categorical(model.A[state_sequence[t - 1], :]))

            if isnothing(X)
                observation_sequence[:, t] = rand(rng, model.B[state_sequence[t]])
            else
                observation_sequence[:, t] = rand(rng, model.B[state_sequence[t]], X[:, t])
            end
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
        autoregressive::Bool=false
    )

Generate `n` sammples from an HMM.
"""
function Random.rand(
    model::HiddenMarkovModel,
    X::Union{Matrix{<:Real},Nothing}=nothing;
    n::Int,
    autoregressive::Bool=false,
)
    return rand(Random.default_rng(), model, X; n=n, autoregressive=autoregressive)
end

"""
    emission_loglikelihoods!(model::HiddenMarkovModel, FB_storage::ForwardBackward, data...)

Calculate observation likelihoods for each state.
"""
function emission_loglikelihoods!(
    model::HiddenMarkovModel, FB_storage::ForwardBackward, data...
)
    log_likelihoods = FB_storage.loglikelihoods

    @threads for k in 1:model.K
        log_likelihoods[k, :] .= loglikelihood(model.B[k], data...)
    end

    return nothing
end

# Thin wrapper that computes logs once if needed
function forward!(model::AbstractHMM, FB::ForwardBackward)
    return forward!(model, FB, log.(model.A), log.(model.πₖ))
end

"""
    forward!(model, FB, logA, logπ)

Compute α in log-space using precomputed `logA = log.(A)` and `logπ = log.(πₖ)`.
"""
function forward!(
    model::AbstractHMM, FB::ForwardBackward, logA::AbstractMatrix, logπ::AbstractVector
)
    @assert size(logA, 1) == model.K && size(logA, 2) == model.K
    @assert length(logπ) == model.K

    α = FB.α                  # K×T (log)
    ll = FB.loglikelihoods     # K×T (log)
    K, T = size(α, 1), size(α, 2)

    @inbounds @views begin
        # t = 1
        for k in 1:K
            α[k, 1] = logπ[k] + ll[k, 1]
        end

        # t ≥ 2
        for t in 2:T
            # for each destination state k, do logsumexp over i
            for k in 1:K
                # compute logsumexp_i (α[i,t-1] + logA[i,k])
                # do the max pass
                m = -Inf
                for i in 1:K
                    v = α[i, t - 1] + logA[i, k]
                    m = ifelse(v > m, v, m)
                end
                # sum exp pass
                s = 0.0
                if isfinite(m)
                    for i in 1:K
                        s += exp(α[i, t - 1] + logA[i, k] - m)
                    end
                    α[k, t] = (m + log(s)) + ll[k, t]
                else
                    α[k, t] = -Inf
                end
            end
        end
    end
    return nothing
end

# Thin wrapper that computes logA if needed
function backward!(model::AbstractHMM, FB::ForwardBackward)
    return backward!(model, FB, log.(model.A))
end

"""
    backward!(model, FB, logA)

Compute β in log-space using precomputed `logA = log.(A)`.
"""
function backward!(model::AbstractHMM, FB::ForwardBackward, logA::AbstractMatrix)
    β = FB.β
    ll = FB.loglikelihoods
    K, T = size(β, 1), size(β, 2)

    @inbounds @views begin
        # β_T = 0 in log-space
        for i in 1:K
            β[i, T] = 0.0
        end

        for t in (T - 1):-1:1
            # for each source state i, do logsumexp over j
            for i in 1:K
                # compute logsumexp_j (logA[i,j] + ll[j,t+1] + β[j,t+1])
                m = -Inf
                for j in 1:K
                    v = logA[i, j] + ll[j, t + 1] + β[j, t + 1]
                    m = ifelse(v > m, v, m)
                end
                s = 0.0
                if isfinite(m)
                    for j in 1:K
                        s += exp(logA[i, j] + ll[j, t + 1] + β[j, t + 1] - m)
                    end
                    β[i, t] = m + log(s)
                else
                    β[i, t] = -Inf
                end
            end
        end
    end
    return nothing
end

"""
    calculate_γ!(model, FB)

Compute γ = log-normalized α+β (still in log-space).
"""
function calculate_γ!(::AbstractHMM, FB::ForwardBackward)
    γ = FB.γ
    α = FB.α
    β = FB.β
    K, T = size(γ, 1), size(γ, 2)

    @inbounds @views begin
        # γ = α + β
        γ .= α .+ β

        # subtract logsumexp per column
        for t in 1:T
            # find max
            m = -Inf
            for i in 1:K
                v = γ[i, t]
                m = ifelse(v > m, v, m)
            end
            # sum exp
            s = 0.0
            if isfinite(m)
                for i in 1:K
                    s += exp(γ[i, t] - m)
                end
                lZ = m + log(s)
                for i in 1:K
                    γ[i, t] -= lZ
                end
            else
                # all -Inf: leave as -Inf
            end
        end
    end
    return nothing
end

"""
    calculate_ξ!(model::AbstractHMM, FB::ForwardBackward)

Compute sum_t ξ_t(i,j) and store **log** of that sum in `FB.ξ`.

Assumes:
- FB.α, FB.β, FB.γ are in **log-space**
- FB.loglikelihoods are **log p(y_t | state)**
- model.A is the transition matrix in **linear** probabilities (not logs)

Numerically stable: does per-t max-shifts, normalizes each ξ_t to sum to 1.
"""
function calculate_ξ!(model::AbstractHMM, FB::ForwardBackward)
    αlog = FB.α                 # K×T (log-space)
    βlog = FB.β                 # K×T (log-space)
    llik = FB.loglikelihoods    # K×T (log-space)
    A = model.A              # K×K (linear probs)

    K, T = size(αlog)
    ξacc = FB.ξ                 # K×K accumulator (we'll store linear sums first)
    fill!(ξacc, 0.0)

    tmp = similar(ξacc)         # K×K scratch
    ax = similar(αlog, K)      # K scratch vector (α in linear space with shift)
    by = similar(αlog, K)      # K scratch vector (β*lik in linear space with shift)

    for t in 1:(T - 1)
        # ax := exp(αlog[:,t] - max)
        x = @view αlog[:, t]
        mx = maximum(x)
        @. ax = exp(x - mx)

        # by := exp( (βlog[:,t+1] + llik[:,t+1]) - max )
        v = @view βlog[:, t + 1]
        w = @view llik[:, t + 1]
        # First accumulate v+w in by to avoid a temporary:
        m = -Inf
        for j in 1:K
            y = v[j] + w[j]
            by[j] = y
            m = ifelse(y > m, y, m)
        end
        for j in 1:K
            by[j] = exp(by[j] - m)
        end

        # tmp := (ax * by') ⊙ A
        mul!(tmp, ax, transpose(by))  # outer product
        tmp .*= A

        # Normalize ξ_t and accumulate in linear space
        s = sum(tmp)
        if s > 0.0
            ξacc .+= tmp .* (1.0 / s)
        end
        # If s == 0.0, this t contributes nothing (all prob mass underflowed)
    end

    # Convert accumulator to log-space once (matching logaddexp over t)
    for j in 1:K, i in 1:K
        v = ξacc[i, j]
        ξacc[i, j] = v > 0.0 ? log(v) : -Inf
    end

    return nothing
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

    return nothing
end

"""
    loglikelihood(model::HiddenMarkovModel, Y::AbstractMatrix{T}, X::Union{AbstractMatrix{<:Real},Nothing}=nothing) where {T<:Real}

Calculate the log-likelihood of observed data given the HMM.

# Arguments
- `model::HiddenMarkovModel`: The fitted HMM model
- `Y::AbstractMatrix{T}`: The emission data (D × T)
- `X::Union{AbstractMatrix{<:Real},Nothing}=nothing`: Optional input data for switching regression models

# Returns
- `ll::Float64`: The total log-likelihood of the data
"""
function loglikelihood(
    model::HiddenMarkovModel,
    Y::AbstractMatrix{T},
    X::Union{AbstractMatrix{<:Real},Nothing}=nothing,
) where {T<:Real}
    data = X === nothing ? (Y,) : (X, Y)

    # Transpose data to match expected dimensions
    transpose_data = Matrix.(transpose.(data))
    num_obs = size(transpose_data[1], 1)

    # Initialize forward-backward storage
    FB_storage = initialize_forward_backward(model, num_obs, T)

    # Run E-step to compute forward probabilities
    estep!(model, transpose_data, FB_storage)

    # The log-likelihood is the log-sum-exp of the final forward probabilities
    return logsumexp(FB_storage.α[:, end])
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

function update_initial_state_distribution!(
    model::AbstractHMM, FB_storage_vec::Vector{<:ForwardBackward}
)
    num_trials = length(FB_storage_vec)

    return model.πₖ = mean([exp.(FB_storage_vec[i].γ[:, 1]) for i in 1:num_trials])
end

"""
    update_transition_matrix!(model::AbstractHMM, FB_storage::ForwardBackward)

Update the transition matrix of an HMM.
"""
function update_transition_matrix!(model::AbstractHMM, FB_storage::ForwardBackward)
    γ = FB_storage.γ
    ξ = FB_storage.ξ  # eventually use this
    for i in 1:model.K
        for j in 1:model.K
            model.A[i, j] = exp(ξ[i, j] - logsumexp(@view γ[i, 1:(end - 1)]))
        end
    end

    return nothing
end

function update_transition_matrix!(
    model::AbstractHMM, FB_storage_vec::Vector{<:ForwardBackward}
)
    K = model.K

    # Initialize numerator and denominator
    log_num = fill(-Inf, K, K)
    log_denom = fill(-Inf, K)

    for FB_trial in FB_storage_vec
        # Accumulate ξ sum (already in log space)
        log_num .= map((x, y) -> logaddexp(x, y), log_num, FB_trial.ξ)

        # Accumulate γ sum for denominator (exclude last time step)
        for i in 1:K
            log_denom[i] = logaddexp(log_denom[i], logsumexp(FB_trial.γ[i, 1:(end - 1)]))
        end
    end

    # Update A
    for j in 1:K
        for k in 1:K
            model.A[j, k] = exp(log_num[j, k] - log_denom[j])
        end
    end

    return nothing
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

    return nothing
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

    return nothing
end

function mstep!(
    model::AbstractHMM,
    FB_storage_vec::Vector{<:ForwardBackward},
    Aggregate_FB_storage::ForwardBackward,
    data,
)
    # update initial state distribution
    update_initial_state_distribution!(model, FB_storage_vec)
    # update transition matrix
    update_transition_matrix!(model, FB_storage_vec)
    # update regression models
    update_emissions!(model, Aggregate_FB_storage, data)

    return nothing
end

"""
    fit!(
        model::HiddenMarkovModel,
        Y::Matrix{<:Real},
        X::Union{Matrix{<:Real}, Nothing}=nothing
        ;
        max_iters::Int=100,
        tol::Float64=1e-6
    )

Fit the Hidden Markov Model using the EM algorithm.

# Arguments
- `model::HiddenMarkovModel`: The Hidden Markov Model to fit.
- `Y::Matrix{<:Real}`: The emission data.
- `X::Union{Matrix{<:Real}, Nothing}=nothing`: Optional input data for fitting Switching
    Regression Models
- `max_iters::Int=100`: The maximum number of iterations to run the EM algorithm.
- `tol::Float64=1e-6`: When the log likelihood is improving by less than this value, the
    algorithm will stop.
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
    fit!(
        model::HiddenMarkovModel,
        Y::Matrix{<:Real},
        X::Union{Matrix{<:Real}, Nothing}=nothing;
        max_iters::Int=100,
        tol::Float64=1e-6
    )

Fit the Hidden Markov Model to multiple trials of data using the EM algorithm.

# Arguments
- `model::HiddenMarkovModel`: The Hidden Markov Model to fit.
- `Y::Vector{<:Matrix{<:Real}}`: The trialized emission data.
- `X::Union{Vector{<:Matrix{<:Real}}, Nothing}=nothing`: Optional input data for fitting
    Switching Regression Models
- `max_iters::Int=100`: The maximum number of iterations to run the EM algorithm.
- `tol::Float64=1e-6`: When the log likelihood is improving by less than this value, the
    algorithm will stop.
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
    FB_storage_vec = [
        initialize_forward_backward(model, size(trial_tuple[1], 1), T) for
        trial_tuple in zipped_matrices
    ]
    Aggregate_FB_storage = initialize_forward_backward(model, total_obs, T)

    p = Progress(max_iters; desc="Running EM algorithm...", barlen=50, showspeed=true)

    for iter in 1:max_iters
        # broadcast estep!() to all storage structs
        output = estep!.(Ref(model), zipped_matrices, FB_storage_vec)

        # collect storage stucts into one struct for m step
        aggregate_forward_backward!(Aggregate_FB_storage, FB_storage_vec)

        # Calculate log_likelihood
        log_likelihood_current = sum([
            logsumexp(FB_vec.α[:, end]) for FB_vec in FB_storage_vec
        ])
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
    class_probabilities(
        model::HiddenMarkovModel,
        Y::Matrix{<:Real},
        X::Union{Matrix{<:Real},Nothing}=nothing;
    )

Calculate the class probabilities at each time point using forward backward algorithm

# Arguments
- `model::HiddenMarkovModel`: The Hidden Markov Model to fit.
- `Y::Matrix{<:Real}`: The emission data
- `X::Union{Matrix{<:Real},Nothing}=nothing`: Optional input data for fitting Switching
    Regression Models

# Returns
- `class_probabilities::Matrix{Float64}`: The class probabilities at each timepoint
"""
function class_probabilities(
    model::HiddenMarkovModel,
    Y::AbstractMatrix{T},
    X::Union{AbstractMatrix{<:Real},Nothing}=nothing;
) where {T<:Real}
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
    class_probabilities(
        model::HiddenMarkovModel,
        Y::Vector{<:Matrix{<:Real}},
        X::Union{Vector{<:Matrix{<:Real}},Nothing}=nothing;
    )

Calculate the class probabilities at each time point using forward backward algorithm on
multiple trials of data

# Arguments
- `model::HiddenMarkovModel`: The Hidden Markov Model to fit.
- `Y::Vectpr{<:Matrix{<:Real}}`: The trials of emission data
- `X::Union{Vector{<:Matrix{<:Real}},Nothing}=nothing`: Optional trials of input data for
    fitting Switching Regression Models

# Returns
- `class_probabilities::Vector{<:Matrix{Float64}}`: Each trial's class probabilities at each
    timepoint
"""
function class_probabilities(
    model::HiddenMarkovModel,
    Y_trials::Vector{<:Matrix{<:Real}},
    X_trials::Union{Vector{<:Matrix{<:Real}},Nothing}=nothing,
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
    viterbi(
        model::HiddenMarkovModel,
        Y::Matrix{<:Real},
        X::Union{Matrix{<:Real},Nothing}=nothing
    )

Get most likely class labels using the Viterbi algorithm

# Arguments
- `model::HiddenMarkovModel`: The Hidden Markov Model to fit.
- `Y::Matrix{<:Real}`: The emission data
- `X::Union{Matrix{<:Real},Nothing}=nothing`: Optional input data for fitting Switching
    Regression Models

# Returns
- `best_path::Vector{Float64}`: The most likely state label at each timepoint
"""
function viterbi(
    model::HiddenMarkovModel,
    Y::AbstractMatrix{T},
    X::Union{AbstractMatrix{<:Real},Nothing}=nothing,
) where {T<:Real}
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

    for t in (num_timepoints - 1):-1:1
        best_path[t] = backpointers[best_path[t + 1], t + 1]
    end

    return best_path
end

"""
    viterbi(
        model::HiddenMarkovModel,
        Y::Vector{<:Matrix{<:Real}},
        X::Union{Vector{<:Matrix{<:Real}},Nothing}=nothing
    )

Get most likely class labels using the Viterbi algorithm for multiple trials of data

# Arguments
- `model::HiddenMarkovModel`: The Hidden Markov Model to fit.
- `Y::Vectpr{<:Matrix{<:Real}}`: The trials of emission data
- `X::Union{Vector{<:Matrix{<:Real}},Nothing}=nothing`: Optional trials of input data for
    fitting Switching Regression Models

# Returns
- `best_path::Vector{<:Vector{Float64}}`: Each trial's best state path
"""
function viterbi(
    model::HiddenMarkovModel,
    Y::Vector{<:Matrix{<:Real}},
    X::Union{Vector{<:Matrix{<:Real}},Nothing}=nothing,
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
