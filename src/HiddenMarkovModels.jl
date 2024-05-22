export GaussianHMM, baumWelch!, viterbi, sample, initialize_transition_matrix, initialize_state_distribution

"""
    mutable struct GaussianHMM{GaussianEmission} <: AbstractHMM

Represents a Gaussian Hidden Markov Model (HMM).

# Fields
- `A::Matrix{Float64}`: State Transition Matrix.
- `B::Vector{GaussianEmission}`: Emission Model.
- `πₖ::Vector{Float64}`: Initial State Distribution.
- `K::Int`: Latent State Dimension.
- `D::Int`: Dimension of the data.
"""
mutable struct GaussianHMM{GaussianEmission} <: AbstractHMM
    A::Matrix{Float64}  # State Transition Matrix
    B::Vector{GaussianEmission}       # Emission Model
    πₖ ::Vector{Float64} # Initial State Distribution
    K::Int              # Latent State Dimension
    D::Int              # Dimension of the data
end

"""
    GaussianHMM(A::Matrix{Float64}, B::Vector{GaussianEmission}, πₖ::Vector{Float64}, K::Int, D::Int)

Creates a Gaussian Hidden Markov Model (HMM) with the specified parameters.

# Arguments
- `A::Matrix{Float64}`: State Transition Matrix.
- `B::Vector{GaussianEmission}`: Emission Model.
- `πₖ::Vector{Float64}`: Initial State Distribution.
- `K::Int`: Latent State Dimension.
- `D::Int`: Dimension of the data.

# Examples:
```julia
A = [0.7 0.3; 0.4 0.6]
B = [GaussianEmission([0.0, 0.0], [1.0 0.0; 0.0 1.0]), GaussianEmission([1.0, 1.0], [1.0 0.0; 0.0 1.0])]
πₖ = [0.6, 0.4]
K = 2
D = 2
hmm = GaussianHMM(A, B, πₖ, K, D)
```
"""
function GaussianHMM(A::Matrix{Float64}, B::Vector{GaussianEmission}, πₖ::Vector{Float64}, K::Int, D::Int)
    return GaussianHMM{GaussianEmission}(A, B, πₖ, K, D)
end

"""
    GaussianHMM(data::Matrix{Float64}, k_states::Int=2)

Creates a Gaussian Hidden Markov Model (HMM) from data, with the number of latent states specified.

# Arguments
- `data::Matrix{Float64}`: Observational data.
- `k_states::Int=2`: Number of latent states (default is 2).

# Returns
A `GaussianHMM` initialized using k-means clustering on the provided data.

# Examples:
```julia
data = randn(100, 2)
hmm = GaussianHMM(data, 2)
```
"""
function GaussianHMM(data::Matrix{Float64}, k_states::Int=2)
    N, D = size(data)
    # Initialize A
    A = initialize_transition_matrix(k_states)
    # Initialize π
    πₖ = initialize_state_distribution(k_states)
    # use kmeans_clustering to initialize the emission model
    sample_means, labels = kmeans_clustering(data, k_states)
    sample_covs = [cov(data[labels .== i, :]) for i in 1:k_states]
    B = [GaussianEmission(sample_means[:, i], sample_covs[i]) for i in 1:k_states]
    return GaussianHMM(A, B, πₖ, k_states, D)
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

function forward(hmm::AbstractHMM, data::Y) where Y <: AbstractArray
    T = size(data, 1)
    K = size(hmm.A, 1)  # Number of states
    # Initialize an α-matrix 
    α = zeros(Float64, T, K)
    # Calculate α₁
    @threads for k in 1:K
        α[1, k] = log(hmm.πₖ[k]) + loglikelihood(hmm.B[k], data[1, :])
    end
    # Now perform the rest of the forward algorithm for t=2 to T
    for t in 2:T
        @threads for j in 1:K
            values_to_sum = Float64[]
            for i in 1:K
                push!(values_to_sum, log(hmm.A[i, j]) + α[t-1, i])
            end
            log_sum_alpha_a = logsumexp(values_to_sum)
            α[t, j] = log_sum_alpha_a + loglikelihood(hmm.B[j], data[t, :])
        end
    end
    return α
end

function backward(hmm::AbstractHMM, data::Y) where Y <: AbstractArray
    T = size(data, 1)
    K = size(hmm.A, 1)  # Number of states

    # Initialize a β matrix
    β = zeros(Float64, T, K)

    # Set last β values. In log-space, 0 corresponds to a value of 1 in the original space.
    β[T, :] .= 0  # log(1) = 0

    # Calculate β, starting from T-1 and going backward to 1
    for t in T-1:-1:1
        @threads for i in 1:K
            values_to_sum = Float64[]
            for j in 1:K
                push!(values_to_sum, log(hmm.A[i, j]) + loglikelihood(hmm.B[j], data[t+1, :]) + β[t+1, j])
            end
            β[t, i] = logsumexp(values_to_sum)
        end
    end
    return β
end

function calculate_γ(hmm::AbstractHMM, α::Matrix{Float64}, β::Matrix{Float64})
    T = size(α, 1)
    γ = α .+ β
    @threads for t in 1:T
        γ[t, :] .-= logsumexp(γ[t, :])
    end
    return γ
end

function calculate_ξ(hmm::AbstractHMM, α::Matrix{Float64}, β::Matrix{Float64}, data::AbstractArray)
    T = size(α, 1)
    K = size(hmm.A, 1)
    ξ = zeros(Float64, T-1, K, K)
    for t in 1:T-1
        # Array to store the unnormalized ξ values
        log_ξ_unnormalized = zeros(Float64, K, K)
        @threads for i in 1:K
            for j in 1:K
                log_ξ_unnormalized[i, j] = α[t, i] + log(hmm.A[i, j]) + loglikelihood(hmm.B[j], data[t+1, :]) + β[t+1, j]
            end
        end
        # Normalize the ξ values using log-sum-exp operation
        ξ[t, :, :] .= log_ξ_unnormalized .- logsumexp(log_ξ_unnormalized)
    end
    return ξ
end

function update_initial_state_distribution!(hmm::AbstractHMM, γ::Matrix{Float64})
    # Update initial state probabilities
    hmm.πₖ .= exp.(γ[1, :])
end

function update_transition_matrix!(hmm::AbstractHMM, γ::Matrix{Float64}, ξ::Array{Float64, 3})
    K = size(hmm.A, 1)
    T = size(γ, 1)
    # Update transition probabilities
    @threads for i in 1:K
        for j in 1:K
            hmm.A[i, j] = exp(logsumexp(ξ[:, i, j]) - logsumexp(γ[1:T-1, i]))
        end
    end
end

function update_emission_models!(hmm::AbstractHMM, γ::Matrix{Float64}, data::Matrix{Float64})
    K = size(hmm.B, 1)
    # Update emission model
    for k in 1:K
        hmm.B[k] = updateEmissionModel!(hmm.B[k], data, exp.(γ[:, k]))
    end
end

function E_step(hmm::AbstractHMM, data::Matrix{Float64})
    α = forward(hmm, data)
    β = backward(hmm, data)
    γ = calculate_γ(hmm, α, β)
    ξ = calculate_ξ(hmm, α, β, data)
    return γ, ξ, α, β
end

function M_step!(hmm::AbstractHMM, γ::Matrix{Float64}, ξ::Array{Float64, 3}, data::Matrix{Float64})
    # Update initial state probabilities
    update_initial_state_distribution!(hmm, γ)
    # Update transition probabilities
    update_transition_matrix!(hmm, γ, ξ)
    # Update emission model 
    update_emission_models!(hmm, γ, data)
end

function baumWelch!(hmm::AbstractHMM, data::Matrix{Float64}, max_iters::Int=100, tol::Float64=1e-6)
    T, _ = size(data)
    K = size(hmm.A, 1)
    log_likelihood = -Inf
    # Initialize progress bar
    p = Progress(max_iters; dt=1, desc="Computing Baum-Welch...",)
    for iter in 1:max_iters
        # Update the progress bar
        next!(p; showvalues = [(:iteration, iter), (:log_likelihood, log_likelihood)])
        # E-Step
        γ, ξ, α, β = E_step(hmm, data)
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
        M_step!(hmm, γ, ξ, data)
    end
end

function viterbi(hmm::AbstractHMM, data::Matrix{Float64})
    T, _ = size(data)
    K = size(hmm.A, 1)  # Number of states
    # Step 1: Initialization
    viterbi = zeros(Float64, T, K)
    backpointer = zeros(Int, T, K)
    for i in 1:K
        viterbi[1, i] = log(hmm.πₖ[i]) + loglikelihood(hmm.B[i], data[1, :])
        backpointer[1, i] = 0
    end
    
    # Step 2: Recursion
    for t in 2:T
        for j in 1:K
            max_prob, max_state = -Inf, 0
            for i in 1:K
                prob = viterbi[t-1, i] + log(hmm.A[i, j]) + loglikelihood(hmm.B[j], data[t, :])
                if prob > max_prob
                    max_prob = prob
                    max_state = i
                end
            end
            viterbi[t, j] = max_prob
            backpointer[t, j] = max_state
        end
    end

    # Step 3: Termination
    best_path_prob, best_last_state = findmax(viterbi[T, :])
    best_path = [best_last_state]
    for t in T:-1:2
        push!(best_path, backpointer[t, best_path[end]])
    end
    return reverse(best_path)
end

function sample(hmm::GaussianHMM, n::Int)
    # Number of states
    K = size(hmm.A, 1)
    # Initialize state and observation arrays
    states = Vector{Int}(undef, n)
    observations = Matrix{Float64}(undef, n, hmm.D)
    # Start with a random state
    states[1] = rand(1:K)
    # Generate first observation
    observations[1, :] = rand(MvNormal(hmm.B[states[1]].μ, hmm.B[states[1]].Σ))
    for t in 2:n
        # Transition to a new state
        states[t] = StatsBase.sample(1:K, Weights(hmm.A[states[t-1], :]))
        # Generate observation
        observations[t, :] = rand(MvNormal(hmm.B[states[t]].μ, hmm.B[states[t]].Σ))
    end
    return states, observations
end

"""
This set of functions is for the Baum-Welch algorithm but uses the scaling factors version of the forward-backward algorithm. I'm keeping this here for now in case we want to use it later. 
"""
# function forward(hmm::AbstractHMM, data::AbstractArray)
#     T = size(data, 1)
#     K = size(hmm.A, 1)  # Number of states
#     # Initialize the scaled α-matrix and scaling factors
#     α = zeros(Float64, K, T)
#     c = zeros(Float64, T)
#     # Calculate α₁
#     for k in 1:K
#         α[k, 1] = hmm.πₖ[k] * likelihood(hmm.B[k], data[1, :]) # α₁(k) = πₖ(k) * Bₖ(y₁)
#     end
#     c[1] = 1 / (sum(α[:, 1]) + eps(Float64))
#     α[:, 1] *= c[1]
#     # Now perform the rest of the forward algorithm for t=2 to T
#     for t in 2:T
#         for j in 1:K
#             α[j, t] = 0 d
#             for i in 1:K
#                 α[j, t] += α[i, t-1] * hmm.A[i, j] # αⱼ(t) = ∑ᵢ αᵢ(t-1) * Aᵢⱼ
#             end
#             α[j, t] *= likelihood(hmm.B[j], data[t, :])  # αⱼ(t) *= Bⱼ(yₜ)
#         end
#         c[t] = 1 / (sum(α[:, t]) + eps(Float64)) # Scale the α values
#         α[:, t] *= c[t]
#     end
#     return α, c
# end

# function backward(hmm::AbstractHMM, data::AbstractArray, scaling_factors::Vector{Float64})
#     T = size(data, 1)
#     K = size(hmm.A, 1)  # Number of states
#     # Initialize the scaled β matrix
#     β = zeros(Float64, K, T)
#     # Set last β values.
#     β[:, T] .= 1  # βₖ(T) = 1 What should this be?
#     # Calculate β, starting from T-1 and going backward to 1
#     for t in T-1:-1:1
#         for i in 1:K
#             β[i, t] = 0
#             for j in 1:K
#                 β[i, t] += hmm.A[i, j] * likelihood(hmm.B[j], data[t+1, :]) * β[j, t+1] # βᵢ(t) = ∑ⱼ Aᵢⱼ * Bⱼ(yₜ₊₁) * βⱼ(t₊₁)
#             end
#             β[i, t] *= scaling_factors[t+1] # Scale the β values
#         end
#     end
#     return β
# end

# function calculate_γ(hmm::AbstractHMM, α::Matrix{Float64}, β::Matrix{Float64})
#     T = size(α, 2)
#     γ = α .* β # γₖ(t) = αₖ(t) * βₖ(t)
#     for t in 1:T
#         γ[:, t] /= sum(γ[:, t]) # Normalize the γ values
#     end
#     return γ
# end

# function calculate_ξ(hmm::AbstractHMM, α::Matrix{Float64}, β::Matrix{Float64}, scaling_factors::Vector{Float64}, data::AbstractArray)
#     T = size(data, 1)
#     K = size(hmm.A, 1)
#     ξ = zeros(Float64, K, K, T-1)
#     for t in 1:T-1
#         for i in 1:K
#             for j in 1:K
#                 ξ[i, j, t] = (scaling_factors[t+1]) * α[i, t] * hmm.A[i, j] * likelihood(hmm.B[j], data[t+1, :]) * β[j, t+1] # ξᵢⱼ(t) = αᵢ(t) * Aᵢⱼ * Bⱼ(yₜ₊₁) * βⱼ(t₊₁)
#             end
#         end
#         ξ[:, :, t] /= sum(ξ[:, :, t]) # Normalize the ξ values
#     end
#     return ξ
# end

# function Estep(hmm::AbstractHMM, data::Matrix{Float64})
#     α, c = forward(hmm, data)
#     β = backward(hmm, data, c)
#     γ = calculate_γ(hmm, α, β)
#     ξ = calculate_ξ(hmm, α, β, c, data)
#     return γ, ξ, α, β, c
# end

# function M_step!(hmm::AbstractHMM, γ::Matrix{Float64}, ξ::Array{Float64, 3}, data::Matrix{Float64})
#     K = size(hmm.A, 1)
#     T = size(data, 1)
#     # Update initial state probabilities
#     hmm.πₖ .= γ[:, 1] / sum(γ[:, 1])
#     # Update transition probabilities
#     for i in 1:K
#         for j in 1:K
#             hmm.A[i, j] = sum(ξ[i, j, :]) / sum(γ[i, 1:T-1])
#         end
#     end
#     # Update emission model 
#     for k in 1:K
#         hmm.B[k] = updateEmissionModel!(hmm.B[k], data, γ[k,:])
#     end
# end

# function baumWelch!(hmm::AbstractHMM, data::Matrix{Float64}, max_iters::Int=100, tol::Float64=1e-6)
#     # Initialize log-likelihood
#     ll_prev = -Inf
#     # Initialize progress bar
#     p = Progress(max_iters; dt=1, desc="Computing Baum-Welch...",)
#     for iter in 1:max_iters
#         # Update the progress bar
#         next!(p; showvalues = [(:iteration, iter), (:log_likelihood, ll_prev)])
#         # E-Step
#         γ, ξ, α, β, c = Estep(hmm, data)
#         # Compute and update the log-likelihood
#         log_likelihood = sum(log.((1 ./ c)))
#         println(log_likelihood)
#         if abs(log_likelihood - ll_prev) < tol
#             finish!(p)
#             break
#         else
#             ll_prev = log_likelihood
#         end
#         # M-Step
#         M_step!(hmm, γ, ξ, data)
#     end
# end
