export GaussianHMM, baumWelch!, viterbi, sample

"""
GaussianHMM: A hidden markov model with Gaussian emissions.

Args:
    A::Matrix{Float64}: State Transition Matrix
    B::Vector{GaussianEmission}: Emission Model
    πₖ::Vector{Float64}: Initial State Distribution
    D::Int: Latent State Dimension
"""
mutable struct GaussianHMM{GaussianEmission} <: AbstractHMM
    A::Matrix{Float64}  # State Transition Matrix
    B::Vector{GaussianEmission}       # Emission Model
    πₖ ::Vector{Float64} # Initial State Distribution
    K::Int              # Latent State Dimension
    D::Int              # Dimension of the data
end

# Constructor for GaussianHMM when all params are known, really just for sampling/testing
function GaussianHMM(A::Matrix{Float64}, B::Vector{GaussianEmission}, πₖ::Vector{Float64}, K::Int, D::Int)
    return GaussianHMM{GaussianEmission}(A, B, πₖ, K, D)
end

function GaussianHMM(data::Matrix{Float64}, k_states::Int=2)
    N, D = size(data)
    # Initialize A
    A = rand(k_states, k_states)
    A = A ./ sum(A, dims=2)  # normalize rows to ensure they are valid probabilities
    # Initialize π
    πₖ = rand(k_states)
    πₖ = πₖ ./ sum(πₖ) # normalize to ensure it's a valid probability vector
    # Initialize Emission Model

    # use kmeans_clustering to initialize the emission model
    sample_means, labels = kmeans_clustering(data, k_states)
    sample_covs = [cov(data[labels .== i, :]) for i in 1:k_states]
    B = [GaussianEmission(sample_means[:, i], sample_covs[i]) for i in 1:k_states]
    return GaussianHMM(A, B, πₖ, k_states, D)
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
#             α[j, t] = 0 
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

# function MStep!(hmm::AbstractHMM, γ::Matrix{Float64}, ξ::Array{Float64, 3}, data::Matrix{Float64})
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
#         MStep!(hmm, γ, ξ, data)
#     end
# end
"""
This set of functions is for the Baum-Welch algorithm but uses the log-space version of the forward-backward algorithm. I'm keeping this here for now in case we want to use it later. 
"""
function forward(hmm::AbstractHMM, data::Y) where Y <: AbstractArray
    T = size(data, 1)
    K = size(hmm.A, 1)  # Number of states
    # Initialize an α-matrix 
    α = zeros(Float64, K, T)
    # Calculate α₁
    for k in 1:K
        α[k, 1] = log(hmm.πₖ[k]) + loglikelihood(hmm.B[k], data[1, :])
    end
    # Now perform the rest of the forward algorithm for t=2 to T
    for t in 2:T
        for j in 1:K
            values_to_sum = Float64[]
            for i in 1:K
                push!(values_to_sum, log(hmm.A[i, j]) + α[i, t-1])
            end
            log_sum_alpha_a = logsumexp(values_to_sum)
            α[j, t] = log_sum_alpha_a + loglikelihood(hmm.B[j], data[t, :])
        end
    end
    return α
end

function backward(hmm::AbstractHMM, data::Y) where Y <: AbstractArray
    T = size(data, 1)
    K = size(hmm.A, 1)  # Number of states

    # Initialize a β matrix
    β = zeros(Float64, K, T)

    # Set last β values. In log-space, 0 corresponds to a value of 1 in the original space.
    β[:, T] .= 0  # log(1) = 0

    # Calculate β, starting from T-1 and going backward to 1
    for t in T-1:-1:1
        for i in 1:K
            values_to_sum = Float64[]
            for j in 1:K
                push!(values_to_sum, log(hmm.A[i, j]) + loglikelihood(hmm.B[j], data[t+1, :]) + β[j, t+1])
            end
            β[i, t] = logsumexp(values_to_sum)
        end
    end
    return β
end

function calculate_γ(hmm::AbstractHMM, α::Matrix{Float64}, β::Matrix{Float64})
    T = size(α, 2)
    γ = α .+ β
    for t in 1:T
        max_gamma = maximum(γ[:, t])
        log_sum = max_gamma + log(sum(exp.(γ[:, t] .- max_gamma)))
        γ[:, t] .-= log_sum
    end
    return γ
end

function calculate_ξ(hmm::AbstractHMM, α::Matrix{Float64}, β::Matrix{Float64}, data::AbstractArray)
    T = size(data, 1)
    K = size(hmm.A, 1)
    ξ = zeros(Float64, K, K, T-1)
    for t in 1:T-1
        # Array to store the unnormalized ξ values
        log_ξ_unnormalized = zeros(Float64, K, K)
        for i in 1:K
            for j in 1:K
                log_ξ_unnormalized[i, j] = α[i, t] + log(hmm.A[i, j]) + loglikelihood(hmm.B[j], data[t+1, :]) + β[j, t+1]
            end
        end
        # Normalize the ξ values using log-sum-exp operation
        max_ξ = maximum(log_ξ_unnormalized)
        denominator = max_ξ + log(sum(exp.(log_ξ_unnormalized .- max_ξ)))
        ξ[:, :, t] .= log_ξ_unnormalized .- denominator
    end
    return ξ
end

function EStep(hmm::AbstractHMM, data::Matrix{Float64})
    α = forward(hmm, data)
    β = backward(hmm, data)
    γ = calculate_γ(hmm, α, β)
    ξ = calculate_ξ(hmm, α, β, data)
    return γ, ξ, α, β
end

function MStep!(hmm::AbstractHMM, γ::Matrix{Float64}, ξ::Array{Float64, 3}, data::Matrix{Float64})
    K = size(hmm.A, 1)
    T = size(data, 1)
    # Update initial state probabilities
    hmm.πₖ .= exp.(γ[:, 1])
    # Update transition probabilities
    for i in 1:K
        for j in 1:K
            hmm.A[i,j] = exp(log(sum(exp.(ξ[i,j,:]))) - log(sum(exp.(γ[i,1:T-1]))))
        end
    end
    # Update emission model 
    for k in 1:K
        hmm.B[k] = updateEmissionModel!(hmm.B[k], data, exp.(γ[k,:]))
    end
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
        γ, ξ, α, β = EStep(hmm, data)
        # Compute and update the log-likelihood
        log_likelihood_current = logsumexp(α[:, T])
        println(log_likelihood_current)
        if abs(log_likelihood_current - log_likelihood) < tol
            finish!(p)
            break
        else
            log_likelihood = log_likelihood_current
        end
        # M-Step
        MStep!(hmm, γ, ξ, data)
    end
end

function viterbi(hmm::AbstractHMM, data::Matrix{Float64})
    T, _ = size(data)
    K = size(hmm.A, 1)  # Number of states
    # Step 1: Initialization
    viterbi = zeros(Float64, K, T)
    backpointer = zeros(Int, K, T)
    for i in 1:K
        viterbi[i, 1] = log(hmm.πₖ[i]) + loglikelihood(hmm.B[i], data[1, :])
        backpointer[i, 1] = 0
    end
    # Step 2: Recursion
    for t in 2:T
        for j in 1:K
            max_prob, max_state = -Inf, 0
            for i in 1:K
                prob = viterbi[i, t-1] + log(hmm.A[i,j]) + loglikelihood(hmm.B[j], data[t, :])
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
PoissonHMM: A hidden markov model with Poisson emissions.

Args:
    A::Matrix{Float64}: State Transition Matrix
    B::Vector{PoissonEmission}: Emission Model
    πₖ::Vector{Float64}: Initial State Distribution
    D::Int: Latent State Dimension
"""

mutable struct PoissonHMM{PoissonEmission} <: AbstractHMM
    A::Matrix{Float64}  # State Transition Matrix
    B::Vector{PoissonEmission} # Emission Model
    πₖ ::Vector{Float64} # Initial State Distribution
    K::Int # Latent State Dimension
end

function PoissonHMM(data::Matrix{Float64}, K::Int)
    T, D = size(data)
    # Initialize A
    A = rand(K, K)
    A = A ./ sum(A, dims=2)  # normalize rows to ensure they are valid probabilities
    # Initialize π
    πₖ = rand(K)
    πₖ = πₖ ./ sum(πₖ) # normalize to ensure it's a valid probability vector
    # Initialize Emission Model
    #TODO: Finish later
end

