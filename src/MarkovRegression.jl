export SwitchingGaussianRegression, SwitchingBernoulliRegression, fit!, viterbi, log_likelihood

abstract type hmmglm <: AbstractHMM end
"""
    SwitchingGaussianRegression

Struct representing a Gaussian hmm-glm model. This model is specifically a Hidden Markov Model with Gaussian Regression emissions. One can think of this model
as a time-dependent mixture of Gaussian regression models. This is similar to how a vanilla HMM is a time-dependent mixture of Gaussian distributions. Thus,
at each time point we can assess the most likely state and the most likely regression model given the data.

Args:
    A::Matrix{T}: Transition matrix
    B::Vector{RegressionEmissions}: Vector of Gaussian Regression Models
    πₖ::Vector{T}: initial state distribution
    K::Int: number of states
"""
mutable struct SwitchingGaussianRegression{T <: Real} <: hmmglm
    A::Matrix{T} # transition matrix
    B::Vector{RegressionEmissions} # Vector of Gaussian Regression Models
    πₖ::Vector{T} # initial state distribution
    K::Int # number of states
end

function SwitchingGaussianRegression(; A::Matrix{Float64}=Matrix{Float64}(undef, 0, 0), B::Vector{RegressionEmissions}=Vector{RegressionEmissions}(), πₖ::Vector{Float64}=Vector{Float64}(), K::Int)
    # if A matrix is not passed, initialize using Dirichlet 
    isempty(A) ? A = initialize_transition_matrix(K) : nothing
    # if B vector is not passed, initialize using Gaussian Regression
    isempty(B) ? B = [RegressionEmissions(GaussianRegression()) for k in 1:K] : nothing
    # if πₖ vector is not passed, initialize using Dirichlet
    isempty(πₖ) ? πₖ = initialize_state_distribution(K) : nothing
    # return model
    return SwitchingGaussianRegression(A, B, πₖ, K)
end

function update_regression!(model::hmmglm, X::Matrix{Float64}, y::Vector{Float64}, w::Matrix{Float64}=ones(length(y), model.K))
   # update regression models 
    for k in 1:model.K
        update_emissions_model!(model.B[k], X, y, w[:, k])
    end
end

function initialize_regression!(model::hmmglm, X::Matrix{Float64}, y::Vector{Float64})
    # first fit the regression models to all of the data unweighted
    update_regression!(model, X, y)
    # add white noise to the beta coefficients
    for k in 1:model.K
        model.B[k].regression.β += randn(length(model.B[k].regression.β))
    end
end

function forward(hmm::hmmglm, X::Matrix{Float64}, y::Vector{Float64})
    T = length(y)
    K = size(hmm.A, 1)  # Number of states
    # Initialize an α-matrix 
    α = zeros(Float64, T, K)
    # Calculate α₁
    for k in 1:K
        α[1, k] = log(hmm.πₖ[k]) + loglikelihood(hmm.B[k], X[1, :], y[1])
    end
    # Now perform the rest of the forward algorithm for t=2 to T
    for t in 2:T
        for k in 1:K
            values_to_sum = Float64[]
            for i in 1:K
                push!(values_to_sum, log(hmm.A[i, k]) + α[t-1, i])
            end
            log_sum_alpha_a = logsumexp(values_to_sum)
            α[t, k] = log_sum_alpha_a + loglikelihood(hmm.B[k], X[t, :], y[t])
        end
    end
    return α
end

function backward(hmm::hmmglm,  X::Matrix{Float64}, y::Vector{Float64})
    T = length(y)
    K = size(hmm.A, 1)  # Number of states

    # Initialize a β matrix
    β = zeros(Float64, T, K)

    # Set last β values. In log-space, 0 corresponds to a value of 1 in the original space.
    β[T, :] .= 0  # log(1) = 0

    # Calculate β, starting from T-1 and going backward to 1
    for t in T-1:-1:1
        for i in 1:K
            values_to_sum = Float64[]
            for j in 1:K
                push!(values_to_sum, log(hmm.A[i, j]) + loglikelihood(hmm.B[j], X[t+1, :], y[t+1]) + β[t+1, j])
            end
            β[t, i] = logsumexp(values_to_sum)
        end
    end
    return β
end

function calculate_ξ(hmm::hmmglm, α::Matrix{Float64}, β::Matrix{Float64}, X::Matrix{Float64}, y::Vector{Float64})
    T = length(y)
    K = size(hmm.A, 1)
    ξ = zeros(Float64, T-1, K, K)
    for t in 1:T-1
        # Array to store the unnormalized ξ values
        log_ξ_unnormalized = zeros(Float64, K, K)
        for i in 1:K
            for j in 1:K
                log_ξ_unnormalized[i, j] = α[t, i] + log(hmm.A[i, j]) + loglikelihood(hmm.B[j], X[t+1, :], y[t+1]) + β[t+1, j]
            end
        end
        # Normalize the ξ values using log-sum-exp operation
        max_ξ = maximum(log_ξ_unnormalized)
        denominator = max_ξ + log(sum(exp.(log_ξ_unnormalized .- max_ξ)))
        ξ[t, :, :] .= log_ξ_unnormalized .- denominator
    end
    return ξ
end

function E_step(model::hmmglm, X::Matrix{Float64}, y::Vector{Float64})
    # run forward-backward algorithm
    α = forward(model, X, y)
    β = backward(model, X, y)
    γ = calculate_γ(model, α, β)
    ξ = calculate_ξ(model, α, β, X, y)
    return γ, ξ, α, β
end

function M_step!(model::hmmglm, γ::Matrix{Float64}, ξ::Array{Float64, 3}, X::Matrix{Float64}, y::Vector{Float64})
    # update initial state distribution
    update_initial_state_distribution!(model, γ)   
    # update transition matrix
    update_transition_matrix!(model, γ, ξ)
    # update regression models
    update_regression!(model, X, y, exp.(γ)) 
end

function fit!(model::hmmglm, X::Matrix{Float64}, y::Vector{Float64}, max_iter::Int=100, tol::Float64=1e-6, initialize::Bool=true)
    # initialize regression models
    if initialize
        initialize_regression!(model, X, y)
    end
    # ll variable
    lls = [-Inf]
    # Initialize first log-likelihood
    prev_ll = -Inf
    # run EM algorithm
    for i in 1:max_iter
        # E-step
        γ, ξ, α, _ = E_step(model, X, y)
        # Log-likelihood
        ll = logsumexp(α[end, :])
        push!(lls, ll)
        println("Log-Likelihood at iter $i: $ll")
        # M-step
        M_step!(model, γ, ξ, X, y)
        # check for convergence
        if i > 1
            if abs(ll - prev_ll) < tol
                return lls
            end
        end
        prev_ll = ll 
    end
    return lls
end

mutable struct SwitchingBernoulliRegression <: hmmglm
    A::Matrix{Float64} # transition matrix
    B::Vector{RegressionEmissions} # Vector of Bernoulli Regression Models
    πₖ::Vector{Float64} # initial state distribution
    K::Int # number of states
end

function SwitchingBernoulliRegression(; A::Matrix{Float64}=Matrix{Float64}(undef, 0, 0), B::Vector{RegressionEmissions}=Vector{RegressionEmissions}(), πₖ::Vector{Float64}=Vector{Float64}(), K::Int)
    # if A matrix is not passed, initialize using Dirichlet 
    isempty(A) ? A = initialize_transition_matrix(K) : nothing
    # if B vector is not passed, initialize using Gaussian Regression
    isempty(B) ? B = [RegressionEmissions(BernoulliRegression()) for k in 1:K] : nothing
    # if πₖ vector is not passed, initialize using Dirichlet
    isempty(πₖ) ? πₖ = initialize_state_distribution(K) : nothing
    # return model
    return SwitchingBernoulliRegression(A, B, πₖ, K)
end

function viterbi(hmm::hmmglm, X::Matrix{Float64}, y::Vector{Float64})
    T = length(y)
    K = size(hmm.A, 1)  # Number of states

    # Step 1: Initialization
    viterbi = zeros(Float64, T, K)
    backpointer = zeros(Int, T, K)
    for i in 1:K
        viterbi[1, i] = log(hmm.πₖ[i]) + loglikelihood(hmm.B[i], X[1, :], y[1])
        backpointer[1, i] = 0
    end

    # Step 2: Recursion
    for t in 2:T
        for j in 1:K
            max_prob, max_state = -Inf, 0
            for i in 1:K
                prob = viterbi[t-1, i] + log(hmm.A[i, j]) + loglikelihood(hmm.B[j], X[t, :], y[t])
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