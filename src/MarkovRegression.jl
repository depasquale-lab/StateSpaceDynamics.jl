export SwitchingGaussianRegression, SwitchingBernoulliRegression, SwitchingPoissonRegression, fit!, viterbi, log_likelihood

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
    λ::Float64: regularization parameter for the regression models
"""
mutable struct SwitchingGaussianRegression{T <: Real} <: hmmglm
    A::Matrix{T} # transition matrix
    B::Vector{RegressionEmissions} # Vector of Gaussian Regression Models
    πₖ::Vector{T} # initial state distribution
    K::Int # number of states
    λ::Float64 # regularization parameter
end

function SwitchingGaussianRegression(; A::Matrix{Float64}=Matrix{Float64}(undef, 0, 0), B::Vector{RegressionEmissions}=Vector{RegressionEmissions}(), πₖ::Vector{Float64}=Vector{Float64}(), K::Int, λ::Float64=0.0)
    # if A matrix is not passed, initialize using Dirichlet 
    isempty(A) ? A = initialize_transition_matrix(K) : nothing
    # if B vector is not passed, initialize using Gaussian Regression
    isempty(B) ? B = [RegressionEmissions(GaussianRegression(;λ=λ)) for k in 1:K] : nothing
    # if πₖ vector is not passed, initialize using Dirichlet
    isempty(πₖ) ? πₖ = initialize_state_distribution(K) : nothing
    # return model
    return SwitchingGaussianRegression(A, B, πₖ, K, λ)
end

"""
    SwitchingBernoulliRegression

Struct representing a Bernoulli hmm-glm model. This model is specifically a Hidden Markov Model with Bernoulli Regression emissions. One can think of this model
as a time-dependent mixture of Bernoulli regression models. This is similar to how a vanilla HMM is a time-dependent mixture of Bernoulli distributions. Thus,
at each time point we can assess the most likely state and the most likely regression model given the data.

Args:
    A::Matrix{T}: Transition matrix
    B::Vector{RegressionEmissions}: Vector of Bernoulli Regression Models
    πₖ::Vector{T}: initial state distribution
    K::Int: number of states
"""
mutable struct SwitchingBernoulliRegression <: hmmglm
    A::Matrix{Float64} # transition matrix
    B::Vector{RegressionEmissions} # Vector of Bernoulli Regression Models
    πₖ::Vector{Float64} # initial state distribution
    K::Int # number of states
    λ::Float64 # regularization parameter
end

function SwitchingBernoulliRegression(; A::Matrix{Float64}=Matrix{Float64}(undef, 0, 0), B::Vector{RegressionEmissions}=Vector{RegressionEmissions}(), πₖ::Vector{Float64}=Vector{Float64}(), K::Int, λ::Float64=0.0)
    # if A matrix is not passed, initialize using Dirichlet 
    isempty(A) ? A = initialize_transition_matrix(K) : nothing
    # if B vector is not passed, initialize using Gaussian Regression
    isempty(B) ? B = [RegressionEmissions(BernoulliRegression(;λ=λ)) for k in 1:K] : nothing
    # if πₖ vector is not passed, initialize using Dirichlet
    isempty(πₖ) ? πₖ = initialize_state_distribution(K) : nothing
    # return model
    return SwitchingBernoulliRegression(A, B, πₖ, K, λ)
end

"""
    SwitchingPoissonRegression

Struct representing a Poisson hmm-glm model. This model is specifically a Hidden Markov Model with Poisson Regression emissions. One can think of this model
as a time-dependent mixture of Poisson regression models. This is similar to how a vanilla HMM is a time-dependent mixture of Poisson distributions. Thus,
at each time point we can assess the most likely state and the most likely regression model given the data.

Args:
    A::Matrix{T}: Transition matrix
    B::Vector{RegressionEmissions}: Vector of Poisson Regression Models
    πₖ::Vector{T}: initial state distribution
    K::Int: number of states
"""
mutable struct SwitchingPoissonRegression <: hmmglm
    A::Matrix{Float64} # transition matrix
    B::Vector{RegressionEmissions} # Vector of Poisson Regression Models
    πₖ::Vector{Float64} # initial state distribution
    K::Int # number of states
    λ::Float64 # regularization parameter
end

function SwitchingPoissonRegression(; A::Matrix{Float64}=Matrix{Float64}(undef, 0, 0), B::Vector{RegressionEmissions}=Vector{RegressionEmissions}(), πₖ::Vector{Float64}=Vector{Float64}(), K::Int, λ::Float64=0.0)
    # if A matrix is not passed, initialize using Dirichlet 
    isempty(A) ? A = initialize_transition_matrix(K) : nothing
    # if B vector is not passed, initialize using Gaussian Regression
    isempty(B) ? B = [RegressionEmissions(PoissonRegression(; λ=λ)) for k in 1:K] : nothing
    # if πₖ vector is not passed, initialize using Dirichlet
    isempty(πₖ) ? πₖ = initialize_state_distribution(K) : nothing
    # return model
    return SwitchingPoissonRegression(A, B, πₖ, K, λ)
end

function update_regression!(model::hmmglm, X::Matrix{Float64}, y::Vector{Float64}, w::Matrix{Float64}=ones(length(y), model.K))
   # update regression models 
    @threads for k in 1:model.K
        update_emissions_model!(model.B[k], X, y, w[:, k])
    end
end

function update_regression!(model::hmmglm, X::Vector{Matrix{Float64}}, y::Vector{Vector{Float64}}, w::Vector{Matrix{Float64}})
    num_trials = length(X)
    p = size(X[1], 2)
    if model.B[1].regression.include_intercept
        p+=1
    end

    state1_models = Vector{RegressionEmissions}(undef, num_trials)
    state2_models = Vector{RegressionEmissions}(undef, num_trials)

    state1_weights = [w[i][:, 1] for i in eachindex(w)]
    state2_weights = [w[i][:, 2] for i in eachindex(w)]

    # Initialize Gaussian emission models for each trial + state combo
    for trial in 1:num_trials
        state1_models[trial] = RegressionEmissions(GaussianRegression(model.B[1].regression.β, model.B[1].regression.σ², true, model.λ))
        state2_models[trial] = RegressionEmissions(GaussianRegression(model.B[2].regression.β, model.B[2].regression.σ², true, model.λ))
    end

    # for each trial + state combo use built in update emission function
    for trial in 1:num_trials
        update_emissions_model!(state1_models[trial], X[trial], y[trial], state1_weights[trial])
        update_emissions_model!(state2_models[trial], X[trial], y[trial], state2_weights[trial])
    end

    state1_β = zeros(p)
    state1_σ² = 0.0

    state2_β = zeros(p)
    state2_σ² = 0.0

    # average across trials
    for m in state1_models
        state1_β += m.regression.β
        state1_σ² += m.regression.σ²
    end

    for m in state2_models
        state2_β += m.regression.β
        state2_σ² += m.regression.σ²
    end

    # update! model parameters
    model.B[1].regression.β = state1_β ./ num_trials
    # model.B[1].regression.σ² = state1_σ² / num_trials
    model.B[1].regression.σ² = 1.0

    model.B[2].regression.β = state2_β ./ num_trials
    # model.B[2].regression.σ² = state2_σ² / num_trials
    model.B[2].regression.σ² = 1.0

    if isnan(model.B[2].regression.σ²)
        println("nan in variance")
    end
    
    if isnan(model.B[1].regression.σ²)
        println("nan in variance")
    end
    


end

function initialize_regression!(model::hmmglm, X::Matrix{Float64}, y::Vector{Float64})
    # first fit the regression models to all of the data unweighted
    update_regression!(model, X, y)
    # add white noise to the beta coefficients
    @threads for k in 1:model.K
        model.B[k].regression.β += randn(length(model.B[k].regression.β))
    end
end

function random_initialization!(model::hmmglm, X::Vector{Matrix{Float64}})
    # Find number of parameters
    p = size(X[1], 2)
    if model.B[1].regression.include_intercept
        p+=1
    end
    # Randomly initialize regression models
    @threads for k in 1:model.K
        model.B[k].regression.β = randn(p)
        model.B[k].regression.σ² = 1.0
    end
end

function forward(hmm::hmmglm, X::Matrix{Float64}, y::Vector{Float64})
    T = length(y)
    K = size(hmm.A, 1)  # Number of states
    # Initialize an α-matrix 
    α = zeros(Float64, T, K)
    # Calculate α₁
    @threads for k in 1:K
        α[1, k] = log(hmm.πₖ[k]) + loglikelihood(hmm.B[k], X[1, :], y[1])
    end
    # Now perform the rest of the forward algorithm for t=2 to T
    for t in 2:T
        @threads for k in 1:K
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
        @threads for i in 1:K
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
        @threads for i in 1:K
            for j in 1:K
                log_ξ_unnormalized[i, j] = α[t, i] + log(hmm.A[i, j]) + loglikelihood(hmm.B[j], X[t+1, :], y[t+1]) + β[t+1, j]
            end
        end
        # Normalize the ξ values using log-sum-exp operation
        ξ[t, :, :] .= log_ξ_unnormalized .- logsumexp(log_ξ_unnormalized)
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

function M_step!(model::hmmglm, γ::Vector{Matrix{Float64}}, ξ::Vector{Array{Float64, 3}}, X::Vector{Matrix{Float64}}, y::Vector{Vector{Float64}})
    # Update initial state distribution
    update_initial_state_distribution!(model, γ)
    # Update transition matrix
    update_transition_matrix!(model, γ, ξ)
    # Update regression models
    γ_exp = [exp.(γ_trial) for γ_trial in γ]
    update_regression!(model, X, y, γ_exp)
end

function fit!(model::hmmglm, X::Matrix{Float64}, y::Union{Vector{T}, BitVector}, max_iter::Int=100, tol::Float64=1e-6, initialize::Bool=true) where T<: Real
    # convert y to Float64
    y = convert(Vector{Float64}, y)
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
        if isnan(ll)
            println("nan in ll")
        end
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

function fit!(model::hmmglm, X::Vector{Matrix{Float64}}, y::Vector{Vector{Float64}}, max_iter::Int=100, tol::Float64=1e-6, initialize::Bool=true)
    # Randomly Initialize the regression models
    random_initialization!(model, X)
    # Initialize log likelihood
    lls = [-Inf]
    prev_ll = -Inf

    # Storage for parameter tracking
    A_stor = Vector{Matrix{Float64}}()
    π_stor = Vector{Vector{Float64}}()
    β1_stor = Vector{Vector{Float64}}()
    β2_stor = Vector{Vector{Float64}}()
    σ1_stor = Vector{Float64}()
    σ2_stor = Vector{Float64}()

    # Expectation-Maximization
    for i in 1:max_iter
        # E-Step
        output = E_step.(Ref(model), X, y)
        γ, ξ, α = map(x-> x[1], output), map(x-> x[2], output), map(x-> x[3], output)
        # Log-likelihood
        ll = sum(map(α -> SSM.logsumexp(α[end, :]), α))
        push!(lls, ll)
        println("Log-Lieklihood at iter $i: $ll")
        # M-Step
        M_step!(model, γ, ξ, X, y)

        # Track parameters
        push!(A_stor, deepcopy(model.A))
        push!(π_stor, deepcopy(model.πₖ))
        push!(β1_stor, deepcopy(model.B[1].regression.β))
        push!(β2_stor, deepcopy(model.B[2].regression.β))
        push!(σ1_stor, deepcopy(model.B[1].regression.σ²))
        push!(σ2_stor, deepcopy(model.B[2].regression.σ²))

        # check for convergence
        if i > 1
            if abs(ll - prev_ll) < tol
                return lls, A_stor, π_stor, β1_stor, β2_stor, σ1_stor, σ2_stor
            end
        end
        prev_ll = ll 
    end
    return lls, A_stor, π_stor, β1_stor, β2_stor, σ1_stor, σ2_stor
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
