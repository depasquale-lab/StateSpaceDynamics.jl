export SwitchingGaussianRegression, SwitchingBernoulliRegression, SwitchingPoissonRegression, fit!, viterbi, log_likelihood, hmmglm

abstract type hmmglm <: AbstractHMM end
"""
    SwitchingGaussianRegression

Struct representing a Gaussian hmm-glm model. This model is specifically a Hidden Markov Model with Gaussian Regression emissions. One can think of this model
as a time-dependent mixture of Gaussian regression models. This is similar to how a vanilla HMM is a time-dependent mixture of Gaussian distributions. Thus,
at each time point we can assess the most likely state and the most likely regression model given the data.

# Arguments
- `A::Matrix{T}`: Transition matrix
- `B::Vector{RegressionEmissions}`: Vector of Gaussian Regression Models
- `πₖ::Vector{T}`: initial state distribution
- `K::Int`: number of states
- `λ::Float64`: regularization parameter for the regression models
- `input_dim::Int`: number of features
- `num_targets::Int`: number of targets
"""
mutable struct SwitchingGaussianRegression{T <: Real} <: hmmglm
    A::Matrix{T} # transition matrix
    B::Vector{RegressionEmissions} # Vector of Gaussian Regression Models
    πₖ::Vector{T} # initial state distribution
    K::Int # number of states
    λ::Float64 # regularization parameter
    input_dim::Int # number of features
    num_targets::Int # number of targets
end

"""
    SwitchingGaussianRegression(; <keyword arguments>)

Constructor for Switching Gaussian Regression Model.

# Arguments
- `input_dim::Int`: Number of features.
- `num_targets::Int`: Number of targets.
- `A::Matrix{<:Real}`: Transition matrix.
- `B::Vector{RegressionEmissions}`: Vector of Gaussian Regression Models.
- `πₖ::Vector{Float64}`: Initial state distribution.
- `K::Int`: Number of states.
- `λ::Float64`: Regularization parameter for the regression models.

# Examples
```julia
model = SwitchingGaussianRegression(input_dim=2, num_targets=1, K=2)
```
"""
function SwitchingGaussianRegression(; input_dim::Int, num_targets::Int, A::Matrix{<:Real}=Matrix{Float64}(undef, 0, 0), B::Vector{RegressionEmissions}=Vector{RegressionEmissions}(), πₖ::Vector{Float64}=Vector{Float64}(), K::Int, λ::Float64=0.0)
    # if A matrix is not passed, initialize using Dirichlet 
    isempty(A) ? A = initialize_transition_matrix(K) : nothing
    # if B vector is not passed, initialize using Gaussian Regression
    isempty(B) ? B = [RegressionEmissions(GaussianRegression(input_dim=input_dim, num_targets=num_targets, λ=λ)) for k in 1:K] : nothing
    # if πₖ vector is not passed, initialize using Dirichlet
    isempty(πₖ) ? πₖ = initialize_state_distribution(K) : nothing
    # return model
    return SwitchingGaussianRegression(A, B, πₖ, K, λ, input_dim, num_targets)
end

"""
    sample(model::SwitchingGaussianRegression, X::Matrix{<:Real})

Sample from the model. 

# Arguments
- `model::SwitchingGaussianRegression`: Switching gaussian regression model.
- `X::Matrix{<:Real}`: Matrix of features.

# Returns
- `y::Matrix{<:Real}`: Matrix of samples.
- `z::Matrix{<:Real}`: Matrix of states. Each row is a one-hot encoding of the state at that time point.
"""
function sample(model::SwitchingGaussianRegression, X::Matrix{<:Real})
    # sample from the model
    y = zeros(Float64, size(X, 1), model.num_targets)
    z = zeros(Float64, size(X, 1), model.K)
    # sample initial state
    state = rand(Categorical(model.πₖ))
    # sample from the model
    for t in 1:size(X, 1)
        # sample from the regression model
        x_data = reshape(X[t, :], 1, :)
        y[t, :] = sample(model.B[state].regression, x_data)[1, :]
        z[t, state] = 1

        # sample next state
        state = rand(Categorical(model.A[state, :]))
    end
    return y, z
end

"""
    SwitchingBernoulliRegression

Struct representing a Bernoulli hmm-glm model. This model is specifically a Hidden Markov Model with Bernoulli Regression emissions. One can think of this model
as a time-dependent mixture of Bernoulli regression models. This is similar to how a vanilla HMM is a time-dependent mixture of Bernoulli distributions. Thus,
at each time point we can assess the most likely state and the most likely regression model given the data.

# Arguments
- `A::Matrix{T}`: Transition matrix.
- `B::Vector{RegressionEmissions}`: Vector of Bernoulli Regression Models.
- `πₖ::Vector{T}`: Initial state distribution.
- `K::Int`: Number of states.
- `λ::Float64`: Regularization parameter for the regression models.
"""
mutable struct SwitchingBernoulliRegression <: hmmglm
    A::Matrix{<:Real} # transition matrix
    B::Vector{RegressionEmissions} # Vector of Bernoulli Regression Models
    πₖ::Vector{Float64} # initial state distribution
    K::Int # number of states
    λ::Float64 # regularization parameter
end

"""
    SwitchingBernoulliRegression(; <keyword arguments>)

Constructor for Switching Bernoulli Regression Model.

# Arguments
- `A::Matrix{<:Real}`: Transition matrix.
- `B::Vector{RegressionEmissions}`: Vector of Bernoulli Regression Models.
- `πₖ::Vector{Float64}`: Initial state distribution.
- `K::Int`: Number of states.
- `λ::Float64`: Regularization parameter for the regression models.

# Examples
```julia
model = SwitchingBernoulliRegression(K=2)
```
"""
function SwitchingBernoulliRegression(; A::Matrix{<:Real}=Matrix{Float64}(undef, 0, 0), B::Vector{RegressionEmissions}=Vector{RegressionEmissions}(), πₖ::Vector{Float64}=Vector{Float64}(), K::Int, λ::Float64=0.0)
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

# Arguments
- `A::Matrix{T}`: Transition matrix.
- `B::Vector{RegressionEmissions}`: Vector of Poisson Regression Models.
- `πₖ::Vector{T}`: Initial state distribution.
- `K::Int`: Number of states.
- `λ::Float64`: Regularization parameter for the regression models.
"""
mutable struct SwitchingPoissonRegression <: hmmglm
    A::Matrix{<:Real} # transition matrix
    B::Vector{RegressionEmissions} # Vector of Poisson Regression Models
    πₖ::Vector{Float64} # initial state distribution
    K::Int # number of states
    λ::Float64 # regularization parameter
end

"""
    SwitchingPoissonRegression(; <keyword arguments>)

Constructor for Switching Poisson Regression Model.

# Arguments
- `A::Matrix{<:Real}`: Transition matrix.
- `B::Vector{RegressionEmissions}`: Vector of Poisson Regression Models.
- `πₖ::Vector{Float64}`: Initial state distribution.
- `K::Int`: Number of states.
- `λ::Float64`: Regularization parameter for the regression models.

# Examples
```julia
model = SwitchingPoissonRegression(K=2)
```
"""
function SwitchingPoissonRegression(; A::Matrix{<:Real}=Matrix{Float64}(undef, 0, 0), B::Vector{RegressionEmissions}=Vector{RegressionEmissions}(), πₖ::Vector{Float64}=Vector{Float64}(), K::Int, λ::Float64=0.0)
    # if A matrix is not passed, initialize using Dirichlet 
    isempty(A) ? A = initialize_transition_matrix(K) : nothing
    # if B vector is not passed, initialize using Gaussian Regression
    isempty(B) ? B = [RegressionEmissions(PoissonRegression(; λ=λ)) for k in 1:K] : nothing
    # if πₖ vector is not passed, initialize using Dirichlet
    isempty(πₖ) ? πₖ = initialize_state_distribution(K) : nothing
    # return model
    return SwitchingPoissonRegression(A, B, πₖ, K, λ)
end

function update_regression!(model::hmmglm, X::Matrix{<:Real}, y::Union{Vector{Float64}, Matrix{<:Real}}, w::Matrix{<:Real}=ones(size(y, 1), model.K))
   # update regression models 

    @threads for k in 1:model.K
        update_emissions_model!(model.B[k], X, y, w[:, k])
    end

end

function initialize_regression!(model::hmmglm, X::Matrix{<:Real}, y::Union{Vector{Float64}, Matrix{<:Real}})
    # first fit the regression models to all of the data unweighted
    update_regression!(model, X, y)

    # add white noise to the beta coefficients
    @threads for k in 1:model.K
        model.B[k].regression.β += randn(size(model.B[k].regression.β))
    end
end

function forward(hmm::hmmglm, X::Matrix{<:Real}, y::Vector{Float64})
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

function forward(hmm::hmmglm, X::Matrix{<:Real}, y::Matrix{<:Real})
    T = size(y, 1)
    K = size(hmm.A, 1)  # Number of states
    # Initialize an α-matrix 
    α = zeros(Float64, T, K)
    # Calculate α₁
    @threads for k in 1:K
        α[1, k] = log(hmm.πₖ[k]) + loglikelihood(hmm.B[k], row_matrix(X[1, :]), row_matrix(y[1, :]))
    end
    # Now perform the rest of the forward algorithm for t=2 to T
    for t in 2:T
        @threads for k in 1:K
            values_to_sum = Float64[]
            for i in 1:K
                push!(values_to_sum, log(hmm.A[i, k]) + α[t-1, i])
            end
            log_sum_alpha_a = logsumexp(values_to_sum)
            α[t, k] = log_sum_alpha_a + loglikelihood(hmm.B[k], row_matrix(X[t, :]), row_matrix(y[t, :]))
        end
    end
    return α
end

function backward(hmm::hmmglm,  X::Matrix{<:Real}, y::Vector{Float64})
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

function backward(hmm::hmmglm,  X::Matrix{<:Real}, y::Matrix{<:Real})
    T = size(y, 1)
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
                push!(values_to_sum, log(hmm.A[i, j]) + loglikelihood(hmm.B[j], row_matrix(X[t+1, :]), row_matrix(y[t+1, :])) + β[t+1, j])
            end
            β[t, i] = logsumexp(values_to_sum)
        end
    end
    return β
end

function calculate_ξ(hmm::hmmglm, α::Matrix{<:Real}, β::Matrix{<:Real}, X::Matrix{<:Real}, y::Vector{Float64})
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

function calculate_ξ(hmm::hmmglm, α::Matrix{<:Real}, β::Matrix{<:Real}, X::Matrix{<:Real}, y::Matrix{<:Real})
    T = size(y, 1)
    K = size(hmm.A, 1)
    ξ = zeros(Float64, T-1, K, K)
    for t in 1:T-1
        # Array to store the unnormalized ξ values
        log_ξ_unnormalized = zeros(Float64, K, K)
        @threads for i in 1:K
            for j in 1:K
                log_ξ_unnormalized[i, j] = α[t, i] + log(hmm.A[i, j]) + loglikelihood(hmm.B[j], row_matrix(X[t+1, :]), row_matrix(y[t+1, :])) + β[t+1, j]
            end
        end
        # Normalize the ξ values using log-sum-exp operation
        ξ[t, :, :] .= log_ξ_unnormalized .- logsumexp(log_ξ_unnormalized)
    end
    return ξ
end

function E_step(model::hmmglm, X::Matrix{<:Real}, y::Union{Vector{Float64}, Matrix{<:Real}})
    # run forward-backward algorithm
    α = forward(model, X, y)
    β = backward(model, X, y)
    γ = calculate_γ(model, α, β)
    ξ = calculate_ξ(model, α, β, X, y)
    return γ, ξ, α, β
end

function M_step!(model::hmmglm, γ::Matrix{<:Real}, ξ::Array{Float64, 3}, X::Matrix{<:Real}, y::Union{Vector{Float64}, Matrix{<:Real}})
    # update initial state distribution
    update_initial_state_distribution!(model, γ)   
    # update transition matrix
    update_transition_matrix!(model, γ, ξ)
    # update regression models
    update_regression!(model, X, y, exp.(γ)) 
end

"""
    fit!(model::hmmglm, X::Matrix{<:Real}, y::Union{Vector{T}, BitVector, Matrix{<:Real}}, max_iter::Int=100, tol::Float64=1e-6, initialize::Bool=true) where T<: Real

Fits a Switching Regression Model (hmmglm) using the EM algorithm.

# Arguments
- `model::hmmglm`: Markov Regression model.
- `X::Matrix{<:Real}`: Matrix of features. Each row is a feature vector. First row is timestep 1.
- `y::Union{Vector{T}, BitVector, Matrix{<:Real}}`: Vector of targets or matrix of targets. For matrix y: Each row is a target vector. First row is timestep 1.
- `max_iter::Int=100`: Maximum number of iterations.
- `tol::Float64=1e-6`: Tolerance for convergence.
- `initialize=True`: Whether to initialize the regression models.

# Returns
- `lls::Vector{Float64}`: Vector of log-likelihoods at each iteration.

# Examples
```julia
X = randn(100, 2)
y = randn(100)
model = SwitchingGaussianRegression(input_dim=2, num_targets=1, K=2)
lls = fit!(model, X, y)
```
"""
function fit!(model::hmmglm, X::Matrix{<:Real}, y::Union{Vector{T}, BitVector, Matrix{<:Real}}, max_iter::Int=100, tol::Float64=1e-6, initialize::Bool=true) where T<: Real
    # convert y to Float64
    if typeof(y) == BitVector
        y = convert(Vector{Float64}, y)
    end
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

"""
    viterbi(hmm::hmmglm, X::Matrix{<:Real}, y::Vector{Float64})

Viterbi algorithm for Switching Regression models.

# Arguments
- `hmm::hmmglm`: Switching Regression model.
- `X::Matrix{<:Real}`: Matrix of features. Each row is a feature vector.
- `y::Vector{Float64}`: Vector of targets.

# Returns
- `best_path::Vector{Int}`: Vector of most likely states.

# Examples
```julia
X = randn(100, 2)
y = randn(100)
model = SwitchingGaussianRegression(input_dim=2, num_targets=1, K=2)
lls = fit!(model, X, y)
best_path = viterbi(model, X, y)
```
"""
function viterbi(hmm::hmmglm, X::Matrix{<:Real}, y::Vector{Float64})
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


