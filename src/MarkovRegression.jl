export SwitchingGaussianRegression, MarkovRegressionEM
"""
SwitchingGaussianRegression

Struct representing a Gaussian Markov Regression model.

"""
mutable struct SwitchingGaussianRegression{T <: Real} <: AbstractHMM
    y::Vector{T} # observations
    X::Matrix{T} # covariates
    K::Int # number of states
    A::Matrix{T} # transition matrix
    πₖ::Vector{T} # initial state distribution
    B::Vector{RegressionEmissions} # Vector of Gaussian Regression Models
    σ²::Vector{T} # Vector of variances for each state
    weights::Matrix{T} # Vector of weights for each state
end

"""
Constructor for SwitchingGaussianRegression. 
    
We assume there is a set of discrete data-generating regimes 
in the data that can be explained by essentially k-separate regression models from a Gaussian Family. 
The transition matrix A is a k x k matrix where A[i, j] is the probability of transitioning from state 
i to state j. The initial state distribution π_k is a k-dimensional vector where π_k[i] is the probability 
of starting in state i. The number of states K is an integer representing the number of states in the model.

Args:
    y: vector of observations.
    X: matrix of covariates.
    k: number of states.

Returns:
    SwitchingGaussianRegression object with initialized parameters.
"""
function SwitchingGaussianRegression(y::Vector{T}, X::Matrix{T}, k::Int) where T <: Real
    # Initialize regression models
    regression_models = [RegressionEmissions(GaussianRegression(X, y, true)) for _ in 1:k]
    # Initialize emissions model and transition matrix
    πₖ, A = initialize_emissions_model(k, y, regression_models)
    # initialize variance
    σ² = zeros(T, k)
    # Initiliaze weights
    weights = ones(T, (k, length(y)))
    # Create the SwitchingGaussianRegression object with initialized parameters
    model = SwitchingGaussianRegression(y, X, k, A, πₖ, regression_models, σ², weights)
    # Perform the E-step with the initialized model
    _, _, γ, _, _ = EStep(model)
    # Update variance based on the E-step
    update_variance!(model, γ)
    # Print initial parameters for verification
    println("Initial Variance: ", model.σ²)
    return model
end

function initialize_emissions_model(k::Int, data::AbstractVector, regression_emissions::Vector{RegressionEmissions})
    # init with a GMM
    gmm = GMM(k, 1, data)
    fit!(gmm, data, tol=0.1)
    # initialize emission model
    for i in 1:k
        weights = gmm.class_probabilities[:, i]
        regression_emissions[i].regression_model.loss = WLSLoss(weights)
        updateEmissionModel!(regression_emissions[i])
    end
    # use gmm class probabilities to set initial distribution
    πₖ = gmm.class_probabilities[1, :]
    # use gmm class probabilites to estimate an A matrix
    A = estimate_transition_matrix(k, gmm.class_labels)
    return πₖ, A
end

function estimate_transition_matrix(k::Int, class_labels::Vector{Int})
    # Initialize transition matrix with zeros
    A = zeros(k, k)
    # Count transitions from i to j
    for t in 1:length(class_labels)-1
        current_state = class_labels[t]
        next_state = class_labels[t+1]
        A[current_state, next_state] += 1
    end
    # Normalize each row to sum to 1
    for i in 1:k
        row_sum = sum(A[i, :])
        if row_sum != 0  # Avoid division by zero
            A[i, :] ./= row_sum
        end
    end
    return A
end

function EStep(model::SwitchingGaussianRegression)
    # E-Step
    α = forward(model, model.y)
    β = backward(model, model.y)
    γ = calculate_γ(model, α, β)
    ξ = calculate_ξ(model, α, β, model.y)
    log_likelihood = logsumexp(α[:, end])
    return α, β, γ, ξ, log_likelihood
end

function update_mixing_coefficients!(model::SwitchingGaussianRegression, γ::Matrix{T}) where T <: Real
    # update the mixing coefficients
    model.πₖ = exp.(γ[:, 1])
end

function update_variance!(model::SwitchingGaussianRegression, γ::Matrix{T}) where T <: Real
    weights = model.weights
    N = size(weights, 2) - 1
    # update the variance term
    σ² = zeros(T, model.K)
    for k in 1:model.K
        resid = residuals(model.B[k].regression_model)
        weighted_mean = sum(weights[k] .* resid) / sum(weights[k])
        println(sum(weights[k] .* (resid .- weighted_mean).^2))
        σ²[k] = sum(weights[k] .* (resid .- weighted_mean).^2) / sum(weights[k]) / N
    end
    model.σ² = σ²
end

function update_regression_model!(model::SwitchingGaussianRegression, α::Matrix{T}, β::Matrix{T}) where T <: Real
    for k in 1:model.K
        # get weights for wls
        weights = sqrt.(exp.(α[k, :] .+ β[k, :]))
        # normalize the weights
        model.weights[k, :] = weights
        # define weighted loss function
        model.B[k].regression_model.loss = WLSLoss(model.weights[k, :])
        # update the regression model for each state
        updateEmissionModel!(model.B[k])
    end
end

function update_adjacency_matrix!(model::SwitchingGaussianRegression, γ::Matrix{T}, ξ::Array{Float64, 3}) where T <: Real
    # update the HMM adjacency matrix
    for i in 1:model.K
        for j in 1:model.K
            model.A[i, j] = exp(log(sum(exp.(ξ[i, j, :]))) - log(sum(exp.(γ[i, 1:end-1]))))
        end
    end
end

function MStep!(model::SwitchingGaussianRegression, α::Matrix{T}, β::Matrix{T}, γ::Matrix{T}, ξ::Array{Float64, 3}) where T <: Real
    # M-Step
    update_adjacency_matrix!(model, γ, ξ)
    update_mixing_coefficients!(model, γ)
    update_regression_model!(model, α, β)
    update_variance!(model, γ)
end

"""
EM for a SwitchingGaussianRegression model.
"""

function MarkovRegressionEM(model::SwitchingGaussianRegression, max_iters::Int=100, tol::Float64=1e-6)
    # init log-likelihood
    prev_log_likelihood = -Inf
    for i in 1:max_iters
        # E-Step
        α, β, γ, ξ, ll = EStep(model)
        # M-Step
        MStep!(model, α, β, γ, ξ)
        println("Log-likelihood at iteration $i: ", ll)
        # check convergence
        if abs(ll - prev_log_likelihood) < tol
            break
        end
        prev_log_likelihood = ll
    end
end

"""
AutoRegressive HMM
"""
mutable struct AutoRegressiveHMM{T<:Real} <: AbstractHMM
    y::Matrix{T} # observations
    K::Int # number of states
    A::Matrix{T} # transition matrix
    πₖ::Vector{T} # initial state distribution
    B::Vector{AutoRegressiveEmissions} # Vector of AutoRegressive Emissions Models
    σ²::Vector{T} # Vector of variances for each state
    weights::Vector{T} # Vector of weights for each state
    p::Int # order of the autoregressive model
end

"""
Poisson Markov regression, Binomial Markov Regression, Multinomial Markov Regression eventually
"""
mutable struct SwitchingBinomialRegression <: AbstractHMM
    y::Vector{Int} # observations
    X::Matrix{Float64} # covariates
    K::Int # number of states
    A::Matrix{Float64} # transition matrix
    πₖ::Vector{Float64} # initial state distribution
    B::Vector{RegressionEmissions} # Vector of Binomial Regression Models
    weights::Vector{Float64} # Vector of weights for each state
end

function SwitchingBinomialRegression(y::Vector{Int}, X::Matrix{Float64}, k::Int)
    # Initialize regression models
    regression_models = [RegressionEmissions(BinomialRegression(X, y, true)) for _ in 1:k]
    # Initialize emissions model and transition matrix  
end

mutable struct SwitchingPoissonRegression{T<:Real} <: AbstractHMM
    #TODO: Implement Switching Poisson Regression
end

mutable struct SwitchingMultinomialRegression{T<:Real} <: AbstractHMM
    #TODO: Implement Switching Multinomial Regression
end

mutable struct SwitchingNegativeBinomialRegression{T<:Real} <: AbstractHMM
    #TODO: Implement Switching Negative Binomial Regression
end
