export GaussianMarkovRegression, MarkovRegressionEM
"""
I do not know if I want to support this model yet. Definitely want to incorporate the Bernoulli, Poisson, and Multinomial cases though.
#TODO: To finish this model I need to add a variance parameter in the struct I created. I also need to make that as a case of emissions model.
"""

"""
GaussianMarkovRegression

Struct representing a Gaussian Markov Regression model.

"""
mutable struct GaussianMarkovRegression{T <: Real} <: AbstractHMM
    y::Vector{T} # observations
    X::Matrix{T} # covariates
    K::Int # number of states
    A::Matrix{T} # transition matrix
    πₖ::Vector{T} # initial state distribution
    B::Vector{RegressionEmissions} # Vector of Gaussian Regression Models
    σ²::Vector{T} # Vector of variances for each state
    weights::Vector{T} # Vector of weights for each state
end

"""
Constructor for GaussianMarkovRegression. We assume there is a set of discrete data-generating regimes 
in the data that can be explained by essentially k-separate regression models from a Gaussian Family. 
The transition matrix A is a k x k matrix where A[i, j] is the probability of transitioning from state 
i to state j. The initial state distribution π_k is a k-dimensional vector where π_k[i] is the probability 
of starting in state i. The number of states K is an integer representing the number of states in the model.
"""
function GaussianMarkovRegression(y::Vector{T}, X::Matrix{T}, k::Int) where T <: Real
    # Initialize regression models
    regression_models = [RegressionEmissions(GaussianRegression(X, y, true)) for _ in 1:k]
    # Initialize emissions model and transition matrix
    πₖ, A = initialize_emissions_model(k, y, regression_models)
    # initialize variance
    σ² = zeros(T, k)
    # Initiliaze weights
    weights = ones(T, size(y))
    # Create the GaussianMarkovRegression object with initialized parameters
    model = GaussianMarkovRegression(y, X, k, A, πₖ, regression_models, σ², weights)
    # Perform the E-step with the initialized model
    α, β = EStep(model)
    # Update variance based on the E-step
    update_variance!(model, α, β)
    # Print initial parameters for verification
    println("Initial Variance: ", model.σ²)
    println("Initial Transition Matrix: ", model.A)
    println("Initial Mixing Coefficients: ", model.πₖ)
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

function EStep(model::GaussianMarkovRegression)
    # E-Step
    α = forward(model, model.y)
    β = backward(model, model.y)
    log_likelihood = logsumexp(α[:, end])
    return α, β
end

function update_mixing_coefficients!(model::GaussianMarkovRegression, α::Matrix{T}, β::Matrix{T}) where T <: Real
    # update the mixing coefficients
    for i in 1:model.K
        for j in 1:model.K
            model.πₖ[i] = (α[i, 1] * β[i, 1])/sum(α[j, :])
        end
    end
    model.πₖ = model.πₖ ./ sum(model.πₖ)
end

function update_variance!(model::GaussianMarkovRegression, α::Matrix{T}, β::Matrix{T}) where T <: Real
    # update the variance term
    σ² = zeros(T, model.K)
    for k in 1:model.K
        resid = residuals(model.B[k].regression_model).^2
        σ²[k] = sum(α[k, :] .* β[k, :] .* resid) / sum(α[k, :] .* β[k, :])
    end
    model.σ² = σ²
end

function update_regression_model!(model::GaussianMarkovRegression, α::Matrix{T}, β::Matrix{T}) where T <: Real
    for k in 1:model.K
        # get weights for wls
        weights = sqrt.(α[k, :] .* β[k, :])
        # normalize the weights
        model.weights = weights ./ sum(weights)
        # define weighted loss function
        model.B[k].regression_model.loss = WLSLoss(model.weights)
        # update the regression model for each state
        updateEmissionModel!(model.B[k])
    end
end

function update_adjacency_matrix!(model::GaussianMarkovRegression, α::Matrix{T}, β::Matrix{T}) where T <: Real
    # update the HMM adjacency matrix
    for i in 1:model.K
        for j in 1:model.K
            log_numerator = -Inf  # Log-space representation of 0
            log_denominator = logsumexp(α[i, 1:end-1] .+ β[i, 2:end])
            for t in 1:(size(model.y, 1) - 1)
                predicted = predict(model.B[j].regression_model, model.B[j].regression_model.X[t+1, :])
                log_prob_density = logpdf(Normal(predicted, model.σ²[j]), model.y[t+1])
                log_numerator = logsumexp([log_numerator, α[i, t] + log(model.A[i, j]) + log_prob_density + β[j, t+1]])
            end
            model.A[i, j] = exp(log_numerator - log_denominator)
        end
        # Normalize rows after all updates
        model.A[i, :] = model.A[i, :] ./ sum(model.A[i, :])
    end
end

function MStep(model::GaussianMarkovRegression, α::Matrix{T}, β::Matrix{T}) where T <: Real
    # M-Step
    update_adjacency_matrix!(model, α, β)
    update_mixing_coefficients!(model, α, β)
    update_regression_model!(model, α, β)
    update_variance!(model, α, β)
end

"""
Generalized EM for a GaussianMarkovRegression model.
"""

function MarkovRegressionEM(model::GaussianMarkovRegression, max_iters::Int=100, tol::Float64=1e-6)
    # init log-likelihood
    prev_log_likelihood = -Inf
    for i in 1:max_iters
        # E-Step
        α, β = EStep(model)
        # M-Step
        MStep(model, α, β)
        # compute log-likelihood
        log_likelihood = logsumexp(α[:, end])
        println("Log-likelihood at iteration $i: ", log_likelihood)
        # check convergence
        if abs(log_likelihood - prev_log_likelihood) < tol
            break
        end
        prev_log_likelihood = log_likelihood
    end
end

"""
Poisson Markov regression, Binomial Markov Regression, Multinomial Markov Regression eventually

"""

