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
end


"""
Constructor for GaussianMarkovRegression. We assume there is a set of discrete data-generating regimes in the data that can be explained by essentially k-separate regression models from a Gaussian Family. The transition matrix A is a k x k matrix where A[i, j] is the probability of transitioning from state i to state j. The initial state distribution π_k is a k-dimensional vector where π_k[i] is the probability of starting in state i. The number of states K is an integer representing the number of states in the model.
"""
function GaussianMarkovRegression(y::Vector{T}, X::Matrix{T}, k::Int) where T <: Real
    # initialize transition matrix
    A = rand(k, k)
    A = A ./ sum(A, dims=2) # normalize rows to ensure they are valid probabilities
    # initialize initial state distribution
    πₖ = rand(k)
    πₖ = πₖ ./ sum(πₖ) # normalize to ensure it's a valid probability vector
    # initialize regression models
    regression_models = [RegressionEmissions(GaussianRegression(X, y, true)) for _ in 1:k]
    # use a GMM to initialize the emission model
    gmm = GMM(k, 1, y)
    fit!(gmm, y)
    # initialize emission model
    for i in 1:k
        weights = gmm.class_probabilities[:, i]
        loss = WLSLoss(weights)
        updateEmissionModel!(regression_models[i], loss)
    end
    return GaussianMarkovRegression(y, X, k, A, πₖ, regression_models)
end

"""
Generalized EM for a GaussianMarkovRegression model.
"""

function MarkovRegressionEM(model::GaussianMarkovRegression, max_iters::Int=100, tol::Float64=1e-6)
    # the algorithm is taken from the Ph.D. thesis of Moshe Fridman see: 
    # https://www.proquest.com/docview/304089763?fromopenview=true&pq-origsite=gscholar&parentSessionId=W5CCeTcPsuORzBfQbAZ52%2B970F5PJ%2Fd%2FjWIdz2qMNXI%3D 
    # for details.
    # init log-likelihood
    log_likelihood = -Inf
    for i in 1:max_iters
        log_likelihood_prev = log_likelihood
        # E-Step
        α = forward(model, model.y)
        log_likelihood = logsumexp(α[:, end])
        β = backward(model, model.y)
        # M-Step
        for k in 1:model.K
            # get weights for wls
            weights = sqrt.(α[k, :] .* β[k, :])
            # define weighted loss function
            loss = WLSLoss(weights)
            # update the regression model for each state
            updateEmissionModel!(model.B[k], loss)
            # calculate the variance term
            σ² = sum(α[k, :] .* β[k, :] .* (model.y .- predict(model.B[k].regression_model, model.B[k].regression_model.X)).^2) / sum(α[k, :].*β[k, :])
            print(σ²)
            # update the HMM adjacency matrix
            for i in 1:model.K
                model.A[k, i] = sum(α[k, :] .* model.A[k, i] .* pdf(Normal(predict(model.B[k].regression_model, model.B[k].regression_model.X[i, :]), σ²), model.y[i]).* β[i, :]) / sum(α[k, :] .* β[i, :])
            end
        end
    end# init log-likelihood
end

"""
Poisson Markov regression, Binomial Markov Regression, Multinomial Markov Regression eventually

"""

