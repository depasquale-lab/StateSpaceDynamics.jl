export MarkovRegression, GaussianMarkovRegression, MarkovRegressionEM
"""
Abstract type for Markov Regression models.
"""
abstract type MarkovRegression end

"""
I do not know if I want to support this model yet. Definitely want to incorporate the Bernoulli, Poisson, and Multinomial cases though.
#TODO: To finish this model I need to add a variance parameter in the struct I created. I also need to make that as a case of emissions model.
"""

"""
GaussianMarkovRegression

Struct representing a Gaussian Markov Regression model.

"""
mutable struct GaussianMarkovRegression{T <: Real} <: MarkovRegression
    y::Vector{T} # observations
    X::Matrix{T} # covariates
    K::Int # number of states
    RegressionModels::Vector{GaussianRegression} # Vector of Gaussian Regression Models
    HMM::HMM # The HMM model
end


"""
Constructor for GaussianMarkovRegression. We assume there is a set of discrete data-generating regimes in the data that can be explained by essentially k-separate regression models from a Gaussian Family. The transition matrix A is a k x k matrix where A[i, j] is the probability of transitioning from state i to state j. The initial state distribution π_k is a k-dimensional vector where π_k[i] is the probability of starting in state i. The number of states K is an integer representing the number of states in the model.
"""
function GaussianMarkovRegression(y::Vector{T}, X::Matrix{T}, k::Int) where T <: Real
    # initiliaze HMM
    hmm = HMM(X, k, "Regression")
    # initialize regression models by fitting the K=1 case (i.e. regular old linear regression)
    regression_models = [GaussianRegression(X, y, true) for _ in 1:k]
    return GaussianMarkovRegression(y, X, k, regression_models, hmm)

end

"""
Generalized EM for a GaussianMarkovRegression model.
"""

function MarkovRegressionEM(model::GaussianMarkovRegression, max_iters::Int=100, tol::Float64=1e-6)
    # the algorith mis taken from the Ph.D. thesis of Moshe Fridman see: https://www.proquest.com/docview/304089763?fromopenview=true&pq-origsite=gscholar&parentSessionId=W5CCeTcPsuORzBfQbAZ52%2B970F5PJ%2Fd%2FjWIdz2qMNXI%3D for details.
    for i in 1:max_iter
        # init log-likelihood
        log_likelihood = -Inf
        # E-Step
        α = forward(model.HMM, model.y)
        β = backward(model.HMM, model.y)
        # calculate the weights for the WLS regression
        
        # M-Step
        for k in 1:model.K
            # get weights for wls
            weights = α[k, :] .* β[k, :]
            # define weighted loss function
            loss = WLSLoss(weights)
            # update the regression model for each state
            model.RegressionModels[k] = fit!(model.RegressionModels[k], loss)
            # calculate the variance term
            σ² = sum(α[k, :] .* β[k, :] .* (model.y .- model.RegressionModels[k].predict(model.X)).^2) / sum(α[k, :].*β[k, :])
            # update the HMM adjacency matrix
            for i in 1:model.K
                model.A[k, i] = sum(alpha[k, :] .* model.A[k, i] .* pdf(normal(model.X .* model.RegressionModels[k].β), σ²).* beta[i, :]) / sum(alpha[k, :] .* beta[i, :])
            end
            # update responsibilities
        end
    end
        # Maybe finish this.
end

"""
Poisson Markov regression, Binomial Markov Regression, Multinomial Markov Regression eventually

"""

