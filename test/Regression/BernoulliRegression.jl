function test_BernoulliRegression_fit()
    # Generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    p = logistic.(X * true_β)
    y = rand.(Bernoulli.(p))
    
    # Initialize and fit the model
    model = BernoulliRegression()
    fit!(model, X[:, 2:end], y)
    
    # Check if the fitted coefficients are reasonable
    @test length(model.β) == 3
    @test isapprox(model.β, true_β, atol=0.5)
end

function test_BernoulliRegression_loglikelihood()
    # Generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    p = logistic.(X * true_β)
    y = rand.(Bernoulli.(p))
    
    # Initialize and fit the model
    model = BernoulliRegression()
    fit!(model, X[:, 2:end], y)
    # check if the fitted coefficients are close to the true coefficients
    @test length(model.β) == 3
    @test isapprox(model.β, true_β, atol=0.5)
    
    # Check log likelihood
    loglik = SSM.loglikelihood(model, X[:, 2:end], y)
    @test loglik < 0

    #test loglikelihood on a single point
    loglik = SSM.loglikelihood(model, X[1, 2:end], y[1])
    @test loglik < 0
end

function test_BernoulliRegression_empty_model()
    model = BernoulliRegression()
    @test isempty(model.β)
end

function test_BernoulliRegression_intercept()
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    p = logistic.(X * true_β)
    y = rand.(Bernoulli.(p))
    
    model = BernoulliRegression(include_intercept=false)
    fit!(model, X[:, 2:end], y)
    @test length(model.β) == 2
    
    model_with_intercept = BernoulliRegression()
    fit!(model_with_intercept, X[:, 2:end], y)
    @test length(model_with_intercept.β) == 3
    @test isapprox(model_with_intercept.β, true_β, atol=0.5)
end

function test_Bernoulli_ll_gradient()
    # generate data and observations
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    p = logistic.(X * true_β)
    y = rand.(Bernoulli.(p))
    # initialize model
    model = BernoulliRegression()
    model.β = [0., 0., 0.]

    # use ForwardDiff to calculate the gradient
    function objective(β, w)
        return -sum(w .* (y .* log.(logistic.(X * β)) .+ (1 .- y) .* log.(1 .- logistic.(X * β)))) + (model.λ * sum(β.^2))
    end
    grad = ForwardDiff.gradient(x -> objective(x, ones(1000)), model.β)
    # calculate the gradient manually
    grad_analytic = SSM.gradient!([0., 0., 0.], model, X, y)
    # check if the gradients are close
    @test isapprox(grad, grad_analytic, atol=1e-6)

    # now do the same with Weights
    weights = rand(1000)
    grad = ForwardDiff.gradient(x -> objective(x, weights), model.β)
    grad_analytic = SSM.gradient!([0., 0., 0.], model, X, y, weights)
    @test isapprox(grad, grad_analytic, atol=1e-6)

    # finally test when λ is not 0
    model = BernoulliRegression(λ=0.1)
    model.β = rand(3)

    grad = ForwardDiff.gradient(x -> objective(x, ones(1000)), model.β)
    grad_analytic = SSM.gradient!([0., 0., 0.], model, X, y)
    @test isapprox(grad, grad_analytic, atol=1e-6)
end