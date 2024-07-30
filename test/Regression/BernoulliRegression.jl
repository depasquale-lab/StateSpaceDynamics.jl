function test_BernoulliRegression_fit()
    # Generate synthetic data
    X = randn(1000, 2)
    β=[0.5, -1.2, 2.3]
    true_model = BernoulliRegression(β, true)
    y = SSM.sample(true_model, X)

    # Initialize and fit the model
    est_model = BernoulliRegression()
    fit!(est_model, X, y)

    # Check if the fitted coefficients are reasonable
    @test length(est_model.β) == 3
    @test isapprox(est_model.β, true_model.β, atol=0.5)
end

function test_BernoulliRegression_loglikelihood()
    # Generate synthetic data
    X = randn(1000, 2)
    β=[0.5, -1.2, 2.3]
    true_model = BernoulliRegression(β, true)
    y = SSM.sample(true_model, X)

    # Initialize and fit the model
    est_model = BernoulliRegression()
    fit!(est_model, X, y)

    # Check if the fitted coefficients are reasonable
    @test length(est_model.β) == 3
    @test isapprox(est_model.β, true_model.β, atol=0.5)
    
    # Check log likelihood
    loglik = SSM.loglikelihood(est_model, X, y)
    @test loglik < 0

    #test loglikelihood on a single point
    loglik = SSM.loglikelihood(est_model, X[1, :], y[1])
    @test loglik < 0
end

function test_BernoulliRegression_empty_model()
    model = BernoulliRegression()
    @test isempty(model.β)
end

function test_BernoulliRegression_intercept()
    # Generate synthetic data
    X = randn(1000, 2)
    β=[0.5, -1.2, 2.3]
    true_model = BernoulliRegression(β, true)
    y = SSM.sample(true_model, X)

    # Initialize and fit the model without intercept 
    est_model = BernoulliRegression(include_intercept=false)
    fit!(est_model, X, y)

    # Check if the fitted coefficients are reasonable
    @test length(est_model.β) == 2
end

function test_Bernoulli_ll_gradient()
    # Generate synthetic data
    X = randn(1000, 2)
    β=[0.5, -1.2, 2.3]
    true_model = BernoulliRegression(β, true)
    y = SSM.sample(true_model, X)


    # initialize model
    est_model = BernoulliRegression()
    est_model.β = [0., 0., 0.]


    function objective(β, X, w)
        temp_model = BernoulliRegression()
        temp_model.β = β
        temp_model.λ = est_model.λ
        return -SSM.loglikelihood(temp_model, X, y, w) + (temp_model.λ * sum(temp_model.β.^2))
    end


    grad = ForwardDiff.gradient(x -> objective(x, X, ones(1000)), est_model.β)


    # calculate the gradient manually
    grad_analytic = SSM.gradient!([0., 0., 0.], est_model, X, y)


    # check if the gradients are close
    @test isapprox(grad, grad_analytic, atol=1e-6)


    # now do the same with Weights
    weights = rand(1000)
    grad = ForwardDiff.gradient(x -> objective(x, X, weights), est_model.β)
    grad_analytic = SSM.gradient!([0., 0., 0.], est_model, X, y, weights)
    @test isapprox(grad, grad_analytic, atol=1e-6)


    # finally test when λ is not 0
    est_model = BernoulliRegression(λ=0.1)
    est_model.β = rand(3)

    grad = ForwardDiff.gradient(x -> objective(x, X, ones(1000)), est_model.β)
    grad_analytic = SSM.gradient!([0., 0., 0.], est_model, X, y)
    @test isapprox(grad, grad_analytic, atol=1e-6)
end