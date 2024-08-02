function BernoulliRegression_simulation(;include_intercept::Bool=true)
    # Generate synthetic data
    X = randn(1000, 2)
    β=[0.5, -1.2, 2.3]
    true_model = BernoulliRegression(β, true)
    y = SSM.sample(true_model, X)

    # Initialize and fit the model
    est_model = BernoulliRegression(include_intercept=include_intercept)
    fit!(est_model, X, y)

    return est_model, X, y, true_model
end


function test_BernoulliRegression_fit()
    est_model, X, y, true_model = BernoulliRegression_simulation()

    # Check if the fitted coefficients are reasonable
    @test length(est_model.β) == 3
    @test isapprox(est_model.β, true_model.β, atol=0.5)
end

function test_BernoulliRegression_loglikelihood()
    est_model, X, y, true_model = BernoulliRegression_simulation()
    
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
    est_model, X, y, true_model = BernoulliRegression_simulation(include_intercept=false)

    # Check if the fitted coefficients are reasonable
    @test length(est_model.β) == 2
end

function test_Bernoulli_ll_gradient()
    _, X, y, true_model = BernoulliRegression_simulation(include_intercept=false)


    # initialize model
    est_model = BernoulliRegression()


    function objective(model, X, y, w)
        return -SSM.loglikelihood(model, X, y, w) + (model.λ * sum(model.β.^2))
    end

    function objective_wrt_β(model, X, y, w=ones(size(y,1)))
        function change_β(β)
            model.β = β
            return objective(model, X, y, w)
        end
        return change_β
    end 

    # numerical gradient
    grad = ForwardDiff.gradient(objective_wrt_β(est_model, X, y), [0., 0., 0.])
    # analytic gradient
    est_model.β = [0., 0., 0.]
    grad_analytic = SSM.gradient!([0., 0., 0.], est_model, X, y)
    # compare
    @test isapprox(grad, grad_analytic, atol=1e-6)


    # now do the same with Weights
    weights = rand(1000)
    # numerical gradient
    grad = ForwardDiff.gradient(objective_wrt_β(est_model, X, y, weights), [0., 0., 0.])
    # analytic gradient
    est_model.β = [0., 0., 0.]
    grad_analytic = SSM.gradient!([0., 0., 0.], est_model, X, y, weights)
    # compare
    @test isapprox(grad, grad_analytic, atol=1e-6)


    # finally test when λ is not 0
    est_model = BernoulliRegression(λ=0.1)
    est_model.β = rand(3)

    # numerical gradient
    grad = ForwardDiff.gradient(objective_wrt_β(est_model, X, y), [0., 0., 0.])
    # analytic gradient
    est_model.β = [0., 0., 0.]
    grad_analytic = SSM.gradient!([0., 0., 0.], est_model, X, y)
    # compare
    @test isapprox(grad, grad_analytic, atol=1e-6)
end