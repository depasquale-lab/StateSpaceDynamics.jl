function BernoulliRegression_simulation(n::Int)
    # Generate synthetic data
    X = randn(n, 2)
    β = [3, 1, 0.5]
    true_model = BernoulliRegression(β, input_dim=2)
    y = SSM.sample(true_model, X)

    return true_model, X, y
end

# check loglikelihood is negative
function test_BernoulliRegression_loglikelihood()
    n = 1000
    true_model, X, y = BernoulliRegression_simulation(n)
    @test SSM.loglikelihood(true_model, X, y) < 0
end

# check model shape and value from constructor
function test_BernoulliRegression_constructor()
    # test parameter shapes
    model = BernoulliRegression(input_dim=3)
    @test size(model.β, 1) == 4

    model = BernoulliRegression(input_dim=3, include_intercept=false)
    @test size(model.β, 1) == 3

    # test default values
    model = BernoulliRegression(input_dim=3)
    @test model.λ == 0.0
    @test model.include_intercept == true
    @test model.β == zeros(4)
end


# check the objective_grad! is close to numerical gradient
function test_BernoulliRegression_objective_gradient()
    n = 1000
    true_model, X, y = BernoulliRegression_simulation(n)


    est_model = BernoulliRegression(input_dim=2)
    

    # test if analytical gradient is close to numerical gradient
    objective = define_objective(est_model, X, y)
    objective_grad! = define_objective_gradient(est_model, X, y)
    test_gradient(objective, objective_grad!, ones(3, 2))



    # now do the same with Weights
    weights = rand(1000)
    objective = define_objective(est_model, X, y, weights)
    objective_grad! = define_objective_gradient(est_model, X, y, weights)
    test_gradient(objective, objective_grad!, ones(3, 2))



    # finally test when λ is not 0
    est_model.λ = 0.1
    objective = define_objective(est_model, X, y)
    objective_grad! = define_objective_gradient(est_model, X, y)
    test_gradient(objective, objective_grad!, ones(3, 2))
end

# check that a fitted model has a higher loglikelihood than the true model
function test_BernoulliRegression_standard_fit()
    # Generate synthetic data
    n = 5000
    true_model, X, y = BernoulliRegression_simulation(n)

    # Initialize and fit the model
    est_model = BernoulliRegression(input_dim=2)
    fit!(est_model, X, y)

    # confirm that the fitted model has a higher loglikelihood than the true model
    @test SSM.loglikelihood(est_model, X, y) >= SSM.loglikelihood(true_model, X, y)

    # confirm that the fitted model has similar β values to the true model
    @test isapprox(est_model.β, true_model.β, atol=0.5)
end

# check that a regularized model has β values closer to a normal gaussian and the model doesn't perform too much worse
function test_BernoulliRegression_regularized_fit()
    λ = 0.1

    # Generate synthetic data
    n = 1000
    true_model, X, y = BernoulliRegression_simulation(n)

    # Initialize and fit an *unregularized* model
    est_model = BernoulliRegression(input_dim=2)
    fit!(est_model, X, y)

    # Initialize and fit a regularized model
    regularized_est_model = BernoulliRegression(input_dim=2, λ=λ)
    fit!(regularized_est_model, X, y)


    # confirm that the regularized model is not too much worse
    @test isapprox(
        SSM.loglikelihood(regularized_est_model, X, y), 
        SSM.loglikelihood(est_model, X, y), 
        atol=0.1
        )

    # confirm thet the regularized model's parameters are closer to standard normal Distributions
    θ_prior = MvNormal(zeros(3), Matrix{Float64}(I, 3, 3))
    @test logpdf(θ_prior, regularized_est_model.β) > logpdf(θ_prior, est_model.β)
end
























# function test_BernoulliRegression_fit()
#     est_model, X, y, true_model = BernoulliRegression_simulation()

#     # Check if the fitted coefficients are reasonable
#     @test length(est_model.β) == 3
#     @test isapprox(est_model.β, true_model.β, atol=0.5)
# end

# function test_BernoulliRegression_loglikelihood()
#     est_model, X, y, true_model = BernoulliRegression_simulation()
    
#     # Check log likelihood
#     loglik = SSM.loglikelihood(est_model, X, y)
#     @test loglik < 0

#     #test loglikelihood on a single point
#     loglik = SSM.loglikelihood(est_model, X[1, :], y[1])
#     @test loglik < 0
# end

# function test_BernoulliRegression_empty_model()
#     model = BernoulliRegression()
#     @test isempty(model.β)
# end

# function test_BernoulliRegression_intercept()
#     est_model, X, y, true_model = BernoulliRegression_simulation(include_intercept=false)

#     # Check if the fitted coefficients are reasonable
#     @test length(est_model.β) == 2
# end

# function test_Bernoulli_ll_gradient()
#     _, X, y, true_model = BernoulliRegression_simulation(include_intercept=false)


#     # initialize model
#     est_model = BernoulliRegression()


#     function objective(model, X, y, w)
#         return -SSM.loglikelihood(model, X, y, w) + (model.λ * sum(model.β.^2))
#     end

#     function objective_wrt_β(model, X, y, w=ones(size(y,1)))
#         function change_β(β)
#             model.β = β
#             return objective(model, X, y, w)
#         end
#         return change_β
#     end 

#     # numerical gradient
#     grad = ForwardDiff.gradient(objective_wrt_β(est_model, X, y), [0., 0., 0.])
#     # analytic gradient
#     est_model.β = [0., 0., 0.]
#     grad_analytic = SSM.gradient!([0., 0., 0.], est_model, X, y)
#     # compare
#     @test isapprox(grad, grad_analytic, atol=1e-6)


#     # now do the same with Weights
#     weights = rand(1000)
#     # numerical gradient
#     grad = ForwardDiff.gradient(objective_wrt_β(est_model, X, y, weights), [0., 0., 0.])
#     # analytic gradient
#     est_model.β = [0., 0., 0.]
#     grad_analytic = SSM.gradient!([0., 0., 0.], est_model, X, y, weights)
#     # compare
#     @test isapprox(grad, grad_analytic, atol=1e-6)


#     # finally test when λ is not 0
#     est_model = BernoulliRegression(λ=0.1)
#     est_model.β = rand(3)

#     # numerical gradient
#     grad = ForwardDiff.gradient(objective_wrt_β(est_model, X, y), [0., 0., 0.])
#     # analytic gradient
#     est_model.β = [0., 0., 0.]
#     grad_analytic = SSM.gradient!([0., 0., 0.], est_model, X, y)
#     # compare
#     @test isapprox(grad, grad_analytic, atol=1e-6)
# end