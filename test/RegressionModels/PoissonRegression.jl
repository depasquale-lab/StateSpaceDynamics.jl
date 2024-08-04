function PoissonRegression_simulation(n::Int)
    # Generate synthetic data
    Φ = randn(n, 2)
    β = [0.5, -1.2, 2.3]
    true_model = PoissonRegression(β=β, input_dim=2)
    Y = SSM.sample(true_model, Φ)

    return true_model, Φ, Y
end

# check loglikelihood is negative
function test_PoissonRegression_loglikelihood()
    n = 1000
    true_model, Φ, Y = PoissonRegression_simulation(n)
    @test SSM.loglikelihood(true_model, Φ, Y) < 0
end

# check model shape and value from constructor
function test_PoissonRegression_constructor()
    # test parameter shapes
    model = PoissonRegression(input_dim=3)
    @test size(model.β, 1) == 4

    model = PoissonRegression(input_dim=3, include_intercept=false)
    @test size(model.β, 1) == 3

    # test default values
    model = PoissonRegression(input_dim=3)
    @test model.λ == 0.0
    @test model.include_intercept == true
    @test model.β == zeros(4)
end

# check the objective_grad! is close to numerical gradient
function test_PoissonRegression_objective_gradient()
    n = 1000
    true_model, Φ, Y = PoissonRegression_simulation(n)


    est_model = PoissonRegression(input_dim=2)
    

    # test if analytical gradient is close to numerical gradient
    objective = define_objective(est_model, Φ, Y)
    objective_grad! = define_objective_gradient(est_model, Φ, Y)
    test_gradient(objective, objective_grad!, ones(3, 2))



    # now do the same with Weights
    weights = rand(1000)
    objective = define_objective(est_model, Φ, Y, weights)
    objective_grad! = define_objective_gradient(est_model, Φ, Y, weights)
    test_gradient(objective, objective_grad!, ones(3, 2))



    # finally test when λ is not 0
    est_model.λ = 0.1
    objective = define_objective(est_model, Φ, Y)
    objective_grad! = define_objective_gradient(est_model, Φ, Y)
    test_gradient(objective, objective_grad!, ones(3, 2))
end

# check that a fitted model has a higher loglikelihood than the true model
function test_PoissonRegression_standard_fit()
    # Generate synthetic data
    n = 5000
    true_model, Φ, Y = PoissonRegression_simulation(n)

    # Initialize and fit the model
    est_model = PoissonRegression(input_dim=2)
    fit!(est_model, Φ, Y)

    # confirm that the fitted model has a higher loglikelihood than the true model
    @test SSM.loglikelihood(est_model, Φ, Y) >= SSM.loglikelihood(true_model, Φ, Y)

    # confirm that the fitted model has similar β values to the true model
    @test isapprox(est_model.β, true_model.β, atol=0.5)
end

# check that a regularized model has β values closer to a normal gaussian and the model doesn't perform too much worse
function test_PoissonRegression_regularized_fit()
    λ = 0.1

    # Generate synthetic data
    n = 1000
    true_model, Φ, Y = PoissonRegression_simulation(n)

    # Initialize and fit an *unregularized* model
    est_model = PoissonRegression(input_dim=2)
    fit!(est_model, Φ, Y)

    # Initialize and fit a regularized model
    regularized_est_model = PoissonRegression(input_dim=2, λ=λ)
    fit!(regularized_est_model, Φ, Y)


    # confirm that the regularized model is not too much worse
    @test isapprox(
        SSM.loglikelihood(regularized_est_model, Φ, Y), 
        SSM.loglikelihood(est_model, Φ, Y), 
        atol=0.1
        )

    # confirm thet the regularized model's parameters are closer to standard normal Distributions
    θ_prior = MvNormal(zeros(3), Matrix{Float64}(I, 3, 3))
    @test logpdf(θ_prior, regularized_est_model.β) > logpdf(θ_prior, est_model.β)

end