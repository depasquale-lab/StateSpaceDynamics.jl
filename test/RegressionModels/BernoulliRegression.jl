function BernoulliRegression_simulation(n::Int)
    # Generate synthetic data
    Φ = randn(n, 2)
    β = [3, 1, 0.5]
    true_model = BernoulliRegressionEmission(β=β, input_dim=2)
    Y = StateSpaceDynamics.sample(true_model, Φ)

    return true_model, Φ, Y
end

# check loglikelihood is negative
function test_BernoulliRegression_loglikelihood()
    n = 1000
    true_model, Φ, Y = BernoulliRegression_simulation(n)
    @test StateSpaceDynamics.loglikelihood(true_model, Φ, Y) < 0
end

# check model shape and value from constructor
function test_BernoulliRegression_constructor()
    # test parameter shapes
    model = BernoulliRegressionEmission(input_dim=3)
    @test size(model.β, 1) == 4

    model = BernoulliRegressionEmission(input_dim=3, include_intercept=false)
    @test size(model.β, 1) == 3

    # test default values
    model = BernoulliRegressionEmission(input_dim=3)
    @test model.λ == 0.0
    @test model.include_intercept == true
    @test model.β == zeros(4)
end


# check the objective_grad! is close to numerical gradient
function test_BernoulliRegression_objective_gradient()
    n = 1000
    true_model, Φ, Y = BernoulliRegression_simulation(n)


    est_model = BernoulliRegressionEmission(input_dim=2)
    

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
function test_BernoulliRegression_standard_fit()
    # Generate synthetic data
    n = 5000
    true_model, Φ, Y = BernoulliRegression_simulation(n)

    # Initialize and fit the model
    est_model = BernoulliRegressionEmission(input_dim=2)
    fit!(est_model, Φ, Y)

    # confirm that the fitted model has a higher loglikelihood than the true model
    @test StateSpaceDynamics.loglikelihood(est_model, Φ, Y) >= StateSpaceDynamics.loglikelihood(true_model, Φ, Y)

    # confirm that the fitted model has similar β values to the true model
    @test isapprox(est_model.β, true_model.β, atol=0.5)
end

# check that a regularized model has β values closer to a normal gaussian and the model doesn't perform too much worse
function test_BernoulliRegression_regularized_fit()
    λ = 0.1

    # Generate synthetic data
    n = 1000
    true_model, Φ, Y = BernoulliRegression_simulation(n)

    # Initialize and fit an *unregularized* model
    est_model = BernoulliRegressionEmission(input_dim=2)
    fit!(est_model, Φ, Y)

    # Initialize and fit a regularized model
    regularized_est_model = BernoulliRegressionEmission(input_dim=2, λ=λ)
    fit!(regularized_est_model, Φ, Y)


    # confirm that the regularized model is not too much worse
    @test isapprox(
        StateSpaceDynamics.loglikelihood(regularized_est_model, Φ, Y), 
        StateSpaceDynamics.loglikelihood(est_model, Φ, Y), 
        atol=0.1
        )

    # confirm thet the regularized model's parameters are closer to standard normal Distributions
    θ_prior = MvNormal(zeros(3), Matrix{Float64}(I, 3, 3))
    @test logpdf(θ_prior, regularized_est_model.β) > logpdf(θ_prior, est_model.β)
end

# check that the model is a valid emission model

# Please ensure all criteria are met for any new emission model:
# 1. loglikelihood(model, data...; observation_wise=true) must return a Vector{Float64} of the loglikelihood of each observation.
# 2. fit!(model, data..., <weights here>) must fit the model using the weights provided (by maximizing the weighted loglikelihood).
# 3. TimeSeries(model, sample(model, data...; n=<number of samples>)) must return a TimeSeries object of n samples.
# 4. revert_TimeSeries(model, time_series) must return the time_series data converted back to the original sample() format (the inverse of TimeSeries(model, samples)).
function test_BernoulliRegression_valid_emission_model()
    n = 1000
    true_model, Φ, Y = BernoulliRegression_simulation(n)

    # Criteria 1
    loglikelihoods = StateSpaceDynamics.loglikelihood(true_model, Φ, Y, observation_wise=true)
    @test length(loglikelihoods) == n

    # Criteria 2
    weights = rand(n)
    est_model = BernoulliRegressionEmission(input_dim=2)
    fit!(est_model, Φ, Y, weights)

    # Criteria 3
    Y_new = StateSpaceDynamics.sample(est_model, Φ, n=100)
    time_series = StateSpaceDynamics.TimeSeries(est_model, Y_new)
    @test typeof(time_series) == TimeSeries

    # Criteria 4
    @test StateSpaceDynamics.revert_TimeSeries(est_model, time_series) == Y_new
   
end