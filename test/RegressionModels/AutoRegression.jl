function AutoRegression_simulation(n::Int)
    output_dim = 2
    order = 1

    # make a rotation matrix for pi/20 radians
    θ = π / 20
    β = [cos(θ) -sin(θ); sin(θ) cos(θ)]

    true_model = AutoRegressionEmission(;
        order=order, output_dim=output_dim, β=β, include_intercept=false
    )

    Y_prev = randn(order, output_dim)
    Y = StateSpaceDynamics.sample(true_model, Y_prev; n=n)

    return true_model, Y_prev, Y
end

# check loglikelihood is negative
function test_AutoRegression_loglikelihood()
    n = 1000
    true_model, Y_prev, Y = AutoRegression_simulation(n)
    @test StateSpaceDynamics.loglikelihood(true_model, Y_prev, Y) < 0
end

# check covariance matrix is positive definite and hermitian
function test_AutoRegression_Σ()
    n = 1000
    true_model, Φ, Y = AutoRegression_simulation(n)

    est_model = AutoRegressionEmission(; order=1, output_dim=2)
    fit!(est_model, Φ, Y)

    @test valid_Σ(est_model.innerGaussianRegression.Σ)
end

# check model shape and value from constructor
function test_AutoRegression_constructor()
    # test parameter shapes
    model = AutoRegressionEmission(; order=1, output_dim=2)
    @test size(model.innerGaussianRegression.β) == (3, 2)
    @test size(model.innerGaussianRegression.Σ) == (2, 2)

    model = AutoRegressionEmission(; order=2, output_dim=2)
    @test size(model.innerGaussianRegression.β) == (5, 2)
    @test size(model.innerGaussianRegression.Σ) == (2, 2)

    model = AutoRegressionEmission(; order=1, output_dim=2, include_intercept=false)
    @test size(model.innerGaussianRegression.β) == (2, 2)
    @test size(model.innerGaussianRegression.Σ) == (2, 2)

    # test default values
    model = AutoRegressionEmission(; order=1, output_dim=2)
    @test model.innerGaussianRegression.λ == 0.0
    @test model.innerGaussianRegression.include_intercept == true
    @test model.innerGaussianRegression.β == zeros(3, 2)
    @test model.innerGaussianRegression.Σ == Matrix{Float64}(I, 2, 2)
end

# check that a fitted model has a higher loglikelihood than the true model
function test_AutoRegression_standard_fit()
    # Generate synthetic data
    n = 1000
    true_model, Φ, Y = AutoRegression_simulation(n)

    # Initialize and fit the model
    est_model = AutoRegressionEmission(; order=1, output_dim=2, include_intercept=false)
    fit!(est_model, Φ, Y)

    # confirm that the fitted model has a higher loglikelihood than the true model
    @test StateSpaceDynamics.loglikelihood(est_model, Φ, Y) >=
        StateSpaceDynamics.loglikelihood(true_model, Φ, Y)

    # confirm that the fitted model has similar β values to the true model
    @test isapprox(
        est_model.innerGaussianRegression.β, true_model.innerGaussianRegression.β, atol=0.1
    )

    # confirm that the fitted model's Σ values are good
    @test isapprox(
        est_model.innerGaussianRegression.Σ, true_model.innerGaussianRegression.Σ, atol=0.1
    )
    @test valid_Σ(est_model.innerGaussianRegression.Σ)
end

# check that a regularized model has β values closer to a normal gaussian and the model doesn't perform too much worse
function test_AutoRegression_regularized_fit()
    λ = 0.1

    # Generate synthetic data
    n = 1000
    true_model, Φ, Y = AutoRegression_simulation(n)

    # Initialize and fit an *unregularized* model
    est_model = AutoRegressionEmission(; order=1, output_dim=2, include_intercept=false)
    fit!(est_model, Φ, Y)

    # Initialize and fit a regularized model
    regularized_est_model = AutoRegressionEmission(;
        order=1, output_dim=2, include_intercept=false, λ=λ
    )
    fit!(regularized_est_model, Φ, Y)

    # confirm that the regularized model is not too much worse
    @test isapprox(
        StateSpaceDynamics.loglikelihood(regularized_est_model, Φ, Y),
        StateSpaceDynamics.loglikelihood(est_model, Φ, Y),
        atol=0.1,
    )

    # confirm thet the regularized model's parameters are closer to standard normal Distributions
    θ_prior = MvNormal(zeros(4), Matrix{Float64}(I, 4, 4))

    regularized_θ = [flatten(regularized_est_model.innerGaussianRegression.β)...]
    est_θ = [flatten(est_model.innerGaussianRegression.β)...]

    @test logpdf(θ_prior, regularized_θ) > logpdf(θ_prior, est_θ)

    # confirm that the regularized fitted model's Σ values are good
    @test isapprox(
        regularized_est_model.innerGaussianRegression.Σ,
        true_model.innerGaussianRegression.Σ,
        atol=0.1,
    )
    @test valid_Σ(regularized_est_model.innerGaussianRegression.Σ)
end

# check that the model is a valid emission model

# Please ensure all criteria are met for any new emission model:
# 1. loglikelihood(model, data...; observation_wise=true) must return a Vector{Float64} of the loglikelihood of each observation.
# 2. fit!(model, data..., <weights here>) must fit the model using the weights provided (by maximizing the weighted loglikelihood).
# 3. TimeSeries(model, sample(model, data...; n=<number of samples>)) must return a TimeSeries object of n samples.
# 4. revert_TimeSeries(model, time_series) must return the time_series data converted back to the original sample() format (the inverse of TimeSeries(model, samples)).
function test_AutoRegression_valid_emission_model()
    n = 1000
    true_model, Φ, Y = AutoRegression_simulation(n)

    # Criteria 1
    loglikelihoods = StateSpaceDynamics.loglikelihood(
        true_model, Φ, Y; observation_wise=true
    )
    @test length(loglikelihoods) == n

    # Criteria 2
    weights = rand(n)
    est_model = AutoRegressionEmission(; order=1, output_dim=2, include_intercept=false)
    fit!(est_model, Φ, Y, weights)

    # Criteria 3
    Y_new = StateSpaceDynamics.sample(est_model, Φ; n=100)
    time_series = StateSpaceDynamics.TimeSeries(est_model, Y_new)
    @test typeof(time_series) == TimeSeries

    # Criteria 4
    @test StateSpaceDynamics.revert_TimeSeries(est_model, time_series) == Y_new
end
