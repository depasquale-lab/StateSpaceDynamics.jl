function AutoRegression_simulation(n::Int)
    data_dim = 2
    order = 1

    # make a rotation matrix for pi/20 radians
    θ = π/20
    β = [cos(θ) -sin(θ); sin(θ) cos(θ)]

    true_model = AutoRegression(order=order, data_dim=data_dim, β=β, include_intercept=false)

    Y_prev = randn(order, data_dim)
    Y = SSM.sample(true_model, Y_prev, n=n)

    return true_model, Y_prev, Y
end

# check loglikelihood is negative
function test_AutoRegression_loglikelihood()
    n = 1000
    true_model, Y_prev, Y = AutoRegression_simulation(n)
    @test SSM.loglikelihood(true_model, Y_prev, Y) < 0
end

# check covariance matrix is positive definite and hermitian
function test_AutoRegression_Σ()
    n = 1000
    true_model, Φ, Y = AutoRegression_simulation(n)

    est_model = AutoRegression(order=1, data_dim=2)
    fit!(est_model, Φ, Y)

    @test valid_Σ(est_model.innerGaussianRegression.Σ)
end

# check model shape and value from constructor
function test_AutoRegression_constructor()
    # test parameter shapes
    model = AutoRegression(order=1, data_dim=2)
    @test size(model.innerGaussianRegression.β) == (3, 2)
    @test size(model.innerGaussianRegression.Σ) == (2, 2)

    model = AutoRegression(order=2, data_dim=2)
    @test size(model.innerGaussianRegression.β) == (5, 2)
    @test size(model.innerGaussianRegression.Σ) == (2, 2)

    model = AutoRegression(order=1, data_dim=2, include_intercept=false)
    @test size(model.innerGaussianRegression.β) == (2, 2)
    @test size(model.innerGaussianRegression.Σ) == (2, 2)

    # test default values
    model = AutoRegression(order=1, data_dim=2)
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
    est_model = AutoRegression(order=1, data_dim=2, include_intercept=false)
    fit!(est_model, Φ, Y)

    # confirm that the fitted model has a higher loglikelihood than the true model
    @test SSM.loglikelihood(est_model, Φ, Y) >= SSM.loglikelihood(true_model, Φ, Y)

    # confirm that the fitted model has similar β values to the true model
    @test isapprox(est_model.innerGaussianRegression.β, true_model.innerGaussianRegression.β, atol=0.1)

    # confirm that the fitted model's Σ values are good
    @test isapprox(est_model.innerGaussianRegression.Σ, true_model.innerGaussianRegression.Σ, atol=0.1)
    @test valid_Σ(est_model.innerGaussianRegression.Σ)
end

# check that a regularized model has β values closer to a normal gaussian and the model doesn't perform too much worse
function test_AutoRegression_regularized_fit()
    λ = 0.1

    # Generate synthetic data
    n = 1000
    true_model, Φ, Y = AutoRegression_simulation(n)

    # Initialize and fit an *unregularized* model
    est_model = AutoRegression(order = 1, data_dim=2, include_intercept=false)
    fit!(est_model, Φ, Y)

    # Initialize and fit a regularized model
    regularized_est_model = AutoRegression(order = 1, data_dim=2, include_intercept=false, λ=λ)
    fit!(regularized_est_model, Φ, Y)


    # confirm that the regularized model is not too much worse
    @test isapprox(
        SSM.loglikelihood(regularized_est_model, Φ, Y), 
        SSM.loglikelihood(est_model, Φ, Y), 
        atol=0.1
        )

    # confirm thet the regularized model's parameters are closer to standard normal Distributions
    θ_prior = MvNormal(zeros(4), Matrix{Float64}(I, 4, 4))

    regularized_θ = [flatten(regularized_est_model.innerGaussianRegression.β)...]
    est_θ = [flatten(est_model.innerGaussianRegression.β)...]

    @test logpdf(θ_prior, regularized_θ) > logpdf(θ_prior, est_θ)

    # confirm that the regularized fitted model's Σ values are good
    @test isapprox(regularized_est_model.innerGaussianRegression.Σ, true_model.innerGaussianRegression.Σ, atol=0.1)
    @test valid_Σ(regularized_est_model.innerGaussianRegression.Σ)
end