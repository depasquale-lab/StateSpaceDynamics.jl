function test_GaussianEmission()
    # Initialize Gaussian Emission Model
    gaussian_emission = GaussianEmission([0.0, 0.0], [1.0 0.0; 0.0 1.0])
    # Check if parameters are initialized correctly
    @test gaussian_emission.μ == [0.0, 0.0]
    @test gaussian_emission.Σ == [1.0 0.0; 0.0 1.0]
    # Generate random data
    data = randn(100, 2)
    # Calculate log-likelihood
    ll = StateSpaceDynamics.loglikelihood(gaussian_emission, data[1, :])
    # Check if log-likelihood is a scalar
    @test size(ll) == ()
    # Log-likelihood should be a negative float
    @test ll < 0.0
    # Check sample emission
    sample = StateSpaceDynamics.sample_emission(gaussian_emission)
    @test length(sample) == 2
    # Update emission model
    γ = ones(100)
    StateSpaceDynamics.updateEmissionModel!(gaussian_emission, data, γ)
    # Check if parameters are updated correctly
    @test gaussian_emission.μ ≈ mean(data; dims=1)'
    @test gaussian_emission.Σ ≈ cov(data; corrected=false)
end

function test_regression_emissions()
    # generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    true_σ² = 0.5
    # gaussian glm response
    y = X * true_β + rand(Normal(0.0, sqrt(true_σ²)), 1000)
    # poisson glm response
    true_poisson_model = PoissonRegression(true_β, true)
    y_poisson = StateSpaceDynamics.sample(true_poisson_model, X)
    # bernoulli glm response
    true_bernoulli_model = BernoulliRegression(true_β, true)
    y_bernoulli = StateSpaceDynamics.sample(true_bernoulli_model, X)

    # initialize emission models
    gaussian_emission = RegressionEmissions(
        GaussianRegression(; num_features=3, num_targets=1, include_intercept=false)
    )
    poisson_emission = RegressionEmissions(PoissonRegression(; include_intercept=false))
    bernoulli_emission = RegressionEmissions(BernoulliRegression(; include_intercept=false))
    # update emission models
    StateSpaceDynamics.update_emissions_model!(gaussian_emission, X, reshape(y, 1000, 1))
    StateSpaceDynamics.update_emissions_model!(poisson_emission, X, y_poisson)
    StateSpaceDynamics.update_emissions_model!(bernoulli_emission, X, y_bernoulli)
    # check if parameters are updated correctly
    @test isapprox(gaussian_emission.regression.β, reshape(true_β, 3, 1), atol=0.5)
    @test isapprox(gaussian_emission.regression.Σ, reshape([true_σ²], 1, 1), atol=0.1)
    @test isapprox(poisson_emission.regression.β, true_β, atol=0.5)
    @test isapprox(bernoulli_emission.regression.β, true_β, atol=0.5)
    # test the loglikelihood
    # ll_gaussian = StateSpaceDynamics.loglikelihood(gaussian_emission, X[1, :], y[1])
    # ll_poisson = StateSpaceDynamics.loglikelihood(poisson_emission, X[1, :], y_poisson[1])
    # ll_bernoulli = StateSpaceDynamics.loglikelihood(bernoulli_emission, X[1, :], y_bernoulli[1])
    # @test ll_gaussian < 0
    # @test ll_poisson < 0
    # @test ll_bernoulli < 0
end
