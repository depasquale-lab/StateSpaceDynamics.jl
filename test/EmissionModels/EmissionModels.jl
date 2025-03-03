function GaussianEmission_simulation(n::Int)
    emission_1 = Gaussian(; output_dim=2)
    emission_1.μ = [3.0, -2.0]
    emission_1.Σ = [0.8 0.1; 0.1 2.0]

    emission_2 = Gaussian(; output_dim=2)
    emission_2.μ = [10.0, -10.0]
    emission_2.Σ = [0.8 0.1; 0.1 2.0]

    true_model = HiddenMarkovModel(; K=2, B=[emission_1, emission_2])
    true_model.πₖ = [1.0, 0]

    # sample data
    state_sequence, Y = StateSpaceDynamics.sample(true_model; n=n)

    return true_model, state_sequence, Y
end

function test_GaussianEmission()
    n = 5000
    true_model, state_sequence, Y = GaussianEmission_simulation(n)

    em_1 = Gaussian(; output_dim=2)
    em_2 = Gaussian(; output_dim=2)
    est_model = HiddenMarkovModel(; K=2, B=[em_1, em_2])

    est_model.πₖ = [1.0, 0]

    fit!(est_model, Y)

    # confirm that the fitted model has a higher loglikelihood than the true model
    @test StateSpaceDynamics.loglikelihood(est_model, Y) >=
        StateSpaceDynamics.loglikelihood(true_model, Y)

    pred_means = [est_model.B[i].μ for i in 1:2]
    means = [true_model.B[i].μ for i in 1:2]

    # check if the means are close, one by one, with atol 0.2
    @test pred_means[1] ≈ means[1] atol = 0.2
    @test pred_means[2] ≈ means[2] atol = 0.2

    pred_covs = [est_model.B[i].Σ for i in 1:2]
    covs = [true_model.B[i].Σ for i in 1:2]

    # check if the covariances are close, one by one, with atol 0.2
    @test pred_covs[1] ≈ covs[1] atol = 0.2
    @test pred_covs[2] ≈ covs[2] atol = 0.2
end

function test_regression_emissions()
    # generate synthetic data
    Φ = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    true_σ² = 0.5
    # gaussian glm response
    y = Φ * true_β + rand(Normal(0.0, sqrt(true_σ²)), 1000)
    # poisson glm response
    true_poisson_model = PoissonRegression(true_β; input_dim=2)
    y_poisson = StateSpaceDynamics.sample(true_poisson_model, Φ)
    # bernoulli glm response
    true_bernoulli_model = BernoulliRegression(true_β; input_dim=2)
    y_bernoulli = StateSpaceDynamics.sample(true_bernoulli_model, Φ)

    # initialize emission models
    gaussian_emission = RegressionEmissions(
        GaussianRegression(; input_dim=3, output_dim=1, include_intercept=false)
    )
    poisson_emission = RegressionEmissions(
        PoissonRegression(; input_dim=3, include_intercept=false)
    )
    bernoulli_emission = RegressionEmissions(
        BernoulliRegression(; input_dim=3, include_intercept=false)
    )
    # update emission models
    StateSpaceDynamics.update_emissions_model!(gaussian_emission, Φ, reshape(y, 1000, 1))
    StateSpaceDynamics.update_emissions_model!(poisson_emission, Φ, y_poisson)
    StateSpaceDynamics.update_emissions_model!(bernoulli_emission, Φ, y_bernoulli)
    # check if parameters are updated correctly
    @test isapprox(gaussian_emission.regression.β, reshape(true_β, 3, 1), atol=0.5)
    @test isapprox(gaussian_emission.regression.Σ, reshape([true_σ²], 1, 1), atol=0.1)
    @test isapprox(poisson_emission.regression.β, true_β, atol=0.5)
    @test isapprox(bernoulli_emission.regression.β, true_β, atol=0.5)
    # test the loglikelihood
    # ll_gaussian = StateSpaceDynamics.loglikelihood(gaussian_emission, Φ[1, :], y[1])
    # ll_poisson = StateSpaceDynamics.loglikelihood(poisson_emission, Φ[1, :], y_poisson[1])
    # ll_bernoulli = StateSpaceDynamics.loglikelihood(bernoulli_emission, Φ[1, :], y_bernoulli[1])
    # @test ll_gaussian < 0
    # @test ll_poisson < 0
    # @test ll_bernoulli < 0
end
