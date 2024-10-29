function Gaussian_simulation(n::Int)
    true_model = GaussianEmission(output_dim=2)
    true_model.μ = [3.0, -2.0]
    true_model.Σ = [0.8 0.1; 0.1 2.0]

    # sample data
    Y = StateSpaceDynamics.emission_sample(true_model, n=n)

    return true_model, Y
end

# check loglikelihood is negative
function test_Gaussian_loglikelihood()
    n = 1000
    true_model, Y = Gaussian_simulation(n)
    @test StateSpaceDynamics.loglikelihood(true_model, Y) < 0
end




# Confirm that validate_model() is properly implemented and called during:
# 1. During instantiation of the model
# 2. sample()
# 3. loglikelihood()
# 4. fit!() - before and after fitting


# check that a fitted model has a higher loglikelihood than the true model
function test_Gaussian_standard_fit()
    # Generate synthetic data
    n = 1000
    true_model, Y = Gaussian_simulation(n)

    # Initialize and fit the model
    est_model = GaussianEmission(output_dim=2)
    fit!(est_model, Y)

    # confirm that the fitted model has a higher loglikelihood than the true model
    @test StateSpaceDynamics.loglikelihood(est_model, Y) >= StateSpaceDynamics.loglikelihood(true_model, Y)

    # confirm that the fitted model has similar μ values to the true model
    @test isapprox(est_model.μ, true_model.μ, atol=0.1)

    # confirm that the fitted model's Σ values are good
    @test isapprox(est_model.Σ, true_model.Σ, atol=0.2)
end


# check that the model is a valid emission model

# Please ensure all criteria are met for any new emission model:
# 1. loglikelihood(model, data...; observation_wise=true) must return a Vector{Float64} of the loglikelihood of each observation.
# 2. fit!(model, data..., <weights here>) must fit the model using the weights provided (by maximizing the weighted loglikelihood).
# 3. TimeSeries(model, sample(model, data...; n=<number of samples>)) must return a TimeSeries object of n samples.
# 4. revert_TimeSeries(model, time_series) must return the time_series data converted back to the original sample() format (the inverse of TimeSeries(model, samples)).
function test_Gaussian_valid_emission_model()
    n = 1000
    true_model, Y = Gaussian_simulation(n)

    # Criteria 1
    loglikelihoods = StateSpaceDynamics.loglikelihood(true_model, Y, observation_wise=true)
    @test length(loglikelihoods) == n

    # Criteria 2
    weights = rand(n)
    est_model = GaussianEmission(output_dim=2)
    fit!(est_model, Y, weights)

    # Criteria 3
    Y_new = StateSpaceDynamics.emission_sample(est_model, n=100)
    time_series = StateSpaceDynamics.TimeSeries(est_model, Y_new)
    @test typeof(time_series) == TimeSeries

    # Criteria 4
    @test StateSpaceDynamics.revert_TimeSeries(est_model, time_series) == Y_new
   
end