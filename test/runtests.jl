using SSM
using Distributions
using LinearAlgebra
using Random
using Test

Random.seed!(1234)

"""
Tests for MixtureModels.jl
"""

function test_GMM_constructor()
    k_means = 3
    data_dim = 2
    data = randn(100, data_dim)
    gmm = GMM(k_means, data_dim, data)
    @test gmm.k_means == k_means
    @test size(gmm.μ_k) == (data_dim, k_means)
    @test length(gmm.Σ_k) == k_means
    for i in 1:k_means
        @test gmm.Σ_k[i] ≈ I(data_dim)
    end
    @test sum(gmm.π_k) ≈ 1.0
    @test size(gmm.class_labels) == (100,)
    @test size(gmm.class_probabilities) == (100, k_means)
end

function testGMM_EStep()
    # Initialize parameters
    k_means = 3
    data_dim = 2
    data = randn(10, data_dim)
    gmm = GMM(k_means, data_dim, data)
    # Run EStep
    SSM.EStep!(gmm, data)
    # Check dimensions
    @test size(gmm.class_probabilities) == (10, k_means)
    # Check if the row sums are close to 1 (since they represent probabilities)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(gmm.class_probabilities, dims=2))
end

function testGMM_MStep()
    # Initialize parameters
    k_means = 3
    data_dim = 2
    data = randn(10, data_dim)
    gmm = GMM(k_means, data_dim, data)
    SSM.EStep!(gmm, data)

    # Run MStep
    SSM.MStep!(gmm, data)

    # Check dimensions of updated μ and Σ
    @test size(gmm.μ_k) == (data_dim, k_means)
    @test length(gmm.Σ_k) == k_means

    # Check if the covariance matrices are Hermitian
    @test all([ishermitian(Σ) for Σ in gmm.Σ_k])
end

function testGMM_fit()
    # Initialize parameters
    k_means = 3
    data_dim = 2
    data = randn(10, data_dim)
    gmm = GMM(k_means, data_dim, data)

    # Run fit!
    fit!(gmm, data; maxiter=10, tol=1e-3)

    # Check dimensions of updated μ and Σ
    @test size(gmm.μ_k) == (data_dim, k_means)
    @test length(gmm.Σ_k) == k_means

    # Check if the covariance matrices are Hermitian
    @test all([ishermitian(Σ) for Σ in gmm.Σ_k])

    # Check if the mixing coefficients sum to 1
    @test sum(gmm.π_k) ≈ 1.0

    # Check if the class_probabilities add to 1
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(gmm.class_probabilities, dims=2))

    # Check if the class labels are integers in the range 1 to k_means
    @test all(x -> x in 1:k_means, gmm.class_labels)
end

function test_log_likelihood()
    # Initialize parameters
    k_means = 3
    data_dim = 2
    data = randn(10, data_dim)
    gmm = GMM(k_means, data_dim, data)

    # Calculate log-likelihood
    ll = log_likelihood(gmm, data)

    # Check if log-likelihood is a scalar
    @test size(ll) == ()

    # Log-likelihood should be a negative float
    @test ll < 0.0

    # Log-likelihood should monotonically increase with iterations (when using exact EM)
    ll_prev = -Inf
    for i in 1:10
        fit!(gmm, data; maxiter=1, tol=1e-3)
        ll = log_likelihood(gmm, data)
        @test ll > ll_prev || isapprox(ll, ll_prev; atol=1e-6)
        ll_prev = ll
    end
end

function test_GMM_vector()
    # Initialize data
    data = randn(1000,)
    k_means = 2
    # Initialize GMM
    gmm = GMM(k_means, 1, data)
    # Run estep
    SSM.EStep!(gmm, data)
    # Run mstep
    SSM.MStep!(gmm, data)
    # Check if the mixing coefficients sum to 1
    @test sum(gmm.π_k) ≈ 1.0
    # Check if the class_probabilities add to 1
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(gmm.class_probabilities, dims=2))
    # Check if the class labels are integers in the range 1 to k_means
    @test all(x -> x in 1:k_means, gmm.class_labels)
    # Now Run
    fit!(gmm, data; maxiter=10, tol=1e-3)
    # Check if the mixing coefficients sum to 1
    @test sum(gmm.π_k) ≈ 1.0
    # Check if the class_probabilities add to 1
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(gmm.class_probabilities, dims=2))
    # Check if the class labels are integers in the range 1 to k_means
    @test all(x -> x in 1:k_means, gmm.class_labels)
end

@testset "MixtureModels.jl Tests" begin
    test_GMM_constructor()
    testGMM_EStep()
    testGMM_MStep()
    testGMM_fit()
    test_log_likelihood()
    test_GMM_vector()
end


"""
Tests for HiddenMarkovModels.jl
"""
#TODO: Implement tests for HMMs
function test_HMM_constructor()
    k = 3
    data_dim = 2
    # generate random data
    data = randn(100, data_dim)
    # initialize HMM
    hmm = HMM(data, k, "Gaussian")
    # check if parameters are initialized correctly
    println(hmm.A)
    @test isapprox(sum(hmm.A, dims=2), ones(k))
    @test typeof(hmm.B) == Vector{GaussianEmission}
    @test sum(hmm.πₖ) ≈ 1.0
    @test hmm.D == data_dim
end

function test_HMM_forward()
    # Initialize parameters
    k = 3
    data_dim = 2
    data = randn(100, data_dim)
    hmm = HMM(data, k, "Gaussian")
    # Run forward algorithm
    α = SSM.forward(hmm, data)
    # Check dimensions
    @test size(α) == (k, 100)
end

function test_HMM_backward()
    # Initialize parameters
    k = 3
    data_dim = 2
    data = randn(100, data_dim)
    hmm = HMM(data, k, "Gaussian")
    # Run backward algorithm
    β = SSM.backward(hmm, data)
    # Check dimensions
    @test size(β) == (k, 100)
end

#TODO: Add tests for gamma, xi, estep, and mstep

function test_HMM_EM()
    A = [0.7 0.2 0.1; 0.1 0.7 0.2; 0.2 0.1 0.7]
    means = [[0.0, 0.0], [-1.0, 2.0], [3.0, 2.5]]
    covs = [
                [0.1 0.0; 0.0 0.1],  # Covariance matrix for state 1
                [0.1 0.0; 0.0 0.1],  # Covariance matrix for state 2
                [0.1 0.0; 0.0 0.1]   # Covariance matrix for state 3
            ]
    emissions_models = [GaussianEmission(mean, cov) for (mean, cov) in zip(means, covs)]
    simul_hmm = HMM(A, emissions_models, [0.33, 0.33, 0.34], 2)
    states, observations = SSM.sample(simul_hmm, 10000)
    # Initialize HMM
    k = 3
    data_dim = 2
    hmm = HMM(observations, k, "Gaussian")
    # Run EM
    baumWelch!(hmm, observations)
    # Check if the transition matrix is close to the simulated one
    @test hmm.A ≈ A atol=1e-1
    # Check if the means are close to the simulated ones
    pred_means = [hmm.B[i].μ for i in 1:k]
    @test sort(pred_means) ≈ sort(means) atol=1e-1
    # Check if the covariance matrices are close to the simulated ones
    pred_covs = [hmm.B[i].Σ for i in 1:k]
    @test pred_covs ≈ covs atol=1e-1
    # Check viterbi now
    best_path = viterbi(hmm, observations)
    @test length(best_path) == 10000
    @test all(x -> x in 1:k, best_path)
end

@testset "HiddenMarkovModels.jl Tests" begin
    test_HMM_constructor()
    test_HMM_forward()
    test_HMM_backward()
    test_HMM_EM()
end

"""
Tests for LDS.jl
"""
#TODO: Implement tests for LDS

# Create a toy example for all LDS tests. This example represents a pendulum in a frictionless environment.
g = 9.81 # gravity
l = 1.0 # length of pendulum
dt = 0.01 # time step
T = 10.0 # total time
# Discrete-time dynamics
A = [1.0 dt; -g/l*dt 1.0]
# Initial state
x0 = [0.0; 1.0]
# Time vector
t = 0:dt:T
# Define the LDS model parameters
H = I(2)  # Observation matrix (assuming direct observation)
Q = 0.00001 * I(2)  # Process noise covariance
observation_noise_std = 0.5
R = (observation_noise_std^2) * I(2)  # Observation noise covariance
P0 = 0.1*I(2)  # Initial state covariance
x0 = [0.0; 1.0]  # Initial state mean
# Generate true data
x = zeros(2, length(t))
x[:,1] = x0
for i = 2:length(t)
    x[:,i] = A*x[:,i-1]
end
# Generate noisy data
x_noisy = zeros(2, length(t))
x_noisy[:, 1] = x0

noise = rand(Normal(0, observation_noise_std), (2, length(t)))

for i in 2:length(t)
    x_noisy[:, i] = A * x[:, i-1] + noise[:, i]
end

function test_LDS_with_params()
    # Create the Kalman filter parameter vector
    kf = LDS(A,
             H,
             nothing,
             Q, 
             R, 
             x0, 
             P0, 
             nothing, 
             2, 
             2, 
             "Gaussian", 
             Vector([false, false, false, false, false, false, false, false]))
    # confirm parameters are set correctly
    @test kf.A == A
    @test kf.H == H
    @test kf.B === nothing
    @test kf.Q == Q
    @test kf.R == R
    @test kf.x0 == x0
    @test kf.P0 == P0
    @test kf.inputs === nothing
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.emissions == "Gaussian"
    @test kf.fit_bool == Vector([false, false, false, false, false, false, false, false])
    # get the likelihood of the model
    ll = SSM.loglikelihood(kf, x_noisy')
    # check if the likelihood is a scalar
    @test size(ll) == ()
    # check if the likelihood is negative
    @test ll < 0.0
    # run the filter
    x_pred, P, v, F, K = KalmanFilter(kf, x_noisy')
    # check dimensions
    @test size(x_pred) == (length(t), 2)
    @test size(P) == (length(t), 2, 2)
    @test size(v) == (length(t), 2)
    @test size(F) == (length(t), 2, 2)
    @test size(K) == (length(t), 2, 2)
    # run the smoother
    x_smooth, p_smooth = KalmanSmoother(kf, x_noisy')
    # check dimensions
    @test size(x_smooth) == (length(t), 2)
    @test size(p_smooth) == (length(t), 2, 2)
end

function test_LDS_without_params()
    # Create the Kalman filter without any params
    kf = LDS()
    # confirm parameters are set correctly
    @test kf.A !== nothing
    @test kf.H !== nothing
    @test kf.B === nothing
    @test kf.Q !== nothing
    @test kf.R !== nothing
    @test kf.x0 !== nothing
    @test kf.P0 !== nothing
    @test kf.inputs === nothing
    @test kf.obs_dim == 1
    @test kf.latent_dim == 1
    @test kf.emissions == "Gaussian"
    @test kf.fit_bool == Vector([true, true, true, true, true, true, true, true])
end

@testset "LDS.jl Tests" begin
    test_LDS_with_params()
    # test_LDS_without_params()
end

"""
Tests for Regression.jl
"""
# sigmoid function
function sigmoid(x)
    return 1 / (1 + exp(-x))
end

# Generate toy data for Regression Tests
n = 10000
p = 2
X = randn(n, p)
X_concat = hcat(ones(n), X)
β = [1.0, 2.0, 3.0]
y_gaussian = X_concat * β .+ randn(n)
y_poisson = Vector{Float64}(rand.(Poisson.(exp.(X_concat * β))))
y_binomial = Vector{Float64}(rand.(Binomial.(1, sigmoid.(X_concat * β))))

function test_link_functions()
    values = [0.0, 0.5, 1.0]
    @test SSM.link(IdentityLink(), values) == values
    @test SSM.invlink(IdentityLink(), values) == values
    @test SSM.derivlink(IdentityLink(), values) == ones(3)
    @test SSM.link(LogLink(), values) == log.(values)
    @test SSM.invlink(LogLink(), values) == exp.(values)
    @test SSM.derivlink(LogLink(), values) == 1 ./ values
    @test SSM.link(LogitLink(), values) == log.(values ./ (1 .- values))
    @test SSM.invlink(LogitLink(), values) == 1 ./ (1 .+ exp.(-values))
    @test SSM.derivlink(LogitLink(), values) == 1 ./ (values .* (1 .- values))
end

function test_link_functions_edge_cases()
    # Test LogitLink with edge values
    @test_throws DomainError SSM.link(LogitLink(), [-1.0, 2.0])
    @test SSM.invlink(LogitLink(), [-Inf, Inf]) == [0.0, 1.0]
    @test SSM.derivlink(LogitLink(), [0.0, 1.0]) == [Inf, Inf]

    # Test LogLink with edge values
    @test_throws DomainError SSM.link(LogLink(), [-1.0, 0.0])
    @test SSM.invlink(LogLink(), [-Inf, 0.0]) == [0.0, 1.0]
    @test SSM.derivlink(LogLink(), [0.0, 1.0]) == [Inf, 1.0]

    # Test IdentityLink with edge values
    @test SSM.link(IdentityLink(), [-Inf, Inf]) == [-Inf, Inf]
    @test SSM.invlink(IdentityLink(), [-Inf, Inf]) == [-Inf, Inf]
    @test SSM.derivlink(IdentityLink(), [-Inf, Inf]) == [1.0, 1.0]
 
end

function test_Gaussian_regression()
    # Initialize Gaussian Regression Model
    gaussian_reg = GaussianRegression(X, y_gaussian)
    # Check if parameters are initialized correctly
    @test gaussian_reg.X == hcat(ones(n), X)
    @test gaussian_reg.y == y_gaussian
    @test gaussian_reg.β == zeros(p + 1)
    @test gaussian_reg.link == IdentityLink()
    # Fit the model
    fit!(gaussian_reg)
    # Check if parameters are updated correctly
    @test isapprox(gaussian_reg.β, β, atol=1e-1)
    # Check edge cases
    # @test_throws ArgumentError GaussianRegression(Float64[], Float64[])
end

function test_Poisson_regression()
    # Initiliaze Poisson Regression Model
    poisson_reg = PoissonRegression(X, y_poisson)
    # Check if parameters are initialized correctly
    @test poisson_reg.X == hcat(ones(n), X)
    @test poisson_reg.y == y_poisson
    @test poisson_reg.β == zeros(p + 1)
    @test poisson_reg.link == LogLink()
    # Fit the model
    fit!(poisson_reg)
    # Check if parameters are updated correctly
    @test isapprox(poisson_reg.β, β, atol=1e-1)
    # Check edge cases
    # @test_throws ArgumentError PoissonRegression(X, -abs.(y_poisson))
end

function test_Binomial_Regression()
    # Initiliaze Logistic Regression Model
    logistic_reg = BinomialRegression(X, y_binomial)
    # Check if parameters are initialized correctly
    @test logistic_reg.X == hcat(ones(n), X)
    @test logistic_reg.y == y_binomial
    @test logistic_reg.β == zeros(p + 1)
    @test logistic_reg.link == LogitLink()
    # Fit the model
    fit!(logistic_reg)
    # Check if parameters are updated correctly
    @test isapprox(logistic_reg.β, β, atol=1e-1)
    # Check edge cases
    # @test_throws ArgumentError BinomialRegression(X, 2 .* y_binomial)
end

function test_WLS_loss()
    # Test data
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    weights = [1.0, 2.0, 0.5, 1.0]
    # Expected loss calculation
    expected_loss = sum(weights .* (y_true - y_pred).^2)
    # Create a WLSLoss instance
    loss = SSM.WLSLoss(weights)
    # Compute the loss
    computed_loss = SSM.compute_loss(loss, y_pred, y_true)
    # Test for correctness
    @test isapprox(computed_loss, expected_loss, atol=1e-6)
    # Test with empty vectors
    @test SSM.compute_loss(SSM.WLSLoss(Float64[]), Float64[], Float64[]) == 0.0
    # Test with zero weights
    @test SSM.compute_loss(SSM.WLSLoss(zeros(length(y_true))), y_pred, y_true) == 0.0
    # Test with large values/weights
    large_values = [1e6, -1e6, 1e6, -1e6]
    large_weights = [1e6, 1e6, 1e6, 1e6]
    expected_large_loss = sum(large_weights .* (large_values - y_pred).^2)
    @test SSM.compute_loss(SSM.WLSLoss(large_weights), y_pred, large_values) ≈ expected_large_loss atol=1e-3
end

function test_LSE_loss()
    # Test data
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    # Expected loss calculation
    expected_loss = sum((y_true - y_pred).^2)
    # Create a LSELoss instance
    loss = SSM.LSELoss()
    # Compute the loss
    computed_loss = SSM.compute_loss(loss, y_pred, y_true)
    # Test for correctness
    @test isapprox(computed_loss, expected_loss, atol=1e-6)
    # Test with empty vectors
    @test SSM.compute_loss(SSM.LSELoss(), Float64[], Float64[]) == 0.0
    # Test with large values/weights
    large_values = [1e6, -1e6, 1e6, -1e6]
    expected_large_loss = sum((large_values - y_pred).^2)
    @test SSM.compute_loss(SSM.LSELoss(), y_pred, large_values) ≈ expected_large_loss atol=1e-3
end

function test_CrossEntropyLoss()
    # Test data
    y_true = [1.0, 0.0, 1.0, 1.0]
    y_pred = [0.9, 0.1, 0.8, 0.9]
    # Expected loss calculation
    expected_loss = -sum(y_true .* log.(y_pred) .+ (1 .- y_true) .* log.(1 .- y_pred))
    # Create a CrossEntropyLoss instance
    loss = SSM.CrossEntropyLoss()
    # Compute the loss
    computed_loss = SSM.compute_loss(loss, y_pred, y_true)
    # Test for correctness
    @test isapprox(computed_loss, expected_loss, atol=1e-6)
    # Test with empty vectors
    @test SSM.compute_loss(SSM.CrossEntropyLoss(), Float64[], Float64[]) == 0.0
    # Test with large values/weights
    large_values = [1e6, -1e6, 1e6, -1e6]
    expected_large_loss = -sum(large_values .* log.(y_pred) .+ (1 .- large_values) .* log.(1 .- y_pred))
    @test SSM.compute_loss(SSM.CrossEntropyLoss(), y_pred, large_values) ≈ expected_large_loss atol=1e-3
end

function test_PoissonLoss()
    # EPSILON
    EPSILON = 1e-15
    # Test data
    y_true = [1.0, 0.0, 1.0, 1.0]
    y_pred = [0.9, 0.1, 0.8, 0.9]
    # Corrected expected loss calculation
    expected_loss = sum(-y_true .* log.(y_pred .+ EPSILON) + y_pred)  # Added EPSILON for stability
    # Create a PoissonLoss instance
    loss = SSM.PoissonLoss()
    # Compute the loss
    computed_loss = SSM.compute_loss(loss, y_pred, y_true)
    # Test for correctness
    @test isapprox(computed_loss, expected_loss, atol=1e-6)
    # Test with empty vectors
    @test SSM.compute_loss(SSM.PoissonLoss(), Float64[], Float64[]) == 0.0
    # Test with large values
    large_values = [1e6, 1e6, 1e6, 1e6]
    # TODO: Fix this test; need better handling of large values.
    # expected_large_loss = sum(-y_true .* log.(y_pred .+ EPSILON) + large_values)
    # @test SSM.compute_loss(SSM.PoissonLoss(), y_pred, large_values) ≈ expected_large_loss atol=1e-3
end

function test_predictions()
    # Initialize Gaussian Regression Model
    gaussian_reg = GaussianRegression(X, y_gaussian)
    # Fit the model
    fit!(gaussian_reg)
    # Check if predictions are correct
    @test isapprox(predict(gaussian_reg, gaussian_reg.X), gaussian_reg.X * gaussian_reg.β, atol=1e-1)
    # Initialize Poisson Regression Model
    poisson_reg = PoissonRegression(X, y_poisson)
    # Fit the model
    fit!(poisson_reg)
    # Check if predictions are correct
    @test isapprox(predict(poisson_reg, poisson_reg.X), exp.(poisson_reg.X * poisson_reg.β), atol=1e-1)
    # Initialize Logistic Regression Model
    logistic_reg = BinomialRegression(X, y_binomial)
    # Fit the model
    fit!(logistic_reg)
    # Check if predictions are correct
    @test isapprox(predict(logistic_reg, logistic_reg.X), sigmoid.(logistic_reg.X * logistic_reg.β), atol=1e-1)
    # Test predictions on a vector
    # TODO: Fix this test; need to handle vector data; add multiple dispatch for link functions.
    # @test isapprox(predict(gaussian_reg, gaussian_reg.X[1,:]),  gaussian_reg.β * gaussian_reg.X[1,:]', atol=1e-1)
    # @test isapprox(predict(poisson_reg, poisson_reg.X[1,:]), exp.( poisson_reg.β * poisson_reg.X[1,:]'), atol=1e-1)
    # @test isapprox(predict(logistic_reg, logistic_reg.X[1,:]), sigmoid.(logistic_reg.β * logistic_reg.X[1,:]'), atol=1e-1)
end

function test_residuals()
    # Initialize Gaussian Regression Model
    gaussian_reg = GaussianRegression(X, y_gaussian)
    # Fit the model
    fit!(gaussian_reg)
    # Check if residuals are correct
    @test isapprox(residuals(gaussian_reg), y_gaussian - gaussian_reg.X * gaussian_reg.β, atol=1e-1)
    # Initialize Poisson Regression Model
    poisson_reg = PoissonRegression(X, y_poisson)
    # Fit the model
    fit!(poisson_reg)
    # Check if residuals are correct
    @test isapprox(residuals(poisson_reg), y_poisson - exp.(poisson_reg.X * poisson_reg.β), atol=1e-1)
    # Initialize Logistic Regression Model
    logistic_reg = BinomialRegression(X, y_binomial)
    # Fit the model
    fit!(logistic_reg)
    # Check if residuals are correct
    @test isapprox(residuals(logistic_reg), y_binomial - sigmoid.(logistic_reg.X * logistic_reg.β), atol=1e-1)
end


@testset "Regression.jl Tests" begin
    test_link_functions()
    test_link_functions_edge_cases()
    test_Gaussian_regression()
    test_Poisson_regression()
    test_Binomial_Regression()
    test_WLS_loss()
    test_LSE_loss()
    test_CrossEntropyLoss()
    test_PoissonLoss()
    test_predictions()
    test_residuals()
end

"""
Tests for Emissions.jl
"""

function test_GaussianEmission()
    # Initialize Gaussian Emission Model
    gaussian_emission = GaussianEmission([0.0, 0.0], [1.0 0.0; 0.0 1.0])
    # Check if parameters are initialized correctly
    @test gaussian_emission.μ == [0.0, 0.0]
    @test gaussian_emission.Σ == [1.0 0.0; 0.0 1.0]
    # Generate random data
    data = randn(100, 2)
    # Calculate log-likelihood
    ll = SSM.loglikelihood(gaussian_emission, data[1, :])
    # Check if log-likelihood is a scalar
    @test size(ll) == ()
    # Log-likelihood should be a negative float
    @test ll < 0.0
    # Check sample emission
    sample = SSM.sample_emission(gaussian_emission)
    @test length(sample) == 2
    # Update emission model
    γ = rand(100)
    SSM.updateEmissionModel!(gaussian_emission, data, γ)
    # Check if parameters are updated correctly
    
end

function test_regression_emissions()
    # Initialize a GLM
    glm = GaussianRegression(X, y_gaussian)
    # Initialize Regression Emission Model
    regression_emission = RegressionEmissions(glm)
    # Check if parameters are initialized correctly
    @test regression_emission.regression_model.X == hcat(ones(n), X)
    @test regression_emission.regression_model.y == y_gaussian
    @test regression_emission.regression_model.β == zeros(p + 1)
    @test regression_emission.regression_model.link == IdentityLink()
    # Fit the model
    fit!(regression_emission.regression_model)
    # Check if parameters are updated correctly
    @test isapprox(regression_emission.regression_model.β, β, atol=1e-1)
end

@testset "Emissions.jl Tests" begin
    test_GaussianEmission()
    test_regression_emissions()
end


"""
Tests for Utilities.jl
"""

function test_euclidean_distance()
    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]
    @test SSM.euclidean_distance(x, y) == sqrt(27.0)
end

function test_kmeanspp_initialization()
    # Generate random data
    data = randn(100, 2)
    # Initialize centroids
    k_means = 3
    centroids = kmeanspp_initialization(data, k_means)
    # Check dimensions
    @test size(centroids) == (2, k_means)
end

function test_kmeans_clustering()
    # Generate random data
    data = randn(100, 2)
    # Initialize centroids
    k_means = 3
    centroids, labels = kmeans_clustering(data, k_means)
    # Check dimensions
    @test size(centroids) == (2, k_means)
    @test length(labels) == 100
end

@testset "Utilities.jl Tests" begin
    test_euclidean_distance()
    test_kmeanspp_initialization()
    test_kmeans_clustering()
end
