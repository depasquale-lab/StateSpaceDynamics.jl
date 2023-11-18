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
end

function testGMM_EStep()
    # Initialize parameters
    k_means = 3
    data_dim = 2
    data = randn(10, data_dim)
    gmm = GMM(k_means, data_dim, data)
    # Run EStep
    γ = SSM.EStep!(gmm, data)
    # Check dimensions
    @test size(γ) == (10, k_means)
    # Check if the row sums are close to 1 (since they represent probabilities)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(γ, dims=2))
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

@testset "MixtureModels.jl Tests" begin
    test_GMM_constructor()
    testGMM_EStep()
    testGMM_MStep()
    testGMM_fit()
    test_log_likelihood()
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

function test_HMM_EM()
    A = [0.9 0.02 0.08; 0.1 0.9 0.0; 0.0 0.1 0.9]
    means = [[0.0, 0.0], [3.0, 2.5], [-1.0, 2.0]]
    covs = [
                [0.1 0.0; 0.0 0.1],  # Covariance matrix for state 1
                [0.1 0.0; 0.0 0.1],  # Covariance matrix for state 2
                [0.1 0.0; 0.0 0.1]   # Covariance matrix for state 3
            ]
    emissions_models = [GaussianEmission(mean, cov) for (mean, cov) in zip(means, covs)]
    simul_hmm = HMM(A, emissions_models, [0.33, 0.33, 0.34], 2)
    states, observations = SSM.sample(simul_hmm, 1000)
    # Initialize HMM
    k = 3
    data_dim = 2
    hmm = HMM(observations, k, "Gaussian")
    # Run EM
    # TODO: Finish later.
end

@testset "HiddenMarkovModels.jl Tests" begin
    test_HMM_constructor()
    test_HMM_forward()
    test_HMM_backward()
end



"""
Tests for LDS.jl
"""
#TODO: Implement tests for LDS

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
end

@testset "Regression.jl Tests" begin
    test_link_functions()
    test_Gaussian_regression()
    test_Poisson_regression()
    test_Binomial_Regression()
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
