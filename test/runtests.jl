using SSM
using LinearAlgebra
using Test

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
#TODO Implement tests for Regression


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
