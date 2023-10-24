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
    γ = SSM.MixtureModels.EStep!(gmm, data)
    # Check dimensions
    @test size(γ) == (10, k_means)
    # Check if the row sums are close to 1 (since they represent probabilities)
    @test all(isapprox(sum(γ, dims=2), 1.0; atol=1e-6))
end

function testGMM_MStep()
    # Initialize parameters
    k_means = 3
    data_dim = 2
    data = randn(10, data_dim)
    gmm = GMM(k_means, data_dim, data)
    γ = SSM.MixtureModels.EStep!(gmm, data)

    # Run MStep
    SSM.MixtureModels.MStep!(gmm, data, γ)

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

