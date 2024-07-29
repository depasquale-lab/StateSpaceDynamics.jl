using SSM
using Distributions
using ForwardDiff
using LinearAlgebra
using Random
using StatsFuns
using SpecialFunctions
using Test

Random.seed!(1234)

"""
Tests for MixtureModels.jl
"""

include("MixtureModels/GaussianMixtureModel.jl")
include("MixtureModels/PoissonMixtureModel.jl")

@testset "MixtureModels.jl Tests" begin
    # Test GaussianMixtureModel

    
    # Initialize test models


    # Standard GaussianMixtureModel model

    # Number of clusters
    k = 3
    # Dimension of data points
    data_dim = 2
    # Construct gmm
    standard_gmm = GaussianMixtureModel(k, data_dim)
    # Generate sample data
    standard_data = randn(10, data_dim)

    # Test constructor method of GaussianMixtureModel
    test_GaussianMixtureModel_properties(standard_gmm, k, data_dim)



    # Vector-data GaussianMixtureModel model

    # Number of clusters
    k = 2
    # Dimension of data points
    data_dim = 1
    # Construct gmm
    vector_gmm = GaussianMixtureModel(k, data_dim)
    # Generate sample data
    vector_data = randn(1000,)
    # Test constructor method of GaussianMixtureModel
    test_GaussianMixtureModel_properties(vector_gmm, k, data_dim)
  
    # Test EM methods of the GaussianMixtureModels

    # Paired data and GaussianMixtureModels to test
    tester_set = [
        (standard_gmm, standard_data), 
        (vector_gmm, vector_data),
        ]

    for (gmm, data) in tester_set
        k = gmm.k
        data_dim = size(data, 2)

        gmm = GaussianMixtureModel(k, data_dim)
        testGaussianMixtureModel_EStep(gmm, data)

        gmm = GaussianMixtureModel(k, data_dim)
        testGaussianMixtureModel_MStep(gmm, data)

        gmm = GaussianMixtureModel(k, data_dim)
        testGaussianMixtureModel_fit(gmm, data)

        gmm = GaussianMixtureModel(k, data_dim)
        test_log_likelihood(gmm, data)
    end
  
    # Test PoissonMixtureModel
    k = 3  # Number of clusters
    
    # Simulate some Poisson-distributed data using the sample function
    # First, define a temporary PMM for sampling purposes
    temp_pmm = PoissonMixtureModel(k)
    temp_pmm.λₖ = [5.0, 10.0, 15.0]  # Assign some λ values for generating data
    temp_pmm.πₖ = [1/3, 1/3, 1/3]  # Equal mixing coefficients for simplicity
    data = SSM.sample(temp_pmm, 300)  # Generate sample data
    
    standard_pmm = PoissonMixtureModel(k)
    
    # Conduct tests
    test_PoissonMixtureModel_properties(standard_pmm, k)
    
    tester_set = [(standard_pmm, data)]
    
    for (pmm, data) in tester_set
        pmm = PoissonMixtureModel(k)
        testPoissonMixtureModel_EStep(pmm, data)
        pmm = PoissonMixtureModel(k)
        testPoissonMixtureModel_MStep(pmm, data)
        pmm = PoissonMixtureModel(k)
        testPoissonMixtureModel_fit(pmm, data)
        pmm = PoissonMixtureModel(k)
        test_log_likelihood(pmm, data)
    end
end

"""
Tests for HiddenMarkovModels.jl
"""

include("HiddenMarkovModels/HiddenMarkovModels.jl")

@testset "HiddenMarkovModels.jl Tests" begin
    test_toy_HMM()
    test_GaussianHMM_constructor()
    test_HMM_forward_and_back()
    test_HMM_gamma_xi()
    test_HMM_E_step()
    test_HMM_M_step()
    test_HMM_EM()
end

"""
Tests for LDS.jl
"""

include("LDS/LDS.jl")

@testset "LDS.jl Tests" begin
    test_LDS_with_params()
    test_LDS_without_params()
    test_LDS_E_Step()
    test_LDS_M_Step!()
    test_LDS_EM()
    test_direct_smoother()
    test_LDS_gradient()
    test_LDS_Hessian()
end

"""
Tests for Regression.jl
""" 

include("Regression/GaussianRegression.jl")

@testset "GaussianRegression Tests" begin
    test_GaussianRegression_fit()
    test_GaussianRegression_loglikelihood()
    test_GaussianRegression_default_model()
    test_GaussianRegression_intercept()
    test_Gaussian_ll_gradient()
end

include("Regression/BernoulliRegression.jl")

@testset "BernoulliRegression Tests" begin
    test_BernoulliRegression_fit()
    test_BernoulliRegression_loglikelihood()
    test_BernoulliRegression_empty_model()
    test_BernoulliRegression_intercept()
    test_Bernoulli_ll_gradient()
end

include("Regression/PoissonRegression.jl")

@testset "PoissonRegression Tests" begin
    test_PoissonRegression_fit()
    test_PoissonRegression_loglikelihood()
    test_PoissonRegression_empty_model()
    test_PoissonRegression_intercept()
    test_Poisson_ll_gradient()
end

"""
Tests for Emissions.jl
"""

include("Emissions/Emissions.jl")

@testset "Emissions.jl Tests" begin
    test_GaussianEmission()
    test_regression_emissions()
end

"""
Tests for Utilities.jl
"""

include("Utilities/Utilities.jl")

@testset "Utilities.jl Tests" begin
    test_euclidean_distance()
    test_kmeanspp_initialization()
    test_kmeans_clustering()
    test_block_tridgm()
    test_interleave_reshape()
end

"""
Tests for Preprocessing.jl
"""

function test_PPCA_with_params()
    # Set parameters
    W = randn(3, 2)
    σ² = 0.5
    # create "data"
    X = randn(100, 3)
    μ = mean(X, dims=1)
    # create PPCA object
    ppca = ProbabilisticPCA(W, σ², μ, 2, 3, Matrix{Float64}(undef, 0, 0))
    # Check if parameters are set correctly
    @test ppca.W === W
    @test ppca.σ² === σ²
    @test ppca.μ === μ
    @test ppca.D === 3
    @test ppca.K === 2
    @test isempty(ppca.z)
end

function test_PPCA_without_params()
    # create ppca object
    ppca = ProbabilisticPCA(;K=2, D=3)
    # Check if parameters are set correctly
    @test size(ppca.W) == (3, 2)
    @test ppca.σ² > 0
    @test isempty(ppca.μ)
    @test ppca.D == 3
    @test ppca.K == 2
    @test isempty(ppca.z)
end

function test_PPCA_E_and_M_Step()
    # create ppca object
    ppca = ProbabilisticPCA(;K=2, D=3)
    # create data
    X = randn(100, 3)
    # assign μ, normally fit! does this
    μ = mean(X, dims=1)
    ppca.μ = μ
    # run E-step
    E_z, E_zz = SSM.E_Step(ppca, X)
    # check dimensions
    @test size(E_z) == (100, 2)
    @test size(E_zz) == (100, 2, 2)
    # run M-step, but first save the old parameters
    W_old = ppca.W
    σ²_old = ppca.σ²
    SSM.M_Step!(ppca, X, E_z, E_zz)
    # check if the parameters are updated
    @test ppca.W !== W_old
    @test ppca.σ² !== σ²_old
    @test ppca.μ === μ
end

function test_PPCA_fit()
    # create ppca object
    ppca = ProbabilisticPCA(;K=2, D=3)
    # create data
    X = randn(100, 3)
    # fit the model
    ll = fit!(ppca, X)
    # check if the parameters are updated
    @test ppca.σ² > 0
    @test size(ppca.W) == (3, 2)
    @test size(ppca.μ) == (1, 3)
    @test size(ppca.z) == (100, 2)
    # check loglikelihood only increases
    @test all(diff(ll) .> 0)
    # also check that the loglikelihood is a scalar
    ll = SSM.loglikelihood(ppca, X)
    @test size(ll) == ()
end

@testset "PPCA Tests" begin
    test_PPCA_with_params()
    test_PPCA_without_params()
    test_PPCA_E_and_M_Step()
    test_PPCA_fit()
end

"""
Tests for MarkovRegression.jl
"""

include("MarkovRegression/MarkovRegression.jl")

@testset "SwitchingRegression Tests" begin
    test_HMMGLM_initialization()
end