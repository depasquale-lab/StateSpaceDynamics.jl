using StateSpaceDynamics
using Distributions
using ForwardDiff
using LinearAlgebra
using Random
using StatsFuns
using SpecialFunctions
using Test
using Base.Iterators: flatten
using Documenter

Random.seed!(1234)

include("helper_functions.jl")

"""
Tests for HiddenMarkovModels.jl
"""

include("HiddenMarkovModels/HiddenMarkovModels.jl")

@testset "HiddenMarkovModels.jl Tests" begin
    test_HiddenMarkovModel_E_step()
    #test_viterbi() -- don't have time to finish, need emission matching logic
    test_GaussianHMM()
    # test_AutoRegressionHMM()
    test_trialized_GaussianHMM()
    test_trialized_SwitchingGaussianRegression()
end

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
    data = StateSpaceDynamics.sample(temp_pmm, 300)  # Generate sample data
    
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
Tests for RegressionModels.jl
""" 

include("RegressionModels/GaussianRegression.jl")

@testset "GaussianRegression Tests" begin
    test_GaussianRegression_initialization()
    test_GaussianRegression_loglikelihood()
    test_GaussianRegression_fit()
    test_GaussianRegression_sample()
    test_GaussianRegression_optimization()
    test_GaussianRegression_regularization()
end

include("RegressionModels/BernoulliRegression.jl")

@testset "BernoulliRegression Tests" begin
    test_BernoulliRegression_initialization()
    test_BernoulliRegression_loglikelihood()
    test_BernoulliRegression_fit()
    test_BernoulliRegression_sample()
    test_BernoulliRegression_optimization()
    test_BernoulliRegression_regularization()
end

include("RegressionModels/PoissonRegression.jl")

@testset "PoissonRegression Tests" begin
    test_PoissonRegression_initialization()
    test_PoissonRegression_loglikelihood()
    test_PoissonRegression_fit()
    test_PoissonRegression_sample()
    test_PoissonRegression_optimization()
    test_PoissonRegression_regularization()
    test_PoissonRegression_regularization()
end

include("RegressionModels/AutoRegression.jl")

@testset "AutoRegression Tests" begin
    test_AutoRegression_loglikelihood()
    # test_AutoRegression_Σ()
    # test_AutoRegression_constructor()
    test_AutoRegression_standard_fit()
    test_AutoRegression_regularized_fit()
end

"""
Tests for Emissions.jl
"""

# include("Emissions/Emissions.jl")

# @testset "Emissions.jl Tests" begin
#     test_GaussianEmission()
#     test_regression_emissions()
# end

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

include("Preprocessing/Preprocessing.jl")

@testset "PPCA Tests" begin
    test_PPCA_with_params()
    test_PPCA_without_params()
    test_PPCA_E_and_M_Step()
    test_PPCA_fit()
end