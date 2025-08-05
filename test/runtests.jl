using StateSpaceDynamics
using Distributions
using ForwardDiff
using LinearAlgebra
using Optim
using Random
using StatsFuns
using SpecialFunctions
using Test
using Aqua
using JET
using CSV
using DataFrames
using MAT

"""
Package Wide Tests
"""

@testset "Package Wide Tests" begin
    Aqua.test_all(StateSpaceDynamics; ambiguities=false)
    @test isempty(Test.detect_ambiguities(StateSpaceDynamics))
end

@testset "Code linting using JET " begin
    if VERSION >= v"1.11"
        JET.test_package(StateSpaceDynamics; target_defined_modules=true)
    end
end

include("helper_functions.jl")
"""
Tests for SLDS.jl
"""

include("LinearDynamicalSystems//SLDS.jl")

@testset "SLDS Tests" begin
    @testset "Constructor Tests" begin
        test_init()
        test_sample()
    end

    @testset "vEM Tests" begin
        test_vEstep()
    end
end

"""
Tests for LDS.jl
"""

include("LinearDynamicalSystems//GaussianLDS.jl")

@testset "GaussianLDS Tests" begin
    @testset "Constructor Tests" begin
        test_lds_with_params()
        test_gaussian_obs_constructor_type_preservation()
        test_gaussian_lds_constructor_type_preservation()
        test_gaussian_sample_type_preservation()
        test_gaussian_fit_type_preservation()
        test_gaussian_loglikelihood_type_preservation()
    end
    @testset "Smoother tests" begin
        test_Gradient()
        test_Hessian()
        test_smooth()
    end
    @testset "EM tests" begin
        test_estep()
        # test when ntrials=1
        test_initial_observation_parameter_updates()
        test_state_model_parameter_updates()
        test_obs_model_params_updates()
        # test when ntrials>1
        test_initial_observation_parameter_updates(3)
        test_state_model_parameter_updates(3)
        test_obs_model_params_updates(3)
        # test fit method using n=1 and n=3
        test_EM()
        test_EM(3)
    end
end

"""
Tests for PoissonLDS.jl
"""

include("LinearDynamicalSystems//PoissonLDS.jl")

@testset "PoissonLDS Tests" begin
    @testset "Constructor Tests" begin
        test_PoissonLDS_with_params()
        test_pobs_constructor_type_preservation()
        test_plds_constructor_type_preservation()
        test_poisson_sample_type_preservation()
        test_poisson_fit_type_preservation()
        test_poisson_loglikelihood_type_preservation()
    end
    @testset "Smoother Tests" begin
        test_Gradient()
        test_Hessian()
        test_smooth()
    end
    @testset "EM Tests" begin
        test_parameter_gradient()
        # test when ntrials=1
        test_initial_observation_parameter_updates()
        test_state_model_parameter_updates()
        # test when n_trials>1
        test_initial_observation_parameter_updates(3)
        test_state_model_parameter_updates(3)
        # test fit method using 1 trial and three trials
        test_EM()
        test_EM(3)
        # test resutlts are same as matlab code
        test_EM_matlab()
    end
end

"""
Tests for Switching Regression Models
"""

include("HiddenMarkovModels/GaussianHMM.jl")

@testset "GaussianHMM Tests" begin
    test_SwitchingGaussian_fit()
    test_SwitchingGaussian_SingleState_fit()
    test_kmeans_init()
    test_trialized_GaussianHMM()
    test_SwitchingGaussian_fit_float32()
end

include("HiddenMarkovModels/SwitchingGaussianRegression.jl")

@testset "Switching Gaussian Regression Tests" begin
    test_SwitchingGaussianRegression_fit()
    test_SwitchingGaussianRegression_SingleState_fit()
    test_trialized_SwitchingGaussianRegression()
    # test_SwitchingGaussianRegression_fit_float32()
end

include("HiddenMarkovModels/SwitchingPoissonRegression.jl")

@testset "Switching Poisson Regression Tests" begin
    test_SwitchingPoissonRegression_fit()
    test_trialized_SwitchingPoissonRegression()
    # test_SwitchingPoissonRegression_fit_float32()
end

include("HiddenMarkovModels/SwitchingBernoulliRegression.jl")

@testset "Switching Bernoulli Regression Tests" begin
    test_SwitchingBernoulliRegression()
    test_trialized_SwitchingBernoulliRegression()
end

"""
Tests for MixtureModels.jl
"""

include("MixtureModels/GaussianMixtureModel.jl")
include("MixtureModels/PoissonMixtureModel.jl")

@testset "MixtureModels.jl Tests" begin
    # GaussianMixtureModel Tests
    k = 3
    D = 2  # feature dimension
    standard_gmm = GaussianMixtureModel(k, D)
    standard_data = rand(standard_gmm, 100)  # 100 samples, 2D each
    test_GaussianMixtureModel_properties(standard_gmm, k, D)

    k = 2
    D = 1
    vector_gmm = GaussianMixtureModel(k, D)
    vector_data = rand(vector_gmm, 1000)  # scalar data
    test_GaussianMixtureModel_properties(vector_gmm, k, D)

    tester_set = [(standard_gmm, standard_data), (vector_gmm, vector_data)]

    for (gmm, data) in tester_set
        data_matrix = isa(data, Vector) ? reshape(data, :, 1) : data
        D = size(data_matrix, 1)
        k = gmm.k

        gmm = GaussianMixtureModel(k, D)
        testGaussianMixtureModel_EStep(gmm, data)

        gmm = GaussianMixtureModel(k, D)
        testGaussianMixtureModel_MStep(gmm, data)

        gmm = GaussianMixtureModel(k, D)
        testGaussianMixtureModel_fit(gmm, data)

        gmm = GaussianMixtureModel(k, D)
        test_loglikelihood(gmm, data)
    end

    # PoissonMixtureModel Tests
    k = 3
    temp_pmm = PoissonMixtureModel(k)
    temp_pmm.λₖ = [5.0, 10.0, 15.0]
    temp_pmm.πₖ = [1/3, 1/3, 1/3]
    data = rand(temp_pmm, 300)  # returns Vector{Int}

    standard_pmm = PoissonMixtureModel(k)
    test_PoissonMixtureModel_properties(standard_pmm, k)

    for (pmm, d) in [(standard_pmm, data)]
        pmm = PoissonMixtureModel(k)
        testPoissonMixtureModel_EStep(pmm, d)

        pmm = PoissonMixtureModel(k)
        testPoissonMixtureModel_MStep(pmm, d)

        pmm = PoissonMixtureModel(k)
        testPoissonMixtureModel_fit(pmm, d)

        pmm = PoissonMixtureModel(k)
        test_loglikelihood_pmm(pmm, d)
    end
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
    test_GaussianRegression_sklearn()
end

include("RegressionModels/BernoulliRegression.jl")

@testset "BernoulliRegression Tests" begin
    test_BernoulliRegression_initialization()
    test_BernoulliRegression_loglikelihood()
    test_BernoulliRegression_fit()
    test_BernoulliRegression_sample()
    test_BernoulliRegression_optimization()
    test_BernoulliRegression_sklearn()
end

include("RegressionModels/PoissonRegression.jl")

@testset "PoissonRegression Tests" begin
    test_PoissonRegression_initialization()
    test_PoissonRegression_loglikelihood()
    test_PoissonRegression_fit()
    test_PoissonRegression_sample()
    test_PoissonRegression_optimization()
    test_PoissonRegression_sklearn()
end

include("RegressionModels/AutoRegression.jl")

@testset "AutoRegression Tests" begin
    test_AR_emission_initialization()
end

include("HiddenMarkovModels/AutoRegressionHMM.jl")

@testset "AutoRegressive HMM Tests" begin
    test_ARHMM_sampling()
    test_ARHMM_fit()
    test_timeseries_to_AR_feature_matrix()
    test_trialized_timeseries_to_AR_feature_matrix()
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
    test_autoregressive_setters_and_getters()
end

"""
Tests for PrettyPrinting.jl 
"""

include("PrettyPrinting/PrettyPrinting.jl")

@testset "PrettyPrinting.jl Tests" begin
    test_pretty_printing()
end

"""
Tests for Preprocessing.jl
"""

include("Preprocessing/Preprocessing.jl")

@testset "PPCA Tests" begin
    test_PPCA_with_params()
    test_PPCA_E_and_M_Step()
    test_PPCA_fit()
    test_PPCA_samples()
end

include("HiddenMarkovModels/State_Labellers.jl")

@testset "Viterbi and Class Probability Tests" begin
    test_viterbi_GaussianHMM()
    test_class_probabilities()
end
