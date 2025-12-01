using Aqua
using CSV
using DataFrames
using Distributions
using ForwardDiff
using JET
using JuliaFormatter
using LinearAlgebra
using MAT
using Optim
using Random
using StateSpaceDynamics
using SparseArrays
using StatsFuns
using SpecialFunctions
using Test

const CHECKED_TYPES = [Float32, Float64]

# Helper functions
include("helper_functions.jl")

@testset verbose=true "StateSpaceDynamics.jl" begin
    # Package-wide quality tests
    @testset verbose=true "Package Quality" begin
        @testset "Aqua.jl" begin
            Aqua.test_all(StateSpaceDynamics; ambiguities=false)
            @test isempty(Test.detect_ambiguities(StateSpaceDynamics))
        end

        @testset "Blue Formatting" begin
            @test JuliaFormatter.format(StateSpaceDynamics; verbose=false, overwrite=false)
        end

        @testset "JET.jl Code Linting" begin
            if VERSION >= v"1.11"
                JET.test_package(StateSpaceDynamics; target_defined_modules=true)
            end
        end
    end

    # Linear Dynamical Systems Tests
    @testset verbose=true "Linear Dynamical Systems" begin
        include("LinearDynamicalSystems/SLDS.jl")
        @testset "SLDS" begin
            @testset "Validation" begin
                test_valid_SLDS_happy_path()
                test_valid_SLDS_dimension_mismatches()
                test_valid_SLDS_nonstochastic_rows_and_invalid_Z0()
                test_valid_SLDS_mixed_observation_model_types()
                test_valid_SLDS_inconsistent_latent_or_obs_dims()
                test_SLDS_sampling_gaussian()
                test_SLDS_sampling_poisson()
                test_SLDS_deterministic_transitions()
                test_SLDS_single_trial()
                test_SLDS_reproducibility()
                test_SLDS_single_state_edge_case()
                test_SLDS_minimal_dimensions()
                test_valid_SLDS_probability_helper_functions()
            end

            @testset "Gradient and Hessian" begin
                test_SLDS_gradient_numerical()
                test_SLDS_hessian_numerical()
                test_SLDS_gradient_reduces_to_single_LDS()
                test_SLDS_hessian_block_structure()
                test_SLDS_gradient_weight_normalization()
            end

            @testset "Smoothing" begin
                test_SLDS_smooth_basic()
                test_SLDS_smooth_reduces_to_single_LDS()
                test_SLDS_smooth_with_realistic_weights()
                test_SLDS_smooth_consistency_with_gradients()
                test_SLDS_smooth_entropy_calculation()
                test_SLDS_smooth_covariance_symmetry()
                test_SLDS_smooth_different_weight_patterns()
            end

            @testset "Weighted M-step" begin
                test_weighted_update_initial_state_mean()
                test_weighted_update_A_b()
                test_weighted_update_Q()
                test_weighted_gradient_linearity()
                test_zero_weights_behavior()
            end

            @testset "EM Algorithm" begin
                test_SLDS_sample_posterior_basic()
                test_SLDS_estep_basic()
                test_SLDS_mstep_updates_parameters()
                test_SLDS_fit_runs_to_completion()
                test_SLDS_fit_elbo_generally_increases()
                test_SLDS_fit_multitrial()
                test_SLDS_estep_elbo_components()
            end

            @testset "Poisson SLDS" begin
                test_SLDS_sampling_poisson_extended()
                test_SLDS_gradient_numerical_poisson()
                test_SLDS_hessian_block_structure_poisson()
                test_SLDS_smooth_basic_poisson()
                test_SLDS_estep_basic_poisson()
                test_SLDS_mstep_updates_parameters_poisson()
                test_SLDS_fit_runs_to_completion_poisson()
                test_SLDS_fit_elbo_generally_increases_poisson()
                test_SLDS_fit_multitrial_poisson()
                test_SLDS_poisson_count_validation()
                test_SLDS_poisson_log_d_interpretation()
                test_SLDS_gradient_weight_normalization_poisson()
            end
        end

        include("LinearDynamicalSystems/GaussianLDS.jl")
        @testset "Gaussian LDS" begin
            @testset "Constructors" begin
                test_lds_with_params()
                test_gaussian_obs_constructor_type_preservation()
                test_gaussian_lds_constructor_type_preservation()
                test_gaussian_sample_type_preservation()
                test_gaussian_fit_type_preservation()
                test_gaussian_loglikelihood_type_preservation()
            end

            @testset "Smoothing" begin
                test_Gradient()
                test_Hessian()
                test_smooth()
            end

            @testset "EM Algorithm" begin
                test_estep()
                test_initial_observation_parameter_updates()
                test_state_model_parameter_updates()
                test_obs_model_parameter_updates()
                test_initial_observation_parameter_updates(3)
                test_state_model_parameter_updates(3)
                test_obs_model_parameter_updates(3)
                test_EM()
                test_EM(3)
                test_gaussian_iw_priors_shape_map_and_R_sanity()
                test_gaussian_update_R_matches_residual_cov()
                test_gaussian_weighting_equiv_to_duplication()
            end
        end

        include("LinearDynamicalSystems/PoissonLDS.jl")
        @testset "Poisson LDS" begin
            @testset "Constructors" begin
                test_PoissonLDS_with_params()
                test_pobs_constructor_type_preservation()
                test_plds_constructor_type_preservation()
                test_poisson_sample_type_preservation()
                test_poisson_fit_type_preservation()
                test_poisson_loglikelihood_type_preservation()
            end

            @testset "Smoothing" begin
                test_Gradient()
                test_Hessian()
                test_smooth()
            end

            @testset "EM Algorithm" begin
                test_parameter_gradient()
                test_initial_observation_parameter_updates()
                test_state_model_parameter_updates()
                test_initial_observation_parameter_updates(3)
                test_state_model_parameter_updates(3)
                test_EM()
                test_EM(3)
                test_EM_matlab()
                test_poisson_map_step_improves_Q()
                test_poisson_gradient_shape_and_finiteness()
            end
        end
    end

    # Hidden Markov Models Tests
    @testset verbose=true "Hidden Markov Models" begin
        include("HiddenMarkovModels/GaussianHMM.jl")
        @testset "Gaussian HMM" begin
            test_SwitchingGaussian_fit()
            test_SwitchingGaussian_SingleState_fit()
            test_kmeans_init()
            test_trialized_GaussianHMM()
            test_SwitchingGaussian_fit_float32()
        end

        include("HiddenMarkovModels/SwitchingGaussianRegression.jl")
        @testset "Switching Gaussian Regression" begin
            test_SwitchingGaussianRegression_fit()
            test_SwitchingGaussianRegression_SingleState_fit()
            test_trialized_SwitchingGaussianRegression()
        end

        include("HiddenMarkovModels/SwitchingPoissonRegression.jl")
        @testset "Switching Poisson Regression" begin
            test_SwitchingPoissonRegression_fit()
            test_trialized_SwitchingPoissonRegression()
        end

        include("HiddenMarkovModels/SwitchingBernoulliRegression.jl")
        @testset "Switching Bernoulli Regression" begin
            test_SwitchingBernoulliRegression()
            test_trialized_SwitchingBernoulliRegression()
        end

        include("HiddenMarkovModels/AutoRegressionHMM.jl")
        @testset "Autoregressive HMM" begin
            test_ARHMM_sampling()
            test_ARHMM_fit()
            test_timeseries_to_AR_feature_matrix()
            test_trialized_timeseries_to_AR_feature_matrix()
        end

        include("HiddenMarkovModels/State_Labellers.jl")
        @testset "Viterbi and Class Probabilities" begin
            test_viterbi_GaussianHMM()
            test_class_probabilities()
        end
    end

    # Mixture Models Tests
    @testset verbose=true "Mixture Models" begin
        include("MixtureModels/GaussianMixtureModel.jl")
        include("MixtureModels/PoissonMixtureModel.jl")

        @testset "Gaussian Mixture Model" begin
            k = 3
            D = 2
            standard_gmm = GaussianMixtureModel(k, D)
            standard_data = rand(standard_gmm, 100)
            test_GaussianMixtureModel_properties(standard_gmm, k, D)

            k = 2
            D = 1
            vector_gmm = GaussianMixtureModel(k, D)
            vector_data = rand(vector_gmm, 1000)
            test_GaussianMixtureModel_properties(vector_gmm, k, D)

            for (gmm, data) in [(standard_gmm, standard_data), (vector_gmm, vector_data)]
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
        end

        @testset "Poisson Mixture Model" begin
            k = 3
            temp_pmm = PoissonMixtureModel(k)
            temp_pmm.λₖ = [5.0, 10.0, 15.0]
            temp_pmm.πₖ = [1/3, 1/3, 1/3]
            data = rand(temp_pmm, 300)

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
    end

    # Regression Models Tests
    @testset verbose=true "Regression Models" begin
        include("RegressionModels/GaussianRegression.jl")
        @testset "Gaussian Regression" begin
            test_GaussianRegression_initialization()
            test_GaussianRegression_loglikelihood()
            test_GaussianRegression_fit()
            test_GaussianRegression_sample()
            test_GaussianRegression_optimization()
            test_GaussianRegression_sklearn()
        end

        include("RegressionModels/BernoulliRegression.jl")
        @testset "Bernoulli Regression" begin
            test_BernoulliRegression_initialization()
            test_BernoulliRegression_loglikelihood()
            test_BernoulliRegression_fit()
            test_BernoulliRegression_sample()
            test_BernoulliRegression_optimization()
            test_BernoulliRegression_sklearn()
        end

        include("RegressionModels/PoissonRegression.jl")
        @testset "Poisson Regression" begin
            test_PoissonRegression_initialization()
            test_PoissonRegression_loglikelihood()
            test_PoissonRegression_fit()
            test_PoissonRegression_sample()
            test_PoissonRegression_optimization()
            test_PoissonRegression_sklearn()
        end

        include("RegressionModels/AutoRegression.jl")
        @testset "Autoregression" begin
            test_AR_emission_initialization()
        end
    end

    # Utilities Tests
    @testset verbose=true "Utilities" begin
        include("Utilities/Utilities.jl")
        test_euclidean_distance()
        test_kmeanspp_initialization()
        test_kmeans_clustering()
        test_block_tridgm()
        test_autoregressive_setters_and_getters()
        test_gaussian_entropy()
    end

    # Preprocessing Tests
    @testset verbose=true "Preprocessing" begin
        include("Preprocessing/Preprocessing.jl")
        @testset verbose=true "PPCA" begin
            test_PPCA_with_params()
            test_PPCA_E_and_M_Step()
            test_PPCA_fit()
            test_PPCA_samples()
        end
    end

    # Pretty Printing Tests
    @testset verbose=true "Pretty Printing" begin
        include("PrettyPrinting/PrettyPrinting.jl")
        test_pretty_printing()
    end
end
