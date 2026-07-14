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
using PDMats
using Pkg
using Printf
using Random
using StableRNGs
using StateSpaceDynamics
const SSD = StateSpaceDynamics
using SparseArrays
using StatsFuns
using SpecialFunctions
using Test

# Run docs/examples headless (no display window).
ENV["GKSwstype"] = "100"

# In-repo sub-package of assertion helpers shared by `docs/examples/`
# tutorials. Pattern lifted from HiddenMarkovModels.jl's `HMMTest`.
Pkg.develop(; path=joinpath(dirname(@__DIR__), "libs", "SSDTest"))
using SSDTest

@testset verbose = true "StateSpaceDynamics.jl" begin
    # Package-wide quality tests
    @testset verbose = true "Package Quality" begin
        @testset "Aqua.jl" begin
            Aqua.test_all(StateSpaceDynamics; ambiguities=false)
            @test isempty(Test.detect_ambiguities(StateSpaceDynamics))
        end

        @testset "Code Formatting" begin
            @test JuliaFormatter.format(StateSpaceDynamics; verbose=false, overwrite=false)
        end

        @testset "JET.jl Code Linting" begin
            # Skip pre-release versions of Julia, which JET does not yet support.
            if (v"1.11" <= VERSION) && isempty(VERSION.prerelease)
                JET.test_package(StateSpaceDynamics; target_modules=(StateSpaceDynamics,))
            end
        end
    end

    # Linear Dynamical Systems Tests
    @testset verbose = true "Linear Dynamical Systems" begin
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
                test_SLDS_rand_integer_overload()
                test_valid_SLDS_probability_helper_functions()
            end

            @testset "Gradient and Hessian" begin
                test_SLDS_gradient_numerical()
                test_SLDS_hessian_numerical()
                test_SLDS_gradient_single_timestep_gaussian()
                test_SLDS_gradient_single_timestep_poisson()
                test_SLDS_hessian_single_timestep_gaussian()
                test_SLDS_hessian_single_timestep_poisson()
                test_SLDS_gradient_reduces_to_single_LDS()
                test_SLDS_hessian_block_structure_gaussian()
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
                test_SLDS_elbo_matches_LDS_marginal_K1()
                test_SLDS_no_priors_zero_prior_logdensity()
                test_SLDS_joint_sample_reproduces_cross_covariance()
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
                test_SLDS_poisson_d_interpretation()
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

            @testset "Log-likelihood" begin
                test_joint_loglikelihood_matches_mvnormal()
                test_gaussian_gradient_nondiag()
                test_gaussian_hessian_nondiag()
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
                test_td_mn_priors_shrink()
                test_td_with_obs_inputs()
                test_td_ragged_multi_trial()
                test_td_weighted_aggregator_matches_unweighted_with_inputs()
                test_mn_prior_type_decoupled_from_model_matrix()
            end
        end

        include("LinearDynamicalSystems/KalmanLDS.jl")
        @testset "LDS smoother + marginal LL" begin
            test_td_covariance_shared_across_trials()
            test_td_shared_cov_matches_per_trial_path()
            test_lds_with_B_input_equivalent_to_bias()
            test_td_fit_with_latent_input()
            test_td_sampling_zero_input_matches_no_input()
            test_td_fit_missing_u_errors()
            test_marginal_loglikelihood()
        end

        @testset "Kalman-path EM (information form)" begin
            test_kalman_fit_basic()
            test_kalman_fit_with_inputs()
            test_kalman_fit_with_priors()
            test_kalman_fit_bool_combinations()
            test_kalman_validate_inputs_errors()
            test_kalman_marginal_loglikelihood_internals()
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

            @testset "Log-likelihood" begin
                test_joint_loglikelihood_matches_distributions()
                test_joint_loglikelihood_multitrial()
                test_newton_objective_is_joint_loglikelihood()
                test_poisson_gradient_nondiag()
                test_poisson_hessian_nondiag()
            end

            @testset "Priors - Poisson LDS" begin
                test_poisson_map_step_improves_Q()
                test_poisson_gradient_shape_and_finiteness()
                test_poisson_cd_prior_shrink()
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

    # Optimization primitives (line search + Newton)
    @testset verbose = true "Optimization" begin
        include("Optimization/Optimization.jl")
        test_backtracking_min_sense_decreases()
        test_backtracking_returns_best_on_exhaustion()
        test_newton_smooth_no_linesearch_converges()
        test_newton_smooth_returns_false_on_linesearch_stall()
        test_newton_smooth_returns_false_on_max_iter()
    end

    # Utilities Tests
    @testset verbose = true "Utilities" begin
        include("Utilities/Utilities.jl")
        test_block_tridgm()
        test_gaussian_entropy()
        test_valid_Σ()

        @testset "Block Tridiagonal Inverse" begin
            test_block_tridiagonal_inverse_mutating()
            test_block_tridiagonal_inverse_logdet()
            test_block_tridiagonal_solve()
            test_block_tridiagonal_solve_spd()
        end

        @testset "Covariance info-form update" begin
            test_info_update()
        end
    end

    # Conjugate-prior helpers (IW / MN MAP + log-prior terms)
    @testset verbose = true "Priors" begin
        include("Priors/Priors.jl")
        test_iw_prior_helpers()
        test_mn_prior_helpers()
    end

    # Validation Tests
    @testset verbose = true "Validation" begin
        include("Validation/Valid.jl")

        @testset "Probability Vector Validation" begin
            test_validate_probvec()
        end

        @testset "LDS Validation" begin
            test_validate_LDS_gaussian()
            test_validate_LDS_poisson()
            test_validate_LDS_dimension_mismatch()
            test_validate_LDS_non_positive_definite()
            test_validate_LDS_wrong_fit_bool_length()
            test_validate_LDS_poisson_extreme_d()
            test_validate_LDS_asymmetric_covariance()
            test_validate_LDS_poisson_fit_bool_length()
        end

        @testset "Model Validators (per-field)" begin
            test_validate_state_model_fields()
            test_validate_obs_model_gaussian_fields()
            test_validate_obs_model_poisson_fields()
        end

        @testset "Validation Error Messages" begin
            test_validation_error_messages()
        end
    end

    # Preprocessing Tests
    @testset verbose = true "Preprocessing" begin
        include("Preprocessing/Preprocessing.jl")
        @testset verbose = true "PPCA" begin
            test_PPCA_with_params()
            test_PPCA_E_and_M_Step()
            test_PPCA_fit()
            test_PPCA_samples()
        end
    end

    # Pretty Printing Tests
    @testset verbose = true "Pretty Printing" begin
        include("PrettyPrinting/PrettyPrinting.jl")
        test_pretty_printing()
    end

    # Docs/examples tests. Pattern lifted from HiddenMarkovModels.jl.
    @testset verbose = true "Docs Examples" begin
        examples_src = joinpath(dirname(@__DIR__), "docs", "examples")
        for file in sort(readdir(examples_src))
            endswith(file, ".jl") || continue
            @testset "Example - $file" begin
                include(joinpath(examples_src, file))
            end
        end
    end
end
