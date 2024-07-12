using SSM
using Distributions
using ForwardDiff
using LinearAlgebra
using Optim
using Random
using StatsFuns
using SpecialFunctions
using Test

# Random.seed!(1234)

"""
Tests for MixtureModels.jl
"""

# Test general properties of GaussianMixtureModel
function test_GaussianMixtureModel_properties(gmm::GaussianMixtureModel, k::Int, data_dim::Int)
    @test gmm.k == k
    @test size(gmm.μₖ) == (k, data_dim)

    for Σ in gmm.Σₖ
        @test size(Σ) == (data_dim, data_dim)
        @test ishermitian(Σ)
    end

    @test length(gmm.πₖ) == k
    @test sum(gmm.πₖ) ≈ 1.0
end

function testGaussianMixtureModel_EStep(gmm::GaussianMixtureModel, data::Union{Matrix{Float64}, Vector{Float64}})

    k::Int = gmm.k
    data_dim::Int = size(data, 2)
    
    # Run EStep
    class_probabilities = SSM.EStep(gmm, data)
    # Check dimensions
    @test size(class_probabilities) == (size(data, 1), k)
    # Check if the row sums are close to 1 (since they represent probabilities)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(class_probabilities, dims=2))
    
    test_GaussianMixtureModel_properties(gmm, k, data_dim)
end

function testGaussianMixtureModel_MStep(gmm::GaussianMixtureModel, data::Union{Matrix{Float64}, Vector{Float64}})

    k::Int = gmm.k
    data_dim::Int = size(data, 2)

    class_probabilities = SSM.EStep(gmm, data)

    # Run MStep
    SSM.MStep!(gmm, data, class_probabilities)

    test_GaussianMixtureModel_properties(gmm, k, data_dim)
end

function testGaussianMixtureModel_fit(gmm::GaussianMixtureModel, data::Union{Matrix{Float64}, Vector{Float64}})

    k::Int = gmm.k
    data_dim::Int = size(data, 2)

    # Run fit!
    fit!(gmm, data; maxiter=10, tol=1e-3)

    test_GaussianMixtureModel_properties(gmm, k, data_dim)
end

function test_log_likelihood(gmm::GaussianMixtureModel, data::Union{Matrix{Float64}, Vector{Float64}})

    # Calculate log-likelihood
    ll = log_likelihood(gmm, data)

    # Check if log-likelihood is a scalar
    @test size(ll) == ()

    # Log-likelihood should be a negative float
    @test ll < 0.0

    # Log-likelihood should monotonically increase with iterations (when using exact EM)

    #repeatedly applying fit! without initializtion, so first initialize means
    # Initialize k means of gmm
	gmm.μₖ = permutedims(kmeanspp_initialization(data, gmm.k))
    
    ll_prev = -Inf
    for i in 1:10
        fit!(gmm, data; maxiter=1, tol=1e-3, initialize_kmeans=false)
        ll = log_likelihood(gmm, data)
        @test ll > ll_prev || isapprox(ll, ll_prev; atol=1e-6)
        ll_prev = ll
    end
end

"""
Tests for PoissonMixtureModel
"""

# Test general properties of PoissonMixtureModel
function test_PoissonMixtureModel_properties(pmm::PoissonMixtureModel, k::Int)
    @test pmm.k == k
    @test length(pmm.λₖ) == k
    @test length(pmm.πₖ) == k
    @test sum(pmm.πₖ) ≈ 1.0
end

function testPoissonMixtureModel_EStep(pmm::PoissonMixtureModel, data::Union{Matrix{Int}, Vector{Int}})
    k::Int = pmm.k
    
    # Run EStep
    class_probabilities = SSM.EStep(pmm, data)
    # Check dimensions
    @test size(class_probabilities) == (size(data, 1), k)
    # Check if the row sums are close to 1 (since they represent probabilities)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(class_probabilities, dims=2))
    
    test_PoissonMixtureModel_properties(pmm, k)
end

function testPoissonMixtureModel_MStep(pmm::PoissonMixtureModel, data::Union{Matrix{Int}, Vector{Int}})
    k::Int = pmm.k

    class_probabilities = SSM.EStep(pmm, data)

    # Run MStep
    SSM.MStep!(pmm, data, class_probabilities)

    test_PoissonMixtureModel_properties(pmm, k)
end

function testPoissonMixtureModel_fit(pmm::PoissonMixtureModel, data::Union{Matrix{Int}, Vector{Int}})
    k::Int = pmm.k

    # Run fit!
    fit!(pmm, data; maxiter=10, tol=1e-3)

    test_PoissonMixtureModel_properties(pmm, k)
end

function test_log_likelihood(pmm::PoissonMixtureModel, data::Union{Matrix{Int}, Vector{Int}})
    # Calculate log-likelihood
    ll = log_likelihood(pmm, data)

    # Check if log-likelihood is a scalar
    @test size(ll) == ()

    # Log-likelihood should not necessarily be negative for Poisson models

    # Initialize λₖ with kmeans_init
    λₖ_matrix = permutedims(kmeanspp_initialization(Float64.(data), pmm.k))
    pmm.λₖ = vec(λₖ_matrix)

    # Log-likelihood should monotonically increase with iterations (when using exact EM)
    ll_prev = -Inf
    for i in 1:10
        fit!(pmm, data; maxiter=1, tol=1e-3, initialize_kmeans=false)
        ll = log_likelihood(pmm, data)
        @test ll > ll_prev || isapprox(ll, ll_prev; atol=1e-6)
        ll_prev = ll
    end
end

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

function toy_HMM(k::Int=3, data_dim::Int=2, n::Int=1000)
    # create random data
    data = randn(n, data_dim)
    # fit hmm
    hmm = GaussianHMM(data, k)
    return hmm, data
end

function test_toy_HMM()
    hmm, data = toy_HMM()
    @test size(data, 2) == hmm.D
    @test size(data, 1) == 1000
    @test hmm.K == 3
end

function test_HMM_properties(hmm::GaussianHMM)
    @test isapprox(sum(hmm.A, dims=2), ones(hmm.K))
    @test typeof(hmm.B) == Vector{GaussianEmission}
    @test sum(hmm.πₖ) ≈ 1.0
end

function test_GaussianHMM_constructor()
    hmm, _ = toy_HMM()
    test_HMM_properties(hmm)
end

function test_HMM_forward_and_back()
    hmm, data = toy_HMM()
    α = SSM.forward(hmm, data)
    @test size(α) == (size(data, 1), hmm.K)
    β = SSM.backward(hmm, data)
    @test size(β) == (size(data, 1), hmm.K)
end

function test_HMM_gamma_xi()
    hmm, data = toy_HMM()
    α = SSM.forward(hmm, data)
    β = SSM.backward(hmm, data)
    γ = SSM.calculate_γ(hmm, α, β)
    ξ = SSM.calculate_ξ(hmm, α, β, data)
    @test size(γ) == (size(data, 1), hmm.K)
    @test size(ξ) == (size(data, 1) - 1, hmm.K, hmm.K)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(exp.(γ), dims=2))
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(exp.(ξ), dims=(2, 3)))
end

function test_HMM_E_step()
    hmm, data = toy_HMM()
    γ, ξ, α, β = SSM.E_step(hmm, data)
    @test size(γ) == (size(data, 1), hmm.K)
    @test size(ξ) == (size(data, 1) - 1, hmm.K, hmm.K)
end

function test_HMM_M_step()
    hmm, data = toy_HMM()
    # test indiviudal M-step functions
    γ, ξ, α, β = SSM.E_step(hmm, data)
    SSM.update_initial_state_distribution!(hmm, γ)
    @test sum(hmm.πₖ) ≈ 1.0
    SSM.update_transition_matrix!(hmm, γ, ξ)
    @test isapprox(sum(hmm.A, dims=2), ones(hmm.K))
    SSM.update_emission_models!(hmm, γ, data)
    @test typeof(hmm.B) == Vector{GaussianEmission}
    # test M-step
    γ, ξ, α, β = SSM.E_step(hmm, data)
    SSM.M_step!(hmm, γ, ξ, data)
    test_HMM_properties(hmm)
end

function test_HMM_EM()
    Random.seed!(1234)
    A = [0.7 0.2 0.1; 0.1 0.7 0.2; 0.2 0.1 0.7]
    means = [[0.0, 0.0], [-1.0, 2.0], [3.0, 2.5]]
    covs = [
        [0.1 0.0; 0.0 0.1], 
        [0.1 0.0; 0.0 0.1], 
        [0.1 0.0; 0.0 0.1]
    ]
    emissions_models = [GaussianEmission(mean, cov) for (mean, cov) in zip(means, covs)]
    simul_hmm = GaussianHMM(A, emissions_models, [0.33, 0.33, 0.34], 3, 2)
    states, observations = SSM.sample(simul_hmm, 10000)
    hmm = GaussianHMM(observations, 3)
    baumWelch!(hmm, observations, 100)
    pred_means = [hmm.B[i].μ for i in 1:3]
    @test sort(pred_means) ≈ sort(means) atol=0.2
    pred_covs = [hmm.B[i].Σ for i in 1:3]
    @test pred_covs ≈ covs atol=0.1
    best_path = viterbi(hmm, observations)
    @test length(best_path) == 10000
    @test all(x -> x in 1:3, best_path)
end
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
# Create a toy example for all LDS tests. This example represents a pendulum in a frictionless environment.
g = 9.81
l=1
# time differential
dt = 0.01
# transition matrix
A = [1.0 dt; -g/l*dt 1.0]
# Initial state
x0 = [0.0, 1.0]
# Define the LDS model parameters
H = Matrix{Float64}(I(2))  # Observation matrix (assuming direct observation)
Q = 0.01 * Matrix{Float64}(I(2))  # Process noise covariance
observation_noise_std = 0.5
R = (observation_noise_std^2) * Matrix{Float64}(I(2))  # Observation noise covariance
p0 = 0.1 * Matrix{Float64}(I(2))  # Initial state covariance

lds = SSM.LDS(;A=A, H=H, Q=Q, R=R, x0=x0, p0=p0, obs_dim=2, latent_dim=2)

x_true, x_noisy = SSM.sample(lds, 100)

function test_LDS_with_params()
    # Create the Kalman filter parameter vector
    kf = LDS(A,
             H,
             nothing,
             Q, 
             R, 
             x0, 
             p0, 
             nothing, 
             2, 
             2, 
             Vector([false, false, false, false, false, false, false, false]))
    # confirm parameters are set correctly
    @test kf.A == A
    @test kf.H == H
    @test kf.B === nothing
    @test kf.Q == Q
    @test kf.R == R
    @test kf.x0 == x0
    @test kf.p0 == p0
    @test kf.inputs === nothing
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == Vector([false, false, false, false, false, false, false, false])
    # run the filter
    x_filt, p_filt, x_pred, p_pred, v, F, K, ml = KalmanFilter(kf, x_noisy)
    # check dimensions
    @test size(x_filt) == (size(x_noisy, 1), 2)
    @test size(p_filt) == (size(x_noisy, 1), 2, 2)
    @test size(x_pred) == (size(x_noisy, 1), 2)
    @test size(p_pred) == (size(x_noisy, 1), 2, 2)
    @test size(v) == (size(x_noisy, 1), 2)
    @test size(F) == (size(x_noisy, 1), 2, 2)
    @test size(K) == (size(x_noisy, 1), 2, 2)
    # run the smoother
    x_smooth, p_smooth = KalmanSmoother(kf, x_noisy)
    # check dimensions
    @test size(x_smooth) == (size(x_noisy, 1), 2)
    @test size(p_smooth) == (size(x_noisy, 1), 2, 2)
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
    @test kf.p0 !== nothing
    @test kf.inputs === nothing
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == fill(true, 7)
end

function test_LDS_EStep()
    # Create the Kalman filter parameter vector
    kf = LDS(A,
             H,
             nothing,
             Q, 
             R, 
             x0, 
             p0, 
             nothing, 
             2, 
             2, 
             Vector([true, true, true, true, true, true, true]))
    # run the EStep
    x_smooth, p_smooth, E_z, E_zz, E_zz_prev, ml = SSM.E_Step(kf, x_noisy)
    x_smooth_dir, p_smooth_dir, E_z_dir, E_zz_dir, E_zz_prev_dir, ml_dir = SSM.E_Step(kf, x_noisy, SSM.DirectSmoothing())
    # check dimensions
    @test size(x_smooth) == (size(x_noisy, 1), 2)
    @test size(p_smooth) == (size(x_noisy, 1), 2, 2)
    @test size(E_z) == (size(x_noisy, 1), 2)
    @test size(E_zz) == (size(x_noisy, 1), 2, 2)
    @test size(E_zz_prev) == (size(x_noisy, 1), 2, 2)
    @test size(ml) == ()
    # check that both methods are the same
    @test isapprox(x_smooth, x_smooth_dir; atol=1e-6)
    @test isapprox(p_smooth, p_smooth_dir; atol=1e-6)
    @test isapprox(E_z, E_z_dir; atol=1e-6)
    @test isapprox(E_zz, E_zz_dir; atol=1e-6)
    println("Maximum of sufficient statistics:", sum((E_zz_prev - E_zz_prev_dir).^2))
    @test_broken isapprox(E_zz_prev, E_zz_prev_dir; atol=1e-6)
end

function test_LDS_MStep!()
    # Create the Kalman filter parameter vector
    kf = LDS(A,
             H,
             nothing,
             Q, 
             R, 
             x0, 
             p0, 
             nothing, 
             2, 
             2, 
             Vector([true, true, true, true, true, true, true]))
    # run the EStep
    x_smooth, p_smooth, E_z, E_zz, E_zz_prev, ml = SSM.E_Step(kf, x_noisy)
    # run the MStep
    SSM.M_Step!(kf, E_z, E_zz, E_zz_prev, x_noisy)
    # check if the parameters are updated
    @test kf.A !== A
    @test kf.H !== H
    @test kf.B === nothing
    @test kf.Q !== Q
    @test kf.R !== R
    @test kf.x0 !== x0
    @test kf.p0 !== p0
    @test kf.inputs === nothing
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == Vector([true, true, true, true, true, true, true])
end

function test_LDS_EM()
    kf = LDS(A,
             H,
             nothing,
             Q, 
             R, 
             x0, 
             p0, 
             nothing, 
             2, 
             2, 
             Vector([true, true, true, true, true, true, true]))
    # run the EM
    mls = SSM.KalmanFilterEM!(kf, x_noisy, 100)
    @test all(diff(mls) .>= 0)
    # check if the parameters are updated
    @test kf.A !== A
    @test kf.H !== H
    @test kf.B === nothing
    @test kf.Q !== Q
    @test kf.R !== R
    @test kf.x0 !== x0
    @test kf.p0 !== p0
    @test kf.inputs === nothing
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == Vector([true, true, true, true, true, true, true]) 
end

function test_EM_numeric_RTS()
    # create a naive kalman filter to optimize via a numerical EM
    kf = LDS(;latent_dim=2, obs_dim=2, fit_bool=fill(true, 7))
    # create a second kalman filter to optimize using EM
    kf_prime = LDS(;A=kf.A, Q=kf.Q, H=kf.H, R=kf.R, x0=kf.x0, p0=kf.p0, obs_dim=kf.obs_dim, latent_dim=kf.latent_dim, fit_bool=kf.fit_bool)
    # set up an optimziation problem for three iterations
    for i in 1:1
        # smooth observations
        x_smooth, p_smooth, E_z, E_zz, E_zz_prev, ml = SSM.E_Step(kf, x_noisy)
        # optimize kf_prime
        SSM.KalmanFilterEM!(kf_prime, x_noisy, 1)

        # optimize x0 and p0 first of kf
        res_1 = optimize(x0 -> -SSM.Q(kf.A, kf.Q, kf.H, kf.R, kf.p0, x0, E_z, E_zz, E_zz_prev, x_noisy), kf.x0, LBFGS())
        res_2 = optimize(p0 -> -SSM.Q(kf.A, kf.Q, kf.H, kf.R, p0, kf.x0, E_z, E_zz, E_zz_prev, x_noisy), kf.p0, LBFGS())
        # test the optimization
        @test isapprox(res_1.minimizer, kf_prime.x0; atol=1e-6)
        @test_broken isapprox(res_2.minimizer * res_2.minimizer', kf_prime.p0; atol=1e-6)

        # now update A and Q
        res_3 = optimize(A -> -SSM.Q(A, kf.Q, kf.H, kf.R, kf.p0, kf.x0, E_z, E_zz, E_zz_prev, x_noisy), kf.A, LBFGS())
        # test if A is updated
        @test isapprox(res_3.minimizer, kf_prime.A; atol=1e-6)
        kf.A = res_3.minimizer
        res_4 = optimize(Q -> -SSM.Q(kf.A, Q, kf.H, kf.R, kf.p0, kf.x0, E_z, E_zz, E_zz_prev, x_noisy), kf.Q, LBFGS())
        # test if Q is updated
        @test isapprox(res_4.minimizer * res_4.minimizer', kf_prime.Q; atol=1e-6)

        # now update H and R
        res_5 = optimize(H -> -SSM.Q(kf.A, kf.Q, H, kf.R, kf.p0, kf.x0, E_z, E_zz, E_zz_prev, x_noisy), kf.H, LBFGS())
        # test if H is updated
        @test isapprox(res_5.minimizer, kf_prime.H; atol=1e-6)
        kf.H = res_5.minimizer
        res_6 = optimize(R -> -SSM.Q(kf.A, kf.Q, kf.H, R, kf.p0, kf.x0, E_z, E_zz, E_zz_prev, x_noisy), kf.R, LBFGS())
        # test if R is updated
        @test isapprox(res_6.minimizer * res_6.minimizer', kf_prime.R; atol=1e-6)
    end
end

function test_EM_numeric_Direct()
    # create a naive kalman filter to optimize via a numerical EM
    kf = LDS(;latent_dim=2, obs_dim=2, fit_bool=fill(true, 7))
    # create a second kalman filter to optimize using EM
    kf_prime = LDS(;A=kf.A, Q=kf.Q, H=kf.H, R=kf.R, x0=kf.x0, p0=kf.p0, obs_dim=kf.obs_dim, latent_dim=kf.latent_dim, fit_bool=kf.fit_bool)
    # set up an optimziation problem for three iterations
    for i in 1:1
        # smooth observations
        x_smooth, p_smooth, E_z, E_zz, E_zz_prev, ml = SSM.E_Step(kf, x_noisy, SSM.DirectSmoothing())
        # optimize kf_prime
        SSM.KalmanFilterEM!(kf_prime, x_noisy, 1, 1e-12, SSM.DirectSmoothing())

        # optimize x0 and p0 first of kf
        res_1 = optimize(x0 -> -SSM.Q(kf.A, kf.Q, kf.H, kf.R, kf.p0, x0, E_z, E_zz, E_zz_prev, x_noisy), kf.x0, LBFGS())
        res_2 = optimize(p0 -> -SSM.Q(kf.A, kf.Q, kf.H, kf.R, p0, kf.x0, E_z, E_zz, E_zz_prev, x_noisy), kf.p0, LBFGS())
        # test the optimization
        @test isapprox(res_1.minimizer, kf_prime.x0; atol=1e-6)
        @test_broken isapprox(res_2.minimizer * res_2.minimizer', kf_prime.p0; atol=1e-6)

        # now update A and Q
        res_3 = optimize(A -> -SSM.Q(A, kf.Q, kf.H, kf.R, kf.p0, kf.x0, E_z, E_zz, E_zz_prev, x_noisy), kf.A, LBFGS())
        # test if A is updated
        @test isapprox(res_3.minimizer, kf_prime.A; atol=1e-6)
        kf.A = res_3.minimizer
        res_4 = optimize(Q -> -SSM.Q(kf.A, Q, kf.H, kf.R, kf.p0, kf.x0, E_z, E_zz, E_zz_prev, x_noisy), kf.Q, LBFGS())
        # test if Q is updated
        @test isapprox(res_4.minimizer * res_4.minimizer', kf_prime.Q; atol=1e-6)

        # now update H and R
        res_5 = optimize(H -> -SSM.Q(kf.A, kf.Q, H, kf.R, kf.p0, kf.x0, E_z, E_zz, E_zz_prev, x_noisy), kf.H, LBFGS())
        # test if H is updated
        @test isapprox(res_5.minimizer, kf_prime.H; atol=1e-6)
        kf.H = res_5.minimizer
        res_6 = optimize(R -> -SSM.Q(kf.A, kf.Q, kf.H, R, kf.p0, kf.x0, E_z, E_zz, E_zz_prev, x_noisy), kf.R, LBFGS())
        # test if R is updated
        @test isapprox(res_6.minimizer * res_6.minimizer', kf_prime.R; atol=1e-6)
    end
end

@testset "LDS Tests" begin
    test_LDS_with_params()
    test_LDS_without_params()
    test_LDS_EStep()
    test_LDS_MStep!()
    test_LDS_EM()
    test_EM_numeric_RTS()
    test_EM_numeric_Direct()
end

function toy_PoissonLDS()
    T = 100
    # create a PLDS model
    x0 = [1.0, -1.0]
    p0 = Matrix(Diagonal([0.001, 0.001]))
    A = [cos(0.1) -sin(0.1); sin(0.1) cos(0.1)]
    Q = Matrix(Diagonal([0.001, 0.001]))
    C = [0.5 0.5; 0.5 0.1; 0.1 0.1]
    log_d = [0.5, 0.5, 0.5]
    D = Matrix(Diagonal([0., 0., 0.]))
    b = ones(T, 2) * 0.0

    plds = PoissonLDS(A=A, C=C, Q=Q, D=D, b=b, log_d=log_d, x0=x0, p0=p0, refractory_period=1, obs_dim=3, latent_dim=2)
    # sample data
    x, y = SSM.sample(plds, T, 3)
    return plds, x, y
end

function test_PLDS_constructor_with_params()
    # create a set of parameters to test with
    obs_dim = 10
    latent_dim = 5

    A = randn(latent_dim, latent_dim)
    C = randn(obs_dim, latent_dim)
    Q = I(latent_dim)
    x0 = randn(latent_dim)
    p0 = I(latent_dim)
    refrac = 1
    log_d = randn(obs_dim)
    D = randn(obs_dim, obs_dim)
    fit_bool=Vector([true, true, true, true, true, true])

    # create the PLDS model
    plds = PoissonLDS(;A=A, C=C, Q=Q, D=D, log_d=log_d, x0=x0, p0=p0, refractory_period=refrac, obs_dim=obs_dim, latent_dim=latent_dim, fit_bool=fit_bool)

    # test model
    @test plds.A == A
    @test plds.C == C
    @test plds.Q == Q
    @test plds.x0 == x0
    @test plds.p0 == p0
    @test plds.log_d == log_d
    @test plds.D == D
    @test plds.refractory_period == 1
    @test plds.obs_dim == obs_dim
    @test plds.latent_dim == latent_dim
end

function test_PLDS_constructor_without_params()
    # create the PLDS model
    plds = PoissonLDS(;obs_dim=10, latent_dim=5)

    # test parameters are not empty
    @test !isempty(plds.A)
    @test !isempty(plds.C)
    @test !isempty(plds.Q)
    @test !isempty(plds.x0)
    @test !isempty(plds.p0)
    @test !isempty(plds.log_d)
    @test !isempty(plds.D)
    @test plds.refractory_period == 1
    @test plds.obs_dim == 10
    @test plds.latent_dim == 5
    @test plds.fit_bool == fill(true, 6)

    # test dims of parameters
    @test size(plds.A) == (5, 5)
    @test size(plds.C) == (10, 5)
    @test size(plds.Q) == (5, 5)
    @test size(plds.x0) == (5,)
    @test size(plds.p0) == (5, 5)
    @test size(plds.log_d) == (10,)
    @test size(plds.D) == (10, 10)
    @test isempty(plds.b)
end

function test_countspikes()
    # create a set of observations that is a matrix of spikes/events
    obs = [0 0 1; 1 1 1; 0 1 0]
    # count the spikes when window=1
    count = SSM.countspikes(obs, 1)
    # check the count
    @test count == [0 0 0; 0 0 1; 1 1 1]
    # count spikes when window=2
    count_2 = SSM.countspikes(obs, 2)
    # check the count
    @test count_2 == [0 0 0; 0 0 1; 1 1 2]
end

function test_logposterior()
    # create a plds model
    plds = PoissonLDS(;obs_dim=10, latent_dim=5)
    # create some observations
    obs = rand(Bool, 100, 10)
    # create latent state
    x = randn(100, 5)
    b = randn(100, 5)
    plds.b = b
    # calculate the log posterior
    logpost = SSM.logposterior(x, plds, obs)
    # check the dimensions
    @test logpost isa Float64
end

function test_gradient_plds()
    # create a plds model
    plds = PoissonLDS(;obs_dim=10, latent_dim=5)
    # create some observations
    obs = rand(Bool, 3, 10)
    # create initial latent state for gradient calculation
    x = randn(3, 5)
    b = zeros(3, 5)
    plds.b = b
    # calculate the gradient
    grad = SSM.Gradient(x, plds, obs)
    # check the dimensions
    @test size(grad) == (3, 5)
    # check the gradients using autodiff
    obj = x -> SSM.logposterior_nonthreaded(x, plds, obs)
    grad_autodiff = ForwardDiff.gradient(obj, x)
    @test grad ≈ grad_autodiff atol=1e-12
end

function test_hessian_plds()
    # create a plds model
    plds = PoissonLDS(;obs_dim=10, latent_dim=5)
    # create some observations
    obs = rand(Bool, 3, 10)
    # create initial latent state for hessian calculation
    x = randn(3, 5)
    b = zeros(3, 5)
    plds.b = b
    # calculate the hessian
    hess, main, super, sub = SSM.Hessian(x, plds, obs)
    # check the dimensions
    @test length(main) == 3
    @test length(super) == 2
    @test length(sub) == 2
    @test size(hess) == (15, 15)
    # check the hessian using autodiff
    function obj_logposterior(x::Vector)
        x = SSM.interleave_reshape(x, 3, 5)
        return SSM.logposterior_nonthreaded(x, plds, obs)
    end
    hess_autodiff = ForwardDiff.hessian(obj_logposterior, reshape(x', 15))
    @test hess ≈ hess_autodiff atol=1e-12
end

function test_direct_smoother()
    # create a plds model
    plds = PoissonLDS(;obs_dim=10, latent_dim=5)
    # create some observations
    obs = rand(Bool, 10, 10)
    # create inputs
    b = rand(10, 5)
    plds.b = b
    # run the direct smoother
    x_smooth, p_smooth = SSM.directsmooth(plds, obs)
    # check the dimensions
    @test size(x_smooth) == (10, 5)
    @test size(p_smooth) == (10, 5, 5)
end

function test_smooth()
    plds, x, y = toy_PoissonLDS()
    # run the smoother
    x_smooth, p_smooth = SSM.smooth(plds, y)
    # check the dimensions
    @test size(x_smooth) == (3, 100, 2)
    @test size(p_smooth) == (3, 100, 2, 2)
end

function test_analytical_parameter_updates()
    # create a dummy data
    dummy_plds, x, y = toy_PoissonLDS()
    # now create a random plds model
    plds = PoissonLDS(;obs_dim=3, latent_dim=2)
    # save the old parameters from the model
    A = copy(plds.A)
    Q = copy(plds.Q)
    x0 = copy(plds.x0)
    p0 = copy(plds.p0)
    # run E-Step
    E_z, E_zz, E_zz_prev, x_sm, p_sm = SSM.E_Step(plds, y)

    # optimize x0 
    opt_x0 = x0 -> -SSM.Q_initial_obs(x0, plds.p0, E_z, E_zz)
    result_x0 = optimize(opt_x0, plds.x0, LBFGS())
    @test isapprox(result_x0.minimizer, SSM.update_initial_state_mean!(plds, E_z))

    # optimize p0
    opt_p0 = p0 -> -SSM.Q_initial_obs(result_x0.minimizer, p0, E_z, E_zz)
    result_p0 = optimize(opt_p0, plds.p0, LBFGS(), Optim.Options(g_abstol=1e-12))
    @test_broken isapprox(result_p0.minimizer * result_p0.minimizer', SSM.update_initial_state_covariance!(plds, E_zz, E_z), atol=1e-3)

    Q_l = Matrix(cholesky(Q).L)
    # optimize A and Q
    opt_A = A -> -SSM.Q_state_model(A, Q_l, E_zz, E_zz_prev)
    result_A = optimize(opt_A, plds.A, LBFGS(), Optim.Options(g_abstol=1e-12))
    @test_broken isapprox(result_A.minimizer, SSM.update_A_plds!(plds, E_zz, E_zz_prev), atol=1e-6)

    # update the model before update Q
    plds.A = result_A.minimizer
    # optimize Q now
    opt_Q = Q -> -SSM.Q_state_model(plds.A, Q, E_zz, E_zz_prev)
    result_Q = optimize(opt_Q, Q_l, LBFGS(), Optim.Options(g_abstol=1e-12))

    @test isapprox(result_Q.minimizer * result_Q.minimizer', SSM.update_Q_plds!(plds, E_zz, E_zz_prev), atol=1e-6)
end


@testset "PLDS Tests" begin
    test_PLDS_constructor_with_params()
    test_PLDS_constructor_without_params()
    test_countspikes()
    test_logposterior()
    test_gradient_plds()
    test_hessian_plds()
    test_direct_smoother()
    test_smooth()
    test_analytical_parameter_updates()
end

"""
Tests for Regression.jl
"""
function test_GaussianRegression_fit()
    # Generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    y = X * true_β + randn(1000) * 0.5
    
    # Initialize and fit the model
    model = GaussianRegression()
    fit!(model, X[:, 2:end], y)
    
    # Check if the fitted coefficients are close to the true coefficients
    @test isapprox(model.β, true_β, atol=0.5)
    @test model.σ² > 0
    @test isapprox(model.σ², 0.25, atol=0.1)
end

function test_GaussianRegression_loglikelihood()
    # Generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    y = X * true_β + randn(1000) * 0.5
    
    # Initialize and fit the model
    model = GaussianRegression()
    fit!(model, X[:, 2:end], y)
    # check if the fitted coefficients are close to the true coefficients
    @test isapprox(model.β, true_β, atol=0.5)
    @test model.σ² > 0
    @test isapprox(model.σ², 0.25, atol=0.1)
    
    # Check log likelihood
    loglik = SSM.loglikelihood(model, X[:, 2:end], y)
    @test loglik < 0

    # test loglikelihood on a single point
    loglik = SSM.loglikelihood(model, X[1, 2:end], y[1])
    @test loglik < 0
end

function test_GaussianRegression_empty_model()
    model = GaussianRegression()
    @test isempty(model.β)
    @test model.σ² == 0.0
end

function test_GaussianRegression_intercept()
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    y = X * true_β + randn(1000) * 0.5
    
    model = GaussianRegression(include_intercept=false)
    fit!(model, X[:, 2:end], y)
    @test length(model.β) == 2
    
    model_with_intercept = GaussianRegression()
    fit!(model_with_intercept, X[:, 2:end], y)
    @test length(model_with_intercept.β) == 3
    @test isapprox(model_with_intercept.β, true_β, atol=0.5)
end

function test_Gaussian_ll_gradient()
    # generate data and observations
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    y = X * true_β + randn(1000) * 0.5
    # initialize model
    model = GaussianRegression()
    model.β = [0., 0., 0.]
    model.σ² = 1.

    # use ForwardDiff to calculate the gradient
    function objective(β, w)
        return sum(w.*(y - X * β).^2) + (model.λ * sum(β.^2))
    end

    grad = ForwardDiff.gradient(x -> objective(x, ones(1000)), model.β)
    # calculate the gradient manually
    grad_analytic = SSM.gradient!([0., 0., 0.], model, X, y)
    # check if the gradients are close
    @test isapprox(grad, grad_analytic, atol=1e-6)

    # now do the same with Weights
    weights = rand(1000)
    grad = ForwardDiff.gradient(x -> objective(x, weights), model.β)
    grad_analytic = SSM.gradient!([0., 0., 0.], model, X, y, weights)
    @test isapprox(grad, grad_analytic, atol=1e-6)

    # finally test when λ is not 0
    model = GaussianRegression(λ=0.1)
    model.β = rand(3)

    grad = ForwardDiff.gradient(x -> objective(x, ones(1000)), model.β)
    grad_analytic = SSM.gradient!([0., 0., 0.], model, X, y)
    @test isapprox(grad, grad_analytic, atol=1e-6)

end

@testset "GaussianRegression Tests" begin
    test_GaussianRegression_fit()
    test_GaussianRegression_loglikelihood()
    test_GaussianRegression_empty_model()
    test_GaussianRegression_intercept()
    test_Gaussian_ll_gradient()
end

function test_BernoulliRegression_fit()
    # Generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    p = logistic.(X * true_β)
    y = rand.(Bernoulli.(p))
    
    # Initialize and fit the model
    model = BernoulliRegression()
    fit!(model, X[:, 2:end], y)
    
    # Check if the fitted coefficients are reasonable
    @test length(model.β) == 3
    @test isapprox(model.β, true_β, atol=0.5)
end

function test_BernoulliRegression_loglikelihood()
    # Generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    p = logistic.(X * true_β)
    y = rand.(Bernoulli.(p))
    
    # Initialize and fit the model
    model = BernoulliRegression()
    fit!(model, X[:, 2:end], y)
    # check if the fitted coefficients are close to the true coefficients
    @test length(model.β) == 3
    @test isapprox(model.β, true_β, atol=0.5)
    
    # Check log likelihood
    loglik = SSM.loglikelihood(model, X[:, 2:end], y)
    @test loglik < 0

    #test loglikelihood on a single point
    loglik = SSM.loglikelihood(model, X[1, 2:end], y[1])
    @test loglik < 0
end

function test_BernoulliRegression_empty_model()
    model = BernoulliRegression()
    @test isempty(model.β)
end

function test_BernoulliRegression_intercept()
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    p = logistic.(X * true_β)
    y = rand.(Bernoulli.(p))
    
    model = BernoulliRegression(include_intercept=false)
    fit!(model, X[:, 2:end], y)
    @test length(model.β) == 2
    
    model_with_intercept = BernoulliRegression()
    fit!(model_with_intercept, X[:, 2:end], y)
    @test length(model_with_intercept.β) == 3
    @test isapprox(model_with_intercept.β, true_β, atol=0.5)
end

function test_Bernoulli_ll_gradient()
    # generate data and observations
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    p = logistic.(X * true_β)
    y = rand.(Bernoulli.(p))
    # initialize model
    model = BernoulliRegression()
    model.β = [0., 0., 0.]

    # use ForwardDiff to calculate the gradient
    function objective(β, w)
        return -sum(w .* (y .* log.(logistic.(X * β)) .+ (1 .- y) .* log.(1 .- logistic.(X * β)))) + (model.λ * sum(β.^2))
    end
    grad = ForwardDiff.gradient(x -> objective(x, ones(1000)), model.β)
    # calculate the gradient manually
    grad_analytic = SSM.gradient!([0., 0., 0.], model, X, y)
    # check if the gradients are close
    @test isapprox(grad, grad_analytic, atol=1e-6)

    # now do the same with Weights
    weights = rand(1000)
    grad = ForwardDiff.gradient(x -> objective(x, weights), model.β)
    grad_analytic = SSM.gradient!([0., 0., 0.], model, X, y, weights)
    @test isapprox(grad, grad_analytic, atol=1e-6)

    # finally test when λ is not 0
    model = BernoulliRegression(λ=0.1)
    model.β = rand(3)

    grad = ForwardDiff.gradient(x -> objective(x, ones(1000)), model.β)
    grad_analytic = SSM.gradient!([0., 0., 0.], model, X, y)
    @test isapprox(grad, grad_analytic, atol=1e-6)
end

@testset "BernoulliRegression Tests" begin
    test_BernoulliRegression_fit()
    test_BernoulliRegression_loglikelihood()
    test_BernoulliRegression_empty_model()
    test_BernoulliRegression_intercept()
    test_Bernoulli_ll_gradient()
end

function test_PoissonRegression_fit()
    # Generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    λ = exp.(X * true_β)
    y = rand.(Poisson.(λ))
    
    # Initialize and fit the model
    model = PoissonRegression()
    fit!(model, X[:, 2:end], y)

    # Check if the fitted coefficients are close to the true coefficients
    @test length(model.β) == 3
    @test isapprox(model.β, true_β, atol=0.5)
end

function test_PoissonRegression_loglikelihood()
    # Generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    λ = exp.(X * true_β)
    y = rand.(Poisson.(λ))
    
    # Initialize and fit the model
    model = PoissonRegression()
    fit!(model, X[:, 2:end], y)
    # check if the fitted coefficients are close to the true coefficients
    @test length(model.β) == 3
    @test isapprox(model.β, true_β, atol=0.5)
    
    # Check log likelihood
    loglik = SSM.loglikelihood(model, X[:, 2:end], y)
    @test loglik < 0

    #test loglikelihood on a single point
    loglik = SSM.loglikelihood(model, X[1, :], y[1])
    @test loglik < 0
end

function test_PoissonRegression_empty_model()
    model = PoissonRegression()
    @test isempty(model.β)
end

function test_PoissonRegression_intercept()
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    λ = exp.(X * true_β)
    y = rand.(Poisson.(λ))
    
    model = PoissonRegression(include_intercept=false)
    fit!(model, X[:, 2:end], y)
    @test length(model.β) == 2
    
    model_with_intercept = PoissonRegression()
    fit!(model_with_intercept, X[:, 2:end], y)
    @test length(model_with_intercept.β) == 3
    @test isapprox(model_with_intercept.β, true_β, atol=0.5)
end

function test_Poisson_ll_gradient()
    # generate data and observations
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    λ = exp.(X * true_β)
    y = rand.(Poisson.(λ))
    # initialize model
    model = PoissonRegression()
    model.β = [0., 0., 0.]

    # use ForwardDiff to calculate the gradient
    function objective(β, w)
        return sum(w .* (y .* log.(exp.(X * β)) .- exp.(X * β) .- loggamma.(Int.(y) .+ 1))) + (model.λ * sum(β.^2))
    end
    grad = -ForwardDiff.gradient(x -> objective(x, ones(1000)), model.β)
    # calculate the gradient manually
    grad_analytic = SSM.gradient!([0., 0., 0.], model, X, y)
    # check if the gradients are close
    @test isapprox(grad, grad_analytic, atol=1e-6)

    # now do the same with Weights
    weights = rand(1000)
    grad = -ForwardDiff.gradient(x -> objective(x, weights), model.β)
    grad_analytic = SSM.gradient!([0., 0., 0.], model, X, y, weights)
    @test isapprox(grad, grad_analytic, atol=1e-6)

    # finally test when λ is not 0
    model = PoissonRegression(λ=0.1)
    model.β = rand(3)

    grad = -ForwardDiff.gradient(x -> objective(x, ones(1000)), model.β)
    grad_analytic = SSM.gradient!([0., 0., 0.], model, X, y)
end

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
    γ = ones(100)
    SSM.updateEmissionModel!(gaussian_emission, data, γ)
    # Check if parameters are updated correctly
    @test gaussian_emission.μ ≈ mean(data, dims=1)'
    @test gaussian_emission.Σ ≈ cov(data, corrected=false)
end

function test_regression_emissions()
    # generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    true_σ² = 0.5
    # gaussian glm response
    y = X * true_β + rand(Normal(0., sqrt(true_σ²)), 1000)
    # poisson glm response
    y_poisson = rand.(Poisson.(exp.(X * true_β)))
    y_poisson = convert(Vector{Float64}, y_poisson)
    # bernoulli glm response
    y_bernoulli = rand.(Bernoulli.(logistic.(X * true_β)))
    y_bernoulli = convert(Vector{Float64}, y_bernoulli)
    # initialize emission models
    gaussian_emission = RegressionEmissions(GaussianRegression(;include_intercept=false))
    poisson_emission = RegressionEmissions(PoissonRegression(;include_intercept=false))
    bernoulli_emission = RegressionEmissions(BernoulliRegression(;include_intercept=false))
    # update emission models
    SSM.update_emissions_model!(gaussian_emission, X, y)
    SSM.update_emissions_model!(poisson_emission, X, y_poisson)
    SSM.update_emissions_model!(bernoulli_emission, X, y_bernoulli)
    # check if parameters are updated correctly
    @test isapprox(gaussian_emission.regression.β, true_β, atol=0.5)
    @test isapprox(gaussian_emission.regression.σ², true_σ², atol=0.1)
    @test isapprox(poisson_emission.regression.β, true_β, atol=0.5)
    @test isapprox(bernoulli_emission.regression.β, true_β, atol=0.5)
    # test the loglikelihood
    # ll_gaussian = SSM.loglikelihood(gaussian_emission, X[1, :], y[1])
    # ll_poisson = SSM.loglikelihood(poisson_emission, X[1, :], y_poisson[1])
    # ll_bernoulli = SSM.loglikelihood(bernoulli_emission, X[1, :], y_bernoulli[1])
    # @test ll_gaussian < 0
    # @test ll_poisson < 0
    # @test ll_bernoulli < 0
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
    # Now test kmeans on a vector.
    data = randn(100,)
    centroids, labels = kmeans_clustering(data, k_means)
    # Check dimensions
    @test size(centroids) == (1, k_means)
    @test length(labels) == 100
end

function test_block_tridgm()
    # Test with minimal block sizes
    super = [rand(1, 1) for i in 1:1]
    sub = [rand(1, 1) for i in 1:1]
    main = [rand(1, 1) for i in 1:2]
    A = block_tridgm(main, super, sub)
    @test size(A) == (2, 2)
    @test A[1, 1] == main[1][1, 1]
    @test A[2, 2] == main[2][1, 1]
    @test A[1, 2] == super[1][1, 1]
    @test A[2, 1] == sub[1][1, 1]

    # Test with 2x2 blocks and a larger matrix
    super = [rand(2, 2) for i in 1:9]
    sub = [rand(2, 2) for i in 1:9]
    main = [rand(2, 2) for i in 1:10]
    A = block_tridgm(main, super, sub)
    @test size(A) == (20, 20)

    # Check some blocks in the matrix
    for i in 1:10
        @test A[(2i-1):(2i), (2i-1):(2i)] == main[i]
        if i < 10
            @test A[(2i-1):(2i), (2i+1):(2i+2)] == super[i]
            @test A[(2i+1):(2i+2), (2i-1):(2i)] == sub[i]
        end
    end

    # Test with integer blocks
    super = [rand(Int, 2, 2) for i in 1:9]
    sub = [rand(Int, 2, 2) for i in 1:9]
    main = [rand(Int, 2, 2) for i in 1:10]
    A = block_tridgm(main, super, sub)
    @test size(A) == (20, 20)
    for i in 1:10
        @test A[(2i-1):(2i), (2i-1):(2i)] == main[i]
        if i < 10
            @test A[(2i-1):(2i), (2i+1):(2i+2)] == super[i]
            @test A[(2i+1):(2i+2), (2i-1):(2i)] == sub[i]
        end
    end
end

function test_interleave_reshape()
    # Test with valid data and dimensions
    data = collect(1:6)
    t = 2
    d = 3
    X = interleave_reshape(data, t, d)
    @test size(X) == (2, 3)
    @test X == [1 2 3; 4 5 6]

    # Test with another set of valid data and dimensions
    data = collect(1:12)
    t = 4
    d = 3
    X = interleave_reshape(data, t, d)
    @test size(X) == (4, 3)
    @test X == [1 2 3; 4 5 6; 7 8 9; 10 11 12]

    # Test with a longer set of data
    data = collect(1:20)
    t = 4
    d = 5
    X = interleave_reshape(data, t, d)
    @test size(X) == (4, 5)
    @test X == [1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20]

    # Test with float data
    data = collect(1.0:0.5:6.5)
    t = 4
    d = 3
    X = interleave_reshape(data, t, d)
    @test size(X) == (4, 3)
    @test X == [1.0 1.5 2.0; 2.5 3.0 3.5; 4.0 4.5 5.0; 5.5 6.0 6.5]

    # Test with mismatched dimensions (should raise an error)
    data = collect(1:11)
    t = 2
    d = 5
    @test_throws ErrorException interleave_reshape(data, t, d)
end

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

"""
Tests for MarkovRegression.jl
"""
function test_hmmglm_properties(model::SSM.hmmglm)
    # test basic properties of the model
    @test size(model.A) == (model.K, model.K)
    @test length(model.B) == model.K
    @test length(model.πₖ) == model.K
    @test sum(model.πₖ) ≈ 1.0
    @test sum(model.A, dims=2) ≈ ones(model.K)
end

function test_HMMGLM_initialization()
    # initialize models
    K = 3
    gaussian_model = SwitchingGaussianRegression(K=K)
    bernoulli_model = SwitchingBernoulliRegression(K=K)
    poisson_model = SwitchingPoissonRegression(K=K)
    # test properties
    test_hmmglm_properties(gaussian_model)
    test_hmmglm_properties(bernoulli_model)
    test_hmmglm_properties(poisson_model)
end

@testset "SwitchingRegression Tests" begin
    test_HMMGLM_initialization()
end