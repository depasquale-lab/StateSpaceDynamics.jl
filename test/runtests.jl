using SSM
using Distributions
using LinearAlgebra
using Random
using Test
using UnPack

Random.seed!(1234)

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

function test_HMM_properties(hmm::AbstractHMM)
    @test isapprox(sum(hmm.A, dims=2), ones(hmm.K))
    @test typeof(hmm.B) == Vector{GaussianEmission}
    @test sum(hmm.πₖ) ≈ 1.0
end

function test_GaussianHMM_constructor()
    hmm, _ = toy_HMM()
    # check if parameters are initialized correctly
    test_HMM_properties(hmm)
end

function test_HMM_forward_and_back()
    # Initialize toy model and data
    hmm, data = toy_HMM()
    # Run forward algorithm
    α = SSM.forward(hmm, data)
    # Check dimensions
    @test size(α) == (1000, hmm.K)
    # Run backward algorithm
    β = SSM.backward(hmm, data)
    # Check dimensions
    @test size(β) == (1000, hmm.K)
end

function test_HMM_gamma_xi()
    # Initialize toy model and data
    hmm, data = toy_HMM()
    # Run forward and backward algorithms
    α = SSM.forward(hmm, data)
    β = SSM.backward(hmm, data)
    # Calculate gamma and xi
    γ = SSM.calculate_γ(hmm, α, β)
    ξ = SSM.calculate_ξ(hmm, α, β, data)
    # Check dimensions
    @test size(γ) == (1000, hmm.K)
    @test size(ξ) == (999, hmm.K, hmm.K)
    # Check if the row sums of gamma are close to 1
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(exp.(γ), dims=2))
    # Check if the row sums of xi are close to 1
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(exp.(ξ), dims=(2, 3)))
end

function test_HMM_E_step()
end

function test_HMM_EM()
    Random.seed!(1234)
    A = [0.7 0.2 0.1; 0.1 0.7 0.2; 0.2 0.1 0.7]
        means = [[0.0, 0.0], [-1.0, 2.0], [3.0, 2.5]]
        covs = [
                    [0.1 0.0; 0.0 0.1],  # Covariance matrix for state 1
                    [0.1 0.0; 0.0 0.1],  # Covariance matrix for state 2
                    [0.1 0.0; 0.0 0.1]   # Covariance matrix for state 3
                ]
    emissions_models = [GaussianEmission(mean, cov) for (mean, cov) in zip(means, covs)]
    simul_hmm = GaussianHMM(A, emissions_models, [0.33, 0.33, 0.34], 3, 2)
    states, observations = SSM.sample(simul_hmm, 10000)
    # Initialize HMM
    k = 3
    data_dim = 2
    hmm = GaussianHMM(observations, 3)
    baumWelch!(hmm, observations, 100)
    # Check if the transition matrix is close to the simulated one
    # @test hmm.A ≈ A atol=1e-1
    # Check if the means are close to the simulated ones
    pred_means = [hmm.B[i].μ for i in 1:k]
    @test sort(pred_means) ≈ sort(means) atol=2e-1
    # Check if the covariance matrices are close to the simulated ones
    pred_covs = [hmm.B[i].Σ for i in 1:k]
    @test pred_covs ≈ covs atol=1e-1
    # Check viterbi now
    best_path = viterbi(hmm, observations)
    @test length(best_path) == 10000
    @test all(x -> x in 1:k, best_path)
end

@testset "HiddenMarkovModels.jl Tests" begin
    test_GaussianHMM_constructor()
    test_HMM_forward_and_back()
    test_HMM_gamma_xi()
    test_HMM_EM()
end

"""
Tests for LDS.jl
"""
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
p0 = 0.1*I(2)  # Initial state covariance
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
    x_filt, p_filt, x_pred, p_pred, v, F, K, ml = KalmanFilter(kf, x_noisy')
    # check dimensions
    @test size(x_filt) == (length(t), 2)
    @test size(p_filt) == (length(t), 2, 2)
    @test size(x_pred) == (length(t), 2)
    @test size(p_pred) == (length(t), 2, 2)
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
    x_smooth, p_smooth, E_z, E_zz, E_zz_prev, ml = SSM.EStep(kf, x_noisy')
    # check dimensions
    @test size(x_smooth) == (length(t), 2)
    @test size(p_smooth) == (length(t), 2, 2)
    @test size(E_z) == (length(t), 2)
    @test size(E_zz) == (length(t), 2, 2)
    @test size(E_zz_prev) == (length(t), 2, 2)
    @test size(ml) == ()
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
    x_smooth, p_smooth, E_z, E_zz, E_zz_prev, ml = SSM.EStep(kf, x_noisy')
    # run the MStep
    SSM.MStep!(kf, E_z, E_zz, E_zz_prev, x_noisy')
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
    for i in 1:10
        ml_prev = -Inf
        l, ml = SSM.KalmanFilterEM!(kf, x_noisy', 1)
        @test ml > ml_prev
        ml_prev = ml
    end
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


@testset "LDS.jl Tests" begin
    test_LDS_with_params()
    test_LDS_without_params()
    test_LDS_EStep()
    test_LDS_MStep!()
    test_LDS_EM()
end

"""
Tests for Regression.jl
"""

#TODO: implement test for regression

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

@testset "Emissions.jl Tests" begin
    test_GaussianEmission()
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

# create a toy autoregression for testing; AR(3)
α = [0.1, -0.3, 0.2]
x = Vector{Float64}(undef, 2000)
for i in 1:2000
    if i <= 3
        x[i] = randn()
    else
        x[i] = α[1]*x[i-1] + α[2]*x[i-2] + α[3]*x[i-3] + randn()
    end
end

function test_autoregression()
    ar = Autoregression(x, 3)
    # check if parameters are initialized correctly
    @test ar.X == x
    @test ar.p == 3
    @test ar.β == zeros(4)
    @test ar.σ² == 1.0
end

function test_fit_autoregression()
    ar = Autoregression(x, 3)
    # fit the model
    fit!(ar)
    # check if parameters are updated correctly
    @test ar.β ≈ append!(α, 0) atol=2e-1
    @test ar.σ² ≈ 1.0 atol=1e-1
end


@testset "Utilities.jl Tests" begin
    test_euclidean_distance()
    test_kmeanspp_initialization()
    test_kmeans_clustering()
    test_autoregression()
    test_fit_autoregression()
end

"""
Tests for MarkovRegression.jl
"""
