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

function testGaussianMixtureModel_E_Step(gmm::GaussianMixtureModel, data::Union{Matrix{Float64}, Vector{Float64}})

    k::Int = gmm.k
    data_dim::Int = size(data, 2)
    
    # Run E_Step
    class_probabilities = SSM.E_Step(gmm, data)
    # Check dimensions
    @test size(class_probabilities) == (size(data, 1), k)
    # Check if the row sums are close to 1 (since they represent probabilities)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(class_probabilities, dims=2))
    
    test_GaussianMixtureModel_properties(gmm, k, data_dim)
end

function testGaussianMixtureModel_M_Step(gmm::GaussianMixtureModel, data::Union{Matrix{Float64}, Vector{Float64}})

    k::Int = gmm.k
    data_dim::Int = size(data, 2)

    class_probabilities = SSM.E_Step(gmm, data)

    # Run M_Step
    SSM.M_Step!(gmm, data, class_probabilities)

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

function testPoissonMixtureModel_E_Step(pmm::PoissonMixtureModel, data::Union{Matrix{Int}, Vector{Int}})
    k::Int = pmm.k
    
    # Run E_Step
    class_probabilities = SSM.E_Step(pmm, data)
    # Check dimensions
    @test size(class_probabilities) == (size(data, 1), k)
    # Check if the row sums are close to 1 (since they represent probabilities)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(class_probabilities, dims=2))
    
    test_PoissonMixtureModel_properties(pmm, k)
end

function testPoissonMixtureModel_M_Step(pmm::PoissonMixtureModel, data::Union{Matrix{Int}, Vector{Int}})
    k::Int = pmm.k

    class_probabilities = SSM.E_Step(pmm, data)

    # Run M_Step
    SSM.M_Step!(pmm, data, class_probabilities)

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
        testGaussianMixtureModel_E_Step(gmm, data)

        gmm = GaussianMixtureModel(k, data_dim)
        testGaussianMixtureModel_M_Step(gmm, data)

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
        testPoissonMixtureModel_E_Step(pmm, data)
        pmm = PoissonMixtureModel(k)
        testPoissonMixtureModel_M_Step(pmm, data)
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
    kf = LDS(;A=A, H=H, Q=Q, R=R, x0=x0, p0=p0, obs_dim=2, latent_dim=2, fit_bool=Vector([false, false, false, false, false, false, false, false]))
    # confirm parameters are set correctly
    @test kf.A == A
    @test kf.H == H
    @test isapprox(kf.B, zeros(kf.latent_dim, size(kf.inputs, 2)), atol=1e-6)
    @test kf.Q == Q
    @test kf.R == R
    @test kf.x0 == x0
    @test kf.p0 == p0
    @test isapprox(kf.inputs, zeros(1, 1), atol=1e-6)
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
    kf = LDS(; obs_dim=2, latent_dim=2, fit_bool=Vector([true, true, true, true, true, true, true]))
    # confirm parameters are set correctly
    @test !isempty(kf.A)
    @test !isempty(kf.H)
    @test !isempty(kf.B)
    @test !isempty(kf.Q)
    @test !isempty(kf.R)
    @test !isempty(kf.x0)
    @test !isempty(kf.p0)
    @test !isempty(kf.inputs)
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == fill(true, 7)
end

function test_LDS_E_Step()
    # Create the Kalman filter parameter vector
    kf = LDS(;A=A, H=H, Q=Q, R=R, x0=x0, p0=p0, obs_dim=2, latent_dim=2, fit_bool=Vector([true, true, true, true, true, true, true]))
    # run the E_Step
    x_smooth, p_smooth, E_z, E_zz, E_zz_prev, ml = SSM.E_Step(kf, x_noisy')
    # check dimensions
    @test size(x_smooth) == (length(t), 2)
    @test size(p_smooth) == (length(t), 2, 2)
    @test size(E_z) == (length(t), 2)
    @test size(E_zz) == (length(t), 2, 2)
    @test size(E_zz_prev) == (length(t), 2, 2)
    @test size(ml) == ()
end

function test_LDS_M_Step!()
    # Create the Kalman filter parameter vector
    kf = LDS(;A=A, H=H, Q=Q, R=R, x0=x0, p0=p0, obs_dim=2, latent_dim=2, fit_bool=Vector([true, true, true, true, true, true, true]))
    # run the E_Step
    x_smooth, p_smooth, E_z, E_zz, E_zz_prev, ml = SSM.E_Step(kf, x_noisy')
    # run the M_Step
    SSM.M_Step!(kf, E_z, E_zz, E_zz_prev, x_noisy')
    # check if the parameters are updated
    @test kf.A !== A
    @test kf.H !== H
    @test isapprox(kf.B, zeros(kf.latent_dim, 1))
    @test kf.Q !== Q
    @test kf.R !== R
    @test kf.x0 !== x0
    @test kf.p0 !== p0
    @test isapprox(kf.inputs, zeros(1, 1))
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == Vector([true, true, true, true, true, true, true])
end

function test_LDS_EM()
    kf = LDS(;A=A, H=H, Q=Q, R=R, x0=x0, p0=p0, obs_dim=2, latent_dim=2, fit_bool=Vector([true, true, true, true, true, true, true]))
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
    @test isapprox(kf.B, zeros(kf.latent_dim, 1))
    @test kf.Q !== Q
    @test kf.R !== R
    @test kf.x0 !== x0
    @test kf.p0 !== p0
    @test isapprox(kf.inputs, zeros(1, 1))
    @test kf.obs_dim == 2
    @test kf.latent_dim == 2
    @test kf.fit_bool == Vector([true, true, true, true, true, true, true]) 
end

function test_direct_smoother()
    # create kalman filter object
    kf = LDS(;A=A, H=H, Q=Q, R=R, x0=x0, p0=p0, obs_dim=2, latent_dim=2, fit_bool=Vector([true, true, true, true, true, true, true]))
    # run the RTS-Smoother
    x_smooth, p_smooth = KalmanSmoother(kf, x_noisy')
    # run the Direct Smoothing algorithm
    x_smooth_direct, p_smooth_direct = KalmanSmoother(kf, permutedims(x_noisy), "Direct")
    @test size(x_smooth) == size(x_smooth_direct)
    @test size(p_smooth) == size(p_smooth_direct)
    # check if the results are the same
    @test isapprox(x_smooth, x_smooth_direct, atol=1e-6)
    @test isapprox(p_smooth, p_smooth_direct, atol=1e-6)
end

function test_LDS_gradient()
    # create kalman filter object
    kf = LDS(;A=A, H=H, Q=Q, R=R, x0=x0, p0=p0, obs_dim=2, latent_dim=2, fit_bool=Vector([true, true, true, true, true, true, true]))
    # calcualte the gradient
    grad = SSM.Gradient(kf, x_noisy', zeros(size(x_noisy')))
    # check dimensions
    @test size(grad) == (length(t), kf.obs_dim)
    # calculate the gradient using autodiff
    obj(x) = x -> SSM.loglikelihood(x, kf, x_noisy')
    grad_auto = ForwardDiff.gradient(obj(x), zeros(size(x_noisy')))
    # check if the gradients are the same
    @test isapprox(grad, grad_auto, atol=1e-6)
end

function test_LDS_Hessian()
    # create kalman filter object
    kf = LDS(;A=A, H=H, Q=Q, R=R, x0=x0, p0=p0, obs_dim=2, latent_dim=2, fit_bool=Vector([true, true, true, true, true, true, true]))
    # calcualte the Hessian
    hess, main, super, sub = SSM.Hessian(kf, x_noisy[:, 1:3]') # only look at first three observations as hessian is expensive to calculate using autodiff

    # check lengths of main, super, and sub diagonals
    @test typeof(main) == Vector{Matrix{Float64}}
    @test typeof(super) == Vector{Matrix{Float64}}
    @test typeof(sub) == Vector{Matrix{Float64}}
    @test length(main) == 3
    @test length(super) == 2
    @test length(sub) == 2

    # check dimensions
    @test size(hess) == (3*kf.obs_dim, 3*kf.obs_dim)

    # calculate the Hessian using autodiff
    function log_likelihood(x::AbstractArray, l::LDS, y::AbstractArray)
        # this wrapper function just makes it so we can pass a D x T array and not a T x D array. Otherwise the Hessian is out of order.
        x = x'
        ll = SSM.loglikelihood(x, l, y)
        return ll  # Negate the log-likelihood
    end
    obj(x) = x -> log_likelihood(x, kf, zeros(size(x_noisy[:, 1:3]')))
    hess_auto = ForwardDiff.hessian(obj(x), zeros(size(x_noisy[:, 1:3])))
    # check if the Hessian are the same
    @test isapprox(Matrix(hess), hess_auto, atol=1e-6)
end



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

function GaussianRegression_simulation()
    # Generate synthetic data
    n = 1000
    X = hcat(ones(n), randn(n, 2))
    true_β = [0.5, -1.2, 2.3]
    true_β = reshape(true_β, 3, 1)
    true_covariance = reshape([0.25], 1, 1)
    y = X * true_β + rand(MvNormal(zeros(1), true_covariance), n)'

    # Remove the intercept column
    X = X[:, 2:end] 
    
    # Initialize and fit the model
    model = GaussianRegression(num_features=2, num_targets=1)
    model.β = ones(3, 1)
    model.Σ = ones(1, 1)
    fit!(model, X, y)

    return model, X, y, true_β, true_covariance, n
end

function test_GaussianRegression_fit()
    model, X, y, true_β, true_covariance, n = GaussianRegression_simulation()
    
    # Check if the fitted coefficients are close to the true coefficients
    @test isapprox(model.β, true_β, atol=0.5)
    @test isposdef(model.Σ)
    @test isapprox(model.Σ, true_covariance, atol=0.1)
end

function test_GaussianRegression_loglikelihood()
    model, X, y, true_β, true_covariance, n = GaussianRegression_simulation()
    
    # Check log likelihood
    loglik = SSM.loglikelihood(model, X, y)
    @test loglik < 0

    # test loglikelihood on a single point
    loglik = SSM.loglikelihood(model, reshape(X[1, :], 1, :), reshape(y[1,:], 1, :))
    @test loglik < 0
end

function test_GaussianRegression_default_model()
    model = GaussianRegression(num_features=2, num_targets=1, include_intercept=false)
    @test model.β == ones(2, 1)
    @test model.Σ == ones(1, 1)
end

function test_GaussianRegression_intercept()
    model, X, y, true_β, true_covariance, n = GaussianRegression_simulation()
    
    model = GaussianRegression(num_features=2, num_targets=1, include_intercept=false)
    model.β = ones(2, 1)
    model.Σ = ones(1, 1)
    fit!(model, X, y)
    @test length(model.β) == 2
    
end

function test_Gaussian_ll_gradient()
    # Generate synthetic data
    n = 1000
    X = hcat(ones(n), randn(n, 2))
    true_β = [0.5, -1.2, 2.3]
    true_β = reshape(true_β, 3, 1)
    true_covariance = reshape([0.25], 1, 1)
    y = X * true_β + rand(MvNormal(zeros(1), true_covariance), n)'

    
    # Initialize model
    model = GaussianRegression(num_features=2, num_targets=1)
    model.β = ones(3, 1)
    model.Σ = ones(1, 1)

    # use ForwardDiff to calculate the gradient
    function objective(β, w, model)
        # calculate log likelihood
        residuals = y - X * β

        # reshape w for broadcasting
        w = reshape(w, (length(w), 1))

        log_likelihood = -0.5 * sum(broadcast(*, w, residuals.^2)) - (model.λ * sum(β.^2))

        
        return log_likelihood
    end


    w = ones(size(X, 1))

    grad = ForwardDiff.gradient(β -> objective(β, w, model), model.β)
    # calculate the gradient manually
    grad_analytic = ones(size(model.β))
    SSM.surrogate_loglikelihood_gradient!(grad_analytic, model, X[:,2:end], y)

    # check if the gradients are close
    @test isapprox(grad, grad_analytic, atol=1e-6)



    # # now do the same with Weights
    w = rand(size(X, 1))


    grad = ForwardDiff.gradient(β -> objective(β, w, model), model.β)


    # calculate the gradient manually
    grad_analytic = ones(size(model.β))
    SSM.surrogate_loglikelihood_gradient!(grad_analytic, model, X[:,2:end], y, w)

    # check if the gradients are close
    @test isapprox(grad, grad_analytic, atol=1e-6)



    # finally test when λ is not 0
    model = GaussianRegression(num_features=2, num_targets=1, λ=0.1)
    model.β = ones(3, 1)
    model.Σ = ones(1, 1)

    grad = ForwardDiff.gradient(β -> objective(β, w, model), model.β)


    # calculate the gradient manually
    grad_analytic = ones(size(model.β))
    SSM.surrogate_loglikelihood_gradient!(grad_analytic, model, X[:,2:end], y, w)

    # check if the gradients are close
    @test isapprox(grad, grad_analytic, atol=1e-6)

end

@testset "GaussianRegression Tests" begin
    test_GaussianRegression_fit()
    test_GaussianRegression_loglikelihood()
    test_GaussianRegression_default_model()
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
    gaussian_emission = RegressionEmissions(GaussianRegression(num_features=3, num_targets=1;include_intercept=false))
    poisson_emission = RegressionEmissions(PoissonRegression(;include_intercept=false))
    bernoulli_emission = RegressionEmissions(BernoulliRegression(;include_intercept=false))
    # update emission models
    SSM.update_emissions_model!(gaussian_emission, X, reshape(y, 1000, 1))
    SSM.update_emissions_model!(poisson_emission, X, y_poisson)
    SSM.update_emissions_model!(bernoulli_emission, X, y_bernoulli)
    # check if parameters are updated correctly
    @test isapprox(gaussian_emission.regression.β, reshape(true_β, 3, 1), atol=0.5)
    @test isapprox(gaussian_emission.regression.Σ, reshape([true_σ²], 1, 1), atol=0.1)
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
    gaussian_model = SwitchingGaussianRegression(num_features=2, num_targets=1, K=K)
    bernoulli_model = SwitchingBernoulliRegression(K=K)
    poisson_model = SwitchingPoissonRegression(K=K)
    # test properties
    test_hmmglm_properties(gaussian_model)
    test_hmmglm_properties(bernoulli_model)
    test_hmmglm_properties(poisson_model)

    @test gaussian_model.B[1].regression.λ == 0.0
    @test gaussian_model.B[2].regression.λ == 0.0
    @test bernoulli_model.B[1].regression.λ == 0.0
    @test bernoulli_model.B[2].regression.λ == 0.0
    @test poisson_model.B[1].regression.λ == 0.0
    @test poisson_model.B[2].regression.λ == 0.0

end

@testset "SwitchingRegression Tests" begin
    test_HMMGLM_initialization()
end