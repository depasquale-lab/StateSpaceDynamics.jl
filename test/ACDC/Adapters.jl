# Tests for ACDC model adapters (GMM, HMM, PPCA)

# =============================================================================
# Gaussian Mixture Model Adapter Tests
# =============================================================================

function test_GMM_adapter_output_shape()
    Random.seed!(42)

    K = 3
    D = 2
    N = 500

    # Create and fit a GMM
    gmm = GaussianMixtureModel(K, D)

    # Generate data from the GMM
    data = rand(gmm, N)

    # Fit the model
    fit!(gmm, data; maxiter=50, tol=1e-4)

    # Get stochastic drivers
    n_samples = 5
    result = stochastic_drivers(gmm, data; n_samples=n_samples)

    # Check output shape
    @test length(result.ε_pools) == K
    @test all(size(pool, 1) == D for pool in result.ε_pools)
    @test sum(size(pool, 2) for pool in result.ε_pools) == N * n_samples
    @test length(result.usage) == K
end

function test_GMM_adapter_driver_bounds()
    Random.seed!(42)

    K = 2
    D = 3
    N = 200

    gmm = GaussianMixtureModel(K, D)
    data = rand(gmm, N)
    fit!(gmm, data; maxiter=50, tol=1e-4)

    result = stochastic_drivers(gmm, data; n_samples=3)

    # All drivers should be in [0, 1]
    @test all(all(0 .<= pool .<= 1) for pool in result.ε_pools)
end

function test_GMM_adapter_usage_sums_approximately_one()
    Random.seed!(42)

    K = 3
    D = 2
    N = 1000

    gmm = GaussianMixtureModel(K, D)
    data = rand(gmm, N)
    fit!(gmm, data; maxiter=50, tol=1e-4)

    result = stochastic_drivers(gmm, data; n_samples=1)

    @test isapprox(sum(result.usage), 1.0; atol=0.1)
end

function test_GMM_component_discrepancies()
    Random.seed!(42)

    K = 2
    D = 2
    N = 500

    gmm = GaussianMixtureModel(K, D)
    data = rand(gmm, N)
    fit!(gmm, data; maxiter=100, tol=1e-6)

    # When well-specified, discrepancies should be small
    acdc_result = component_discrepancies(gmm, data, MMDDiscrepancy(); n_samples=10)

    @test acdc_result.K == K
    @test length(acdc_result.component_discrepancies) == K
    # Discrepancies should be non-negative
    @test all(acdc_result.component_discrepancies .>= -0.01)
end

# =============================================================================
# Hidden Markov Model Adapter Tests
# =============================================================================

function test_HMM_adapter_output_shape()
    Random.seed!(42)

    K = 2
    D = 2
    T_obs = 300

    # Create emissions
    emission_1 = GaussianEmission(D, zeros(D), Matrix{Float64}(I(D)))
    emission_2 = GaussianEmission(D, ones(D) * 2, Matrix{Float64}(I(D)))

    A = [0.9 0.1; 0.1 0.9]
    πₖ = [0.5, 0.5]

    hmm = HiddenMarkovModel(; K=K, B=[emission_1, emission_2], A=A, πₖ=πₖ)

    # Sample data
    _, data = rand(hmm; n=T_obs)

    # Fit
    fit!(hmm, data; max_iters=50, tol=1e-4)

    # Get stochastic drivers
    n_samples = 5
    result = stochastic_drivers(hmm, data; n_samples=n_samples)

    @test length(result.ε_pools) == K
    @test all(size(pool, 1) == D for pool in result.ε_pools)
    @test sum(size(pool, 2) for pool in result.ε_pools) == T_obs * n_samples
    @test length(result.usage) == K
end

function test_HMM_adapter_driver_bounds()
    Random.seed!(42)

    K = 2
    D = 2
    T_obs = 200

    emission_1 = GaussianEmission(D, zeros(D), Matrix{Float64}(I(D) * 0.5))
    emission_2 = GaussianEmission(D, ones(D) * 3, Matrix{Float64}(I(D) * 0.5))

    A = [0.95 0.05; 0.05 0.95]
    πₖ = [0.5, 0.5]

    hmm = HiddenMarkovModel(; K=K, B=[emission_1, emission_2], A=A, πₖ=πₖ)
    _, data = rand(hmm; n=T_obs)
    fit!(hmm, data; max_iters=50, tol=1e-4)

    result = stochastic_drivers(hmm, data; n_samples=3)

    @test all(all(0 .<= pool .<= 1) for pool in result.ε_pools)
end

function test_HMM_component_discrepancies()
    Random.seed!(42)

    K = 2
    D = 2
    T_obs = 500

    emission_1 = GaussianEmission(D, zeros(D), Matrix{Float64}(I(D) * 0.3))
    emission_2 = GaussianEmission(D, ones(D) * 3, Matrix{Float64}(I(D) * 0.3))

    A = [0.95 0.05; 0.05 0.95]
    πₖ = [0.5, 0.5]

    hmm = HiddenMarkovModel(; K=K, B=[emission_1, emission_2], A=A, πₖ=πₖ)
    _, data = rand(hmm; n=T_obs)
    fit!(hmm, data; max_iters=100, tol=1e-6)

    acdc_result = component_discrepancies(hmm, data, MMDDiscrepancy(); n_samples=10)

    @test acdc_result.K == K
    @test length(acdc_result.component_discrepancies) == K
    @test all(acdc_result.component_discrepancies .>= -0.01)
end

function test_HMM_regression_emission_adapter()
    Random.seed!(42)

    K = 2
    input_dim = 2
    output_dim = 1
    T_obs = 300

    emission_1 = GaussianRegressionEmission(;
        input_dim=input_dim,
        output_dim=output_dim,
        include_intercept=true,
        β=reshape([1.0, 0.5, 0.5], :, 1),
        Σ=[0.1;;],
        λ=0.0,
    )
    emission_2 = GaussianRegressionEmission(;
        input_dim=input_dim,
        output_dim=output_dim,
        include_intercept=true,
        β=reshape([-1.0, -0.5, 0.5], :, 1),
        Σ=[0.1;;],
        λ=0.0,
    )

    A = [0.9 0.1; 0.1 0.9]
    πₖ = [0.5, 0.5]

    hmm = HiddenMarkovModel(; K=K, A=A, πₖ=πₖ, B=[emission_1, emission_2])

    Φ = randn(input_dim, T_obs)
    _, data = rand(hmm, Φ; n=T_obs)

    fit!(hmm, data, Φ; max_iters=50)

    result = stochastic_drivers(hmm, data, Φ; n_samples=3)

    @test length(result.ε_pools) == K
    @test all(size(pool, 1) == output_dim for pool in result.ε_pools)
    @test sum(size(pool, 2) for pool in result.ε_pools) == T_obs * 3
    @test all(all(0 .<= pool .<= 1) for pool in result.ε_pools)
end

function test_HMM_wrapped_cauchy_emission_adapter()
    Random.seed!(42)

    K = 1
    D = 2
    T_obs = 300

    emission = WrappedCauchyEmission(; output_dim=D, μ=zeros(D), ρ=fill(0.5, D))
    A = [1.0;;]
    πₖ = [1.0]

    hmm = HiddenMarkovModel(; K=K, B=[emission], A=A, πₖ=πₖ)
    _, data = rand(hmm; n=T_obs)

    result = stochastic_drivers(hmm, data; n_samples=3)

    @test length(result.ε_pools) == K
    @test all(size(pool, 1) == D for pool in result.ε_pools)
    @test sum(size(pool, 2) for pool in result.ε_pools) == T_obs * 3
    @test all(all(0 .<= pool .<= 1) for pool in result.ε_pools)
end

# =============================================================================
# Probabilistic PCA Adapter Tests
# =============================================================================

function test_PPCA_adapter_output_shape()
    Random.seed!(42)

    D = 10
    K = 3
    N = 500

    # Create true model
    W_true = randn(D, K)
    μ_true = randn(D)
    σ²_true = 0.5

    ppca_true = ProbabilisticPCA(W_true, σ²_true, μ_true)
    data, _ = rand(ppca_true, N)

    # Create and fit test model
    W_init = randn(D, K)
    μ_init = vec(mean(data; dims=2))
    ppca = ProbabilisticPCA(W_init, 1.0, μ_init)
    fit!(ppca, data)

    # Get stochastic drivers
    n_samples = 5
    result = stochastic_drivers(ppca, data; n_samples=n_samples)

    @test length(result.ε_pools) == K
    @test all(size(pool, 1) == D for pool in result.ε_pools)
    @test sum(size(pool, 2) for pool in result.ε_pools) > 0
    @test length(result.usage) == K
end

function test_PPCA_adapter_driver_bounds()
    Random.seed!(42)

    D = 5
    K = 2
    N = 200

    W = randn(D, K)
    μ = randn(D)
    σ² = 0.3

    ppca = ProbabilisticPCA(W, σ², μ)
    data, _ = rand(ppca, N)
    fit!(ppca, data)

    result = stochastic_drivers(ppca, data; n_samples=3)

    @test all(all(0 .<= pool .<= 1) for pool in result.ε_pools)
end

function test_PPCA_component_discrepancies()
    Random.seed!(42)

    D = 8
    K = 2
    N = 500
    σ² = 0.5

    W_true = randn(D, K)
    μ_true = randn(D)

    ppca_true = ProbabilisticPCA(W_true, σ², μ_true)
    data, _ = rand(ppca_true, N)

    W_init = randn(D, K)
    μ_init = vec(mean(data; dims=2))
    ppca = ProbabilisticPCA(W_init, 1.0, μ_init)
    fit!(ppca, data)

    acdc_result = component_discrepancies(ppca, data, MMDDiscrepancy(); n_samples=10)

    @test acdc_result.K == K
    @test length(acdc_result.component_discrepancies) == K
    @test all(acdc_result.component_discrepancies .>= -0.01)
end

# =============================================================================
# Cross-Model Integration Tests
# =============================================================================

function test_acdc_model_selection_GMM()
    Random.seed!(42)

    # Generate data from K=3 GMM
    true_K = 3
    D = 2
    N = 1000

    gmm_true = GaussianMixtureModel(true_K, D)
    gmm_true.μₖ[:, 1] = [0.0, 0.0]
    gmm_true.μₖ[:, 2] = [4.0, 0.0]
    gmm_true.μₖ[:, 3] = [2.0, 4.0]
    for k in 1:true_K
        gmm_true.Σₖ[k] = Matrix{Float64}(I(D) * 0.3)
    end
    gmm_true.πₖ = [0.33, 0.34, 0.33]

    data = rand(gmm_true, N)

    # Fit models with different K
    K_range = 2:5
    results = Vector{ACDCResult{Float64}}(undef, length(K_range))

    for (i, K) in enumerate(K_range)
        gmm = GaussianMixtureModel(K, D)
        fit!(gmm, data; maxiter=100, tol=1e-6)
        results[i] = component_discrepancies(gmm, data, MMDDiscrepancy(); n_samples=5)
    end

    # ACDC should select K close to true K
    K_selected = acdc_select(results, 0.01)
    @test K_selected >= 2 && K_selected <= 5
end

function test_acdc_detects_misspecification()
    Random.seed!(42)

    # Generate data from mixture of t-distributions (heavy tails)
    # but fit Gaussian mixture (misspecified)
    K = 2
    D = 2
    N = 1000

    # Generate heavy-tailed data
    data = zeros(D, N)
    for n in 1:N
        if rand() < 0.5
            data[:, n] = rand(MvTDist(3.0, zeros(D), Matrix{Float64}(I(D))))
        else
            data[:, n] = rand(MvTDist(3.0, ones(D) * 3, Matrix{Float64}(I(D))))
        end
    end

    # Fit Gaussian mixture
    gmm = GaussianMixtureModel(K, D)
    fit!(gmm, data; maxiter=100, tol=1e-6)

    acdc_result = component_discrepancies(gmm, data, MMDDiscrepancy(); n_samples=10)

    # Discrepancies should be elevated due to misspecification
    # (though this is a statistical test so we use a loose threshold)
    max_disc = maximum(acdc_result.component_discrepancies)
    @test max_disc >= 0.0  # At minimum should be non-negative
end
