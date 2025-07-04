function test_PPCA_with_params()
    # Set parameters
    D = 3
    k = 2 
    W = randn(D, k)
    σ² = 0.5
    T = eltype(W)
   
    # create "data"
    num_obs = 100 
    X = randn(D, num_obs)
    μ_vector = vec(mean(X; dims=2))

    # create PPCA struct
    ppca = ProbabilisticPCA(W, σ², μ_vector)

    # Check if parameters are set correctly
    @test size(ppca.W) == (D, k)
    @test ppca.σ² === σ²
    @test ppca.D === D
    @test ppca.k === k
    @test isempty(ppca.z)
end

function test_PPCA_E_and_M_Step()
    # create ppca object 
    D = 3
    k = 2 
    W = randn(D, k)
    σ² = 0.5
    T = eltype(W)
   
    num_obs = 100 
    X = randn(D, num_obs)
    μ_vector = vec(mean(X; dims=2))

    ppca = ProbabilisticPCA(W, σ², μ_vector)

    # run E-step    
    E_z, E_zz = StateSpaceDynamics.estep(ppca, X)
    # check dimensions
    @test size(E_z) == (k, 100)
    @test size(E_zz) == (k, k, 100)
    # run M-step, but first save the old parameters
    W_old = copy(ppca.W)  # ← THIS IS THE KEY FIX
    σ²_old = copy(ppca.σ²)
    StateSpaceDynamics.mstep!(ppca, X, E_z, E_zz)
    # check if the parameters are updated
    @test ppca.W != W_old
    @test ppca.σ² != σ²_old
    @test ppca.μ == μ_vector
end

function test_PPCA_fit()
    # create ppca object
    D = 3
    k = 2 
    W = randn(D, k)
    σ² = 0.5
    T = eltype(W)
   
    num_obs = 100 
    X = randn(D, num_obs)
    μ_vector = vec(mean(X; dims=2))

    M = typeof(W)
    V = typeof(μ_vector)

    ppca = ProbabilisticPCA(W, σ², μ_vector)

    # fit the model
    ll = fit!(ppca, X)
    # check if the parameters are updated
    @test ppca.σ² > 0
    @test size(ppca.W) == (D, k)
    @test size(ppca.μ) == (D,)
    @test size(ppca.z) == (k, num_obs)
    # check loglikelihood only increases
    @test all(diff(ll) .> 0)
    # also check that the loglikelihood is a scalar
    ll = StateSpaceDynamics.loglikelihood(ppca, X)
    @test size(ll) == ()
end

function test_PPCA_samples()
    D = 3
    k = 2 
    W = randn(D, k)
    T = eltype(W)

    σ² = 0.5
    num_obs = 100 
    X = randn(D, num_obs)
    μ = randn(3)

    ppca = ProbabilisticPCA(W, σ², μ)

    X, z = rand(ppca, 10000)

    @test size(X) == (3, 10000)
    @test size(z) == (2, 10000)

    # Test empirical mean
    empirical_mean = mean(X; dims=2)
    @test all(isapprox.(empirical_mean, μ; atol=0.05))

    # Test noise level
    residuals = X .- (W * z .+ μ)
    residual_norm = norm(residuals) / √(ppca.D * size(X, 2))
    @test abs(residual_norm - sqrt(σ²)) < 0.05
end