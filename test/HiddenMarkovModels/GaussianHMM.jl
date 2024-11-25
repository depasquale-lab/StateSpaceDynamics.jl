function test_SwitchingGaussian_fit()
    Random.seed!(1234)
    # Create Guassian Emission Models
    output_dim = 2
    μ = [0.0, 0.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_1 = GaussianEmission(output_dim, μ, Σ)

    μ = [2.0, 1.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_2 = GaussianEmission(output_dim, μ, Σ)

    # Create GaussianHMM
    true_model = StateSpaceDynamics.GaussianHMM(; K=2, output_dim=2)
    true_model.B[1] = emission_1
    true_model.B[2] = emission_2
    true_model.A = [0.9 0.1; 0.2 0.8]

    # Sample from the model
    n = 50000
    true_labels, data = StateSpaceDynamics.sample(true_model; n=n)

    # Fit a gaussian hmm to the data
    test_model = StateSpaceDynamics.GaussianHMM(; K=2, output_dim=2)

    μ = [0.5, 0.1]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_1 = GaussianEmission(output_dim, μ, Σ)

    μ = [1.8, 1.2]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_2 = GaussianEmission(output_dim, μ, Σ)

    test_model.B[1] = emission_1
    test_model.B[2] = emission_2

    test_model.A = [0.8 0.2; 0.05 0.95]
    ll = StateSpaceDynamics.fit!(test_model, data)

    # Test that the transition matrix is correct
    @test isapprox(test_model.A, true_model.A, atol=0.1)

    # Test that the fit is correct
    @test isapprox(test_model.B[1].μ, true_model.B[1].μ; atol=0.1) ||
        isapprox(test_model.B[1].μ, true_model.B[2].μ; atol=0.1)
    @test isapprox(test_model.B[2].μ, true_model.B[2].μ; atol=0.1) ||
        isapprox(test_model.B[2].μ, true_model.B[1].μ; atol=0.1)

    @test isapprox(test_model.B[1].Σ, true_model.B[1].Σ; atol=0.1) ||
        isapprox(test_model.B[1].Σ, true_model.B[2].Σ; atol=0.1)
    @test isapprox(test_model.B[2].Σ, true_model.B[2].Σ; atol=0.1) ||
        isapprox(test_model.B[2].Σ, true_model.B[1].Σ; atol=0.1)

    # Test that the ll is always increasing
    @test any(diff(ll) .< -1e-4) == false
end

function test_SwitchingGaussian_SingleState_fit()
    Random.seed!(1234)
    # Create Guassian Emission Models
    output_dim = 3
    μ = [0.75, -1.25, 1.5]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_1 = GaussianEmission(output_dim, μ, Σ)

    # Create GaussianHMM
    true_model = GaussianHMM(; K=1, output_dim=2)
    true_model.B[1] = emission_1
    # Sample from the model
    n = 20000
    true_labels, data = StateSpaceDynamics.sample(true_model; n=n)

    # Fit a new gaussian hmm to the data
    test_model = GaussianHMM(; K=1, output_dim=3)

    ll = StateSpaceDynamics.fit!(test_model, data)

    # Test that the transition matrix is correct
    @test isapprox(test_model.A, true_model.A, atol=0.1)

    # Test that the fit is correct
    @test isapprox(test_model.B[1].μ, true_model.B[1].μ, atol=0.1)
    @test isapprox(test_model.B[1].Σ, true_model.B[1].Σ, atol=0.1)

    # Test that the ll is always increasing
    @test any(diff(ll) .< 0) == false
end

function test_kmeans_init()
    # Create a dummy dataset that we know the mean and covariance of
    data = [1.0 2.0 11.0 12.0 1.0; 5.0 5.0 -14.0 -13.0 6.0]

    # Define the expected clusters
    μ₁ = [1.33, 5.33]  # Approximate mean of first cluster
    μ₂ = [11.5, -13.5]  # Approximate mean of second cluster]

    Σ₁ = [0.333 -0.166; -0.166 0.333] # Approximate covariance of first cluster
    Σ₂ = [0.5 0.5; 0.5 0.5] # Approximate covariance of second cluster

    # Create a dummy hmm
    model = GaussianHMM(; K=2, output_dim=2)

    # Initialize the model using kmeans
    kmeans_init!(model, data)

    # Test that the means are correct 
    @test isapprox(model.B[1].μ, μ₁, atol=0.1) || isapprox(model.B[1].μ, μ₂, atol=0.1)
    @test isapprox(model.B[2].μ, μ₁, atol=0.1) || isapprox(model.B[2].μ, μ₂, atol=0.1)

    # Test that the covariances are correct
    @test isapprox(model.B[1].Σ, Σ₁, atol=0.1) || isapprox(model.B[1].Σ, Σ₂, atol=0.1)
    @test isapprox(model.B[2].Σ, Σ₁, atol=0.1) || isapprox(model.B[2].Σ, Σ₂, atol=0.1)
end


function test_trialized_GaussianHMM()

    # Create Guassian Emission Models
    output_dim = 2
    μ = [-5.0, -4.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_1 = GaussianEmission(; output_dim=output_dim, μ=μ, Σ=Σ)

    μ = [2.0, 1.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_2 = GaussianEmission(; output_dim=output_dim, μ=μ, Σ=Σ)

    # Create GaussianHMM
    true_model = GaussianHMM(; K=2, output_dim=2)
    true_model.B[1] = emission_1
    true_model.B[2] = emission_2
    true_model.A = [0.9 0.1; 0.8 0.2]

    # Generate trialized synthetic data
    n = 100
    num_trials = 30
    Y = Vector{Matrix{Float64}}(undef, num_trials)
    trial_labels = Vector{Vector{Int}}(undef, num_trials)

    for i in 1:num_trials
        true_labels, data = StateSpaceDynamics.sample(true_model; n=n)  # Generate data and labels
        Y[i] = data  # Store data matrix for the ith trial
    end

    # Fit a model to the trialized synthetic data
    est_model = GaussianHMM(; K=2, output_dim=2)

    # Give the model a warm start 
    est_model.A = [0.75 0.25; 0.01 0.99]
    est_model.B[1] = GaussianEmission(; output_dim=output_dim, μ=[-4.0, -3.0], Σ=0.1 * Matrix{Float64}(I, output_dim, output_dim))
    est_model.B[2] = GaussianEmission(; output_dim=output_dim, μ=[1.5, 1.0], Σ=0.1 * Matrix{Float64}(I, output_dim, output_dim))


    lls = StateSpaceDynamics.fit!(est_model, Y; max_iters=100)

    # Test that model output is correct
    @test isapprox(est_model.A, true_model.A, atol=0.1)

    # @test StateSpaceDynamics.loglikelihood(est_model, Y) >= StateSpaceDynamics.loglikelihood(true_model, Y)
    @test isapprox(est_model.B[1].μ, true_model.B[1].μ; atol=0.1) ||
        isapprox(est_model.B[1].μ, true_model.B[2].μ; atol=0.1)
    @test isapprox(est_model.B[1].Σ, true_model.B[1].Σ; atol=0.1) ||
        isapprox(est_model.B[1].Σ, true_model.B[2].Σ; atol=0.1)
    @test isapprox(est_model.B[2].μ, true_model.B[2].μ; atol=0.1) ||
        isapprox(est_model.B[2].μ, true_model.B[1].μ; atol=0.1)
    @test isapprox(est_model.B[2].Σ, true_model.B[2].Σ; atol=0.1) ||
        isapprox(est_model.B[2].Σ, true_model.B[1].Σ; atol=0.1)
end
