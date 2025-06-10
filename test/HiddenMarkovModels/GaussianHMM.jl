function test_SwitchingGaussian_fit()
    # Define the output dimensionality of the HMM
    output_dim = 2

    # Define the transition matrix and the initial state distribution
    A = [0.99 0.01; 0.05 0.95];
    πₖ = [0.5; 0.5]

    # Initialize the emission models
    μ_1 = [0.0, 0.0]
    Σ_1 = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_1 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)

    μ_2 = [2.0, 1.0]
    Σ_2 = 0.2 * Matrix{Float64}(I, output_dim, output_dim)
    emission_2 = GaussianEmission(output_dim=output_dim, μ=μ_2, Σ=Σ_2)

    # The general HMM constructor is used as follows
    true_model = HiddenMarkovModel(K=2, B=[emission_1, emission_2], A=A, πₖ=πₖ)

    # Sample from the model
    n = 50000
    true_labels, data = rand(true_model; n=n)

    # Initialize a new GaussianHMM
    μ_1 = rand(output_dim)
    Σ_1 = 0.3 * Matrix{Float64}(I, output_dim, output_dim)
    emission_1 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)

    μ_2 = rand(output_dim)
    Σ_2 = 0.5 * Matrix{Float64}(I, output_dim, output_dim)
    emission_2 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)

    A = [0.8 0.2; 0.05 0.95]
    πₖ = [0.6,0.4]
    test_model = HiddenMarkovModel(K=2, B=[emission_1, emission_2], A=A, πₖ=πₖ)

    # Fit the model to recover the original parameters
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
    @test any(diff(ll) .< -1) == false
end

function test_SwitchingGaussian_SingleState_fit()
    # Define the output dimensionality of the HMM
    output_dim = 2

    # Define the transition matrix and the initial state distribution
    A = [1.0;;];
    πₖ = [1.0]

    # Initialize the emission models
    μ_1 = [0.0, 0.0]
    Σ_1 = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_1 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)
    # The general HMM constructor is used as follows
    true_model = HiddenMarkovModel(K=1, B=[emission_1], A=A, πₖ=πₖ)

    # Sample from the model
    n = 20000
    true_labels, data = rand(true_model; n=n)

    # Fit new model
    μ_1 = [1.0, -1.0]
    Σ_1 = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_2 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)
    test_model = HiddenMarkovModel(K=1, B=[emission_2], A=A, πₖ=πₖ)

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

    # Create dummy HMM
    output_dim = 2

    A = [0.99 0.01; 0.05 0.95];
    πₖ = [0.5; 0.5]

    μ_1 = [0.0, 0.0]
    Σ_1 = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_1 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)

    μ_2 = [2.0, 1.0]
    Σ_2 = 0.2 * Matrix{Float64}(I, output_dim, output_dim)
    emission_2 = GaussianEmission(output_dim=output_dim, μ=μ_2, Σ=Σ_2)

    model = HiddenMarkovModel(K=2, B=[emission_1, emission_2], A=A, πₖ=πₖ)

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

    # Define the output dimensionality of the HMM
    output_dim = 2

    # Define the transition matrix and the initial state distribution
    A = [0.99 0.01; 0.05 0.95];
    πₖ = [0.5; 0.5]

    # Initialize the emission models
    μ_1 = [0.0, 0.0]
    Σ_1 = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_1 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)

    μ_2 = [2.0, 1.0]
    Σ_2 = 0.2 * Matrix{Float64}(I, output_dim, output_dim)
    emission_2 = GaussianEmission(output_dim=output_dim, μ=μ_2, Σ=Σ_2)

    # The general HMM constructor is used as follows
    true_model = HiddenMarkovModel(K=2, B=[emission_1, emission_2], A=A, πₖ=πₖ)


    """
    Here, we sample multiple trials of data by looping over our sample function.
    """
    # Number of trials and samples per trial
    n_trials = 100
    n_samples = 1000

    # Preallocate storage for true_labels and data from each trial
    all_true_labels = Vector{Vector{Int}}(undef, n_trials)
    all_data = Vector{Matrix{Float64}}(undef, n_trials)

    # Run 100 sampling trials
    for i in 1:n_trials
        true_labels, data = rand(true_model, n=n_samples)
        all_true_labels[i] = true_labels
        all_data[i] = data
    end

    """
    Finally, we demonstrate that our fit function can handle multiple trials when the data is organized as a Vector{<:Matrix{Float64}}
    """
    # Initialize a new GaussianHMM
    μ_1 = rand(output_dim)
    Σ_1 = 0.3 * Matrix{Float64}(I, output_dim, output_dim)
    emission_1 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)

    μ_2 = rand(output_dim)
    Σ_2 = 0.5 * Matrix{Float64}(I, output_dim, output_dim)
    emission_2 = GaussianEmission(output_dim=output_dim, μ=μ_1, Σ=Σ_1)

    A = [0.8 0.2; 0.05 0.95]
    πₖ = [0.6,0.4]
    test_model = HiddenMarkovModel(K=2, B=[emission_1, emission_2], A=A, πₖ=πₖ)

    lls = StateSpaceDynamics.fit!(test_model, all_data)

    # Test that model output is correct
    @test isapprox(test_model.A, true_model.A, atol=0.1)

    # @test StateSpaceDynamics.loglikelihood(est_model, Y) >= StateSpaceDynamics.loglikelihood(true_model, Y)
    @test isapprox(test_model.B[1].μ, true_model.B[1].μ; atol=0.1) ||
        isapprox(test_model.B[1].μ, true_model.B[2].μ; atol=0.1)
    @test isapprox(test_model.B[1].Σ, true_model.B[1].Σ; atol=0.1) ||
        isapprox(test_model.B[1].Σ, true_model.B[2].Σ; atol=0.1)
    @test isapprox(test_model.B[2].μ, true_model.B[2].μ; atol=0.1) ||
        isapprox(test_model.B[2].μ, true_model.B[1].μ; atol=0.1)
    @test isapprox(test_model.B[2].Σ, true_model.B[2].Σ; atol=0.1) ||
        isapprox(test_model.B[2].Σ, true_model.B[1].Σ; atol=0.1)
end