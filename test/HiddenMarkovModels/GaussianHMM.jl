function test_SwitchingGaussian_fit()
    # Create Guassian Emission Models
    output_dim = 2
    μ = [0.0, 0.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_1 = GaussianEmission(output_dim, μ, Σ)

    μ = [2.0, 1.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_2 = GaussianEmission(output_dim, μ, Σ)

    # Create GaussianHMM
    true_model = GaussianHMM(K=2, output_dim=2)
    true_model.B[1] = emission_1
    true_model.B[2] = emission_2
    true_model.A = [0.9 0.1; 0.8 0.2]

    # Sample from the model
    n=20000
    true_labels, data = StateSpaceDynamics.sample(true_model, n=n)

    # Fit a new gaussian hmm to the data
    test_model = GaussianHMM(K=2, output_dim=2)
    ll = fit!(test_model, data)

    # Test that the transition matrix is correct
    @test isapprox(test_model.A, true_model.A, atol=0.1)

    # Test that the fit is correct
    @test isapprox(test_model.B[1].μ, true_model.B[1].μ, atol=0.1) || isapprox(test_model.B[1].μ, true_model.B[2].μ, atol=0.1)
    @test isapprox(test_model.B[2].μ, true_model.B[2].μ, atol=0.1) || isapprox(test_model.B[2].μ, true_model.B[1].μ, atol=0.1)

    @test isapprox(test_model.B[1].Σ, true_model.B[1].Σ, atol=0.1) || isapprox(test_model.B[1].Σ, true_model.B[2].Σ, atol=0.1)
    @test isapprox(test_model.B[2].Σ, true_model.B[2].Σ, atol=0.1) || isapprox(test_model.B[2].Σ, true_model.B[1].Σ, atol=0.1)

    # Test that the ll is always increasing
    @test any(diff(ll) .< 0) == false

end

function test_SwitchingGaussian_SingleState_fit()
    # Create Guassian Emission Models
    output_dim = 3
    μ = [0.75, -1.25, 1.5]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_1 = GaussianEmission(output_dim, μ, Σ)

    # Create GaussianHMM
    true_model = GaussianHMM(K=1, output_dim=2)
    true_model.B[1] = emission_1

    # Sample from the model
    n=20000
    true_labels, data = StateSpaceDynamics.sample(true_model, n=n)

    # Fit a new gaussian hmm to the data
    test_model = GaussianHMM(K=1, output_dim=3)
    ll = StateSpaceDynamics.fit!(test_model, data)

    # Test that the transition matrix is correct
    @test isapprox(test_model.A, true_model.A, atol=0.1)

    # Test that the fit is correct
    @test isapprox(test_model.B[1].μ, true_model.B[1].μ, atol=0.1)
    @test isapprox(test_model.B[1].Σ, true_model.B[1].Σ, atol=0.1)

    # Test that the ll is always increasing
    @test any(diff(ll) .< 0) == false
end

function test_trialized_GaussianHMM()

    # Create Guassian Emission Models
    output_dim = 2
    μ = [-5.0, -4.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_1 = GaussianEmission(output_dim=output_dim, μ=μ, Σ=Σ)

    μ = [2.0, 1.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_2 = GaussianEmission(output_dim=output_dim, μ=μ, Σ=Σ)

    # Create GaussianHMM
    true_model = GaussianHMM(K=2, output_dim=2)
    true_model.B[1] = emission_1
    true_model.B[2] = emission_2
    true_model.A = [0.9 0.1; 0.8 0.2]

    # Generate trialized synthetic data
    n = 100
    num_trials = 30
    Y = Vector{Matrix{Float64}}(undef, num_trials)
    trial_labels = Vector{Vector{Int}}(undef, num_trials)  

    for i in 1:num_trials
        true_labels, data = StateSpaceDynamics.sample(true_model, n=n)  # Generate data and labels
        Y[i] = data  # Store data matrix for the ith trial
    end

    # Fit a model to the trialized synthetic data
    est_model = GaussianHMM(K=2, output_dim=2)
    lls = StateSpaceDynamics.fit!(est_model, Y, max_iters=100)

    # Test that model output is correct
    # @test StateSpaceDynamics.loglikelihood(est_model, Y) >= StateSpaceDynamics.loglikelihood(true_model, Y)
    @test isapprox(est_model.B[1].μ, true_model.B[1].μ, atol=0.1) || isapprox(est_model.B[1].μ, true_model.B[2].μ, atol=0.1)
    @test isapprox(est_model.B[1].Σ, true_model.B[1].Σ, atol=0.1) || isapprox(est_model.B[1].Σ, true_model.B[2].Σ, atol=0.1)
    @test isapprox(est_model.B[2].μ, true_model.B[2].μ, atol=0.1) || isapprox(est_model.B[2].μ, true_model.B[1].μ, atol=0.1)
    @test isapprox(est_model.B[2].Σ, true_model.B[2].Σ, atol=0.1) || isapprox(est_model.B[2].Σ, true_model.B[1].Σ, atol=0.1)

end