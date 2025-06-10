function test_viterbi_GaussianHMM()
    # Create Guassian Emission Models
    output_dim = 2
    μ = [0.0, 0.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_1 = GaussianEmission(output_dim, μ, Σ)

    output_dim = 2
    μ = [0.0, 1.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_3 = GaussianEmission(output_dim, μ, Σ)

    output_dim = 2
    μ = [3.0, 1.2]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_4 = GaussianEmission(output_dim, μ, Σ)

    μ = [2.0, 1.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_2 = GaussianEmission(output_dim, μ, Σ)

    A = [0.9 0.1; 0.2 0.8]
    πₖ = [0.7; 0.3]
    true_model = HiddenMarkovModel(K=2, B=[emission_1, emission_2], A=A, πₖ=πₖ)

    # Number of trials and samples per trial
    n_trials = 100
    n_samples = 1000

    # Preallocate storage for true_labels and data from each trial
    all_true_labels = Vector{Vector{Int}}(undef, n_trials)
    all_data = Vector{Matrix{Float64}}(undef, n_trials)

    # Run 100 sampling trials
    for i in 1:n_trials
        true_labels, data = StateSpaceDynamics.sample(true_model, n=n_samples)
        all_true_labels[i] = true_labels
        all_data[i] = data
    end

    # Fit a gaussian hmm to the data
    A = [0.9 0.1; 0.2 0.8]
    πₖ = [0.7; 0.3]
    test_model = HiddenMarkovModel(K=2, B=[emission_3, emission_4], A=A, πₖ=πₖ)
    lls = StateSpaceDynamics.fit!(test_model, all_data)

    @test isapprox(test_model.B[1].μ, true_model.B[1].μ, atol=0.1) || isapprox(test_model.B[1].μ, true_model.B[2].μ, atol=0.1)
    @test isapprox(test_model.B[2].μ, true_model.B[2].μ, atol=0.1) || isapprox(test_model.B[2].μ, true_model.B[1].μ, atol=0.1)

    labels = viterbi(test_model, all_data)
    tolerance_percentage = 1.0
    error_percentage = (sum(sum.([abs.(labels[i] - all_true_labels[i]) for i in 1:length(labels)])) / (n_trials*n_samples)) * 100

    @test error_percentage <= tolerance_percentage
end


function test_class_probabilities()
    # Create Guassian Emission Models
    output_dim = 2
    μ = [0.0, 0.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_1 = GaussianEmission(output_dim, μ, Σ)

    output_dim = 2
    μ = [0.0, 1.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_3 = GaussianEmission(output_dim, μ, Σ)

    output_dim = 2
    μ = [3.0, 1.2]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_4 = GaussianEmission(output_dim, μ, Σ)

    μ = [2.0, 1.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_2 = GaussianEmission(output_dim, μ, Σ)

    A = [0.9 0.1; 0.2 0.8]
    πₖ = [0.7; 0.3]
    true_model = HiddenMarkovModel(K=2, B=[emission_1, emission_2], A=A, πₖ=πₖ)

    # Number of trials and samples per trial
    n_trials = 100
    n_samples = 1000

    # Preallocate storage for true_labels and data from each trial
    all_true_labels = Vector{Vector{Int}}(undef, n_trials)
    all_data = Vector{Matrix{Float64}}(undef, n_trials)

    # Run 100 sampling trials
    for i in 1:n_trials
        true_labels, data = StateSpaceDynamics.sample(true_model, n=n_samples)
        all_true_labels[i] = true_labels
        all_data[i] = data
    end

    # Fit a gaussian hmm to the data
    A = [0.9 0.1; 0.2 0.8]
    πₖ = [0.7; 0.3]
    test_model = HiddenMarkovModel(K=2, B=[emission_3, emission_4], A=A, πₖ=πₖ)
    lls = StateSpaceDynamics.fit!(test_model, all_data)

    @test isapprox(test_model.B[1].μ, true_model.B[1].μ, atol=0.1) || isapprox(test_model.B[1].μ, true_model.B[2].μ, atol=0.1)
    @test isapprox(test_model.B[2].μ, true_model.B[2].μ, atol=0.1) || isapprox(test_model.B[2].μ, true_model.B[1].μ, atol=0.1)

    # Compute class probabilities
    p = class_probabilities(test_model, all_data)

    # Test shape of the output probabilities
    @test length(p) == n_trials
    @test all(size(p[i]) == size(all_data[i]) for i in 1:n_trials)

    # Test that probabilities are between 0 and 1
    for i in 1:n_trials
        @test all(p[i] .>= 0.0)
        @test all(p[i] .<= 1.0)
        @test all(abs.(sum(p[i], dims=1) .- 1.0) .< 1e-5)
    end
end
