function test_SwitchingBernoulliRegression()
    emission_1 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([3.0, 1.0, 2.0], :, 1), λ=0.0)
    emission_2 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([-3.0, -2.0, 1.0], :, 1), λ=0.0)
    
    A = [0.9 0.1; 0.2 0.8]
    πₖ = [0.8; 0.2]

    true_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])

    # Sample from the model
    n = 30000
    Φ = randn(2, n)
    true_labels, data = rand(true_model, Φ; n=n)

    # Initialize the test model
    emission_1 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([2.0, 0.5, 1.0], :, 1), λ=0.0)
    emission_2 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([-2.0, -0.5, 2.0], :, 1), λ=0.0)
    
    A = [0.7 0.3; 0.1 0.9]
    πₖ = [0.5; 0.5]

    test_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])

    ll = StateSpaceDynamics.fit!(test_model, data, Φ; max_iters=200)

    # Test the transition matrix
    @test isapprox(true_model.A, test_model.A, atol=0.1)

    # # Test it works alright
    @test all(isapprox.(test_model.B[1].β, true_model.B[1].β, atol=0.2))
    @test all(isapprox.(test_model.B[2].β, true_model.B[2].β, atol=0.2))
    # Test that the ll is always increasing (accept some numerical instability)
    @test any(diff(ll) .< -1e-3) == false
end

function test_trialized_SwitchingBernoulliRegression()
    # Define parameters
    num_trials = 50  # Number of trials
    trial_length = 1000 # Number of time steps per trial

     emission_1 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([3.0, 1.0, 2.0], :, 1), λ=0.0)
    emission_2 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([-3.0, -2.0, 1.0], :, 1), λ=0.0)
    
    A = [0.9 0.1; 0.2 0.8]
    πₖ = [0.8; 0.2]

    true_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])

    # Create lists to hold data and labels for each trial
    Φ_trials = [randn(2, trial_length) for _ in 1:num_trials]  # Input features for each trial
    true_labels_trials = Vector{Vector{Int}}(undef, num_trials)
    data_trials = Vector{Matrix{Float64}}(undef, num_trials)

    # Sample data for each trial
    for i in 1:num_trials
        true_labels_trials[i], data_trials[i] = rand(
            true_model, Φ_trials[i]; n=trial_length
        )
    end

    # Initialize the test model
    emission_1 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([2.0, 0.5, 1.0], :, 1), λ=0.0)
    emission_2 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([-2.0, -0.5, 2.0], :, 1), λ=0.0)
    
    A = [0.7 0.3; 0.1 0.9]
    πₖ = [0.5; 0.5]

    test_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])

    # Fit the model using data from all trials
    ll = StateSpaceDynamics.fit!(test_model, data_trials, Φ_trials; max_iters=200)

    # Test the transition matrix
    @test isapprox(true_model.A, test_model.A, atol=0.1)

    # Check if parameters are approximately recovered

    @test all(isapprox.(test_model.B[1].β, true_model.B[1].β, atol=0.2)) ||
        all(isapprox.(test_model.B[1].β, true_model.B[2].β, atol=0.2))
    @test all(isapprox.(test_model.B[2].β, true_model.B[2].β, atol=0.2)) ||
        all(isapprox.(test_model.B[2].β, true_model.B[1].β, atol=0.2))

    # Test that the ll is always increasing (accept some numerical instability)
    @test any(diff(ll) .< -1e-3) == false
end
