function test_SwitchingPoissonRegression_fit()
    emission_1 = PoissonRegressionEmission(; input_dim=3, output_dim=1, include_intercept=true, β=reshape([4.0, 3.0, 2.0, 4.0], :, 1), λ=0.0)
    emission_2 = PoissonRegressionEmission(; input_dim=3, output_dim=1, include_intercept=true, β=reshape([-4.0, -2.0, 1.0, 3.0], :, 1), λ=0.0)

    A = [0.9 0.1; 0.2 0.8]
    πₖ = [0.7, 0.3]

    # Initialize the SwitchingPoissonRegression
    true_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])

    # Sample from the HMM
    n = 20000
    Φ = randn(3, n)
    true_labels, data = rand(true_model, Φ; n=n)

    emission_1 = PoissonRegressionEmission(; input_dim=3, output_dim=1, include_intercept=true, β=reshape([3.0, 1.0, 1.0, 5.0], :, 1), λ=0.0)
    emission_2 = PoissonRegressionEmission(; input_dim=3, output_dim=1, include_intercept=true, β=reshape([-5.0, -1.0, 0.5, 2.0], :, 1), λ=0.0)

    A = [0.7 0.3; 0.3 0.7]
    πₖ = [0.5, 0.5]
    
    # Initialize the SwitchingPoissonRegression
    test_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])

    ll = StateSpaceDynamics.fit!(test_model, data, Φ; max_iters=200)

    #Test the transition matrix
    @test isapprox(true_model.A, test_model.A, atol=0.1)

    # Test the regression fit
    @test isapprox(test_model.B[1].β, true_model.B[1].β; atol=0.1) ||
        isapprox(test_model.B[1].β, true_model.B[2].β; atol=0.1)
    @test isapprox(test_model.B[2].β, true_model.B[2].β; atol=0.1) ||
        isapprox(test_model.B[2].β, true_model.B[1].β; atol=0.1)

    # Test that the ll is always increasing (accept some numerical instability)
    return any(diff(ll) .< -1e3) == false
end

function test_trialized_SwitchingPoissonRegression()
    # Define parameters
    num_trials = 50  # Number of trials
    trial_length = 1000  # Number of time steps per trial

    emission_1 = PoissonRegressionEmission(; input_dim=3, output_dim=1, include_intercept=true, β=reshape([4.0, 3.0, 2.0, 4.0], :, 1), λ=0.0)
    emission_2 = PoissonRegressionEmission(; input_dim=3, output_dim=1, include_intercept=true, β=reshape([-4.0, -2.0, 1.0, 3.0], :, 1), λ=0.0)

    A = [0.9 0.1; 0.2 0.8]
    πₖ = [0.7, 0.3]

    # Initialize the SwitchingPoissonRegression
    true_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])

    # Create lists to hold data and labels for each trial
    Φ_trials = [randn(3, trial_length) for _ in 1:num_trials]  # Input features for each trial
    true_labels_trials = Vector{Vector{Int}}(undef, num_trials)
    data_trials = Vector{Matrix{Float64}}(undef, num_trials)

    # Sample data for each trial
    for i in 1:num_trials
        true_labels_trials[i], data_trials[i] = rand(
            true_model, Φ_trials[i]; n=trial_length
        )
    end

    emission_1 = PoissonRegressionEmission(; input_dim=3, output_dim=1, include_intercept=true, β=reshape([3.0, 1.0, 1.0, 5.0], :, 1), λ=0.0)
    emission_2 = PoissonRegressionEmission(; input_dim=3, output_dim=1, include_intercept=true, β=reshape([-5.0, -1.0, 0.5, 2.0], :, 1), λ=0.0)

    A = [0.7 0.3; 0.3 0.7]
    πₖ = [0.5, 0.5]
    
    # Initialize the SwitchingPoissonRegression
    test_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])

    # Fit the model using data from all trials
    ll = StateSpaceDynamics.fit!(test_model, data_trials, Φ_trials; max_iters=200)

    # Test the transition matrix
    @test isapprox(true_model.A, test_model.A, atol=0.1)

    # Check if parameters are approximately recovered
    @test isapprox(test_model.B[1].β, true_model.B[1].β; atol=0.1) ||
        isapprox(test_model.B[1].β, true_model.B[2].β; atol=0.1)
    @test isapprox(test_model.B[2].β, true_model.B[2].β; atol=0.1) ||
        isapprox(test_model.B[2].β, true_model.B[1].β; atol=0.1)

    # Test that the ll is always increasing (accept some numerical instability)
    @test any(diff(ll) .< -1e3) == false
end

# Function to sample from initial state and transition matrix
function sample_states(num_samples, initial_probs, transition_matrix)
    Random.seed!(1234)
    states = Vector{Int}(undef, num_samples)
    states[1] = rand(Categorical(initial_probs))  # Initial state
    for i in 2:num_samples
        states[i] = rand(Categorical(transition_matrix[states[i - 1], :]))  # State transitions
    end
    return states
end