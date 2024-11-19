function test_SwitchingPoissonRegression_fit()
    Random.seed!(1234)
    # Create the emission models
    emission_1 = PoissonRegressionEmission(;
        input_dim=3, output_dim=1, include_intercept=true, β=reshape([4, 3, 2, 4], :, 1)
    )
    emission_2 = PoissonRegressionEmission(;
        input_dim=3, output_dim=1, include_intercept=true, β=reshape([-4, -2, 1, 3], :, 1)
    )

    # Initialize the SwitchingPoissonRegression
    true_model = SwitchingPoissonRegression(; K=2, input_dim=3, output_dim=1)

    # Plug in the emission models
    true_model.B[1] = emission_1
    true_model.B[2] = emission_2

    # Sample from the HMM
    n = 20000
    Φ = randn(3, n)
    true_labels, data = StateSpaceDynamics.sample(true_model, Φ; n=n)

    # Create a new SwitchingPoissonRegression and try to recover parameters
    test_model = SwitchingPoissonRegression(; K=2, input_dim=3, output_dim=1)

    # Create the emission models for warm start
    emission_1 = PoissonRegressionEmission(;
        input_dim=3,
        output_dim=1,
        include_intercept=true,
        β=reshape([2.0, 1.0, 4.0, 2.0], :, 1),
    )
    emission_2 = PoissonRegressionEmission(;
        input_dim=3,
        output_dim=1,
        include_intercept=true,
        β=reshape([-5.0, -1.0, 0.0, 2.0], :, 1),
    )
    test_model.B[1], test_model.B[2] = emission_1, emission_2

    ll = StateSpaceDynamics.fit!(test_model, data, Φ; max_iters=200)

    #Test the transition matrix
    @test isapprox(true_model.A, test_model.A, atol=0.1)

    # Test the regression fit
    @test isapprox(test_model.B[1].β, true_model.B[1].β; atol=0.1) ||
        isapprox(test_model.B[1].β, true_model.B[2].β; atol=0.1)
    @test isapprox(test_model.B[2].β, true_model.B[2].β; atol=0.1) ||
        isapprox(test_model.B[2].β, true_model.B[1].β; atol=0.1)

    # Test that the ll is always increasing (accept some numerical instability)
    return any(diff(ll) .< -1e4) == false
end

function test_trialized_SwitchingPoissonRegression()
    Random.seed!(1234)
    # Define parameters
    num_trials = 50  # Number of trials
    trial_length = 400  # Number of time steps per trial

    # Create the emission models
    emission_1 = PoissonRegressionEmission(;
        input_dim=3, output_dim=1, include_intercept=true, β=reshape([4, 3, 2, 4], :, 1)
    )
    emission_2 = PoissonRegressionEmission(;
        input_dim=3, output_dim=1, include_intercept=true, β=reshape([-4, -2, 1, 3], :, 1)
    )

    # Initialize the SwitchingPoissonRegression model
    true_model = SwitchingPoissonRegression(; K=2, input_dim=3, output_dim=1)
    true_model.A = [0.9 0.1; 0.2 0.8]

    # Plug in the emission models
    true_model.B[1] = emission_1
    true_model.B[2] = emission_2

    # Create lists to hold data and labels for each trial
    Φ_trials = [randn(3, trial_length) for _ in 1:num_trials]  # Input features for each trial
    true_labels_trials = Vector{Vector{Int}}(undef, num_trials)
    data_trials = Vector{Matrix{Float64}}(undef, num_trials)

    # Sample data for each trial
    for i in 1:num_trials
        true_labels_trials[i], data_trials[i] = StateSpaceDynamics.sample(
            true_model, Φ_trials[i]; n=trial_length
        )
    end

    # Create a new SwitchingPoissonRegression and try to recover parameters
    test_model = SwitchingPoissonRegression(; K=2, input_dim=3, output_dim=1)

    # Initialize the emission models for warm start
    emission_1 = PoissonRegressionEmission(;
        input_dim=3,
        output_dim=1,
        include_intercept=true,
        β=reshape([2.0, 1.0, 4.0, 2.0], :, 1),
    )
    emission_2 = PoissonRegressionEmission(;
        input_dim=3,
        output_dim=1,
        include_intercept=true,
        β=reshape([-5.0, -1.0, 0.0, 2.0], :, 1),
    )
    test_model.B[1], test_model.B[2] = emission_1, emission_2

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
    @test any(diff(ll) .< -1e4) == false
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
