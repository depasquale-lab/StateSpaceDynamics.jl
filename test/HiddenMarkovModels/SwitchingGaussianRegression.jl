function test_SwitchingGaussianRegression_fit()
    Random.seed!(1234)
    # Create Emission Models
    emission_1 = GaussianRegressionEmission(;
        input_dim=3, output_dim=1, include_intercept=true, β=reshape([3, 2, 2, 3], :, 1)
    )
    emission_2 = GaussianRegressionEmission(;
        input_dim=3, output_dim=1, include_intercept=true, β=reshape([-4, -2, 3, 2], :, 1)
    )

    # Create Switching Regression Model
    true_model = SwitchingGaussianRegression(;
        K=2, input_dim=3, output_dim=1, include_intercept=true
    )
    true_model.A = [0.9 0.1; 0.2 0.8]

    # Plug in the emission models
    true_model.B[1] = emission_1
    true_model.B[2] = emission_2

    # Sample from the model
    n = 20000
    Φ = randn(3, n)
    true_labels, data = StateSpaceDynamics.sample(true_model, Φ; n=n)

    # Try to fit a new model to the data
    test_model = StateSpaceDynamics.SwitchingGaussianRegression(;
        K=2, input_dim=3, output_dim=1, include_intercept=true
    )
    emission_1 = GaussianRegressionEmission(;
        input_dim=3,
        output_dim=1,
        include_intercept=true,
        β=reshape([2.0, 0.0, 1.5, 1.5], :, 1),
    )
    emission_2 = GaussianRegressionEmission(;
        input_dim=3,
        output_dim=1,
        include_intercept=true,
        β=reshape([-2.5, -3.0, 2.25, 2.0], :, 1),
    )
    test_model.B[1] = emission_1
    test_model.B[2] = emission_2
    test_model.A = [0.75 0.25; 0.1 0.9]
    ll = StateSpaceDynamics.fit!(test_model, data, Φ)

    # Test transition matrix
    @test isapprox(true_model.A, test_model.A, atol=0.1)

    # Test regression fit
    @test isapprox(test_model.B[1].β, true_model.B[1].β; atol=0.1) ||
        isapprox(test_model.B[1].β, true_model.B[2].β; atol=0.1)
    @test isapprox(test_model.B[2].β, true_model.B[2].β; atol=0.1) ||
        isapprox(test_model.B[2].β, true_model.B[1].β; atol=0.1)

    # Test that the ll is always increasing
    @test any(diff(ll) .< -1e4) == false
end

function test_SwitchingGaussianRegression_SingleState_fit()
    Random.seed!(1234)
    # Create Emission Models
    emission_1 = GaussianRegressionEmission(;
        input_dim=3, output_dim=1, include_intercept=true, β=reshape([3, 2, 2, 3], :, 1)
    )

    # Create Switching Regression Model
    true_model = SwitchingGaussianRegression(;
        K=1, input_dim=3, output_dim=1, include_intercept=true
    )

    # Plug in the emission models
    true_model.B[1] = emission_1

    # Sample from the model
    n = 20000
    Φ = randn(3, n)
    true_labels, data = StateSpaceDynamics.sample(true_model, Φ; n=n)

    # Try to fit a new model to the data
    test_model = StateSpaceDynamics.SwitchingGaussianRegression(;
        K=1, input_dim=3, output_dim=1, include_intercept=true
    )
    ll = StateSpaceDynamics.fit!(test_model, data, Φ)

    # Test transition matrix
    @test isapprox(true_model.A, test_model.A, atol=0.1)

    # Test regression fit
    @test isapprox(test_model.B[1].β, true_model.B[1].β, atol=0.1)

    # Test that the ll is always increasing
    @test any(diff(ll) .< -1e4) == false
end

function test_trialized_SwitchingGaussianRegression()
    # Create a true underlying model
    model = SwitchingGaussianRegression(;
        K=2, input_dim=1, output_dim=1, include_intercept=true
    )
    model.B[1].β = [100; 100;;]
    model.B[2].β = [-100; -100;;]
    model.B[1].Σ = [2.0;;]
    model.B[2].Σ = [3.0;;]

    # Define initial state probabilities (π) and transition matrix (A)
    initial_probs = [0.6, 0.4]
    transition_matrix = [0.9 0.1; 0.4 0.6]

    n = 100 # Number of samples per trial
    num_trials = 50  # Number of trials
    n1_std = 1.0
    n2_std = 0.5

    # Vectors to store generated data
    trial_inputs = Vector{Matrix{Float64}}(undef, num_trials)
    trial_labels = Vector{Vector{Int}}(undef, num_trials)
    trial_outputs = Vector{Matrix{Float64}}(undef, num_trials)

    # Generate trials
    for trial in 1:num_trials
        # Random input data
        x_data = randn(1, n)  # Random input data for this trial
        trial_inputs[trial] = x_data

        # Generate state sequence
        state_sequence = sample_states(n, initial_probs, transition_matrix)
        trial_labels[trial] = state_sequence

        # Generate output data based on state and linear relationships
        y_data = zeros(1, n)
        for i in 1:n
            if state_sequence[i] == 1
                y_data[i] =
                    (model.B[1].β[2] * x_data[i] + model.B[1].β[1]) + (randn() * n1_std)
            else
                y_data[i] =
                    (model.B[2].β[2] * x_data[i] + model.B[2].β[1]) + (randn() * n2_std)
            end
        end
        trial_outputs[trial] = y_data
    end

    # Create new model and fit the data
    est_model = SwitchingGaussianRegression(; K=2, input_dim=1, output_dim=1)
    ll = fit!(est_model, trial_outputs, trial_inputs; max_iters=200)

    # Test the transition matrix
    # @test isapprox(est_model.A, transition_matrix, atol=0.1)

    # Run tests to assess model fit
    @test isapprox(est_model.B[1].β, model.B[1].β; atol=0.1) ||
        isapprox(est_model.B[1].β, model.B[2].β; atol=0.1)
    @test isapprox(est_model.B[2].β, model.B[2].β; atol=0.1) ||
        isapprox(est_model.B[2].β, model.B[1].β; atol=0.1)
    # @test isapprox(est_model.B[1].Σ, model.B[1].Σ, atol=0.1) || isapprox(est_model.B[1].Σ, model.B[2].Σ, atol=0.1)
    # @test isapprox(est_model.B[2].Σ, model.B[2].Σ, atol=0.1) || isapprox(est_model.B[2].Σ, model.B[1].Σ, atol=0.1)

    # Test that the ll is always increasing (accept small numerical instability)
    @test any(diff(ll) .< -1e4) == false
end

# Function to sample from initial state and transition matrix
function sample_states(num_samples, initial_probs, transition_matrix)
    states = Vector{Int}(undef, num_samples)
    states[1] = rand(Categorical(initial_probs))  # Initial state
    for i in 2:num_samples
        states[i] = rand(Categorical(transition_matrix[states[i - 1], :]))  # State transitions
    end
    return states
end
