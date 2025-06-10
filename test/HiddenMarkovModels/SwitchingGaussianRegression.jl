function test_SwitchingGaussianRegression_fit()
    # Create Emission Models
    emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([3.0, 2.0, 2.0, 3.0], :, 1), Σ=[1.0;;], λ=0.0)
    emission_2 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-4.0, -2.0, 3.0, 2.0], :, 1), Σ=[1.0;;], λ=0.0)

    # Create Switching Regression Model
    A = [0.99 0.01; 0.05 0.95]
    πₖ = [0.8; 0.2]

    true_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])

    # Sample from the model
    n = 20000
    Φ = randn(3, n)
    true_labels, data = StateSpaceDynamics.sample(true_model, Φ, n=n)

    # Fit a new model to the data
    A = [0.8 0.2; 0.1 0.9]
    πₖ = [0.6; 0.4]
    emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([2.0, -1.0, 1.0, 2.0], :, 1), Σ=[2.0;;], λ=0.0)
    emission_2 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-2.5, -1.0, 3.5, 3.0], :, 1), Σ=[0.5;;], λ=0.0)

    test_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])
    lls = StateSpaceDynamics.fit!(test_model, data, Φ)

    # Test transition matrix
    @test isapprox(true_model.A, test_model.A, atol=0.1)

    # Test regression fit
    @test isapprox(test_model.B[1].β, true_model.B[1].β; atol=0.1) ||
        isapprox(test_model.B[1].β, true_model.B[2].β; atol=0.1)
    @test isapprox(test_model.B[2].β, true_model.B[2].β; atol=0.1) ||
        isapprox(test_model.B[2].β, true_model.B[1].β; atol=0.1)

    # Test that the ll is always increasing
    @test any(diff(lls) .< 0.0) == false
end

function test_SwitchingGaussianRegression_SingleState_fit()
    # Create one emission
    emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([3.0, 2.0, 2.0, 3.0], :, 1), Σ=[1.0;;], λ=0.0)

    # Initialize HMM parameters
    A = [1.0;;]
    πₖ = [1.0]

    # Create HMM
    true_model = HiddenMarkovModel(K=1, A=A, πₖ=πₖ, B=[emission_1])

    # Sample from the model
    n = 20000
    Φ = randn(3, n)
    true_labels, data = StateSpaceDynamics.sample(true_model, Φ; n=n)

    # Create test model
    emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([2.0, 1.0, 0.0, 0.0], :, 1), Σ=[0.5;;], λ=0.0)
    test_model = HiddenMarkovModel(K=1, A=A, πₖ=πₖ, B=[emission_1])

    # Recover the original parameters
    ll = StateSpaceDynamics.fit!(test_model, data, Φ)

    # Test transition matrix
    @test isapprox(true_model.A, test_model.A, atol=0.1)

    # Test regression fit
    @test isapprox(test_model.B[1].β, true_model.B[1].β, atol=0.1)

    # Test that the ll is always increasing
    @test any(diff(ll) .< 0.0) == false
end

function test_trialized_SwitchingGaussianRegression()
    # Create Emission Models
    emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([3.0, 2.0, 2.0, 3.0], :, 1), Σ=[1.0;;], λ=0.0)
    emission_2 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-4.0, -2.0, 3.0, 2.0], :, 1), Σ=[1.0;;], λ=0.0)

    # Create Switching Regression Model
    A = [0.99 0.01; 0.05 0.95]
    πₖ = [0.8; 0.2]

    true_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])

    # Sample from the model
    all_data = Vector{Matrix{Float64}}()  # Store each data matrix
    Φ_total = Vector{Matrix{Float64}}()

    num_trials = 10
    n=1000
    all_true_labels = []

    for i in 1:num_trials
        Φ = randn(3, n)
        true_labels, data = StateSpaceDynamics.sample(true_model, Φ, n=n)
        push!(all_true_labels, true_labels)
        push!(all_data, data)
        push!(Φ_total, Φ)
    end

    # Fit a new model to the data
    A = [0.8 0.2; 0.1 0.9]
    πₖ = [0.6; 0.4]
    emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([2.0, -1.0, 1.0, 2.0], :, 1), Σ=[2.0;;], λ=0.0)
    emission_2 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-2.5, -1.0, 3.5, 3.0], :, 1), Σ=[0.5;;], λ=0.0)

    test_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])

    # Fit the model to the sampled data
    lls = StateSpaceDynamics.fit!(test_model, all_data, Φ_total)

    # Run tests to assess model fit
    @test isapprox(test_model.B[1].β, true_model.B[1].β; atol=0.1) ||
        isapprox(test_model.B[1].β, true_model.B[2].β; atol=0.1)
    @test isapprox(test_model.B[2].β, true_model.B[2].β; atol=0.1) ||
        isapprox(test_model.B[2].β, true_model.B[1].β; atol=0.1)
    @test_broken isapprox(test_model.B[1].Σ, true_model.B[1].Σ, atol=0.1) || isapprox(test_model.B[1].Σ, true_model.B[2].Σ, atol=0.1)
    @test_broken isapprox(test_model.B[2].Σ, true_model.B[2].Σ, atol=0.1) || isapprox(test_model.B[2].Σ, true_model.B[1].Σ, atol=0.1)

    # Test that the ll is always increasing (accept small numerical instability)
    @test any(diff(lls) .< 0.0) == false
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
