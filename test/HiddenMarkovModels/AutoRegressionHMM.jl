function test_ARHMM_sampling()
    # Create Emission Models
    emission_1 = GaussianRegressionEmission(input_dim=2, output_dim=1, include_intercept=false, β=reshape([3.0, 2.0], :, 1), Σ=[1.0;;], λ=0.0)

    # Create Switching Regression Model
    A = [1.0;;]
    πₖ = [1.0]

    true_model = HiddenMarkovModel(K=1, A=A, πₖ=πₖ, B=[emission_1])

    # Sample from the model
    n = 2
    Φ = randn(2, n)
    true_labels, data = rand(true_model, Φ, n=n)
    
    order=1
    output_dim = 2
    AR_emission_1 = AutoRegressionEmission(output_dim=output_dim, order=1, include_intercept=false, β=[0.8 -0.6; 0.6 0.8], Σ=[0.001 0.0; 0.0 0.001], λ=0.0);
    AR_emission_2 = AutoRegressionEmission(output_dim=output_dim, order=1, include_intercept=false, β=[0.0 -1.0; 1.0 0.0], Σ=[0.001 0.0; 0.0 0.001], λ=0.0);

    A = [0.9 0.1; 0.2 0.8]
    πₖ = [0.9; 0.1]

    # Create the autoregressive model and plug in emission models
    true_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[AR_emission_1, AR_emission_2]);

    # Sample from the AR HMM using its own emission models and a starting point
    num_points = 1000

    Y = StateSpaceDynamics.construct_AR_feature_matrix(data, order, false)
    X = copy(Y[:, end])
    X = reshape(Y[:, end], 2, 1)
    AR_labels, AR_data = rand(true_model, X, n=num_points, autoregressive=true)

    @test size(AR_data, 2) == num_points
    @test size(AR_data, 1) == output_dim
end

function test_ARHMM_fit()
    # Create Emission Models
    emission_1 = GaussianRegressionEmission(input_dim=2, output_dim=1, include_intercept=false, β=reshape([3.0, 2.0], :, 1), Σ=[1.0;;], λ=0.0)

    # Create Switching Regression Model
    A = [1.0;;]
    πₖ = [1.0]

    true_model = HiddenMarkovModel(K=1, A=A, πₖ=πₖ, B=[emission_1])

    # Sample from the model
    n = 2
    Φ = randn(2, n)
    true_labels, data = rand(true_model, Φ, n=n)
    
    """
    Construct true AR HMM
    """
    # Create autoregressive HMM and sample from it -> one growing, one shrinking
    order = 1
    output_dim = 2
    AR_emission_1 = AutoRegressionEmission(output_dim=output_dim, order=1, include_intercept=false, β=[0.8 -0.6; 0.6 0.8], Σ=[0.001 0.0; 0.0 0.001], λ=0.0);
    AR_emission_2 = AutoRegressionEmission(output_dim=output_dim, order=1, include_intercept=false, β=[0.0 -1.0; 1.0 0.0], Σ=[0.001 0.0; 0.0 0.001], λ=0.0);

    A = [0.9 0.1; 0.2 0.8]
    πₖ = [0.9; 0.1]

    # Create the autoregressive model and plug in emission models
    true_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[AR_emission_1, AR_emission_2]);

    # Sample from the AR HMM using its own emission models and a starting point
    num_points = 10000

    Y = StateSpaceDynamics.construct_AR_feature_matrix(data, order, false)
    X = copy(Y[:, end])
    X = reshape(Y[:, end], 2, 1)
    AR_labels, AR_data = rand(true_model, X, n=2000, autoregressive=true)
    
    
    """
    Create input and output data for AR HMM
    """
    # Define the autoregressive order and dimensions
    order = 1
    num_samples = size(AR_data, 2)
    output_dim = size(AR_data, 1)
    # align  Φ (inputs) and Y (outputs) for regression
    Φ = AR_data[:, 1:(num_points - 1)];  # Input: All time steps except the last
    Y = AR_data[:, 2:num_points];        # Output: All time steps except the first


    """
    Create test model and fit it to the data
    """
    AR_emission_1 = AutoRegressionEmission(output_dim=output_dim, order=1, include_intercept=false, β=[0.5 -0.2; 0.1 0.5], Σ=[0.001 0.0; 0.0 0.001], λ=0.0);
    AR_emission_2 = AutoRegressionEmission(output_dim=output_dim, order=1, include_intercept=false, β=[0.1 -0.5; 2.0 0.2], Σ=[0.001 0.0; 0.0 0.001], λ=0.0);

    A = [0.6 0.4; 0.3 0.7]
    πₖ = [0.7; 0.3]

    # Create the autoregressive model and plug in emission models
    test_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[AR_emission_1, AR_emission_2]);
    ll = StateSpaceDynamics.fit!(test_model, Y, Φ, max_iters=200)

    # Test transition matrix
    @test isapprox(true_model.A, test_model.A, atol=0.1)

    # Test regression fit
    @test isapprox(test_model.B[1].β, true_model.B[1].β; atol=0.1) ||
        isapprox(test_model.B[1].β, true_model.B[2].β; atol=0.1)
    @test isapprox(test_model.B[2].β, true_model.B[2].β; atol=0.1) ||
        isapprox(test_model.B[2].β, true_model.B[1].β; atol=0.1)

    # Test that the ll is always increasing
    @test any(diff(ll) .< 0) == false
end

function test_timeseries_to_AR_feature_matrix()
    # Create Emission Models
    emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([3.0, 2.0, 2.0, 3.0], :, 1), Σ=[1.0;;], λ=0.0)
    emission_2 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-4.0, -2.0, 3.0, 2.0], :, 1), Σ=[1.0;;], λ=0.0)

    # Create Switching Regression Model
    A = [0.99 0.01; 0.05 0.95]
    πₖ = [0.8; 0.2]

    true_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])

    # Sample from the model
    n = 100
    Φ = randn(2, n)
    true_labels, data = rand(true_model, Φ, n=n)

    # Sample from the AR HMM using its own emission models and a starting point
    order = 1
    include_intercept = false
    Y = StateSpaceDynamics.construct_AR_feature_matrix(data, order, include_intercept)

    # Test dimensions work out properly
    @test size(Y, 2) == size(data, 2) - order
    @test size(Y, 1) == size(data, 1) * (order + 1)
end

function test_trialized_timeseries_to_AR_feature_matrix()
    # Create Emission Models
    emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([3.0, 2.0, 2.0, 3.0], :, 1), Σ=[1.0;;], λ=0.0)
    emission_2 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-4.0, -2.0, 3.0, 2.0], :, 1), Σ=[1.0;;], λ=0.0)

    # Create Switching Regression Model
    A = [0.99 0.01; 0.05 0.95]
    πₖ = [0.8; 0.2] 
    true_model = HiddenMarkovModel(K=2, A=A, πₖ=πₖ, B=[emission_1, emission_2])

    all_data = Vector{Matrix{Float64}}()  # Store each data matrix
    Φ_total = Vector{Matrix{Float64}}()
    num_trials = 10
    # Sample from the model
    for i in 1:num_trials
        Φ = randn(2, 100)
        true_labels, data = rand(true_model, Φ, n=100)
        push!(all_data, data)
        push!(Φ_total, Φ)
    end

    # Sample from the AR HMM using its own emission models and a starting point
    order = 1
    Y = StateSpaceDynamics.construct_AR_feature_matrix(all_data, order);

    # Test dimensions work out properly
    @test all(y -> size(y, 2) == size(all_data[1], 2) - order, Y)
    @test all(y -> size(y, 1) == size(all_data[1], 1) * (order + 1), Y)
end