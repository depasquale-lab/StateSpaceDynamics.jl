function test_ARHMM_sampling()
    """
    Use Gaussian Regression to get first point for sampling from AR
    """
    emission_1 = GaussianRegressionEmission(input_dim=2, output_dim=1, include_intercept=false, β=reshape([2.0 1.0 ], :, 1))
    emission_2 = GaussianRegressionEmission(input_dim=2, output_dim=1, include_intercept=false, β=reshape([-2.0 -1.0], :, 1))
    # Create Switching Regression Model
    true_model = SwitchingGaussianRegression(K=2, input_dim=2, output_dim=1, include_intercept=false)
    # Plug in the emission models
    true_model.B[1] = emission_1
    true_model.B[2] = emission_2
    # Sample from the model
    n = 2
    Φ = randn(2, n)
    true_labels, data = rand(true_model, Φ, n=n)
    
    order=1
    output_dim = 2
    # Create autoregressive emission models
    AR_emission_1 = AutoRegressionEmission(output_dim=output_dim, order=order, include_intercept=false, β=[0.8 -0.6; 0.6 0.8])
    AR_emission_2 = AutoRegressionEmission(output_dim=output_dim, order=order, include_intercept=false, β=[0.0 -1.0; 1.0 0.0])

    # Create the autoregressive model and plug in emission models
    true_model = StateSpaceDynamics.SwitchingAutoRegression(K=2, output_dim=output_dim, order=order)
    true_model.A = [0.8 0.2; 0.2 0.8]
    true_model.B[1] = AR_emission_1
    true_model.B[2] = AR_emission_2

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
    """
    Get initial sample from GaussianRegression
    """
    emission_1 = GaussianRegressionEmission(input_dim=2, output_dim=1, include_intercept=false, β=reshape([2.0 1.0 ], :, 1))
    emission_2 = GaussianRegressionEmission(input_dim=2, output_dim=1, include_intercept=false, β=reshape([-2.0 -1.0], :, 1))
    # Create Switching Regression Model
    true_model = SwitchingGaussianRegression(K=2, input_dim=2, output_dim=1, include_intercept=false)
    # Plug in the emission models
    true_model.B[1] = emission_1
    true_model.B[2] = emission_2
    # Sample from the model
    n = 2
    Φ = randn(2, n)
    true_labels, data = rand(true_model, Φ, n=n)
    
    """
    Construct true AR HMM
    """
    # Create autoregressive emission models
    AR_emission_1 = AutoRegressionEmission(output_dim=2, order=1, include_intercept=false, β=[0.8 -0.6; 0.6 0.8])
    AR_emission_2 = AutoRegressionEmission(output_dim=2, order=1, include_intercept=false, β=[0.0 -1.0; 1.0 0.0])
    
    # Create the autoregressive model and plug in emission models
    true_model = StateSpaceDynamics.SwitchingAutoRegression(K=2, output_dim=2, order=1)
    true_model.A = [0.8 0.2; 0.2 0.8]
    true_model.B[1] = AR_emission_1
    true_model.B[2] = AR_emission_2
    
    # Sample from the AR HMM using its own emission models and a starting point
    Y = StateSpaceDynamics.construct_AR_feature_matrix(data, 1, false)
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
    Φ = AR_data[:, 1:(num_samples - 1)]  # Input: All time steps except the last
    Y = AR_data[:, 2:num_samples]        # Output: All time steps except the first


    """
    Create test model and fit it to the data
    """
    AR_emission_1 = AutoRegressionEmission(output_dim=2, order=1, include_intercept=false, β=[1.0 -0.4; 0.5 0.5])
    AR_emission_2 = AutoRegressionEmission(output_dim=2, order=1, include_intercept=false, β=[-0.5 -0.5; -0.5 -0.5])

    # Create the autoregressive model and plug in emission models
    test_model = StateSpaceDynamics.SwitchingAutoRegression(K=2, output_dim=2, order=1)
    test_model.B[1] = AR_emission_1
    test_model.B[2] = AR_emission_2
    test_model.A = [0.6 0.4; 0.3 0.7]
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
    emission_1 = GaussianRegressionEmission(input_dim=2, output_dim=1, include_intercept=false, β=reshape([2.0 1.0 ], :, 1))
    emission_2 = GaussianRegressionEmission(input_dim=2, output_dim=1, include_intercept=false, β=reshape([-2.0 -1.0], :, 1))
    
    # Create Switching Regression Model
    true_model = SwitchingGaussianRegression(K=2, input_dim=2, output_dim=1, include_intercept=false)
    
    # Plug in the emission models
    true_model.B[1] = emission_1
    true_model.B[2] = emission_2
    
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
    emission_1 = GaussianRegressionEmission(input_dim=2, output_dim=1, include_intercept=false, β=reshape([2.0 1.0 ], :, 1))
    emission_2 = GaussianRegressionEmission(input_dim=2, output_dim=1, include_intercept=false, β=reshape([-2.0 -1.0], :, 1))

    # Create Switching Regression Model
    true_model = SwitchingGaussianRegression(K=2, input_dim=2, output_dim=1, include_intercept=false)

    # Plug in the emission models
    true_model.B[1] = emission_1
    true_model.B[2] = emission_2
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