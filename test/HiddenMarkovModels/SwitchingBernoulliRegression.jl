function test_SwitchingBernoulliRegression()
    Random.seed!(1234)
    # Make Emission Models
    emission_1 = StateSpaceDynamics.BernoulliRegressionEmission(;
        input_dim=2, output_dim=1, include_intercept=true, β=reshape([3.0, 1.0, 2.0], :, 1)
    )
    emission_2 = StateSpaceDynamics.BernoulliRegressionEmission(;
        input_dim=2,
        output_dim=1,
        include_intercept=true,
        β=reshape([-3.0, -2.0, 1.0], :, 1),
    )

    # Create Switching Bernoulli Regression and add the emissions
    true_model = StateSpaceDynamics.SwitchingBernoulliRegression(; K=2, input_dim=2)
    true_model.A = [0.9 0.1; 0.2 0.8]
    true_model.B[1] = emission_1
    true_model.B[2] = emission_2

    # Sample from the model
    n = 30000
    Φ = randn(2, n)
    true_labels, data = StateSpaceDynamics.sample(true_model, Φ; n=n)

    # Fit a new Bernoulli Regression Model to the data
    test_model = StateSpaceDynamics.SwitchingBernoulliRegression(; K=2, input_dim=2, λ=1.0)
    test_model.A = [0.75 0.25; 0.1 0.9]
    test_model.B[1] = StateSpaceDynamics.BernoulliRegressionEmission(;
        input_dim=2, output_dim=1, include_intercept=true, β=reshape([2.8, 0.25, 1.5], :, 1)
    )
    test_model.B[2] = StateSpaceDynamics.BernoulliRegressionEmission(;
        input_dim=2,
        output_dim=1,
        include_intercept=true,
        β=reshape([-3.4, -1.8, 0.8], :, 1),
    )
    ll = StateSpaceDynamics.fit!(test_model, data, Φ; max_iters=200)

    # Test the transition matrix
    @test isapprox(true_model.A, test_model.A, atol=0.1)

    # # Test it works alright
    @test all(isapprox.(test_model.B[1].β, true_model.B[1].β, atol=0.2))
    @test all(isapprox.(test_model.B[2].β, true_model.B[2].β, atol=0.2))
    # Test that the ll is always increasing (accept some numerical instability)
    @test any(diff(ll) .< -1) == false
end

function test_trialized_SwitchingBernoulliRegression()
    # Define parameters
    num_trials = 50  # Number of trials
    trial_length = 1000 # Number of time steps per trial

    # Create the emission models
    emission_1 = StateSpaceDynamics.BernoulliRegressionEmission(;
        input_dim=2, output_dim=1, include_intercept=true, β=reshape([3, 1, 2], :, 1)
    )
    emission_2 = StateSpaceDynamics.BernoulliRegressionEmission(;
        input_dim=2, output_dim=1, include_intercept=true, β=reshape([-3, -2, 0.1], :, 1)
    )

    # Initialize the SwitchingPoissonRegression model
    true_model = SwitchingBernoulliRegression(; K=2, input_dim=2, output_dim=1)
    true_model.A = [0.9 0.1; 0.2 0.8]

    # Plug in the emission models
    true_model.B[1] = emission_1
    true_model.B[2] = emission_2

    # Create lists to hold data and labels for each trial
    Φ_trials = [randn(2, trial_length) for _ in 1:num_trials]  # Input features for each trial
    true_labels_trials = Vector{Vector{Int}}(undef, num_trials)
    data_trials = Vector{Matrix{Float64}}(undef, num_trials)

    # Sample data for each trial
    for i in 1:num_trials
        true_labels_trials[i], data_trials[i] = StateSpaceDynamics.sample(
            true_model, Φ_trials[i]; n=trial_length
        )
    end

    # Create a new SwitchingPoissonRegression and try to recover parameters
    test_model = SwitchingBernoulliRegression(; K=2, input_dim=2, output_dim=1)
    test_model.A = [0.75 0.25; 0.1 0.9]

    # Initialize the emission models for warm start
    test_model.B[1] = StateSpaceDynamics.BernoulliRegressionEmission(;
        input_dim=2, output_dim=1, include_intercept=true, β=reshape([2.5, 0.25, 1.0], :, 1)
    )
    test_model.B[2] = StateSpaceDynamics.BernoulliRegressionEmission(;
        input_dim=2,
        output_dim=1,
        include_intercept=true,
        β=reshape([-3.4, -1.8, -1.0], :, 1),
    )

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
    @test any(diff(ll) .< -1) == false
end

function test_sample_non_float64()

    emission_1 = StateSpaceDynamics.BernoulliRegressionEmission(;
        input_dim=2, output_dim=1, include_intercept=true, β=reshape([3.0, 1.0, 2.0], :, 1)
    )
    emission_2 = StateSpaceDynamics.BernoulliRegressionEmission(;
        input_dim=2,
        output_dim=1,
        include_intercept=true,
        β=reshape([-3.0, -2.0, 1.0], :, 1),
    )

    # Create Switching Bernoulli Regression and add the emissions
    true_model = StateSpaceDynamics.SwitchingBernoulliRegression(; K=2, input_dim=2)
    true_model.A = [0.9 0.1; 0.2 0.8]
    true_model.B[1] = emission_1
    true_model.B[2] = emission_2

    # Sample from the model
    n = 30000
    Φ_int = randn(2, n)
    Φ_f32 = Float32.(randn(2, n))

    # int features
    labels_int, Y_int = StateSpaceDynamics.sample(true_model, Φ_int; n=n)
    @test length(labels_int) == n
    @test eltype(Y_int) == Float64

    # Float32 features
    labels_f32, Y_f32 = StateSpaceDynamics.sample(true_model, Φ_f32; n=n)
    @test length(labels_f32) == n
    @test eltype(Y_f32) == Float64
end

function test_fit_non_float_SwitchingBernoulliRegression()

    num_trials = 50  # Number of trials
    trial_length = 1000 # Number of time steps per trial

    # Create the emission models
    emission_1 = StateSpaceDynamics.BernoulliRegressionEmission(;
        input_dim=2, output_dim=1, include_intercept=true, β=reshape([3, 1, 2], :, 1)
    )
    emission_2 = StateSpaceDynamics.BernoulliRegressionEmission(;
        input_dim=2, output_dim=1, include_intercept=true, β=reshape([-3, -2, 0.1], :, 1)
    )

    # Initialize the SwitchingPoissonRegression model
    true_model = SwitchingBernoulliRegression(; K=2, input_dim=2, output_dim=1)
    true_model.A = [0.9 0.1; 0.2 0.8]

    # Plug in the emission models
    true_model.B[1] = emission_1
    true_model.B[2] = emission_2

    # Create lists to hold data and labels for each trial
    Φ_trials = [randn(2, trial_length) for _ in 1:num_trials]  # Input features for each trial
    true_labels_trials = Vector{Vector{Int}}(undef, num_trials)
    data_trials = Vector{Matrix{Float64}}(undef, num_trials)

    # Sample data for each trial
    for i in 1:num_trials
        true_labels_trials[i], data_trials[i] = StateSpaceDynamics.sample(
            true_model, Φ_trials[i]; n=trial_length
        )
    end

    all_Y_int = [Int.(round.(Y)) for Y in data_trials]
    all_Y_f32 = [Float32.(Y)       for Y in data_trials]

    all_Φ_int    = [Int.(round.(Φ))   for Φ in Φ_trials]
    all_Φ_f32    = [Float32.(Φ)       for Φ in Φ_trials]

    # Int model
    model_int = SwitchingBernoulliRegression(; K=2, input_dim=2, output_dim=1)
    ll_int = StateSpaceDynamics.fit!(model_int, all_Y_int, all_Φ_int; max_iters=100)
    @test eltype(ll_int) == Float64
    @test eltype(model_int.B[1].β) == Float64
    @test eltype(model_int.A) == Float64

    # Float32 data
    model_f32 = SwitchingBernoulliRegression(; K=2, input_dim=2, output_dim=1)
    ll_f32 = StateSpaceDynamics.fit!(model_f32, all_Y_f32, all_Φ_f32; max_iters=100)
    @test eltype(ll_f32) == Float64
    @test eltype(model_f32.B[1].β) == Float64
    @test eltype(model_f32.A) == Float64
end