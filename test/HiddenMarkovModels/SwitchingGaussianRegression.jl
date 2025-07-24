function test_SwitchingGaussianRegression_fit()
    # Create Emission Models
    emission_1_true = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([3.0, 2.0, 2.0, 3.0], :, 1), Σ=[1.0;;], λ=0.0)
    emission_2_true = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-4.0, -2.0, 3.0, 2.0], :, 1), Σ=[1.0;;], λ=0.0)

    # Create Switching Regression Model
    A_true = [0.99 0.01; 0.05 0.95]
    πₖ_true = [0.8; 0.2]
    true_model = HMM(πₖ_true, A_true, [emission_1_true, emission_2_true])

    print(true_model)

    # Sample from the model
    n = 20000
    control_seq = Fill(nothing, n)
    simulated_data = rand(true_model, control_seq)
    state_seq, obs_seq = simulated_data.state_seq, simulated_data.obs_seq 

    # Fit a new model to the data
    A = [0.8 0.2; 0.1 0.9]
    πₖ = [0.6; 0.4]
    emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([2.0, -1.0, 1.0, 2.0], :, 1), Σ=[2.0;;], λ=0.0)
    emission_2 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-2.5, -1.0, 3.5, 3.0], :, 1), Σ=[0.5;;], λ=0.0)

    test_model = HMM(πₖ, A, [emission_1, emission_2])
    hmm, lls = baum_welch(test_model, obs_seq, control_seq; max_iterations=200)

    # Test transition matrix
    @test isapprox(true_model.trans, hmm.trans, atol=0.1)

    # Test regression fit
    @test isapprox(hmm.dists[1].β, true_model.dists[1].β; atol=0.1) ||
        isapprox(hmm.dists[1].β, true_model.dists[2].β; atol=0.1)
    @test isapprox(hmm.dists[2].β, true_model.dists[2].β; atol=0.1) ||
        isapprox(hmm.dists[2].β, true_model.dists[1].β; atol=0.1)

    # Test that the ll is always increasing
    @test any(diff(lls) .< -1e-3) == false
end

function test_SwitchingGaussianRegression_SingleState_fit()
    # Create one emission
    emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([3.0, 2.0, 2.0, 3.0], :, 1), Σ=[1.0;;], λ=0.0)

    # Initialize HMM parameters
    A = [1.0;;]
    πₖ = [1.0]

    # Create HMM
    true_model = HMM(πₖ, A, [emission_1])

    # Sample from the model
    n = 20000
    control_seq = Fill(nothing, n)
    simulated_data = rand(true_model, control_seq)
    state_seq, obs_seq = simulated_data.state_seq, simulated_data.obs_seq 

    # Create test model
    emission_guess = GaussianRegressionEmission(3, 1, true, reshape([2.0, 1.0, 0.0, 0.0], :, 1), [0.5;;], 0.0)
    test_model = HMM([1.0], [1.0;;], [emission_guess])

    # Recover the original parameters
    
    hmm, lls = baum_welch(test_model, obs_seq, control_seq; max_iterations=200)

    # Test transition matrix
    @test isapprox(true_model.trans, hmm.trans, atol=0.1)

    # Test regression fit
    @test isapprox(hmm.dists[1].β, true_model.dists[1].β, atol=0.1)

    # Test that the ll is always increasing
    @test any(diff(ll) .< -1e-3) == false
end

function test_trialized_SwitchingGaussianRegression()
    # Create Emission Models
    emission_1_true = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([3.0, 2.0, 2.0, 3.0], :, 1), Σ=[1.0;;], λ=0.0)
    emission_2_true = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-4.0, -2.0, 3.0, 2.0], :, 1), Σ=[1.0;;], λ=0.0)

    # Create Switching Regression Model
    A_true = [0.99 0.01; 0.05 0.95]
    πₖ_true = [0.8; 0.2]

    true_model = HMM(πₖ_true, A_true, [emission_1_true, emission_2_true])
    
    # Sample from the model
    num_trials = 10
    T=1000
    Φ_trials = [randn(3, T) for _ in 1:num_trials]
    obs_trials = Vector{Matrix{Float64}}(undef, num_trials)
    state_trials = Vector{Vector{Int}}(undef, num_trials)

    for i in 1:num_trials
        Φ = Φ_trials[i]
        sim = rand(true_model, eachcol(Φ))  
        obs_trials[i] = sim.obs_seq
        state_trials[i] = sim.state_seq
    end

    # Fit a new model to the data
    A = [0.8 0.2; 0.1 0.9]
    πₖ = [0.6; 0.4]
    emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([2.0, -1.0, 1.0, 2.0], :, 1), Σ=[2.0;;], λ=0.0)
    emission_2 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-2.5, -1.0, 3.5, 3.0], :, 1), Σ=[0.5;;], λ=0.0)

    test_model = HMM(πₖ, A, [emission_1, emission_2])

    # Fit the model to the sampled data
    hmm, lls = baum_welch(test_model, obs_trials, Φ_trials; max_iterations=200)

    # Run tests to assess model fit
    @test isapprox(hmm.dists[1].β, true_model.dists[1].β; atol=0.1) ||
        isapprox(hmm.dists[1].β, true_model.dists[2].β; atol=0.1)
    @test isapprox(hmm.dists[2].β, true_model.dists[2].β; atol=0.1) ||
        isapprox(hmm.dists[2].β, true_model.dists[1].β; atol=0.1)
    @test_broken isapprox(hmm.dists[1].Σ, true_model.dists[1].Σ, atol=0.1) || isapprox(hmm.dists[1].Σ, true_model.dists[2].Σ, atol=0.1)
    @test_broken isapprox(hmm.dists[2].Σ, true_model.dists[2].Σ, atol=0.1) || isapprox(hmm.dists[2].Σ, true_model.dists[1].Σ, atol=0.1)

    # Test that the ll is always increasing (accept small numerical instability)
    @test any(diff(lls) .< -1e-3) == false
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