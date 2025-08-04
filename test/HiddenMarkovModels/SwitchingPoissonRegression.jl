function test_SwitchingPoissonRegression_fit()
    emission_1 = PoissonRegressionEmission(; input_dim=3, output_dim=1, include_intercept=true, β=reshape([4.0, 3.0, 2.0, 4.0], :, 1), λ=0.0)
    emission_2 = PoissonRegressionEmission(; input_dim=3, output_dim=1, include_intercept=true, β=reshape([-4.0, -2.0, 1.0, 3.0], :, 1), λ=0.0)

    A_true = [0.9 0.1; 0.2 0.8]
    πₖ_true = [0.7, 0.3]

    # Initialize the SwitchingPoissonRegression
    true_model = HMM(πₖ_true, A_true, [emission_1, emission_2])

    # Sample from the HMM
    n = 20000
    control_seq = Fill(nothing, n)

    simulated_data = Random.rand(default_rng(), true_model, control_seq)
    state_seq, obs_seq = simulated_data.state_seq, simulated_data.obs_seq 

    emission_1 = PoissonRegressionEmission(; input_dim=3, output_dim=1, include_intercept=true, β=reshape([3.0, 1.0, 1.0, 5.0], :, 1), λ=0.0)
    emission_2 = PoissonRegressionEmission(; input_dim=3, output_dim=1, include_intercept=true, β=reshape([-5.0, -1.0, 0.5, 2.0], :, 1), λ=0.0)

    A = [0.7 0.3; 0.3 0.7]
    πₖ = [0.5, 0.5]
    
    # Initialize the SwitchingPoissonRegression
    test_model = HMM(πₖ, A, [emission_1, emission_2])

    hmm, lls = baum_welch(test_model, obs_seq, control_seq; max_iterations=200, loglikelihood_increasing=false)

    #Test the transition matrix
    @test isapprox(true_model.trans, hmm.trans, atol=0.1)

    # Test the regression fit
    @test isapprox(hmm.dists[1].β, true_model.dists[1].β; atol=0.1) ||
        isapprox(hmm.dists[1].β, true_model.dists[2].β; atol=0.1)
    @test isapprox(hmm.dists[2].β, true_model.dists[2].β; atol=0.1) ||
        isapprox(hmm.dists[2].β, true_model.dists[1].β; atol=0.1)

    # Test that the ll is always increasing (accept some numerical instability)
    return any(diff(lls) .< -1e3) == false
end

function test_trialized_SwitchingPoissonRegression()
    # Define parameters
    num_trials = 50  # Number of trials
    trial_length = 1000  # Number of time steps per trial

    emission_1 = PoissonRegressionEmission(; input_dim=3, output_dim=1, include_intercept=true, β=reshape([4.0, 3.0, 2.0, 4.0], :, 1), λ=0.0)
    emission_2 = PoissonRegressionEmission(; input_dim=3, output_dim=1, include_intercept=true, β=reshape([-4.0, -2.0, 1.0, 3.0], :, 1), λ=0.0)

    A_true = [0.9 0.1; 0.2 0.8]
    πₖ_true = [0.7, 0.3]

    # Initialize the SwitchingPoissonRegression
    true_model = HMM(πₖ_true, A_true, [emission_1, emission_2])

    # Create lists to hold data and labels for each trial
    obs_trials = Vector{Vector{Tuple{Vector{Float64}, Vector{Float64}}}}(undef, num_trials)
    state_trials = Vector{Vector{Int}}(undef, num_trials)
    control_seq = [fill(nothing, trial_length) for _ in 1:num_trials] 

    # Sample data for each trial
    for i in 1:num_trials
        sim_data = rand(true_model, control_seq[i]) 
        obs_trials[i] = sim_data.obs_seq
        state_trials[i] = sim_data.state_seq
    end

    obs_seq = vcat(obs_trials...)
    seq_ends = cumsum(length.(obs_trials))

    emission_1 = PoissonRegressionEmission(; input_dim=3, output_dim=1, include_intercept=true, β=reshape([3.0, 1.0, 1.0, 5.0], :, 1), λ=0.0)
    emission_2 = PoissonRegressionEmission(; input_dim=3, output_dim=1, include_intercept=true, β=reshape([-5.0, -1.0, 0.5, 2.0], :, 1), λ=0.0)

    A = [0.7 0.3; 0.3 0.7]
    πₖ = [0.5, 0.5]
    
    # Initialize the SwitchingPoissonRegression
    test_model = HMM(πₖ, A, [emission_1, emission_2])

    # Fit the model using data from all trials
    hmm, lls = baum_welch(test_model, obs_seq; max_iterations=200, loglikelihood_increasing=false, seq_ends=seq_ends)

    # Test the transition matrix
    @test isapprox(true_model.trans, hmm.trans, atol=0.1)

    # Check if parameters are approximately recovered
    @test isapprox(hmm.dists[1].β, true_model.dists[1].β; atol=0.1) ||
        isapprox(hmm.dists[1].β, true_model.dists[2].β; atol=0.1)
    @test isapprox(hmm.dists[2].β, true_model.dists[2].β; atol=0.1) ||
        isapprox(hmm.dists[2].β, true_model.dists[1].β; atol=0.1)

    # Test that the ll is always increasing (accept some numerical instability)
    @test any(diff(lls) .< -1e3) == false
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