function test_SwitchingBernoulliRegression()
    emission_1 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([3.0, 1.0, 2.0], :, 1), λ=0.0)
    emission_2 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([-3.0, -2.0, 1.0], :, 1), λ=0.0)
    
    A = [0.9 0.1; 0.2 0.8]
    πₖ = [0.8; 0.2]
    true_model = HMM(πₖ, A, [emission_1, emission_2])

    # Sample from the model
    n = 30000
    control_seq = Fill(nothing, n)

    simulated_data = Random.rand(default_rng(), true_model, control_seq)
    state_seq, obs_seq = simulated_data.state_seq, simulated_data.obs_seq

    # Initialize the test model
    emission_1 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([2.0, 0.5, 1.0], :, 1), λ=0.0)
    emission_2 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([-2.0, -0.5, 2.0], :, 1), λ=0.0)

    A = [0.7 0.3; 0.1 0.9]
    πₖ = [0.5; 0.5]

    test_model = HMM(πₖ, A, [emission_1, emission_2])
    
    hmm, lls = baum_welch(test_model, obs_seq, control_seq; max_iterations=200, loglikelihood_increasing=false)

    # Test the transition matrix
    @test isapprox(true_model.trans, hmm.trans, atol=0.1)

    # # Test it works alright
    @test all(isapprox.(hmm.dists[1].β, true_model.dists[1].β, atol=0.2))
    @test all(isapprox.(hmm.dists[2].β, true_model.dists[2].β, atol=0.2))
    # Test that the ll is always increasing (accept some numerical instability)
    @test any(diff(lls) .< -1e-3) == false
end

function test_trialized_SwitchingBernoulliRegression()
    # Define parameters
    num_trials = 50  # Number of trials
    trial_length = 1000 # Number of time steps per trial

    emission_1_true = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([3.0, 1.0, 2.0], :, 1), λ=0.0)
    emission_2_true = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([-3.0, -2.0, 1.0], :, 1), λ=0.0)
    
    A_true = [0.9 0.1; 0.2 0.8]
    πₖ_true = [0.8; 0.2]

    true_model = HMM(πₖ_true, A_true, [emission_1_true, emission_2_true])

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

    # Initialize the test model
    emission_1 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([2.5, 0.75, 1.5], :, 1), λ=0.0)
    emission_2 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([-2.5, -0.75, 1.5], :, 1), λ=0.0)
    
    A = [0.7 0.3; 0.1 0.9]
    πₖ = [0.5; 0.5]

    test_model = HMM(πₖ, A, [emission_1, emission_2])

    # Fit the model using data from all trials
    hmm, lls = baum_welch(test_model, obs_seq; max_iterations=200, loglikelihood_increasing=false, seq_ends=seq_ends)

    # Test the transition matrix
    @test isapprox(true_model.trans, hmm.trans, atol=0.1)

    # Check if parameters are approximately recovered

    @test all(isapprox.(hmm.dists[1].β, true_model.dists[1].β, atol=0.2)) ||
        all(isapprox.(hmm.dists[1].β, true_model.dists[2].β, atol=0.2))
    @test all(isapprox.(hmm.dists[2].β, true_model.dists[2].β, atol=0.2)) ||
        all(isapprox.(hmm.dists[2].β, true_model.dists[1].β, atol=0.2))

    # Test that the ll is always increasing (accept some numerical instability)
    @test any(diff(lls) .< -1e-3) == false
end
