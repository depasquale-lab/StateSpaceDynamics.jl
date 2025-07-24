function test_SwitchingBernoulliRegression()
    emission_1 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([3.0, 1.0, 2.0], :, 1), λ=0.0)
    emission_2 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([-3.0, -2.0, 1.0], :, 1), λ=0.0)
    
    A = [0.9 0.1; 0.2 0.8]
    πₖ = [0.8; 0.2]

    true_model = HMM(πₖ, A, [emission_1, emission_2])

    # Sample from the model
    n = 30000
    Φ = randn(2, n)
    control_seq = Fill(nothing, n)
    simulated_data = rand(true_model, control_seq) 
    state_seq, obs_seq = simulated_data.state_seq, simulated_data.obs_seq

    # Initialize the test model
    emission_1 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([2.0, 0.5, 1.0], :, 1), λ=0.0)
    emission_2 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([-2.0, -0.5, 2.0], :, 1), λ=0.0)
    
    A = [0.7 0.3; 0.1 0.9]
    πₖ = [0.5; 0.5]

    test_model = HMM(init=πₖ, trans=A, dists=[emission_1, emission_2])
    
    hmm, lls = baum_welch(test_model, obs_seq, control_seq; max_iterations=200)

    # Test the transition matrix
    @test isapprox(hmm.trans, true_model.trans, atol=0.1)

    # # Test it works alright
    @test all(isapprox.(hmm.dists[1].β, true_model.dists[1].β, atol=0.2))
    @test all(isapprox.(hmm.dists[2].β, true_model.dists[2].β, atol=0.2))
    # Test that the ll is always increasing (accept some numerical instability)
    @test any(diff(ll) .< -1e-3) == false
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
    Φ_trials = [randn(2, trial_length) for _ in 1:num_trials]  # Input features for each trial
    obs_trials = Matrix{Float64}[]
    state_trials = Vector{Int}[]

    # Sample data for each trial
    for i in 1:num_trials
        Φ = Φ_trials[i]
        sample = rand(true_model, Φ)
        push!(state_trials, sample.state_seq)
        push!(obs_trials, sample.obs_seq)
    end

    # Initialize the test model
    emission_1 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([2.5, 0.75, 1.5], :, 1), λ=0.0)
    emission_2 = BernoulliRegressionEmission(; input_dim=2, output_dim=1, include_intercept=true, β=reshape([-2.5, -0.75, 1.5], :, 1), λ=0.0)
    
    A = [0.7 0.3; 0.1 0.9]
    πₖ = [0.5; 0.5]

    test_model = HMM(πₖ, A, [emission_1, emission_2])

    # Fit the model using data from all trials
    hmm, lls = baum_welch(test_model, obs_trials, Φ_trials; max_iterations=200)

    # Test the transition matrix
    @test isapprox(true_model.trans, hmm.trans, atol=0.1)

    # Check if parameters are approximately recovered

    @test all(isapprox.(hmm.dists[1].β, true_model.dists[1].β, atol=0.2)) ||
        all(isapprox.(hmm.dists[1].β, true_model.dists[2].β, atol=0.2))
    @test all(isapprox.(hmm.dists[2].β, true_model.dists[2].β, atol=0.2)) ||
        all(isapprox.(hmm.dists[2].β, true_model.dists[1].β, atol=0.2))

    # Test that the ll is always increasing (accept some numerical instability)
    @test any(diff(ll) .< -1e-3) == false
end
