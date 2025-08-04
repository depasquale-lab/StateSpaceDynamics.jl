function HiddenMarkovModels.rand(rng::AbstractRNG, hmm::HiddenMarkovModels.AbstractHMM, control_seq::AbstractVector)
    T = length(control_seq)
    dummy_log_probas = fill(-Inf, length(hmm))

    init = initialization(hmm)
    state_seq = Vector{Int}(undef, T)
    state1 = rand(rng, HiddenMarkovModels.LightCategorical(init, dummy_log_probas))
    state_seq[1] = state1

    @views for t in 1:(T - 1)
        trans = transition_matrix(hmm, control_seq[t + 1])
        state_seq[t + 1] = rand(
            rng, HiddenMarkovModels.LightCategorical(trans[state_seq[t], :], dummy_log_probas)
        )
    end

    dists1 = HiddenMarkovModels.obs_distributions(hmm, control_seq[1])
    obs1 = rand(rng, dists1[state1])
    obs_seq = Vector{typeof(obs1)}(undef, T)
    obs_seq[1] = obs1

    for t in 2:T
        dists = HiddenMarkovModels.obs_distributions(hmm, control_seq[t])
        obs_seq[t] = rand(rng, dists[state_seq[t]])
    end
    return (; state_seq=state_seq, obs_seq=obs_seq)
end

function test_SwitchingGaussianRegression_fit()
    # Create Emission Models
    emission_1_true = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([3.0, 2.0, 2.0, 3.0], :, 1), Σ=[1.0;;], λ=0.0)
    emission_2_true = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-4.0, -2.0, 3.0, 2.0], :, 1), Σ=[1.0;;], λ=0.0)

    # Create Switching Regression Model
    A_true = [0.99 0.01; 0.05 0.95]
    πₖ_true = [0.8; 0.2]
    true_model = HMM(πₖ_true, A_true, [emission_1_true, emission_2_true])

    # Sample from the model
    n = 20000
    control_seq = Fill(nothing, n)

    simulated_data = Random.rand(default_rng(), true_model, control_seq)
    state_seq, obs_seq = simulated_data.state_seq, simulated_data.obs_seq 

    # Fit a new model to the data
    A = [0.8 0.2; 0.1 0.9]
    πₖ = [0.6; 0.4]
    emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([2.0, -1.0, 1.0, 2.0], :, 1), Σ=[2.0;;], λ=0.0)
    emission_2 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-2.5, -1.0, 3.5, 3.0], :, 1), Σ=[0.5;;], λ=0.0)

    test_model = HMM(πₖ, A, [emission_1, emission_2])
    hmm, lls = baum_welch(test_model, obs_seq, control_seq; max_iterations=200, loglikelihood_increasing=false)

    # Test transition matrix
    @test isapprox(true_model.trans, hmm.trans, atol=0.1)

    # Test regression fit
    @test isapprox(hmm.dists[1].β, true_model.dists[1].β; atol=0.1) ||
        isapprox(hmm.dists[1].β, true_model.dists[2].β; atol=0.1)
    @test isapprox(hmm.dists[2].β, true_model.dists[2].β; atol=0.1) ||
        isapprox(hmm.dists[2].β, true_model.dists[1].β; atol=0.1)

    # Test that the ll is always increasing
    # @test any(diff(lls) .< -1e-3) == false
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
    emission_guess = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([2.0, 1.0, 0.0, 0.0], :, 1), Σ=[0.5;;], λ=0.0)
    test_model = HMM(πₖ, A, [emission_guess])

    # Recover the original parameters    
    hmm, lls = baum_welch(test_model, obs_seq, control_seq; max_iterations=200, loglikelihood_increasing=false)

    # Test transition matrix
    @test isapprox(true_model.trans, hmm.trans, atol=0.1)

    # Test regression fit
    @test isapprox(hmm.dists[1].β, true_model.dists[1].β, atol=0.1)

    # Test that the ll is always increasing
    # @test any(diff(lls) .< -1e-3) == false
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

    obs_trials = Vector{Vector{Tuple{Vector{Float64}, Vector{Float64}}}}(undef, num_trials)
    state_trials = Vector{Vector{Int}}(undef, num_trials)
    control_seq = [fill(nothing, T) for _ in 1:num_trials] 

    for i in 1:num_trials
        sim_data = rand(true_model, control_seq[i]) 
        obs_trials[i] = sim_data.obs_seq
        state_trials[i] = sim_data.state_seq
    end

    obs_seq = vcat(obs_trials...)
    seq_ends = cumsum(length.(obs_trials))

    # Fit a new model to the data
    A = [0.8 0.2; 0.1 0.9]
    πₖ = [0.6; 0.4]
    emission_1 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([2.0, -1.0, 1.0, 2.0], :, 1), Σ=[2.0;;], λ=0.0)
    emission_2 = GaussianRegressionEmission(input_dim=3, output_dim=1, include_intercept=true, β=reshape([-2.5, -1.0, 3.5, 3.0], :, 1), Σ=[0.5;;], λ=0.0)

    test_model = HMM(πₖ, A, [emission_1, emission_2])

    # Fit the model to the sampled data
    hmm, lls = baum_welch(test_model, obs_seq; max_iterations=200, loglikelihood_increasing=false, seq_ends=seq_ends)

    # Run tests to assess model fit
    @test isapprox(hmm.dists[1].β, true_model.dists[1].β; atol=0.5) ||
        isapprox(hmm.dists[1].β, true_model.dists[2].β; atol=0.5)
    @test isapprox(hmm.dists[2].β, true_model.dists[2].β; atol=0.5) ||
        isapprox(hmm.dists[2].β, true_model.dists[1].β; atol=0.5)
    @test isapprox(hmm.dists[1].Σ, true_model.dists[1].Σ, atol=0.5) || isapprox(hmm.dists[1].Σ, true_model.dists[2].Σ, atol=0.5)
    @test_broken isapprox(hmm.dists[2].Σ, true_model.dists[2].Σ, atol=0.5) || isapprox(hmm.dists[2].Σ, true_model.dists[1].Σ, atol=0.5)

    # Test that the ll is always increasing (accept small numerical instability)
    @test any(diff(lls) .< -1e-3) == false
end

# Function to sample from initial state and transition matrix
# function sample_states(num_samples, initial_probs, transition_matrix)
#     states = Vector{Int}(undef, num_samples)
#     states[1] = rand(Categorical(initial_probs))  # Initial state
#     for i in 2:num_samples
#         states[i] = rand(Categorical(transition_matrix[states[i - 1], :]))  # State transitions
#     end
#     return states
# end