function GaussianHMM_simulation(n::Int)
    Random.seed!(1234)

    output_dim = 2

    μ = [0.0, 0.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)

    emission_1 = GaussianEmission(output_dim=output_dim, μ=μ, Σ=Σ)

    μ = [2.0, 1.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)

    emission_2 = GaussianEmission(μ=μ, Σ=Σ, output_dim=output_dim)

    μ = [-1.0, 2.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)

    emission_3 = GaussianEmission(μ=μ, Σ=Σ, output_dim=output_dim)

    # make the true_model
    true_model = HiddenMarkovModel(K=3, B=[emission_1, emission_2, emission_3])
    true_model.πₖ = [1.0, 0, 0]
    true_model.A = [0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9]

    # generate some data
    state_sequence, Y = StateSpaceDynamics.sample(true_model, n=n)

    return true_model, state_sequence, Y
end

function test_GaussianHMM()
    n = 1000
    true_model, state_sequence, Y = GaussianHMM_simulation(n)

    est_model = HiddenMarkovModel(K=3, emission=GaussianEmission(output_dim=2))
    weighted_initialization(est_model, Y)
    fit!(est_model, Y)

    # confirm that the fitted model has a higher loglikelihood than the true model
    @test StateSpaceDynamics.loglikelihood(est_model, Y) >= StateSpaceDynamics.loglikelihood(true_model, Y)
end

function test_trialized_GaussianHMM()

    # Create Guassian Emission Models
    output_dim = 2
    μ = [-5.0, -4.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_1 = GaussianEmission(output_dim=output_dim, μ=μ, Σ=Σ)

    μ = [2.0, 1.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_2 = GaussianEmission(output_dim=output_dim, μ=μ, Σ=Σ)

    # Create GaussianHMM
    true_model = StateSpaceDynamics.GaussianHMM(K=2, output_dim=2)
    true_model.B[1] = emission_1
    true_model.B[2] = emission_2
    true_model.A = [0.9 0.1; 0.8 0.2]

    # Generate trialized synthetic data
    n = 100
    num_trials = 30
    Y = Vector{Matrix{Float64}}(undef, num_trials)
    trial_labels = Vector{Vector{Int}}(undef, num_trials)  

    for i in 1:num_trials
        true_labels, data = StateSpaceDynamics.sample(true_model, n=n)  # Generate data and labels
        Y[i] = data  # Store data matrix for the ith trial
    end

    # Fit a model to the trialized synthetic data
    est_model = StateSpaceDynamics.GaussianHMM(K=2, output_dim=2)
    lls = fit!(est_model, Y, max_iters=100)

    # Test that model output is correct
    # @test StateSpaceDynamics.loglikelihood(est_model, Y) >= StateSpaceDynamics.loglikelihood(true_model, Y)
    @test isapprox(est_model.B[1].μ, true_model.B[1].μ, atol=0.1) || isapprox(est_model.B[1].μ, true_model.B[2].μ, atol=0.1)
    @test isapprox(est_model.B[1].Σ, true_model.B[1].Σ, atol=0.1) || isapprox(est_model.B[1].Σ, true_model.B[2].Σ, atol=0.1)
    @test isapprox(est_model.B[2].μ, true_model.B[2].μ, atol=0.1) || isapprox(est_model.B[2].μ, true_model.B[1].μ, atol=0.1)
    @test isapprox(est_model.B[2].Σ, true_model.B[2].Σ, atol=0.1) || isapprox(est_model.B[2].Σ, true_model.B[1].Σ, atol=0.1)

end

function test_trialized_SwitchingGaussianRegression()
    # Create a true underlying model
    model = SwitchingGaussianRegression(K=2, input_dim=1, output_dim=1)
    model.B[1].β = [100; 100;;]
    model.B[2].β = [-100; -100;;]
    model.B[1].Σ = [2.0;;]
    model.B[2].Σ = [3.0;;]

    # Define initial state probabilities (π) and transition matrix (A)
    initial_probs = [0.6, 0.4]  # Probability to start in state 1 or state 2
    transition_matrix = [0.9 0.1; 0.4 0.6]  # State transition probabilities

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
        x_data = randn(n, 1)  # Random input data for this trial
        trial_inputs[trial] = x_data

        # Generate state sequence
        state_sequence = sample_states(n, initial_probs, transition_matrix)
        trial_labels[trial] = state_sequence

        # Generate output data based on state and linear relationships
        y_data = zeros(n, 1)
        for i in 1:n
            if state_sequence[i] == 1
                y_data[i] = (model.B[1].β[2] * x_data[i] + model.B[1].β[1]) + (randn() * n1_std)
            else
                y_data[i] = (model.B[2].β[2] * x_data[i] + model.B[2].β[1]) + (randn() * n2_std)
            end
        end
        trial_outputs[trial] = y_data
    end

    # Create new model and fit the data
    est_model = SwitchingGaussianRegression(K=2, input_dim=1, output_dim=1)
    ll = fit!(est_model, trial_outputs, trial_inputs, max_iters=200)

    # Run tests to assess model fit
    @test isapprox(est_model.B[1].β, model.B[1].β, atol=0.1) || isapprox(est_model.B[1].β, model.B[2].β, atol=0.1)
    @test isapprox(est_model.B[2].β, model.B[2].β, atol=0.1) || isapprox(est_model.B[2].β, model.B[1].β, atol=0.1)
    # @test isapprox(est_model.B[1].Σ, model.B[1].Σ, atol=0.1) || isapprox(est_model.B[1].Σ, model.B[2].Σ, atol=0.1)
    # @test isapprox(est_model.B[2].Σ, model.B[2].Σ, atol=0.1) || isapprox(est_model.B[2].Σ, model.B[1].Σ, atol=0.1)
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