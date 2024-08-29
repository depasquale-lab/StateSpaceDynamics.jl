function GaussianHMM_simulation(n::Int)
    Random.seed!(1234)

    output_dim = 2

    μ = [0.0, 0.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)

    emission_1 = Gaussian(output_dim=output_dim, μ=μ, Σ=Σ)

    μ = [2.0, 1.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)

    emission_2 = Gaussian(μ=μ, Σ=Σ, output_dim=output_dim)

    μ = [-1.0, 2.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)

    emission_3 = Gaussian(μ=μ, Σ=Σ, output_dim=output_dim)

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

    est_model = HiddenMarkovModel(K=3, emission=Gaussian(output_dim=2))
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
    emission_1 = GaussianEmission(Gaussian(output_dim=output_dim, μ=μ, Σ=Σ))

    μ = [2.0, 1.0]
    Σ = 0.1 * Matrix{Float64}(I, output_dim, output_dim)
    emission_2 = GaussianEmission(Gaussian(μ=μ, Σ=Σ, output_dim=output_dim))

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
    lls = fit!(est_model, Y, trials=true, max_iters=100)

    # Test that model output is correct
    # @test StateSpaceDynamics.loglikelihood(est_model, Y) >= StateSpaceDynamics.loglikelihood(true_model, Y)
    @test isapprox(est_model.B[1].μ, true_model.B[1].μ, atol=0.1) || isapprox(est_model.B[1].μ, true_model.B[2].μ, atol=0.1)
    @test isapprox(est_model.B[1].Σ, true_model.B[1].Σ, atol=0.1) || isapprox(est_model.B[1].Σ, true_model.B[2].Σ, atol=0.1)
    @test isapprox(est_model.B[2].μ, true_model.B[2].μ, atol=0.1) || isapprox(est_model.B[2].μ, true_model.B[1].μ, atol=0.1)
    @test isapprox(est_model.B[2].Σ, true_model.B[2].Σ, atol=0.1) || isapprox(est_model.B[2].Σ, true_model.B[1].Σ, atol=0.1)

end