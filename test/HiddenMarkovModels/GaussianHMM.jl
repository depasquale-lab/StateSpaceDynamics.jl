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
    state_sequence, Y = SSM.sample(true_model, n=n)

    return true_model, state_sequence, Y
end

function test_GaussianHMM()
    n = 1000
    true_model, state_sequence, Y = GaussianHMM_simulation(n)

    est_model = HiddenMarkovModel(K=3, emission=Gaussian(output_dim=2))
    weighted_initialization(est_model, Y)
    fit!(est_model, Y)

    # confirm that the fitted model has a higher loglikelihood than the true model
    @test SSM.loglikelihood(est_model, Y) >= SSM.loglikelihood(true_model, Y)
end