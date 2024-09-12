function toy_HMM(k::Int=3, data_dim::Int=2, n::Int=1000)
    # create random data
    data = randn(n, data_dim)
    # fit hmm
    hmm = GaussianHMM(data, k)
    return hmm, data
end

function test_toy_HMM()
    hmm, data = toy_HMM()
    @test size(data, 2) == hmm.D
    @test size(data, 1) == 1000
    @test hmm.K == 3
end

function test_HMM_properties(hmm::GaussianHMM)
    @test isapprox(sum(hmm.A; dims=2), ones(hmm.K))
    @test typeof(hmm.B) == Vector{GaussianEmission}
    @test sum(hmm.πₖ) ≈ 1.0
end

function test_GaussianHMM_constructor()
    hmm, _ = toy_HMM()
    return test_HMM_properties(hmm)
end

function test_HMM_forward_and_back()
    hmm, data = toy_HMM()
    α = StateSpaceDynamics.forward(hmm, data)
    @test size(α) == (size(data, 1), hmm.K)
    β = StateSpaceDynamics.backward(hmm, data)
    @test size(β) == (size(data, 1), hmm.K)
end

function test_HMM_gamma_xi()
    hmm, data = toy_HMM()
    α = StateSpaceDynamics.forward(hmm, data)
    β = StateSpaceDynamics.backward(hmm, data)
    γ = StateSpaceDynamics.calculate_γ(hmm, α, β)
    ξ = StateSpaceDynamics.calculate_ξ(hmm, α, β, data)
    @test size(γ) == (size(data, 1), hmm.K)
    @test size(ξ) == (size(data, 1) - 1, hmm.K, hmm.K)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(exp.(γ); dims=2))
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(exp.(ξ); dims=(2, 3)))
end

function test_HMM_E_step()
    hmm, data = toy_HMM()
    γ, ξ, α, β = StateSpaceDynamics.E_step(hmm, data)
    @test size(γ) == (size(data, 1), hmm.K)
    @test size(ξ) == (size(data, 1) - 1, hmm.K, hmm.K)
end

function test_HMM_M_step()
    hmm, data = toy_HMM()
    # test indiviudal M-step functions
    γ, ξ, α, β = StateSpaceDynamics.E_step(hmm, data)
    StateSpaceDynamics.update_initial_state_distribution!(hmm, γ)
    @test sum(hmm.πₖ) ≈ 1.0
    StateSpaceDynamics.update_transition_matrix!(hmm, γ, ξ)
    @test isapprox(sum(hmm.A; dims=2), ones(hmm.K))
    StateSpaceDynamics.update_emission_models!(hmm, γ, data)
    @test typeof(hmm.B) == Vector{GaussianEmission}
    # test M-step
    γ, ξ, α, β = StateSpaceDynamics.E_step(hmm, data)
    StateSpaceDynamics.M_step!(hmm, γ, ξ, data)
    return test_HMM_properties(hmm)
end

function test_HMM_EM()
    Random.seed!(1234)
    A = [0.7 0.2 0.1; 0.1 0.7 0.2; 0.2 0.1 0.7]
    means = [[0.0, 0.0], [-1.0, 2.0], [3.0, 2.5]]
    covs = [[0.1 0.0; 0.0 0.1], [0.1 0.0; 0.0 0.1], [0.1 0.0; 0.0 0.1]]
    emissions_models = [GaussianEmission(mean, cov) for (mean, cov) in zip(means, covs)]
    simul_hmm = GaussianHMM(A, emissions_models, [0.33, 0.33, 0.34], 3, 2)
    states, observations = StateSpaceDynamics.sample(simul_hmm, 10000)
    hmm = GaussianHMM(observations, 3)
    baumWelch!(hmm, observations, 100)
    pred_means = [hmm.B[i].μ for i in 1:3]
    @test sort(pred_means) ≈ sort(means) atol = 0.2
    pred_covs = [hmm.B[i].Σ for i in 1:3]
    @test pred_covs ≈ covs atol = 0.1
    best_path = viterbi(hmm, observations)
    @test length(best_path) == 10000
    @test all(x -> x in 1:3, best_path)
end
