function HiddenMarkovModel_simulation(n::Int)
    Φ = randn(n, 2)
    Σ = [0.1 0.05;
            0.05 0.1]
    β = [3 3;
        1 0.5;
        0.5 1]
    true_model = GaussianRegression(β=β, Σ=Σ, input_dim=2, output_dim=2)
    Y = SSM.sample(true_model, Φ)

    return true_model, Φ, Y
end

function test_HiddenMarkovModel_E_step()
    model, data = HiddenMarkovModel_simulation()
    α = SSM.forward(model, data)
    @test size(α) == (size(data, 1), model.K)

    β = SSM.backward(model, data)
    @test size(β) == (size(data, 1), model.K)

    γ = SSM.calculate_γ(hmm, α, β)
    @test size(γ) == (size(data, 1), hmm.K)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(exp.(γ), dims=2))

    ξ = SSM.calculate_ξ(hmm, α, β, data)
    @test size(ξ) == (size(data, 1) - 1, hmm.K, hmm.K)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(exp.(ξ), dims=(2, 3)))
end

function test_HiddenMarkovModel_standard_fit()
    # copy all basically from gaussian regression
end 