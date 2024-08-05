function HiddenMarkovModel_simulation(time_steps::Int)
    Φ = randn(time_steps, 2)
    Σ = [0.1 0.05;
            0.05 0.1]
    β = [10 -10;
        1 0.5;
        0.5 1]

    emmission_1 = GaussianRegression(input_dim=2, output_dim=2)
    emmission_2 = GaussianRegression(β=β, Σ=Σ, input_dim=2, output_dim=2)

    true_model = HiddenMarkovModel(K=2, B=[emmission_1, emmission_2])

    Y = SSM.sample(true_model, Φ, time_steps=time_steps)

    return true_model, Φ, Y
end

function test_HiddenMarkovModel_E_step()
    time_steps = 1000
    model, data... = HiddenMarkovModel_simulation(time_steps)

    γ, ξ, α, β = E_step(model, data)

    # test α
    @test size(α) == (size(data, 1), model.K)

    # test β
    @test size(β) == (size(data, 1), model.K)

    # test γ
    @test size(γ) == (size(data, 1), hmm.K)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(exp.(γ), dims=2))

    # test ξ
    @test size(ξ) == (size(data, 1) - 1, hmm.K, hmm.K)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(exp.(ξ), dims=(2, 3)))
end

function test_HiddenMarkovModel_standard_fit()
    # copy all basically from gaussian regression
end 