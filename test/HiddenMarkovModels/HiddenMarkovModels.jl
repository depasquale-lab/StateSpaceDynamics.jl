include("GaussianHMM.jl")
include("AutoRegressionHMM.jl")

function test_HiddenMarkovModel_E_step()
    n = 1000
    true_model, state_sequence, Y = GaussianHMM_simulation(n)

    est_model = HiddenMarkovModel(K=3, emission=GaussianEmission(output_dim=2))
    kmeans_init!(est_model, Y)

    γ, ξ, α, β = E_step(est_model, (Y,))

    # test α
    @test size(α) == (size(Y, 1), est_model.K)

    # test β
    @test size(β) == (size(Y, 1), est_model.K)

    # test γ
    @test size(γ) == (size(Y, 1), est_model.K)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(exp.(γ), dims=2))

    # test ξ
    @test size(ξ) == (size(Y, 1) - 1, est_model.K, est_model.K)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(exp.(ξ), dims=(2, 3)))
end

function test_viterbi()
    n = 1000
    true_model, state_sequence, Y = GaussianHMM_simulation(n)

    est_model = HiddenMarkovModel(K=3, emission=GaussianEmission(output_dim=2))
    kmeans_init!(est_model, Y)
    fit!(est_model, Y)

    # compare the viterbi path to the true state sequence
    viterbi_path = viterbi(est_model, Y)
    # This test does NOT work, but viterbi DOES. The mappings between emissions are scrambled.
    @test sum(viterbi_path .== state_sequence) / n > 0.99
end