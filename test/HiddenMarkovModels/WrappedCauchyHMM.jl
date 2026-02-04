function test_WrappedCauchyHMM_fit()
    Random.seed!(42)

    output_dim = 2
    K = 1

    μ_true = [0.2, -1.0]
    ρ_true = [0.7, 0.3]

    emission_true = WrappedCauchyEmission(; output_dim=output_dim, μ=μ_true, ρ=ρ_true)
    A = [1.0;;]
    πₖ = [1.0]

    true_model = HiddenMarkovModel(; K=K, B=[emission_true], A=A, πₖ=πₖ)

    n = 5000
    _, data = rand(true_model; n=n)

    μ_init = [1.0, 2.0]
    ρ_init = [0.2, 0.2]
    emission_init = WrappedCauchyEmission(; output_dim=output_dim, μ=μ_init, ρ=ρ_init)
    test_model = HiddenMarkovModel(; K=K, B=[emission_init], A=A, πₖ=πₖ)

    ll = StateSpaceDynamics.fit!(test_model, data)

    function circ_dist(a, b)
        return abs(atan(sin(a - b), cos(a - b)))
    end

    @test circ_dist(test_model.B[1].μ[1], μ_true[1]) < 0.1
    @test circ_dist(test_model.B[1].μ[2], μ_true[2]) < 0.1

    @test isapprox(test_model.B[1].ρ[1], ρ_true[1]; atol=0.1)
    @test isapprox(test_model.B[1].ρ[2], ρ_true[2]; atol=0.1)

    @test all(diff(ll) .>= -1e-6)
end
