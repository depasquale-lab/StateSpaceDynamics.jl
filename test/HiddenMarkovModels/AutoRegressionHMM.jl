function AutoRegressionHMM_simulation(n::Int)
    output_dim = 2
    order = 1

    # make a rotation matrix for pi/20 radians
    θ = π / 20
    β = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    Σ = 0.001 * Matrix{Float64}(I, output_dim, output_dim)

    emission_1 = AutoRegressionEmission(;
        order=order, output_dim=output_dim, β=β, Σ=Σ, include_intercept=false
    )

    # make a rotation matrix for -pi/10 radians
    θ = -π / 10
    β = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    Σ = 0.001 * Matrix{Float64}(I, output_dim, output_dim)

    emission_2 = AutoRegressionEmission(;
        order=order, output_dim=output_dim, β=β, Σ=Σ, include_intercept=false
    )

    # make the HMM
    true_model = HiddenMarkovModel(; K=2, B=[emission_1, emission_2])
    true_model.πₖ = [1.0, 0]
    true_model.A = [0.9 0.1; 0.1 0.9]

    Y_prev = reshape([1.0, 0.0], 1, :)

    # sample data
    states, Y = StateSpaceDynamics.sample(true_model, Y_prev; n=n)

    return true_model, Y_prev, Y
end

function test_AutoRegressionHMM()
    n = 1000
    true_model, Y_prev, Y = AutoRegressionHMM_simulation(n)

    est_model = HiddenMarkovModel(;
        K=2,
        emission=AutoRegressionEmission(; order=1, output_dim=2, include_intercept=false),
    )
    # weighted_initialization(est_model, Y_prev, Y) #TODO: add an initialization strategy later
    fit!(est_model, Y, Y_prev)  # flipped these

    # check the β values
    similar_β = false
    if isapprox(true_model.B[1].β, est_model.B[1].β; atol=0.1) &&
        isapprox(true_model.B[2].β, est_model.B[2].β; atol=0.1)
        # the emissions were paired one to one
        similar_β = true
    elseif isapprox(true_model.B[2].β, est_model.B[1].β; atol=0.1) &&
        isapprox(true_model.B[1].β, est_model.B[2].β; atol=0.1)
        # the emissions were paired one to one
        similar_β = true
    else
        similar_β = false
    end

    @test similar_β
end
