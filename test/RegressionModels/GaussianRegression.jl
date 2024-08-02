function GaussianRegression_simulation(n::Int)
    X = randn(n, 2)
    Σ = [0.1 0;
            0 0.1]
    β = [3 3;
        1 0.5;
        0.5 1]
    true_model = GaussianRegression(β, Σ, input_dim=2, output_dim=2)
    y = SSM.sample(true_model, X)

    return true_model, X, y
end

# check loglikelihood is negative
function test_GaussianRegression_loglikelihood()
    n = 1000
    true_model, X, y = GaussianRegression_simulation(n)
    @test SSM.loglikelihood(true_model, X, y) < 0
end

# check covariance matrix is positive definite and hermitian
function test_GaussianRegression_Σ()
    n = 1000
    true_model, X, y = GaussianRegression_simulation(n)

    est_model = GaussianRegression(input_dim=2, output_dim=2)
    fit!(est_model, X, y)

    @test isposdef(est_model.Σ)
    @test ishermitian(est_model.Σ)
end

# check model shape and value from constructor
function test_GaussianRegression_constructor()
    # test parameter shapes
    model = GaussianRegression(input_dim=3, output_dim=2)
    @test size(model.β) == (4, 2)
    @test size(model.Σ) == (2, 2)

    model = GaussianRegression(input_dim=3, output_dim=2, include_intercept=false)
    @test size(model.β) == (3, 2)
    @test size(model.Σ) == (2, 2)

    # test default values
    model = GaussianRegression(input_dim=3, output_dim=2)
    @test model.λ == 0.0
    @test model.include_intercept == true
    @test model.β == zeros(4, 2)
    @test model.Σ == Matrix{Float64}(I, 2, 2)
end

function test_gradient(objective, objective_grad!, β; atol::Float64=1e-6)
    # numerical gradient
    G_numeric = ForwardDiff.gradient(objective, β)
    # analytic gradient
    G_analytic = zeros(3, 2)
    objective_grad!(G_analytic, β)
    # compare
    @test isapprox(G_numeric, G_analytic, atol=atol)
end

# check surrogate_loglikelihood_gradient is close to numerical gradient
function test_GaussianRegression_objective_gradient()
    n = 1000
    true_model, X, y = GaussianRegression_simulation(n)


    est_model = GaussianRegression(input_dim=2, output_dim=2)
    

    # test if analytical gradient is close to numerical gradient
    objective = define_objective(est_model, X, y)
    objective_grad! = define_objective_gradient(est_model, X, y)
    test_gradient(objective, objective_grad!, ones(3, 2))



    # now do the same with Weights
    weights = rand(1000)
    objective = define_objective(est_model, X, y, weights)
    objective_grad! = define_objective_gradient(est_model, X, y, weights)
    test_gradient(objective, objective_grad!, ones(3, 2))



    # finally test when λ is not 0
    est_model.λ = 0.1
    objective = define_objective(est_model, X, y)
    objective_grad! = define_objective_gradient(est_model, X, y)
    test_gradient(objective, objective_grad!, ones(3, 2))
end

# check that a fitted model has a higher loglikelihood than the true model
function test_GaussianRegression_standard_fit(;include_intercept::Bool=true)
    # Generate synthetic data
    n = 1000
    true_model, X, y = GaussianRegression_simulation(n)

    # Initialize and fit the model
    est_model = GaussianRegression(input_dim=2, output_dim=2, include_intercept=include_intercept)
    fit!(est_model, X, y)

    @test SSM.loglikelihood(est_model, X, y) >= SSM.loglikelihood(true_model, X, y)
end

# check that a regularized model has β values closer to a normal gaussian and the model doesn't perform too much worse
function test_GaussianRegression_regularized_fit(;include_intercept::Bool=true)
    λ = 0.1

    # Generate synthetic data
    n = 1000
    true_model, X, y = GaussianRegression_simulation(n)

    # Initialize and fit an *unregularized* model
    est_model = GaussianRegression(input_dim=2, output_dim=2, include_intercept=true)
    fit!(est_model, X, y)

    # Initialize and fit a regularized model
    regularized_est_model = GaussianRegression(input_dim=2, output_dim=2, include_intercept=true, λ=λ)
    fit!(regularized_est_model, X, y)


    # confirm that the regularized model is not too much worse
    @test isapprox(
        SSM.loglikelihood(regularized_est_model, X, y), 
        SSM.loglikelihood(est_model, X, y), 
        atol=0.1
        )

    # confirm that the regularized model has smaller absolute values for beta
    @test all(abs.(regularized_est_model.β) .<= abs.(est_model.β))
end