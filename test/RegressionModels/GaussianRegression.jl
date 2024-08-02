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

# check surrogate_loglikelihood_gradient is close to numerical gradient
function test_GaussianRegression_surrogate_loglikelihood_gradient()
    n = 1000
    true_model, X, y = GaussianRegression_simulation(n)

    est_model = GaussianRegression(input_dim=2, output_dim=2)
    
    function objective_wrt_β(model, X, y, w=ones(size(y,1)))
        function change_β(β)
            model.β = β
            return SSM.surrogate_loglikelihood(model, X, y, w)
        end
        return change_β
    end 

    
    # numerical gradient
    grad = ForwardDiff.gradient(objective_wrt_β(est_model, X, y), ones(3, 2))
    # analytic gradient
    est_model.β = ones(3, 2)
    grad_analytic = zeros(3, 2)
    surrogate_loglikelihood_gradient!(grad_analytic, est_model, X, y)
    # compare
    @test isapprox(grad, grad_analytic, atol=1e-6)



    # now do the same with Weights
    weights = rand(1000)
    # numerical gradient
    grad = ForwardDiff.gradient(objective_wrt_β(est_model, X, y, weights), ones(3, 2))
    # analytic gradient
    est_model.β = ones(3, 2)
    grad_analytic = zeros(3, 2)
    surrogate_loglikelihood_gradient!(grad_analytic, est_model, X, y, weights)
    # compare
    @test isapprox(grad, grad_analytic, atol=1e-6)



    # finally test when λ is not 0
    est_model.λ = 0.1

    # numerical gradient
    grad = ForwardDiff.gradient(objective_wrt_β(est_model, X, y), ones(3, 2))
    # analytic gradient
    est_model.β = ones(3, 2)
    grad_analytic = zeros(3, 2)
    surrogate_loglikelihood_gradient!(grad_analytic, est_model, X, y)
    # compare
    @test isapprox(grad, grad_analytic, atol=1e-6)
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

# check that a regularized fitted model has a higher loglikelihood than an unregularized fitted model (when evaluated on NEW data)
function test_GaussianRegression_regularized_fit(;include_intercept::Bool=true)
    # Test if this λ value improves model generalization
    λ = 0.1

    # Generate synthetic data
    n = 1000
    true_model, X, y = GaussianRegression_simulation(n)

    # Initialize and fit an *unregularized* model
    est_model = GaussianRegression(input_dim=2, output_dim=2, include_intercept=include_intercept)
    fit!(est_model, X, y)

    # Initialize and fit a regularized model
    regularized_est_model = GaussianRegression(input_dim=2, output_dim=2, include_intercept=include_intercept, λ=λ)
    fit!(regularized_est_model, X, y)


    # Generate testing data
    X_test = randn(n, 2)
    y_test = SSM.sample(true_model, X_test)

    # Confirm the regularized model generalizes better 
    @test SSM.loglikelihood(regularized_est_model, X_test, y_test) >= SSM.loglikelihood(est_model, X_test, y_test)
end