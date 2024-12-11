function GaussianRegression_simulation(; include_intercept::Bool=true)
    # Generate synthetic data
    n = 1000
    X = randn(n, 2)  # Remove intercept from data generation
    true_β = [-1.2, 2.3]
    if include_intercept
        true_β = vcat(0.5, true_β)
    end
    true_β = reshape(true_β, :, 1)
    true_covariance = reshape([0.25], 1, 1)

    # Generate y with or without intercept
    X_with_intercept = include_intercept ? hcat(ones(n), X) : X
    y = X_with_intercept * true_β + rand(MvNormal(zeros(1), true_covariance), n)'

    return X, y, true_β, true_covariance, n
end

function test_GaussianRegression_initialization()
    # Test with default parameters
    model = GaussianRegressionEmission(; input_dim=2, output_dim=1)

    @test model.input_dim == 2
    @test model.output_dim == 1
    @test model.include_intercept == true
    @test size(model.β) == (3, 1)  # input_dim + 1 (intercept) × output_dim
    @test size(model.Σ) == (1, 1)
    @test model.λ == 0.0

    # Test without intercept
    model_no_intercept = GaussianRegressionEmission(;
        input_dim=2, output_dim=1, include_intercept=false
    )
    @test size(model_no_intercept.β) == (2, 1)
end

function test_GaussianRegression_fit()
    X, y, true_β, true_covariance, n = GaussianRegression_simulation()

    model = GaussianRegressionEmission(; input_dim=2, output_dim=1)
    fit!(model, X, y)

    # Check if the fitted coefficients are close to the true coefficients
    @test isapprox(model.β, true_β, atol=0.5)
    @test isposdef(model.Σ)
    @test isapprox(model.Σ, true_covariance, atol=0.1)

    # Test with weights
    w = ones(n)
    fit!(model, X, y, w)
    @test isapprox(model.β, true_β, atol=0.5)
end

function test_GaussianRegression_loglikelihood()
    X, y, true_β, true_covariance, n = GaussianRegression_simulation()

    model = GaussianRegressionEmission(; input_dim=2, output_dim=1)
    fit!(model, X, y)

    # Test full dataset loglikelihood
    ll = StateSpaceDynamics.loglikelihood(model, X, y)
    @test length(ll) == n
    @test all(isfinite.(ll))

    # Test single observation
    single_ll = StateSpaceDynamics.loglikelihood(model, X[1:1, :], y[1:1, :])
    @test length(single_ll) == 1
    @test isfinite(single_ll[1])

    # Test with weights
    w = ones(n)
    weighted_ll = StateSpaceDynamics.loglikelihood(model, X, y, w)
    @test length(weighted_ll) == n
    @test all(isfinite.(weighted_ll))
end

function test_GaussianRegression_sample()
    model = GaussianRegressionEmission(; input_dim=2, output_dim=1)

    # Test single sample
    X_test = randn(1, 2)
    sample_single = StateSpaceDynamics.sample(model, X_test)
    @test size(sample_single) == (1, 1)

    # Test multiple samples
    X_test = randn(10, 2)
    samples = StateSpaceDynamics.sample(model, X_test)
    @test size(samples) == (10, 1)
end

function test_GaussianRegression_optimization()
    X, y, true_β, true_covariance, n = GaussianRegression_simulation()

    model = GaussianRegressionEmission(;
        input_dim=2,
        output_dim=1,
        λ=0.1,  # Add regularization for testing
    )

    # Test optimization problem creation
    opt_problem = StateSpaceDynamics.create_optimization(model, X, y)

    # Test objective function
    β_vec = vec(model.β)
    obj_val = StateSpaceDynamics.objective(opt_problem, β_vec)
    @test isfinite(obj_val)

    # Test gradient calculation
    G = similar(β_vec)
    StateSpaceDynamics.objective_gradient!(G, opt_problem, β_vec)
    @test length(G) == length(β_vec)
    @test all(isfinite.(G))

    # Compare with ForwardDiff
    grad_fd = ForwardDiff.gradient(β -> StateSpaceDynamics.objective(opt_problem, β), β_vec)
    @test isapprox(G, grad_fd, rtol=1e-5)
end

function test_GaussianRegression_sklearn()
    # create fake data that we fit in sklearn as well
    X = reshape([1.0, 2.0, 3.0, 4.1], 4, 1)
    y = reshape([0.5, 0.9, 0.99, 1.4], 4, 1)
    w = [1.0, 1.0, 1.0, 0.1]

    # plain model i.e., no weights, no regularization
    base_model = GaussianRegressionEmission(; input_dim=1, output_dim=1, λ=0.0)
    fit!(base_model, X, y)

    @test isapprox(base_model.β, [0.2623, 0.2713]; atol=1e-3)

    # test with regularization
    regularized_model = GaussianRegressionEmission(; input_dim=1, output_dim=1, λ=1.0)
    fit!(regularized_model, X, y)

    @test isapprox(regularized_model.β, [0.3709, 0.2283]; atol=1e-3)

    # test model with weight but no regularization
    weighted_model = GaussianRegressionEmission(; input_dim=1, output_dim=1, λ=0.0)
    fit!(weighted_model, X, y, w)

    @test isapprox(weighted_model.β, [0.2941, 0.2524], atol=1e-3)

    # test model with weights and regularized_model
    rw_model = GaussianRegressionEmission(; input_dim=1, output_dim=1, λ=1.0)
    fit!(rw_model, X, y, w)

    @test isapprox(rw_model.β, [0.4464, 0.1787], atol=1e-3)
end
