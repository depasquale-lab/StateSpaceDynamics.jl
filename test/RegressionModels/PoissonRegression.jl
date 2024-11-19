# PoissonRegression.jl
function PoissonRegression_simulation(; include_intercept::Bool=true)
    # Generate synthetic data
    n = 1000
    X = randn(n, 2)  # Remove intercept from data generation
    true_β = [-1.2, 2.3]
    if include_intercept
        true_β = vcat(0.5, true_β)
    end
    true_β = reshape(true_β, :, 1)

    # Generate y with or without intercept
    X_with_intercept = include_intercept ? hcat(ones(n), X) : X
    λ = exp.(X_with_intercept * true_β)
    y = reshape(rand.(Poisson.(λ)), :, 1)

    return X, y, true_β, n
end

function test_PoissonRegression_initialization()
    # Test with default parameters
    model = PoissonRegressionEmission(; input_dim=2, output_dim=1)

    @test model.input_dim == 2
    @test model.output_dim == 1
    @test model.include_intercept == true
    @test size(model.β) == (3, 1)  # input_dim + 1 (intercept) × output_dim
    @test model.λ == 0.0

    # Test without intercept
    model_no_intercept = PoissonRegressionEmission(;
        input_dim=2, output_dim=1, include_intercept=false
    )
    @test size(model_no_intercept.β) == (2, 1)
end

function test_PoissonRegression_fit()
    X, y, true_β, n = PoissonRegression_simulation()

    model = PoissonRegressionEmission(; input_dim=2, output_dim=1)
    fit!(model, X, y)

    # Check if the fitted coefficients are close to the true coefficients
    @test isapprox(model.β, true_β, atol=0.5)

    # Test with weights
    w = ones(n)
    fit!(model, X, y, w)
    @test isapprox(model.β, true_β, atol=0.5)
end

function test_PoissonRegression_loglikelihood()
    X, y, true_β, n = PoissonRegression_simulation()

    model = PoissonRegressionEmission(; input_dim=2, output_dim=1)
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

function test_PoissonRegression_optimization()
    X, y, true_β, n = PoissonRegression_simulation()

    model = PoissonRegressionEmission(;
        input_dim=2,
        output_dim=1,
        λ=0.1,  # Add regularization for testing
    )

    # Test objective function
    β_vec = vec(model.β)
    opt_problem = StateSpaceDynamics.create_optimization(model, X, y)
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

function test_PoissonRegression_sample()
    model = PoissonRegressionEmission(; input_dim=2, output_dim=1)

    # Test single sample
    X_test = randn(1, 2)
    sample_single = StateSpaceDynamics.sample(model, X_test)
    @test size(sample_single) == (1, 1)

    # Test multiple samples
    X_test = randn(10, 2)
    samples = StateSpaceDynamics.sample(model, X_test)
    @test size(samples) == (10, 1)
end

function test_PoissonRegression_sklearn()
    # create a set of dummy data we will fit equivalently in sklearn
    X = reshape([1.0, 2.0, 3.0, 4.1], 4, 1)
    y = reshape([0.0, 0.0, 2.0, 1.0], 4, 1)
    w = [1.0, 1.0, 1.0, 0.1]

    # test the regression with no weights and no regularization
    base_model = PoissonRegressionEmission(; input_dim=1, output_dim=1, λ=0.0)
    fit!(base_model, X, y)

    @test isapprox(base_model.β, [-2.405, 0.7125], atol=1e-3)

    # now test with regularization
    regularized_model = PoissonRegressionEmission(; input_dim=1, output_dim=1, λ=1.0)
    fit!(regularized_model, X, y)

    @test_broken isapprox(regularized_model.β, [-1.1618, 0.3195], atol=1e-3)

    # test with regularization
    weighted_model = PoissonRegressionEmission(; input_dim=1, output_dim=1)
    fit!(weighted_model, X, y, w)

    @test isapprox(weighted_model.β, [-3.8794, 1.3505], atol=1e-3)

    # test with weights and refularization
    rw_model = PoissonRegressionEmission(; input_dim=1, output_dim=1, λ=1.0)
    fit!(rw_model, X, y, w)

    @test_broken isapprox(rw_model.β, [-1.3611, 0.4338], atol=1e-3)
end
