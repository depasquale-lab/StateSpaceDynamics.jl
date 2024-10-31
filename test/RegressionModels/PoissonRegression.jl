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
    y = StateSpaceDynamics.sample(PoissonRegressionEmission(vec(true_β), true), X)
    
    return X, y, true_β, n
end

function test_PoissonRegression_initialization()
    # Test with default parameters
    model = PoissonRegressionEmission(input_dim=2, output_dim=1)
    
    @test model.input_dim == 2
    @test model.output_dim == 1
    @test model.include_intercept == true
    @test size(model.β) == (3, 1)  # input_dim + 1 (intercept) × output_dim
    @test model.λ == 0.0
    
    # Test without intercept
    model_no_intercept = PoissonRegressionEmission(
        input_dim=2,
        output_dim=1,
        include_intercept=false
    )
    @test size(model_no_intercept.β) == (2, 1)
end

function test_PoissonRegression_fit()
    X, y, true_β, n = PoissonRegression_simulation()
    
    model = PoissonRegressionEmission(input_dim=2, output_dim=1)
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
    
    model = PoissonRegressionEmission(input_dim=2, output_dim=1)
    fit!(model, X, y)
    
    # Test full dataset loglikelihood
    ll = StateSpaceDynamics.loglikelihood(model, X, y)
    @test length(ll) == n
    @test all(isfinite.(ll))
    
    # Test single observation
    single_ll = StateSpaceDynamics.loglikelihood(model, X[1:1,:], y[1:1])
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
    
    model = PoissonRegressionEmission(
        input_dim=2,
        output_dim=1,
        λ=0.1  # Add regularization for testing
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

function test_PoissonRegression_regularization()
    X, y, true_β, n = PoissonRegression_simulation()
    
    # Test model with different regularization values
    λ_values = [0.0, 0.1, 1.0]
    
    for λ in λ_values
        model = PoissonRegressionEmission(
            input_dim=2,
            output_dim=1,
            λ=λ
        )
        
        fit!(model, X, y)
        
        # Higher regularization should result in smaller coefficients
        if λ > 0
            @test norm(model.β) < norm(true_β)
        end
    end
end