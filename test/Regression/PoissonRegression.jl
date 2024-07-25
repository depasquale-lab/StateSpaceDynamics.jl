function test_PoissonRegression_fit()
    # Generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    λ = exp.(X * true_β)
    y = rand.(Poisson.(λ))
    
    # Initialize and fit the model
    model = PoissonRegression()
    fit!(model, X[:, 2:end], y)

    # Check if the fitted coefficients are close to the true coefficients
    @test length(model.β) == 3
    @test isapprox(model.β, true_β, atol=0.5)
end

function test_PoissonRegression_loglikelihood()
    # Generate synthetic data
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    λ = exp.(X * true_β)
    y = rand.(Poisson.(λ))
    
    # Initialize and fit the model
    model = PoissonRegression()
    fit!(model, X[:, 2:end], y)
    # check if the fitted coefficients are close to the true coefficients
    @test length(model.β) == 3
    @test isapprox(model.β, true_β, atol=0.5)
    
    # Check log likelihood
    loglik = SSM.loglikelihood(model, X[:, 2:end], y)
    @test loglik < 0

    #test loglikelihood on a single point
    loglik = SSM.loglikelihood(model, X[1, :], y[1])
    @test loglik < 0
end

function test_PoissonRegression_empty_model()
    model = PoissonRegression()
    @test isempty(model.β)
end

function test_PoissonRegression_intercept()
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    λ = exp.(X * true_β)
    y = rand.(Poisson.(λ))
    
    model = PoissonRegression(include_intercept=false)
    fit!(model, X[:, 2:end], y)
    @test length(model.β) == 2
    
    model_with_intercept = PoissonRegression()
    fit!(model_with_intercept, X[:, 2:end], y)
    @test length(model_with_intercept.β) == 3
    @test isapprox(model_with_intercept.β, true_β, atol=0.5)
end

function test_Poisson_ll_gradient()
    # generate data and observations
    X = hcat(ones(1000), randn(1000, 2))
    true_β = [0.5, -1.2, 2.3]
    λ = exp.(X * true_β)
    y = rand.(Poisson.(λ))
    # initialize model
    model = PoissonRegression()
    model.β = [0., 0., 0.]

    # use ForwardDiff to calculate the gradient
    function objective(β, w)
        return sum(w .* (y .* log.(exp.(X * β)) .- exp.(X * β) .- loggamma.(Int.(y) .+ 1))) + (model.λ * sum(β.^2))
    end
    grad = -ForwardDiff.gradient(x -> objective(x, ones(1000)), model.β)
    # calculate the gradient manually
    grad_analytic = SSM.gradient!([0., 0., 0.], model, X, y)
    # check if the gradients are close
    @test isapprox(grad, grad_analytic, atol=1e-6)

    # now do the same with Weights
    weights = rand(1000)
    grad = -ForwardDiff.gradient(x -> objective(x, weights), model.β)
    grad_analytic = SSM.gradient!([0., 0., 0.], model, X, y, weights)
    @test isapprox(grad, grad_analytic, atol=1e-6)

    # finally test when λ is not 0
    model = PoissonRegression(λ=0.1)
    model.β = rand(3)

    grad = -ForwardDiff.gradient(x -> objective(x, ones(1000)), model.β)
    grad_analytic = SSM.gradient!([0., 0., 0.], model, X, y)
end
