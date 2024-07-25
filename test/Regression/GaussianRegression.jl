function GaussianRegression_simulation()
    # Generate synthetic data
    n = 1000
    X = hcat(ones(n), randn(n, 2))
    true_β = [0.5, -1.2, 2.3]
    true_β = reshape(true_β, 3, 1)
    true_covariance = reshape([0.25], 1, 1)
    y = X * true_β + rand(MvNormal(zeros(1), true_covariance), n)'

    # Remove the intercept column
    X = X[:, 2:end] 
    
    # Initialize and fit the model
    model = GaussianRegression(num_features=2, num_targets=1)
    model.β = ones(3, 1)
    model.Σ = ones(1, 1)
    fit!(model, X, y)

    return model, X, y, true_β, true_covariance, n
end

function test_GaussianRegression_fit()
    model, X, y, true_β, true_covariance, n = GaussianRegression_simulation()
    
    # Check if the fitted coefficients are close to the true coefficients
    @test isapprox(model.β, true_β, atol=0.5)
    @test isposdef(model.Σ)
    @test isapprox(model.Σ, true_covariance, atol=0.1)
end

function test_GaussianRegression_loglikelihood()
    model, X, y, true_β, true_covariance, n = GaussianRegression_simulation()
    
    # Check log likelihood
    loglik = SSM.loglikelihood(model, X, y)
    @test loglik < 0

    # test loglikelihood on a single point
    loglik = SSM.loglikelihood(model, reshape(X[1, :], 1, :), reshape(y[1,:], 1, :))
    @test loglik < 0
end

function test_GaussianRegression_default_model()
    model = GaussianRegression(num_features=2, num_targets=1, include_intercept=false)
    @test model.β == ones(2, 1)
    @test model.Σ == ones(1, 1)
end

function test_GaussianRegression_intercept()
    model, X, y, true_β, true_covariance, n = GaussianRegression_simulation()
    
    model = GaussianRegression(num_features=2, num_targets=1, include_intercept=false)
    model.β = ones(2, 1)
    model.Σ = ones(1, 1)
    fit!(model, X, y)
    @test length(model.β) == 2
    
end

function test_Gaussian_ll_gradient()
    # Generate synthetic data
    n = 1000
    X = hcat(ones(n), randn(n, 2))
    true_β = [0.5, -1.2, 2.3]
    true_β = reshape(true_β, 3, 1)
    true_covariance = reshape([0.25], 1, 1)
    y = X * true_β + rand(MvNormal(zeros(1), true_covariance), n)'

    
    # Initialize model
    model = GaussianRegression(num_features=2, num_targets=1)
    model.β = ones(3, 1)
    model.Σ = ones(1, 1)

    # use ForwardDiff to calculate the gradient
    function objective(β, w, model)
        # calculate log likelihood
        residuals = y - X * β

        # reshape w for broadcasting
        w = reshape(w, (length(w), 1))

        log_likelihood = -0.5 * sum(broadcast(*, w, residuals.^2)) - (model.λ * sum(β.^2))

        
        return log_likelihood
    end


    w = ones(size(X, 1))

    grad = ForwardDiff.gradient(β -> objective(β, w, model), model.β)
    # calculate the gradient manually
    grad_analytic = ones(size(model.β))
    SSM.surrogate_loglikelihood_gradient!(grad_analytic, model, X[:,2:end], y)

    # check if the gradients are close
    @test isapprox(grad, grad_analytic, atol=1e-6)



    # # now do the same with Weights
    w = rand(size(X, 1))


    grad = ForwardDiff.gradient(β -> objective(β, w, model), model.β)


    # calculate the gradient manually
    grad_analytic = ones(size(model.β))
    SSM.surrogate_loglikelihood_gradient!(grad_analytic, model, X[:,2:end], y, w)

    # check if the gradients are close
    @test isapprox(grad, grad_analytic, atol=1e-6)



    # finally test when λ is not 0
    model = GaussianRegression(num_features=2, num_targets=1, λ=0.1)
    model.β = ones(3, 1)
    model.Σ = ones(1, 1)

    grad = ForwardDiff.gradient(β -> objective(β, w, model), model.β)


    # calculate the gradient manually
    grad_analytic = ones(size(model.β))
    SSM.surrogate_loglikelihood_gradient!(grad_analytic, model, X[:,2:end], y, w)

    # check if the gradients are close
    @test isapprox(grad, grad_analytic, atol=1e-6)

end