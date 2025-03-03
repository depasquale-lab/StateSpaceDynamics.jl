# Function to generate a random rotation matrix
function random_rotation_matrix(dim::Int)
    # Generate a random matrix with normally distributed entries
    A = randn(dim, dim)
    # Perform QR decomposition
    Q, R = qr(A)
    # Ensure Q is a proper rotation matrix by adjusting the sign
    Q *= sign(det(Matrix(Q)))
    return Matrix(Q)
end

# Function to generate a random covariance matrix
function random_Σ(dim::Int)
    # Set the random seed for reproducibility
    Random.seed!(1234)

    # Step 1: Generate a random rotation matrix
    R = random_rotation_matrix(dim)

    # Step 2: Generate positive eigenvalues from a gamma distribution
    shape = 2.0
    scale = 1.0
    eigenvalues = rand(Gamma(shape, scale), dim)

    # ensure the eigenvalues are not too small
    eigenvalues = max.(eigenvalues, 1e-12)

    # Step 3: Create a diagonal matrix with these eigenvalues
    Λ = Diagonal(eigenvalues)

    # Step 4: Construct the symmetric positive definite matrix
    A = R * Λ * R'

    return A
end

# Test the analytical gradient against the numerical gradient
function test_gradient(objective, objective_grad!, params; atol::Float64=1e-6)
    # numerical gradient
    G_numeric = ForwardDiff.gradient(objective, params)
    # analytic gradient
    G_analytic = zeros(3, 2)
    objective_grad!(G_analytic, params)
    # compare
    @test isapprox(G_numeric, G_analytic, atol=atol)
end

function print_models(true_model, est_model, data...)
    println("True Model:")
    println("true_model: ", true_model)
    println("loglikelihood: ", SSM.loglikelihood(true_model, data...))
    println()
    println("Estimated Model:")
    println("est_model: ", est_model)
    return println("loglikelihood: ", SSM.loglikelihood(est_model, data...))
end
