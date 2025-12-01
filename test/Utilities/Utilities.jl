function test_euclidean_distance()
    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]
    @test StateSpaceDynamics.euclidean_distance(x, y) == sqrt(27.0)
end

function test_kmeanspp_initialization()
    # Generate random data
    data = randn(2, 100)
    # Initialize centroids
    k_means = 3
    centroids = kmeanspp_initialization(data, k_means)
    # Check dimensions
    @test size(centroids) == (2, k_means)
end

function test_kmeans_clustering()
    # Generate random data
    data = randn(2, 100)
    # Initialize centroids
    k_means = 3
    centroids, labels = kmeans_clustering(data, k_means)
    # Check dimensions
    @test size(centroids) == (2, k_means)
    @test length(labels) == 100
    # Now test kmeans on a vector.
    data = randn(100)
    centroids, labels = kmeans_clustering(data, k_means)
    # Check dimensions
    @test size(centroids) == (1, k_means)
    @test length(labels) == 100
end

function test_block_tridgm()
    # Test with minimal block sizes
    super = [rand(1, 1) for i in 1:1]
    sub = [rand(1, 1) for i in 1:1]
    main = [rand(1, 1) for i in 1:2]
    A = block_tridgm(main, super, sub)
    @test size(A) == (2, 2)
    @test A[1, 1] == main[1][1, 1]
    @test A[2, 2] == main[2][1, 1]
    @test A[1, 2] == super[1][1, 1]
    @test A[2, 1] == sub[1][1, 1]

    # Test with 2x2 blocks and a larger matrix
    super = [rand(2, 2) for i in 1:9]
    sub = [rand(2, 2) for i in 1:9]
    main = [rand(2, 2) for i in 1:10]
    A = block_tridgm(main, super, sub)
    @test size(A) == (20, 20)

    # Check some blocks in the matrix
    for i in 1:10
        @test A[(2i - 1):(2i), (2i - 1):(2i)] == main[i]
        if i < 10
            @test A[(2i - 1):(2i), (2i + 1):(2i + 2)] == super[i]
            @test A[(2i + 1):(2i + 2), (2i - 1):(2i)] == sub[i]
        end
    end

    # Test with integer blocks
    super = [rand(Int, 2, 2) for i in 1:9]
    sub = [rand(Int, 2, 2) for i in 1:9]
    main = [rand(Int, 2, 2) for i in 1:10]
    A = block_tridgm(main, super, sub)
    @test size(A) == (20, 20)
    for i in 1:10
        @test A[(2i - 1):(2i), (2i - 1):(2i)] == main[i]
        if i < 10
            @test A[(2i - 1):(2i), (2i + 1):(2i + 2)] == super[i]
            @test A[(2i + 1):(2i + 2), (2i - 1):(2i)] == sub[i]
        end
    end
end

function test_autoregressive_setters_and_getters()

    # Define AR emission
    AR = AutoRegressionEmission(;
        output_dim=2, order=1, include_intercept=false, β=rand(4, 4), Σ=rand(2, 2), λ=0.0
    )

    # Define parameters
    β = rand(4, 4)
    Σ = rand(4, 4)
    λ = 1.0

    # Set parameters of inner gaussian regression of AR emission using defined parameters
    AR.β, AR.Σ, AR.λ = β, Σ, λ

    # Test that the inner gaussian has these parameters
    @test AR.innerGaussianRegression.β == β
    @test AR.innerGaussianRegression.Σ == Σ
    @test AR.innerGaussianRegression.λ == λ

    # Test the getters now 
    @test AR.innerGaussianRegression.β == AR.β
    @test AR.innerGaussianRegression.Σ == AR.Σ
    @test AR.innerGaussianRegression.λ == AR.λ

    # Test setting innerGaussianRegression directly
    GR = GaussianRegressionEmission(;
        input_dim=1,
        output_dim=2,
        include_intercept=false,
        β=2*rand(1, 2),
        Σ=rand(2, 2),
        λ=0.0,
    )
    AR.innerGaussianRegression = GR
    @test AR.innerGaussianRegression == GR
end

function test_gaussian_entropy()
    n = 3
    A = sprandn(n, n, 0.6)
    Λ = Symmetric(A' * A) + 1e-8I

    F = cholesky(Λ)
    Σ = Symmetric(Matrix(F \ Matrix{Float64}(I, n, n)))

    gaus_entropy_dist = entropy(MvNormal(zeros(n), Σ))
    gauss_entropy_ssd = gaussian_entropy(-Λ)
    @test isapprox(gaus_entropy_dist, gauss_entropy_ssd; atol=1e-6)
end

function test_block_tridiagonal_inverse()
    # Test basic functionality with simple matrices
    n = 3
    block_size = 2

    # Create simple block tridiagonal matrix
    A = [Matrix{Float64}(I, block_size, block_size) for _ in 1:(n - 1)]
    B = [2.0 * Matrix{Float64}(I, block_size, block_size) for _ in 1:n]
    C = [0.5 * Matrix{Float64}(I, block_size, block_size) for _ in 1:(n - 1)]

    λii, λij = StateSpaceDynamics.block_tridiagonal_inverse(A, B, C)

    # Check output dimensions
    @test size(λii) == (block_size, block_size, n)
    @test size(λij) == (block_size, block_size, n - 1)

    # Check that diagonal blocks are well-defined (not NaN or Inf)
    @test all(isfinite.(λii))
    @test all(isfinite.(λij))

    # For this specific case with identity matrices, check rough magnitude
    # The diagonal blocks should be inverses of the diagonal elements
    for i in 1:n
        @test all(abs.(λii[:, :, i]) .< 10.0)  # Reasonable magnitude
    end
end

function test_block_tridiagonal_inverse_type_preservation()
    # Test that Float32 is preserved
    n = 3
    block_size = 2

    A = [Matrix{Float32}(I, block_size, block_size) for _ in 1:(n - 1)]
    B = [2.0f0 * Matrix{Float32}(I, block_size, block_size) for _ in 1:n]
    C = [0.5f0 * Matrix{Float32}(I, block_size, block_size) for _ in 1:(n - 1)]

    λii, λij = StateSpaceDynamics.block_tridiagonal_inverse(A, B, C)

    @test eltype(λii) == Float32
    @test eltype(λij) == Float32
end

function test_block_tridiagonal_inverse_single_block()
    # Test with a single block (n=1)
    block_size = 3
    A = Vector{Matrix{Float64}}()
    B = [randn(block_size, block_size)]
    B[1] = B[1] + 5.0 * I  # Make it well-conditioned
    C = Vector{Matrix{Float64}}()

    λii, λij = StateSpaceDynamics.block_tridiagonal_inverse(A, B, C)

    # Should just be the inverse of the single block
    @test isapprox(λii[:, :, 1], inv(B[1]); atol=1e-10)
    @test size(λij, 3) == 0  # No off-diagonal blocks
end

function test_block_tridiagonal_inverse_vs_static()
    # Compare non-static vs static implementation
    n = 4
    block_size = 2

    A = [randn(block_size, block_size) for _ in 1:(n - 1)]
    B = [
        5.0 * Matrix{Float64}(I, block_size, block_size) + randn(block_size, block_size) for
        _ in 1:n
    ]
    C = [randn(block_size, block_size) for _ in 1:(n - 1)]

    # Non-static version
    λii_reg, λij_reg = StateSpaceDynamics.block_tridiagonal_inverse(A, B, C)

    # Static version
    A_static = [SMatrix{block_size,block_size}(a) for a in A]
    B_static = [SMatrix{block_size,block_size}(b) for b in B]
    C_static = [SMatrix{block_size,block_size}(c) for c in C]
    λii_static, λij_static = StateSpaceDynamics.block_tridiagonal_inverse_static(
        A_static, B_static, C_static, Val(block_size)
    )

    # Results should be reasonably close (allowing for numerical differences in the algorithms)
    @test isapprox(λii_reg, λii_static; atol=1e-6)
    @test isapprox(λij_reg, λij_static; atol=1e-6)

    # Verify both implementations produce finite values
    @test all(isfinite.(λii_reg))
    @test all(isfinite.(λij_reg))
    @test all(isfinite.(λii_static))
    @test all(isfinite.(λij_static))
end
