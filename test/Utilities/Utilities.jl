function test_euclidean_distance()
    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]
    @test StateSpaceDynamics.euclidean_distance(x, y) == sqrt(27.0)
end

function test_kmeanspp_initialization()
    # Generate random data
    data = randn(100, 2)
    # Initialize centroids
    k_means = 3
    centroids = kmeanspp_initialization(data, k_means)
    # Check dimensions
    @test size(centroids) == (2, k_means)
end

function test_kmeans_clustering()
    # Generate random data
    data = randn(100, 2)
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

function test_interleave_reshape()
    # Test with valid data and dimensions
    data = collect(1:6)
    t = 2
    d = 3
    X = interleave_reshape(data, t, d)
    @test size(X) == (2, 3)
    @test X == [1 2 3; 4 5 6]

    # Test with another set of valid data and dimensions
    data = collect(1:12)
    t = 4
    d = 3
    X = interleave_reshape(data, t, d)
    @test size(X) == (4, 3)
    @test X == [1 2 3; 4 5 6; 7 8 9; 10 11 12]

    # Test with a longer set of data
    data = collect(1:20)
    t = 4
    d = 5
    X = interleave_reshape(data, t, d)
    @test size(X) == (4, 5)
    @test X == [1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20]

    # Test with float data
    data = collect(1.0:0.5:6.5)
    t = 4
    d = 3
    X = interleave_reshape(data, t, d)
    @test size(X) == (4, 3)
    @test X == [1.0 1.5 2.0; 2.5 3.0 3.5; 4.0 4.5 5.0; 5.5 6.0 6.5]

    # Test with mismatched dimensions (should raise an error)
    data = collect(1:11)
    t = 2
    d = 5
    @test_throws ErrorException interleave_reshape(data, t, d)
end
