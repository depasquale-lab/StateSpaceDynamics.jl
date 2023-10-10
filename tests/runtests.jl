function test_GMM_constructor()
    k_means = 3
    data_dim = 2
    data = randn(100, data_dim)
    
    gmm = GMM(k_means, data_dim, data)
    
    @test gmm.k_means == k_means
    @test size(gmm.μ_k) == (data_dim, k_means)
    @test length(gmm.Σ_k) == k_means
    for i in 1:k_means
        @test gmm.Σ_k[i] ≈ I(data_dim)
    end
    @test sum(gmm.π_k) ≈ 1.0
end

test_GMM_constructor()