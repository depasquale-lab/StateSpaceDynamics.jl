function test_init()
    # Test the random initialization of the model with default parameters
    model = initialize_slds()
    
    # Check basic model dimensions
    @test size(model.A) == (model.K, model.K)
    @test length(model.B) == model.K
    @test length(model.πₖ) == model.K
    
    # Check transition matrix properties
    @test all(sum(model.A, dims=2) .≈ 1)  # Row sums should be 1
    @test all(model.A .>= 0)  # Non-negative probabilities
    @test model.A[1,1] ≈ 0.96  # Check specific values
    @test model.A[1,2] ≈ 0.04
    @test model.A[2,2] ≈ 0.96
    @test model.A[2,1] ≈ 0.04
    
    # Check initial state distribution
    @test sum(model.πₖ) ≈ 1
    @test all(model.πₖ .>= 0)
    
    # Check LDS components
    for lds in model.B
        @test lds.latent_dim == 2  # Default latent dimension
        @test lds.obs_dim == 10    # Default observation dimension
        @test size(lds.state_model.A) == (2, 2)
        @test size(lds.state_model.Q) == (2, 2)
        @test size(lds.obs_model.C) == (10, 2)
        @test size(lds.obs_model.R) == (10, 10)
        @test all(isposdef(lds.state_model.Q))  # Covariance matrices should be positive definite
        @test all(isposdef(lds.obs_model.R))
    end
    
    # # Test initialization with different parameters
    # model_alt = initialize_slds(K=3, d=3, p=15)
    # @test size(model_alt.A) == (3, 3)
    # @test length(model_alt.B) == 3
    # @test model_alt.B[1].latent_dim == 3
    # @test model_alt.B[1].obs_dim == 15
    
    # Test seed reproducibility
    model1 = initialize_slds(seed=42)
    model2 = initialize_slds(seed=42)
    @test all(model1.A .== model2.A)
    @test all(model1.πₖ .== model2.πₖ)
    @test all(model1.B[1].obs_model.C .== model2.B[1].obs_model.C)
end

function test_sample()
    model = StateSpaceDynamics.initialize_slds()
   
    # sample from the model
    x, y, z = StateSpaceDynamics.sample(model, 1000)  # 1000 time steps

    # check the dimensions of the samples
    @test size(x) == (model.B[1].latent_dim, 1000)
    @test size(y) == (model.B[1].obs_dim, 1000)
    @test size(z) == (1000,)
end
