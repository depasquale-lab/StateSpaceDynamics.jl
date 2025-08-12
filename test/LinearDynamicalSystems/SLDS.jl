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
    @test all(sum(model.A, dims=2) .≈ 1)  # Each row should sum to 1

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
    model1 = initialize_slds(; seed=42)
    model2 = initialize_slds(; seed=42)
    @test all(model1.A .== model2.A)
    @test all(model1.πₖ .== model2.πₖ)
    @test all(model1.B[1].obs_model.C .== model2.B[1].obs_model.C)
end

function test_sample()
    model = StateSpaceDynamics.initialize_slds()

    # sample from the model
    x, y, z = rand(model, 1000)  # 1000 time steps

    # check the dimensions of the samples
    @test size(x) == (model.B[1].latent_dim, 1000)
    @test size(y) == (model.B[1].obs_dim, 1000)
    @test size(z) == (1000,)
end

function test_vEstep()
    # get a model
    model = StateSpaceDynamics.initialize_slds()
    # sample from the model
    x, y, z = rand(model, 1000)  # 1000 time steps

    ty = typeof(y[1])

    # just a few vars
    K = model.K
    T_step = 1000
    # initialize the FS and FB structs
    FS = [StateSpaceDynamics.initialize_FilterSmooth(model.B[k], T_step) for k in 1:K]
    FB = StateSpaceDynamics.initialize_forward_backward(model, T_step, ty)

    # run the vEstep
    ml_total, mls = variational_expectation!(model, y, FB, FS)

    # Test 1: Check ELBO increases
    if !all(diff(mls) .>= -1e-10)
        bad_vals = findall(diff(mls) .< -1e-3) # 
        # print the vad values
        println("Bad values: ", diff(mls)[bad_vals])
    end
    @test all(diff(mls) .>= -1e-3)  # Allow for small numerical instability. this is a bit too permissive, most exteeme i've seen is -1e-6, but this calculation is a wee bit unstable

    # Test 2: Check gamma normalization and properties
    γ = exp.(FB.γ)
    @test size(γ) == (K, T_step)
    @test all(isapprox.(sum(γ, dims=1), 1, rtol=1e-5))
    @test all(γ .>= 0)  # Probabilities should be non-negative

    # Test 3: Check FB properties
    @test size(FB.α) == (K, T_step)  # Forward messages
    @test size(FB.β) == (K, T_step)  # Backward messages
    @test size(FB.ξ) == (K, K, T_step-1)  # Transition expectations

    # Test 4: Check FS properties for each state
    for k in 1:K
        # Test dimensions
        @test size(FS[k].x_smooth) == (model.B[k].latent_dim, T_step)
        @test size(FS[k].p_smooth) == (model.B[k].latent_dim, model.B[k].latent_dim, T_step)

        # Test sufficient statistics dimensions
        @test size(FS[k].E_z) == (model.B[k].latent_dim, T_step, 1)
        @test size(FS[k].E_zz) == (model.B[k].latent_dim, model.B[k].latent_dim, T_step, 1)
        @test size(FS[k].E_zz_prev) ==
            (model.B[k].latent_dim, model.B[k].latent_dim, T_step, 1)

        # Test covariance properties
        for t in 1:T_step
            P_t = FS[k].p_smooth[:, :, t]
            @test isposdef(P_t)  # Covariance should be positive definite
            @test ishermitian(P_t)  # Covariance should be symmetric
        end

        # Test smoothed state properties
        @test !any(isnan.(FS[k].x_smooth))  # No NaN values
        @test !any(isinf.(FS[k].x_smooth))  # No Inf values

        # Test sufficient statistics properties
        @test !any(isnan.(FS[k].E_z))
        @test !any(isinf.(FS[k].E_z))
        @test all(isposdef.(eachslice(FS[k].E_zz[:, :, :, 1], dims=3)))  # Each time slice should be positive definite
    end

    # Test 5: Check numerical properties of the final ELBO
    @test !isnan(ml_total)
    @test !isinf(ml_total)
    @test ml_total > -Inf # ELBO should be finite

    # Test 6: Check convergence
    @test length(mls) > 1  # Should have multiple iterations
    @test abs(mls[end] - mls[end - 1]) < 1e-6  # Should have converged
end
