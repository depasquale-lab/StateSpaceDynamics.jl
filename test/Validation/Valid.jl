# Tests for Valid.jl validation functions

function test_isvalid_probvec()
    # Valid probability vectors
    @test isvalid_probvec([0.3, 0.7])
    @test isvalid_probvec([0.25, 0.25, 0.25, 0.25])
    @test isvalid_probvec([1.0])
    @test isvalid_probvec(Float32[0.3, 0.7])

    # Invalid probability vectors
    @test !isvalid_probvec([0.3, 0.8])  # Sum > 1
    @test !isvalid_probvec([0.3, 0.5])  # Sum < 1
    @test !isvalid_probvec([-0.1, 1.1])  # Negative value
    @test !isvalid_probvec([0.5, 0.5, 0.5])  # Sum > 1
    @test !isvalid_probvec([0.0, 0.0])  # Sum = 0
end

function test_isvalid_LDS_gaussian()
    # Create a valid Gaussian LDS
    A = Matrix{Float64}(I, 2, 2)
    C = Matrix{Float64}(I, 2, 2)
    Q = Matrix{Float64}(I, 2, 2)
    R = Matrix{Float64}(I, 2, 2)
    x0 = zeros(Float64, 2)
    P0 = Matrix{Float64}(I, 2, 2)
    b = zeros(Float64, 2)
    d = zeros(Float64, 2)

    gsm = GaussianStateModel(; A=A, Q=Q, x0=x0, P0=P0, b=b)
    gom = GaussianObservationModel(; C=C, R=R, d=d)
    lds = LinearDynamicalSystem(;
        state_model=gsm, obs_model=gom, latent_dim=2, obs_dim=2, fit_bool=fill(true, 6)
    )

    @test isvalid_LDS(lds)
end

function test_isvalid_LDS_poisson()
    # Create a valid Poisson LDS
    A = Matrix{Float64}(I, 2, 2)
    C = randn(3, 2)
    Q = Matrix{Float64}(I, 2, 2)
    log_d = zeros(Float64, 3)
    x0 = zeros(Float64, 2)
    P0 = Matrix{Float64}(I, 2, 2)
    b = zeros(Float64, 2)

    gsm = GaussianStateModel(; A=A, Q=Q, x0=x0, P0=P0, b=b)
    pom = PoissonObservationModel(; C=C, log_d=log_d)
    lds = LinearDynamicalSystem(;
        state_model=gsm, obs_model=pom, latent_dim=2, obs_dim=3, fit_bool=fill(true, 5)
    )

    @test isvalid_LDS(lds)
end

function test_isvalid_LDS_dimension_mismatch()
    # Test dimension mismatches
    A = Matrix{Float64}(I, 2, 2)
    C = Matrix{Float64}(I, 3, 2)  # obs_dim = 3
    Q = Matrix{Float64}(I, 2, 2)
    R = Matrix{Float64}(I, 2, 2)  # Should be 3x3!
    x0 = zeros(Float64, 2)
    P0 = Matrix{Float64}(I, 2, 2)
    b = zeros(Float64, 2)
    d = zeros(Float64, 2)  # Should be length 3!

    gsm = GaussianStateModel(; A=A, Q=Q, x0=x0, P0=P0, b=b)
    gom = GaussianObservationModel(; C=C, R=R, d=d)
    lds = LinearDynamicalSystem(;
        state_model=gsm, obs_model=gom, latent_dim=2, obs_dim=3, fit_bool=fill(true, 6)
    )

    @test !isvalid_LDS(lds)
end

function test_isvalid_LDS_non_positive_definite()
    # Test non-positive definite Q matrix
    A = Matrix{Float64}(I, 2, 2)
    C = Matrix{Float64}(I, 2, 2)
    Q = [1.0 0.0; 0.0 -0.1]  # Negative eigenvalue
    R = Matrix{Float64}(I, 2, 2)
    x0 = zeros(Float64, 2)
    P0 = Matrix{Float64}(I, 2, 2)
    b = zeros(Float64, 2)
    d = zeros(Float64, 2)

    gsm = GaussianStateModel(; A=A, Q=Q, x0=x0, P0=P0, b=b)
    gom = GaussianObservationModel(; C=C, R=R, d=d)
    lds = LinearDynamicalSystem(;
        state_model=gsm, obs_model=gom, latent_dim=2, obs_dim=2, fit_bool=fill(true, 6)
    )

    @test !isvalid_LDS(lds)
end

function test_isvalid_LDS_wrong_fit_bool_length()
    # Test wrong fit_bool length
    A = Matrix{Float64}(I, 2, 2)
    C = Matrix{Float64}(I, 2, 2)
    Q = Matrix{Float64}(I, 2, 2)
    R = Matrix{Float64}(I, 2, 2)
    x0 = zeros(Float64, 2)
    P0 = Matrix{Float64}(I, 2, 2)
    b = zeros(Float64, 2)
    d = zeros(Float64, 2)

    gsm = GaussianStateModel(; A=A, Q=Q, x0=x0, P0=P0, b=b)
    gom = GaussianObservationModel(; C=C, R=R, d=d)
    lds = LinearDynamicalSystem(;
        state_model=gsm,
        obs_model=gom,
        latent_dim=2,
        obs_dim=2,
        fit_bool=fill(true, 5),  # Should be 6!
    )

    @test !isvalid_LDS(lds)
end

function test_isvalid_LDS_poisson_extreme_log_d()
    # Test Poisson with extreme log_d values
    A = Matrix{Float64}(I, 2, 2)
    C = randn(3, 2)
    Q = Matrix{Float64}(I, 2, 2)
    log_d = [100.0, 0.0, 0.0]  # Extremely large value
    x0 = zeros(Float64, 2)
    P0 = Matrix{Float64}(I, 2, 2)
    b = zeros(Float64, 2)

    gsm = GaussianStateModel(; A=A, Q=Q, x0=x0, P0=P0, b=b)
    pom = PoissonObservationModel(; C=C, log_d=log_d)
    lds = LinearDynamicalSystem(;
        state_model=gsm, obs_model=pom, latent_dim=2, obs_dim=3, fit_bool=fill(true, 5)
    )

    @test !isvalid_LDS(lds)
end

function test_isvalid_LDS_asymmetric_covariance()
    # Test asymmetric Q matrix (should fail)
    A = Matrix{Float64}(I, 2, 2)
    C = Matrix{Float64}(I, 2, 2)
    Q = [1.0 0.5; 0.3 1.0]  # Asymmetric
    R = Matrix{Float64}(I, 2, 2)
    x0 = zeros(Float64, 2)
    P0 = Matrix{Float64}(I, 2, 2)
    b = zeros(Float64, 2)
    d = zeros(Float64, 2)

    gsm = GaussianStateModel(; A=A, Q=Q, x0=x0, P0=P0, b=b)
    gom = GaussianObservationModel(; C=C, R=R, d=d)
    lds = LinearDynamicalSystem(;
        state_model=gsm, obs_model=gom, latent_dim=2, obs_dim=2, fit_bool=fill(true, 6)
    )

    @test !isvalid_LDS(lds)
end
