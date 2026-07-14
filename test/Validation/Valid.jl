# Tests for Valid.jl validation functions

function test_validate_probvec()
    # Valid probability vectors - should not throw
    @test validate_probvec([0.3, 0.7]) === nothing
    @test validate_probvec([0.25, 0.25, 0.25, 0.25]) === nothing
    @test validate_probvec([1.0]) === nothing
    @test validate_probvec(Float32[0.3, 0.7]) === nothing

    #=
    A12: Float32 vectors whose sum isn't exactly 1.0f0. `fill(0.1f0, 10)` sums to
    1.0000001f0 and a normalizedrandom Float32 vector generally lands ~1e-7 off.
    =#
    @test validate_probvec(fill(0.1f0, 10)) === nothing
    v32 = rand(StableRNG(7), Float32, 8)
    v32 ./= sum(v32)
    @test validate_probvec(v32) === nothing

    # Invalid probability vectors - should throw
    @test_throws InvalidProbabilityVectorError validate_probvec([0.3, 0.8])  # Sum > 1
    @test_throws InvalidProbabilityVectorError validate_probvec([0.3, 0.5])  # Sum < 1
    @test_throws InvalidProbabilityVectorError validate_probvec([-0.1, 1.1])  # Negative value
    @test_throws InvalidProbabilityVectorError validate_probvec([0.5, 0.5, 0.5])  # Sum > 1
    @test_throws InvalidProbabilityVectorError validate_probvec([0.0, 0.0])  # Sum = 0
    # The loosened tolerance still rejects a genuinely-invalid Float32 vector.
    @test_throws InvalidProbabilityVectorError validate_probvec(Float32[0.3, 0.8])
end

function test_validate_LDS_gaussian()
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

    @test validate_LDS(lds) === nothing
end

function test_validate_LDS_poisson()
    # Create a valid Poisson LDS
    A = Matrix{Float64}(I, 2, 2)
    C = randn(3, 2)
    Q = Matrix{Float64}(I, 2, 2)
    d = zeros(Float64, 3)
    x0 = zeros(Float64, 2)
    P0 = Matrix{Float64}(I, 2, 2)
    b = zeros(Float64, 2)

    gsm = GaussianStateModel(; A=A, Q=Q, x0=x0, P0=P0, b=b)
    pom = PoissonObservationModel(; C=C, d=d)
    lds = LinearDynamicalSystem(;
        state_model=gsm, obs_model=pom, latent_dim=2, obs_dim=3, fit_bool=fill(true, 5)
    )

    @test validate_LDS(lds) === nothing
end

function test_validate_LDS_dimension_mismatch()
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

    @test_throws DimensionMismatchError validate_LDS(lds)
end

function test_validate_LDS_non_positive_definite()
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

    @test_throws NotPositiveDefiniteError validate_LDS(lds)
end

function test_validate_LDS_wrong_fit_bool_length()
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

    @test_throws DimensionMismatchError validate_LDS(lds)
end

function test_validate_LDS_poisson_extreme_d()
    # Test Poisson with extreme d values
    A = Matrix{Float64}(I, 2, 2)
    C = randn(3, 2)
    Q = Matrix{Float64}(I, 2, 2)
    d = [100.0, 0.0, 0.0]  # Extremely large value
    x0 = zeros(Float64, 2)
    P0 = Matrix{Float64}(I, 2, 2)
    b = zeros(Float64, 2)

    gsm = GaussianStateModel(; A=A, Q=Q, x0=x0, P0=P0, b=b)
    pom = PoissonObservationModel(; C=C, d=d)
    lds = LinearDynamicalSystem(;
        state_model=gsm, obs_model=pom, latent_dim=2, obs_dim=3, fit_bool=fill(true, 5)
    )

    @test_throws NumericalStabilityError validate_LDS(lds)
end

function test_validate_LDS_asymmetric_covariance()
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

    @test_throws NotSymmetricError validate_LDS(lds)
end

function test_validate_state_model_fields()
    D = 2
    function good()
        return GaussianStateModel(;
            A=Matrix{Float64}(I, D, D),
            Q=Matrix{Float64}(I, D, D),
            b=zeros(D),
            x0=zeros(D),
            P0=Matrix{Float64}(I, D, D),
        )
    end

    @test StateSpaceDynamics._validate_state_model(good(), D) === nothing

    sm = good()
    sm.A = Matrix{Float64}(I, D + 1, D + 1)            # A wrong shape
    @test_throws DimensionMismatchError StateSpaceDynamics._validate_state_model(sm, D)

    sm = good()
    sm.B = zeros(D + 1, 2)                              # B rows ≠ latent_dim
    @test_throws DimensionMismatchError StateSpaceDynamics._validate_state_model(sm, D)

    sm = good()
    sm.Q = Matrix{Float64}(I, D + 1, D + 1)            # Q wrong shape
    @test_throws DimensionMismatchError StateSpaceDynamics._validate_state_model(sm, D)

    sm = good()
    sm.b = zeros(D + 1)                                 # b wrong length
    @test_throws DimensionMismatchError StateSpaceDynamics._validate_state_model(sm, D)

    sm = good()
    sm.x0 = zeros(D + 1)                                # x0 wrong length
    @test_throws DimensionMismatchError StateSpaceDynamics._validate_state_model(sm, D)

    sm = good()
    sm.P0 = Matrix{Float64}(I, D + 1, D + 1)           # P0 wrong shape
    @test_throws DimensionMismatchError StateSpaceDynamics._validate_state_model(sm, D)

    sm = good()
    sm.P0 = [1.0 0.5; 0.3 1.0]                          # P0 asymmetric
    @test_throws NotSymmetricError StateSpaceDynamics._validate_state_model(sm, D)

    sm = good()
    sm.P0 = [1.0 0.0; 0.0 -0.1]                         # P0 non-PSD
    @test_throws NotPositiveDefiniteError StateSpaceDynamics._validate_state_model(sm, D)
end

function test_validate_obs_model_gaussian_fields()
    D, p = 2, 3
    function good()
        return GaussianObservationModel(;
            C=randn(p, D), R=Matrix{Float64}(I, p, p), d=zeros(p)
        )
    end

    @test StateSpaceDynamics._validate_obs_model(good(), p, D) === nothing

    om = good()
    om.C = randn(p + 1, D)                              # C wrong shape
    @test_throws DimensionMismatchError StateSpaceDynamics._validate_obs_model(om, p, D)

    om = good()
    om.R = Matrix{Float64}(I, p + 1, p + 1)            # R wrong shape
    @test_throws DimensionMismatchError StateSpaceDynamics._validate_obs_model(om, p, D)

    om = good()
    om.R = [1.0 0.5 0.0; 0.3 1.0 0.0; 0.0 0.0 1.0]      # R asymmetric
    @test_throws NotSymmetricError StateSpaceDynamics._validate_obs_model(om, p, D)

    om = good()
    om.R = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 -0.1]     # R non-PSD
    @test_throws NotPositiveDefiniteError StateSpaceDynamics._validate_obs_model(om, p, D)

    om = good()
    om.d = zeros(p + 1)                                 # d wrong length
    @test_throws DimensionMismatchError StateSpaceDynamics._validate_obs_model(om, p, D)
end

function test_validate_obs_model_poisson_fields()
    D, p = 2, 3
    good() = PoissonObservationModel(; C=randn(p, D), d=zeros(p))

    @test StateSpaceDynamics._validate_obs_model(good(), p, D) === nothing

    om = good()
    om.C = randn(p + 1, D)                              # C wrong shape
    @test_throws DimensionMismatchError StateSpaceDynamics._validate_obs_model(om, p, D)

    om = good()
    om.d = zeros(p + 1)                                 # d wrong length
    @test_throws DimensionMismatchError StateSpaceDynamics._validate_obs_model(om, p, D)

    om = good()
    om.d = [0.0, 0.0, 100.0]                            # extreme d → overflow risk
    @test_throws NumericalStabilityError StateSpaceDynamics._validate_obs_model(om, p, D)

    om = good()
    om.d = [50.0, 0.0, 0.0]                             # |d| == 50 is the boundary (check is > 50)
    @test StateSpaceDynamics._validate_obs_model(om, p, D) === nothing
end

function test_validate_LDS_poisson_fit_bool_length()
    # Poisson expects a length-5 fit_bool; a length-6 vector must be rejected.
    D, p = 2, 3
    gsm = GaussianStateModel(;
        A=Matrix{Float64}(I, D, D),
        Q=Matrix{Float64}(I, D, D),
        b=zeros(D),
        x0=zeros(D),
        P0=Matrix{Float64}(I, D, D),
    )
    pom = PoissonObservationModel(; C=randn(p, D), d=zeros(p))
    lds = LinearDynamicalSystem(;
        state_model=gsm, obs_model=pom, latent_dim=D, obs_dim=p, fit_bool=fill(true, 6)
    )
    @test_throws DimensionMismatchError validate_LDS(lds)
end

function test_validation_error_messages()
    msg(e) = sprint(showerror, e)

    @test occursin(
        "DimensionMismatchError", msg(DimensionMismatchError("A", (2, 2), (3, 3)))
    )
    @test occursin("not positive definite", msg(NotPositiveDefiniteError("Q", -0.5)))
    @test occursin("not symmetric", msg(NotSymmetricError("P0", 0.1)))
    @test occursin("NumericalStabilityError", msg(NumericalStabilityError("d", "overflow")))

    @test occursin("Sum is", msg(InvalidProbabilityVectorError("v", 0.9, false, false)))
    @test occursin("negative", msg(InvalidProbabilityVectorError("v", 1.0, true, false)))
    @test occursin("> 1.0", msg(InvalidProbabilityVectorError("v", 1.0, false, true)))

    err = try
        validate_probvec([0.2, 0.2]; name="myvec")
        nothing
    catch e
        e
    end
    @test err isa InvalidProbabilityVectorError
    @test occursin("myvec", msg(err))
end
