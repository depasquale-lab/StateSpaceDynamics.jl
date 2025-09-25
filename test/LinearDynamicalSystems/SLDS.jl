# Build a Gaussian LDS with given dims
function _make_gaussian_lds(latent_dim::Int, obs_dim::Int)
    A  = rand(latent_dim, latent_dim)
    Q  = Matrix(0.1 * I(latent_dim))
    b  = zeros(latent_dim)
    x0 = zeros(latent_dim)
    P0 = Matrix(1.0 * I(latent_dim))

    C = zeros(obs_dim, latent_dim)
    R = Matrix(1.0 * I(obs_dim))
    d = zeros(obs_dim)

    gsm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)       # state model
    gom = GaussianObservationModel(; C=C, R=R, d=d)               # obs model
    return LinearDynamicalSystem(;
        state_model=gsm, obs_model=gom,
        latent_dim=latent_dim, obs_dim=obs_dim, fit_bool=fill(true, 6)
    )
end

# Build a Poisson-observation LDS (Gaussian state model)
function _make_poisson_lds(latent_dim::Int, obs_dim::Int)
    A  = rand(latent_dim, latent_dim)
    Q  = Matrix(0.1 * I(latent_dim))
    b  = zeros(latent_dim)
    x0 = zeros(latent_dim)
    P0 = Matrix(1.0 * I(latent_dim))

    C = zeros(obs_dim, latent_dim)
    log_d = zeros(obs_dim)

    gsm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
    pom = PoissonObservationModel(; C=C, log_d=log_d)
    return LinearDynamicalSystem(;
        state_model=gsm, obs_model=pom,
        latent_dim=latent_dim, obs_dim=obs_dim, fit_bool=fill(true, 6)
    )
end

# Simple probability vector / row-stochastic makers
_probvec(K) = fill(1.0 / K, K)
function _rowstochastic(K)
    A = fill(0.0, K, K)
    for i in 1:K
        A[i, :] .= _probvec(K)
    end
    return A
end

function test_valid_SLDS_happy_path()
    K = 3
    lds = _make_gaussian_lds(2, 4)
    s = SLDS(A=_rowstochastic(K), Z₀=_probvec(K), LDSs=fill(lds, K))
    @test begin
        valid_SLDS(s)    # should not throw
        true
    end
end

function test_valid_SLDS_dimension_mismatches()
    K = 2
    lds = _make_gaussian_lds(2, 3)

    # size(A,1)=K, but length(Z₀) ≠ K
    s_badZ0 = SLDS(A=_rowstochastic(K), Z₀=_probvec(K+1), LDSs=fill(lds, K))
    @test_throws AssertionError valid_SLDS(s_badZ0)

    # size(A,1)=K, but number of LDSs ≠ K
    s_badLDSs = SLDS(A=_rowstochastic(K), Z₀=_probvec(K), LDSs=fill(lds, K+1))
    @test_throws AssertionError valid_SLDS(s_badLDSs)
end

function test_valid_SLDS_nonstochastic_rows_and_invalid_Z0()
    K = 3
    lds = _make_gaussian_lds(2, 2)

    # Non-probability row in A (negative entry)
    A_bad = _rowstochastic(K)
    A_bad[2, :] .= (-0.1, 0.5, 0.6)  # sums to 1 but has a negative entry
    s_badA = SLDS(A=A_bad, Z₀=_probvec(K), LDSs=fill(lds, K))
    @test_throws AssertionError valid_SLDS(s_badA)

    # Z0 does not sum to 1
    Z0_bad = _probvec(K); Z0_bad[1] += 0.1
    s_badZ0 = SLDS(A=_rowstochastic(K), Z₀=Z0_bad, LDSs=fill(lds, K))
    @test_throws AssertionError valid_SLDS(s_badZ0)
end

function test_valid_SLDS_mixed_observation_model_types()
    # mode 1..K-1 Gaussian obs; last mode Poisson obs → should assert (type mismatch)
    K = 3
    lds_g = _make_gaussian_lds(2, 2)
    lds_p = _make_poisson_lds(2, 2)  # different obs model type
    s = SLDS(A=_rowstochastic(K), Z₀=_probvec(K), LDSs=[lds_g, lds_g, lds_p])
    @test_throws AssertionError valid_SLDS(s)
end

function test_valid_SLDS_inconsistent_latent_or_obs_dims()
    K = 2
    lds_a = _make_gaussian_lds(2, 3)
    lds_b_state = _make_gaussian_lds(3, 3) # different latent_dim
    lds_b_obs   = _make_gaussian_lds(2, 4) # different obs_dim

    s_bad_state = SLDS(A=_rowstochastic(K), Z₀=_probvec(K), LDSs=[lds_a, lds_b_state])
    @test_throws AssertionError valid_SLDS(s_bad_state)

    s_bad_obs = SLDS(A=_rowstochastic(K), Z₀=_probvec(K), LDSs=[lds_a, lds_b_obs])
    @test_throws AssertionError valid_SLDS(s_bad_obs)
end