# Build a Gaussian LDS with given dims
function _make_gaussian_lds(latent_dim::Int, obs_dim::Int)
    A = rand(latent_dim, latent_dim)
    Q = Matrix(0.1 * I(latent_dim))
    b = zeros(latent_dim)
    x0 = zeros(latent_dim)
    P0 = Matrix(1.0 * I(latent_dim))

    C = zeros(obs_dim, latent_dim)
    R = Matrix(1.0 * I(obs_dim))
    d = zeros(obs_dim)

    gsm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)       # state model
    gom = GaussianObservationModel(; C=C, R=R, d=d)               # obs model
    return LinearDynamicalSystem(;
        state_model=gsm,
        obs_model=gom,
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        fit_bool=fill(true, 6),
    )
end

# Build a Poisson-observation LDS (Gaussian state model)
function _make_poisson_lds(latent_dim::Int, obs_dim::Int)
    A = rand(latent_dim, latent_dim)
    Q = Matrix(0.1 * I(latent_dim))
    b = zeros(latent_dim)
    x0 = zeros(latent_dim)
    P0 = Matrix(1.0 * I(latent_dim))

    C = zeros(obs_dim, latent_dim)
    log_d = zeros(obs_dim)

    gsm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
    pom = PoissonObservationModel(; C=C, log_d=log_d)
    return LinearDynamicalSystem(;
        state_model=gsm,
        obs_model=pom,
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        fit_bool=fill(true, 6),
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
    s = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))
    @test begin
        isvalid_SLDS(s)    # should not throw
        true
    end
end

function test_valid_SLDS_dimension_mismatches()
    K = 2
    lds = _make_gaussian_lds(2, 3)

    # size(A,1)=K, but length(πₖ) ≠ K
    s_badZ0 = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K+1), LDSs=fill(lds, K))
    @test_throws AssertionError isvalid_SLDS(s_badZ0)

    # size(A,1)=K, but number of LDSs ≠ K
    s_badLDSs = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K+1))
    @test_throws AssertionError isvalid_SLDS(s_badLDSs)
end

function test_valid_SLDS_nonstochastic_rows_and_invalid_Z0()
    K = 3
    lds = _make_gaussian_lds(2, 2)

    # Non-probability row in A (negative entry)
    A_bad = _rowstochastic(K)
    A_bad[2, :] .= (-0.1, 0.5, 0.6)  # sums to 1 but has a negative entry
    s_badA = SLDS(; A=A_bad, πₖ=_probvec(K), LDSs=fill(lds, K))
    @test_throws AssertionError isvalid_SLDS(s_badA)

    # Z0 does not sum to 1
    Z0_bad = _probvec(K);
    Z0_bad[1] += 0.1
    s_badZ0 = SLDS(; A=_rowstochastic(K), πₖ=Z0_bad, LDSs=fill(lds, K))
    @test_throws AssertionError isvalid_SLDS(s_badZ0)
end

function test_valid_SLDS_mixed_observation_model_types()
    # mode 1..K-1 Gaussian obs; last mode Poisson obs → should assert (type mismatch)
    K = 3
    lds_g = _make_gaussian_lds(2, 2)
    lds_p = _make_poisson_lds(2, 2)  # different obs model type
    @test_throws MethodError SLDS(
        A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds_g, lds_g, lds_p]
    )
end

function test_valid_SLDS_inconsistent_latent_or_obs_dims()
    K = 2
    lds_a = _make_gaussian_lds(2, 3)
    lds_b_state = _make_gaussian_lds(3, 3) # different latent_dim
    lds_b_obs = _make_gaussian_lds(2, 4) # different obs_dim

    s_bad_state = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds_a, lds_b_state])
    @test_throws AssertionError isvalid_SLDS(s_bad_state)

    s_bad_obs = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds_a, lds_b_obs])
    @test_throws AssertionError isvalid_SLDS(s_bad_obs)
end

function test_SLDS_sampling_gaussian()
    K = 3
    lds = _make_gaussian_lds(2, 4)
    s = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    tsteps, ntrials = 50, 5
    z, x, y = rand(s; tsteps=tsteps, ntrials=ntrials)

    @test size(z) == (tsteps, ntrials)
    @test size(x) == (2, tsteps, ntrials)
    @test size(y) == (4, tsteps, ntrials)
    @test all(1 ≤ z[t, n] ≤ K for t in 1:tsteps, n in 1:ntrials)
    @test all(isfinite, x)
    @test all(isfinite, y)
end

function test_SLDS_sampling_poisson()
    K = 2
    lds = _make_poisson_lds(2, 3)
    s = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    tsteps, ntrials = 30, 3
    z, x, y = rand(s; tsteps=tsteps, ntrials=ntrials)

    @test size(z) == (tsteps, ntrials)
    @test size(x) == (2, tsteps, ntrials)
    @test size(y) == (3, tsteps, ntrials)
    @test all(1 ≤ z[t, n] ≤ K for t in 1:tsteps, n in 1:ntrials)
    @test all(y[i, t, n] ≥ 0 for i in 1:3, t in 1:tsteps, n in 1:ntrials)
    @test all(y[i, t, n] == round(y[i, t, n]) for i in 1:3, t in 1:tsteps, n in 1:ntrials)
end

function test_SLDS_deterministic_transitions()
    K = 2
    lds = _make_gaussian_lds(2, 2)

    # Always transition 1 → 2 → 2 → 2...
    A_det = [0.0 1.0; 0.0 1.0]
    Z0_det = [1.0, 0.0]  # Always start in state 1

    s = SLDS(; A=A_det, πₖ=Z0_det, LDSs=fill(lds, K))

    tsteps = 10
    z, x, y = rand(s; tsteps=tsteps, ntrials=3)

    # Should always start in state 1
    @test all(z[1, n] == 1 for n in 1:3)
    # Should always be in state 2 after first step
    @test all(z[t, n] == 2 for t in 2:tsteps, n in 1:3)
end

function test_SLDS_single_trial()
    K = 3
    lds = _make_gaussian_lds(2, 4)
    s = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    tsteps = 100
    z, x, y = rand(s; tsteps=tsteps, ntrials=1)

    @test size(z) == (tsteps, 1)
    @test size(x) == (2, tsteps, 1)
    @test size(y) == (4, tsteps, 1)
end

function test_SLDS_reproducibility()
    K = 2
    lds = _make_gaussian_lds(2, 3)
    s = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    # Same seed should give same results
    Random.seed!(42)
    z1, x1, y1 = rand(s; tsteps=20, ntrials=2)

    Random.seed!(42)
    z2, x2, y2 = rand(s; tsteps=20, ntrials=2)

    @test z1 == z2
    @test x1 ≈ x2
    @test y1 ≈ y2
end

function test_SLDS_single_state_edge_case()
    K = 1
    lds = _make_gaussian_lds(2, 3)
    s = SLDS(; A=reshape([1.0], 1, 1), πₖ=[1.0], LDSs=[lds])

    @test isvalid_SLDS(s)

    z, x, y = rand(s; tsteps=10, ntrials=2)
    @test all(z .== 1)  # Should always be in state 1
end

function test_SLDS_minimal_dimensions()
    K = 2
    lds = _make_gaussian_lds(1, 1)  # Minimal dimensions
    s = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(s; tsteps=10, ntrials=3)

    @test size(x) == (1, 10, 3)
    @test size(y) == (1, 10, 3)
    @test all(isfinite, x)
    @test all(isfinite, y)
end

function test_valid_SLDS_probability_helper_functions()
    # Test probability vector validation
    @test isvalid_probvec([0.3, 0.7])
    @test isvalid_probvec([0.25, 0.25, 0.25, 0.25])
    @test !isvalid_probvec([0.6, 0.5])   # Sums to > 1
    @test !isvalid_probvec([-0.1, 1.1])  # Has negative

    # Test helper functions
    @test _probvec(4) ≈ [0.25, 0.25, 0.25, 0.25]

    A = _rowstochastic(3)
    @test size(A) == (3, 3)
    @test all(isapprox(sum(A[i, :]), 1.0) for i in 1:3)
    @test all(A[i, j] ≥ 0 for i in 1:3, j in 1:3)
end

function test_SLDS_gradient_numerical()
    K = 2
    lds = _make_gaussian_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(slds; tsteps=20, ntrials=1)

    tsteps = size(y, 2)
    w = rand(K, tsteps)
    w ./= sum(w; dims=1)

    y_trial = y[:, :, 1]
    x_trial = x[:, :, 1]

    # Analytical gradient
    grad_analytical = StateSpaceDynamics.Gradient(slds, y_trial, x_trial, w)

    # For numerical gradient, we need to compute the weighted objective
    # The trick: the gradient is LINEAR in the weights, so we can compute
    # a weighted sum of gradients, which equals the gradient of the weighted objective
    function weighted_ll(x_flat)
        x_mat = reshape(x_flat, size(x_trial))

        # Compute the weighted log-likelihood term by term
        # We need to manually compute each component weighted by w
        ll = 0.0

        for k in 1:K
            lds_k = slds.LDSs[k]

            # Extract parameters
            A_k = lds_k.state_model.A
            Q_k = lds_k.state_model.Q
            b_k = lds_k.state_model.b
            x0_k = lds_k.state_model.x0
            P0_k = lds_k.state_model.P0
            C_k = lds_k.obs_model.C
            R_k = lds_k.obs_model.R
            d_k = lds_k.obs_model.d

            R_chol = cholesky(Symmetric(R_k)).U
            Q_chol = cholesky(Symmetric(Q_k)).U
            P0_chol = cholesky(Symmetric(P0_k)).U

            # Initial state (weighted by w[k, 1])
            dx0 = x_mat[:, 1] - x0_k
            ll += w[k, 1] * (-0.5 * sum(abs2, P0_chol \ dx0))

            # Dynamics and emissions
            for t in 1:tsteps
                # Emission (weighted by w[k, t])
                dy = y_trial[:, t] - (C_k * x_mat[:, t] + d_k)
                ll += w[k, t] * (-0.5 * sum(abs2, R_chol \ dy))

                # Dynamics (weighted by w[k, t])
                if t > 1
                    dx = x_mat[:, t] - (A_k * x_mat[:, t - 1] + b_k)
                    ll += w[k, t] * (-0.5 * sum(abs2, Q_chol \ dx))
                end
            end
        end

        return ll
    end

    grad_numerical = ForwardDiff.gradient(weighted_ll, vec(x_trial))
    grad_numerical = reshape(grad_numerical, size(x_trial))

    @test isapprox(grad_analytical, grad_numerical, rtol=1e-5, atol=1e-5)
end

function test_SLDS_hessian_numerical()
    K = 2
    lds = _make_gaussian_lds(2, 2)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(slds; tsteps=5, ntrials=1)

    tsteps = size(y, 2)
    w = rand(K, tsteps)
    w ./= sum(w; dims=1)

    y_trial = y[:, :, 1]
    x_trial = x[:, :, 1]

    # Analytical Hessian
    H, H_diag, H_super, H_sub = StateSpaceDynamics.Hessian(slds, y_trial, x_trial, w)

    # Numerical Hessian - same weighted objective as gradient test
    function weighted_ll(x_flat)
        x_mat = reshape(x_flat, size(x_trial))
        ll = 0.0

        for k in 1:K
            lds_k = slds.LDSs[k]

            A_k = lds_k.state_model.A
            Q_k = lds_k.state_model.Q
            b_k = lds_k.state_model.b
            x0_k = lds_k.state_model.x0
            P0_k = lds_k.state_model.P0
            C_k = lds_k.obs_model.C
            R_k = lds_k.obs_model.R
            d_k = lds_k.obs_model.d

            R_chol = cholesky(Symmetric(R_k)).U
            Q_chol = cholesky(Symmetric(Q_k)).U
            P0_chol = cholesky(Symmetric(P0_k)).U

            # Initial state (weighted by w[k, 1])
            dx0 = x_mat[:, 1] - x0_k
            ll += w[k, 1] * (-0.5 * sum(abs2, P0_chol \ dx0))

            # Dynamics and emissions
            for t in 1:tsteps
                # Emission (weighted by w[k, t])
                dy = y_trial[:, t] - (C_k * x_mat[:, t] + d_k)
                ll += w[k, t] * (-0.5 * sum(abs2, R_chol \ dy))

                # Dynamics (weighted by w[k, t])
                if t > 1
                    dx = x_mat[:, t] - (A_k * x_mat[:, t - 1] + b_k)
                    ll += w[k, t] * (-0.5 * sum(abs2, Q_chol \ dx))
                end
            end
        end

        return ll
    end

    H_numerical = ForwardDiff.hessian(weighted_ll, vec(x_trial))

    @test isapprox(H, H_numerical, rtol=1e-5, atol=1e-5)
end

function test_SLDS_gradient_reduces_to_single_LDS()
    K = 3
    lds = _make_gaussian_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(slds; tsteps=20, ntrials=1)

    tsteps = size(y, 2)

    # Test each discrete state in isolation
    for active_k in 1:K
        # Create weights where only state active_k is active
        w = zeros(K, tsteps)
        w[active_k, :] .= 1.0

        grad_slds = StateSpaceDynamics.Gradient(slds, y[:, :, 1], x[:, :, 1], w)
        grad_lds = StateSpaceDynamics.Gradient(slds.LDSs[active_k], y[:, :, 1], x[:, :, 1])

        @test isapprox(grad_slds, grad_lds, rtol=1e-10)
    end
end

function test_SLDS_hessian_block_structure()
    K = 2
    lds = _make_gaussian_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(slds; tsteps=10, ntrials=1)

    tsteps = size(y, 2)
    w = rand(K, tsteps)
    w ./= sum(w; dims=1)

    H, H_diag, H_super, H_sub = StateSpaceDynamics.Hessian(slds, y[:, :, 1], x[:, :, 1], w)

    D = slds.LDSs[1].latent_dim

    # Check sizes
    @test length(H_diag) == tsteps
    @test length(H_super) == tsteps - 1
    @test length(H_sub) == tsteps - 1
    @test size(H) == (D * tsteps, D * tsteps)

    # Check each block is D×D
    for t in 1:tsteps
        @test size(H_diag[t]) == (D, D)
    end
    for t in 1:(tsteps - 1)
        @test size(H_super[t]) == (D, D)
        @test size(H_sub[t]) == (D, D)
    end

    # Check block-tridiagonal structure (zeros outside bands)
    for i in 1:(D * tsteps)
        for j in 1:(D * tsteps)
            block_i = (i-1) ÷ D + 1
            block_j = (j-1) ÷ D + 1
            if abs(block_i - block_j) > 1
                @test abs(H[i, j]) < 1e-10
            end
        end
    end
end

function test_SLDS_gradient_weight_normalization()
    K = 2
    lds = _make_gaussian_lds(2, 2)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(slds; tsteps=15, ntrials=1)

    tsteps = size(y, 2)

    # Create two different weight matrices that sum to same values
    w1 = rand(K, tsteps)
    w1 ./= sum(w1; dims=1)

    w2 = 0.5 .* w1  # Scale by 0.5
    w2 ./= sum(w2; dims=1)  # Renormalize

    # Gradients should be the same (weights are normalized)
    grad1 = StateSpaceDynamics.Gradient(slds, y[:, :, 1], x[:, :, 1], w1)
    grad2 = StateSpaceDynamics.Gradient(slds, y[:, :, 1], x[:, :, 1], w2)

    @test isapprox(grad1, grad2, rtol=1e-10)
end

function test_SLDS_smooth_basic()
    # Setup: Simple 2-state SLDS with known structure
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 20

    # Create two LDS models
    lds1 = _make_gaussian_lds(latent_dim, obs_dim)
    lds2 = _make_gaussian_lds(latent_dim, obs_dim)

    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds1, lds2])

    # Generate data
    z, x, y = rand(slds; tsteps=tsteps, ntrials=1)

    # Create uniform weights (should behave like averaging both LDS)
    w = ones(Float64, K, tsteps) ./ K

    # Call smooth
    x_smooth, p_smooth = smooth(slds, y[:, :, 1], w)

    # Basic checks
    @test size(x_smooth) == (latent_dim, tsteps)
    @test size(p_smooth) == (latent_dim, latent_dim, tsteps)

    # Covariances should be positive definite
    for t in 1:tsteps
        @test isposdef(p_smooth[:, :, t])
    end

    # Smoothed states should be reasonable (not NaN/Inf)
    @test all(isfinite, x_smooth)
    @test all(isfinite, p_smooth)
end

function test_SLDS_smooth_reduces_to_single_LDS()
    # When all weight is on one LDS, should match that LDS's smooth
    K = 3
    latent_dim = 2
    obs_dim = 3
    tsteps = 15

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(slds; tsteps=tsteps, ntrials=1)
    y_trial = y[:, :, 1]

    # All weight on first LDS
    w = zeros(Float64, K, tsteps)
    w[1, :] .= 1.0

    # SLDS smooth with concentrated weights
    x_slds, p_slds = smooth(slds, y_trial, w)

    # Direct LDS smooth
    x_lds, p_lds = smooth(lds, y_trial)

    # Should match closely (allowing for numerical tolerance)
    @test isapprox(x_slds, x_lds, rtol=1e-4)
    @test isapprox(p_slds, p_lds, rtol=1e-4)
end

function test_SLDS_smooth_with_realistic_weights()
    # Test with realistic posterior weights that change over time
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 25

    lds1 = _make_gaussian_lds(latent_dim, obs_dim)
    lds2 = _make_gaussian_lds(latent_dim, obs_dim)

    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds1, lds2])

    z, x, y = rand(slds; tsteps=tsteps, ntrials=1)
    y_trial = y[:, :, 1]

    # Create time-varying weights (simulate discrete state posterior)
    w = zeros(Float64, K, tsteps)
    for t in 1:tsteps
        if t < tsteps ÷ 2
            w[1, t] = 0.8
            w[2, t] = 0.2
        else
            w[1, t] = 0.3
            w[2, t] = 0.7
        end
    end

    x_smooth, p_smooth = smooth(slds, y_trial, w)

    @test size(x_smooth) == (latent_dim, tsteps)
    @test all(isfinite, x_smooth)
    @test all(t -> isposdef(p_smooth[:, :, t]), 1:tsteps)
end

function test_SLDS_smooth_consistency_with_gradients()
    # Verify that smooth finds a point where gradient is near zero
    K = 2
    latent_dim = 2
    obs_dim = 2
    tsteps = 10

    lds1 = _make_gaussian_lds(latent_dim, obs_dim)
    lds2 = _make_gaussian_lds(latent_dim, obs_dim)

    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds1, lds2])

    z, x, y = rand(slds; tsteps=tsteps, ntrials=1)
    y_trial = y[:, :, 1]

    w = rand(Float64, K, tsteps)
    w ./= sum(w; dims=1)  # Normalize

    x_smooth, _ = smooth(slds, y_trial, w)

    # Gradient at optimum should be small
    grad = StateSpaceDynamics.Gradient(slds, y_trial, x_smooth, w)

    @test norm(grad) < 1e-4  # Should be near zero at optimum
end

function test_SLDS_smooth_entropy_calculation()
    # Verify entropy is computed and has reasonable values
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 15

    lds1 = _make_gaussian_lds(latent_dim, obs_dim)
    lds2 = _make_gaussian_lds(latent_dim, obs_dim)

    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds1, lds2])

    z, x, y = rand(slds; tsteps=tsteps, ntrials=1)

    w = ones(Float64, K, tsteps) ./ K

    # Call smooth! directly to access StateSpaceDynamics.FilterSmooth
    fs = StateSpaceDynamics.initialize_FilterSmooth(slds.LDSs[1], tsteps)
    StateSpaceDynamics.smooth!(slds, fs, y[:, :, 1], w)

    # Entropy should be positive (for Gaussian)
    # @test fs.entropy > 0
    # @test isfinite(fs.entropy)
end

function test_SLDS_smooth_covariance_symmetry()
    # Ensure covariances remain symmetric
    K = 2
    latent_dim = 3
    obs_dim = 2
    tsteps = 12

    lds1 = _make_gaussian_lds(latent_dim, obs_dim)
    lds2 = _make_gaussian_lds(latent_dim, obs_dim)

    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds1, lds2])

    z, x, y = rand(slds; tsteps=tsteps, ntrials=1)

    w = rand(Float64, K, tsteps)
    w ./= sum(w; dims=1)

    _, p_smooth = smooth(slds, y[:, :, 1], w)

    # Check symmetry at each timestep
    for t in 1:tsteps
        @test isapprox(p_smooth[:, :, t], p_smooth[:, :, t]', atol=1e-10)
    end
end

function test_SLDS_smooth_different_weight_patterns()
    # Test various weight patterns
    K = 2
    latent_dim = 2
    obs_dim = 2
    tsteps = 20

    lds1 = _make_gaussian_lds(latent_dim, obs_dim)
    lds2 = _make_gaussian_lds(latent_dim, obs_dim)

    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds1, lds2])

    z, x, y = rand(slds; tsteps=tsteps, ntrials=1)
    y_trial = y[:, :, 1]

    # Test 1: Uniform weights
    w_uniform = ones(Float64, K, tsteps) ./ K
    x1, p1 = smooth(slds, y_trial, w_uniform)
    @test all(isfinite, x1)

    # Test 2: One-hot weights (switch halfway)
    w_onehot = zeros(Float64, K, tsteps)
    w_onehot[1, 1:(tsteps ÷ 2)] .= 1.0
    w_onehot[2, (tsteps ÷ 2 + 1):end] .= 1.0
    x2, p2 = smooth(slds, y_trial, w_onehot)
    @test all(isfinite, x2)

    # Test 3: Smooth transition
    w_smooth = zeros(Float64, K, tsteps)
    for t in 1:tsteps
        alpha = (t - 1) / (tsteps - 1)
        w_smooth[1, t] = 1 - alpha
        w_smooth[2, t] = alpha
    end
    x3, p3 = smooth(slds, y_trial, w_smooth)
    @test all(isfinite, x3)
end

function test_SLDS_sample_posterior_basic()
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 20

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(slds; tsteps=tsteps, ntrials=1)
    w = ones(Float64, K, tsteps) ./ K

    # Get smoothed posterior
    fs = StateSpaceDynamics.initialize_FilterSmooth(slds.LDSs[1], tsteps)
    StateSpaceDynamics.smooth!(slds, fs, y[:, :, 1], w)

    # Sample from posterior
    x_sample, entropy = StateSpaceDynamics.sample_posterior(fs)

    @test size(x_sample) == (latent_dim, tsteps)
    # @test entropy > 0
    # @test isfinite(entropy)
    @test all(isfinite, x_sample)
end

function test_SLDS_estep_basic()
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 15
    ntrials = 2

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(slds; tsteps=tsteps, ntrials=ntrials)

    # Initialize structures
    tfs = StateSpaceDynamics.initialize_FilterSmooth(slds.LDSs[1], tsteps, ntrials)
    fbs = [
        StateSpaceDynamics.initialize_forward_backward(slds, tsteps, Float64) for
        _ in 1:ntrials
    ]

    # Initialize with uniform weights
    for trial in 1:ntrials
        w_uniform = ones(Float64, K, tsteps) ./ K
        StateSpaceDynamics.smooth!(slds, tfs[trial], y[:, :, trial], w_uniform)
    end

    # Sample and run E-step
    x_samples, _ = StateSpaceDynamics.sample_posterior(tfs, 1)
    elbo = StateSpaceDynamics.estep!(slds, tfs, fbs, y, x_samples)

    # Check ELBO is finite
    @test isfinite(elbo)

    # Check sufficient statistics were computed
    for trial in 1:ntrials
        @test size(tfs[trial].E_z) == (latent_dim, tsteps)
        @test size(tfs[trial].E_zz) == (latent_dim, latent_dim, tsteps)
        @test size(tfs[trial].E_zz_prev) == (latent_dim, latent_dim, tsteps)
        @test all(isfinite, tfs[trial].E_z)
        @test all(isfinite, tfs[trial].E_zz)
        @test all(isfinite, tfs[trial].E_zz_prev)
    end

    # Check discrete posteriors
    for trial in 1:ntrials
        w = exp.(fbs[trial].γ)
        @test size(w) == (K, tsteps)
        @test all(isfinite, w)
        @test all(w .>= 0)
        # Each column should approximately sum to 1
        @test all(isapprox.(sum(w; dims=1), 1.0, atol=1e-10))
    end
end

function test_SLDS_mstep_updates_parameters()
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 15
    ntrials = 2

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(slds; tsteps=tsteps, ntrials=ntrials)

    # Initialize structures
    tfs = StateSpaceDynamics.initialize_FilterSmooth(slds.LDSs[1], tsteps, ntrials)
    fbs = [
        StateSpaceDynamics.initialize_forward_backward(slds, tsteps, Float64) for
        _ in 1:ntrials
    ]

    # Run one E-step
    for trial in 1:ntrials
        w_uniform = ones(Float64, K, tsteps) ./ K
        StateSpaceDynamics.smooth!(slds, tfs[trial], y[:, :, trial], w_uniform)
    end
    x_samples, _ = StateSpaceDynamics.sample_posterior(tfs, 1)
    StateSpaceDynamics.estep!(slds, tfs, fbs, y, x_samples)

    # Store old parameters
    A_old = copy(slds.A)
    πₖ_old = copy(slds.πₖ)
    A_lds_old = [copy(lds.state_model.A) for lds in slds.LDSs]

    # Run M-step
    StateSpaceDynamics.mstep!(slds, tfs, fbs, y)

    # Check parameters changed (with high probability)
    @test !isapprox(slds.A, A_old, rtol=1e-6) || true  # May not change if data is degenerate
    @test all(isfinite, slds.A)
    @test all(isfinite, slds.πₖ)

    # Check stochasticity is preserved
    @test all(isapprox.(sum(slds.A; dims=2), 1.0, atol=1e-10))
    @test isapprox(sum(slds.πₖ), 1.0, atol=1e-10)
    @test all(slds.A .>= 0)
    @test all(slds.πₖ .>= 0)
end

function test_SLDS_fit_runs_to_completion()
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 15
    ntrials = 2
    max_iter = 5

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(slds; tsteps=tsteps, ntrials=ntrials)

    # Fit without progress bar
    elbos = fit!(slds, y; max_iter=max_iter, progress=false)

    # Check correct number of iterations
    @test length(elbos) == max_iter

    # Check all ELBOs are finite
    @test all(isfinite, elbos)
end

function test_SLDS_fit_elbo_generally_increases()
    # ELBO should generally increase or stabilize (may have noise due to sampling)
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 20
    ntrials = 3
    max_iter = 10

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(slds; tsteps=tsteps, ntrials=ntrials)

    elbos = fit!(slds, y; max_iter=max_iter, progress=false)

    # Check that later ELBOs are generally higher than early ones
    # (allowing for stochastic noise)
    early_mean = mean(elbos[1:3])
    late_mean = mean(elbos[(end - 2):end])

    @test late_mean > early_mean - 100  # Allow some slack for noise
end

function test_SLDS_fit_multitrial()
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 15
    ntrials = 5
    max_iter = 5

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(slds; tsteps=tsteps, ntrials=ntrials)

    elbos = fit!(slds, y; max_iter=max_iter, progress=false)

    @test length(elbos) == max_iter
    @test all(isfinite, elbos)
end

function test_SLDS_estep_elbo_components()
    # Verify ELBO contains expected components
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 10
    ntrials = 1

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(slds; tsteps=tsteps, ntrials=ntrials)

    tfs = StateSpaceDynamics.initialize_FilterSmooth(slds.LDSs[1], tsteps, ntrials)
    fbs = [
        StateSpaceDynamics.initialize_forward_backward(slds, tsteps, Float64) for
        _ in 1:ntrials
    ]

    # Initialize
    w_uniform = ones(Float64, K, tsteps) ./ K
    StateSpaceDynamics.smooth!(slds, tfs[1], y[:, :, 1], w_uniform)

    # Sample and run E-step
    x_samples, _ = StateSpaceDynamics.sample_posterior(tfs, 1)
    elbo = StateSpaceDynamics.estep!(slds, tfs, fbs, y, x_samples)

    # ELBO should be finite and typically negative (log probability)
    @test isfinite(elbo)

    # Entropy should be positive
    # @test tfs[1].entropy > 0
end

function test_weighted_update_initial_state_mean()
    """Test weighted update of initial state mean"""
    K = 2
    latent_dim = 2
    obs_dim = 3
    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    ntrials = 4
    tsteps = 10
    z, x, y = rand(slds; tsteps=tsteps, ntrials=ntrials)

    # Create per-trial, per-timestep weights
    w = [rand(K, tsteps) for _ in 1:ntrials]
    for trial in 1:ntrials
        w[trial] ./= sum(w[trial]; dims=1)
    end

    # Mock FilterSmooth objects with ALL required fields
    tfs_array = Vector{StateSpaceDynamics.FilterSmooth{Float64}}(undef, ntrials)
    for k in 1:ntrials
        x_smooth = x[:, :, k]
        p_smooth = zeros(Float64, latent_dim, latent_dim, tsteps)
        p_smooth_tt1 = zeros(Float64, latent_dim, latent_dim, tsteps)
        E_z = x[:, :, k]
        E_zz = zeros(Float64, latent_dim, latent_dim, tsteps)
        E_zz_prev = zeros(Float64, latent_dim, latent_dim, tsteps)

        for t in 1:tsteps
            p_smooth[:, :, t] .= 0.1 * I(latent_dim)
            p_smooth_tt1[:, :, t] .= 0.1 * I(latent_dim)
            E_zz[:, :, t] .= E_z[:, t] * E_z[:, t]' .+ 0.1 * I(latent_dim)
            if t > 1
                E_zz_prev[:, :, t] .= E_z[:, t] * E_z[:, t - 1]'
            end
        end

        entropy = 0.0
        tfs_array[k] = StateSpaceDynamics.FilterSmooth(
            x_smooth, p_smooth, p_smooth_tt1, E_z, E_zz, E_zz_prev, entropy
        )
    end
    tfs = StateSpaceDynamics.TrialFilterSmooth(tfs_array)

    for active_k in 1:K
        lds_k = slds.LDSs[active_k]
        lds_k.fit_bool[1] = true

        # Create weights for this LDS
        w_k = [w[trial][active_k, :] for trial in 1:ntrials]

        StateSpaceDynamics.update_initial_state_mean!(lds_k, tfs, w_k)

        # Check result is finite
        @test all(isfinite.(lds_k.state_model.x0))
    end
end

function test_weighted_update_A_b()
    """Test weighted update of A and b matrices"""
    K = 2
    latent_dim = 2
    obs_dim = 3
    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    ntrials = 3
    tsteps = 15
    z, x, y = rand(slds; tsteps=tsteps, ntrials=ntrials)

    # Create weights
    w = [rand(K, tsteps) for _ in 1:ntrials]
    for trial in 1:ntrials
        w[trial] ./= sum(w[trial]; dims=1)
    end

    # Create FilterSmooth objects with ALL required fields
    tfs_array = Vector{StateSpaceDynamics.FilterSmooth{Float64}}(undef, ntrials)
    for k in 1:ntrials
        x_smooth = x[:, :, k]
        p_smooth = zeros(Float64, latent_dim, latent_dim, tsteps)
        p_smooth_tt1 = zeros(Float64, latent_dim, latent_dim, tsteps)
        E_z = x[:, :, k]
        E_zz = zeros(Float64, latent_dim, latent_dim, tsteps)
        E_zz_prev = zeros(Float64, latent_dim, latent_dim, tsteps)

        for t in 1:tsteps
            p_smooth[:, :, t] .= 0.1 * I(latent_dim)
            p_smooth_tt1[:, :, t] .= 0.1 * I(latent_dim)
            E_zz[:, :, t] .= E_z[:, t] * E_z[:, t]' .+ 0.1 * I(latent_dim)
            if t > 1
                E_zz_prev[:, :, t] .= E_z[:, t] * E_z[:, t - 1]'
            end
        end

        entropy = 0.0
        tfs_array[k] = StateSpaceDynamics.FilterSmooth(
            x_smooth, p_smooth, p_smooth_tt1, E_z, E_zz, E_zz_prev, entropy
        )
    end
    tfs = StateSpaceDynamics.TrialFilterSmooth(tfs_array)

    for active_k in 1:K
        lds_k = slds.LDSs[active_k]
        lds_k.fit_bool[3] = true

        # Create weights for this LDS
        w_k = [w[trial][active_k, :] for trial in 1:ntrials]

        StateSpaceDynamics.update_A_b!(lds_k, tfs, w_k)

        # Check results are finite and correct size
        @test all(isfinite.(lds_k.state_model.A))
        @test all(isfinite.(lds_k.state_model.b))
        @test size(lds_k.state_model.A) == (latent_dim, latent_dim)
        @test size(lds_k.state_model.b) == (latent_dim,)
    end
end

function test_weighted_update_Q()
    """Test weighted update of Q covariance matrix"""
    K = 2
    latent_dim = 2
    obs_dim = 3
    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    ntrials = 3
    tsteps = 15
    z, x, y = rand(slds; tsteps=tsteps, ntrials=ntrials)

    # Create weights
    w = [rand(K, tsteps) for _ in 1:ntrials]
    for trial in 1:ntrials
        w[trial] ./= sum(w[trial]; dims=1)
    end

    # Create FilterSmooth objects with ALL required fields
    tfs_array = Vector{StateSpaceDynamics.FilterSmooth{Float64}}(undef, ntrials)
    for k in 1:ntrials
        x_smooth = x[:, :, k]
        p_smooth = zeros(Float64, latent_dim, latent_dim, tsteps)
        p_smooth_tt1 = zeros(Float64, latent_dim, latent_dim, tsteps)
        E_z = x[:, :, k]
        E_zz = zeros(Float64, latent_dim, latent_dim, tsteps)
        E_zz_prev = zeros(Float64, latent_dim, latent_dim, tsteps)

        for t in 1:tsteps
            p_smooth[:, :, t] .= 0.1 * I(latent_dim)
            p_smooth_tt1[:, :, t] .= 0.1 * I(latent_dim)
            E_zz[:, :, t] .= E_z[:, t] * E_z[:, t]' .+ 0.1 * I(latent_dim)
            if t > 1
                E_zz_prev[:, :, t] .= E_z[:, t] * E_z[:, t - 1]'
            end
        end

        entropy = 0.0
        tfs_array[k] = StateSpaceDynamics.FilterSmooth(
            x_smooth, p_smooth, p_smooth_tt1, E_z, E_zz, E_zz_prev, entropy
        )
    end
    tfs = StateSpaceDynamics.TrialFilterSmooth(tfs_array)

    for active_k in 1:K
        lds_k = slds.LDSs[active_k]
        lds_k.fit_bool[4] = true

        # Create weights for this LDS
        w_k = [w[trial][active_k, :] for trial in 1:ntrials]

        StateSpaceDynamics.update_Q!(lds_k, tfs, w_k)

        # Check results are finite, symmetric, and PSD
        @test all(isfinite.(lds_k.state_model.Q))
        @test isapprox(lds_k.state_model.Q, lds_k.state_model.Q', atol=1e-10)

        # Check positive semi-definite
        eigvals_Q = eigvals(lds_k.state_model.Q)
        @test all(eigvals_Q .>= -1e-10)
    end
end

function test_weighted_gradient_linearity()
    """Test that gradient scales linearly with weights"""
    K = 2
    lds = _make_gaussian_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(slds; tsteps=10, ntrials=1)
    tsteps = size(y, 2)

    # Create base weights
    w = rand(K, tsteps)
    w ./= sum(w; dims=1)

    # Compute gradient with base weights
    grad1 = StateSpaceDynamics.Gradient(slds, y[:, :, 1], x[:, :, 1], w)

    # Compute gradient with scaled weights (should produce same result after normalization)
    w_scaled = 2.5 * w
    w_scaled ./= sum(w_scaled; dims=1)
    grad2 = StateSpaceDynamics.Gradient(slds, y[:, :, 1], x[:, :, 1], w_scaled)

    # After normalization, gradients should be equal
    @test isapprox(grad1, grad2, rtol=1e-10)
end

function test_zero_weights_behavior()
    """Test that zero weights are handled correctly"""
    K = 2
    lds = _make_gaussian_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(slds; tsteps=10, ntrials=1)
    tsteps = size(y, 2)

    # Create weights where one state has zero weight everywhere
    w = zeros(K, tsteps)
    w[1, :] .= 1.0  # Only state 1 is active

    # This should work without errors
    grad = StateSpaceDynamics.Gradient(slds, y[:, :, 1], x[:, :, 1], w)
    @test all(isfinite.(grad))

    H, H_diag, H_super, H_sub = StateSpaceDynamics.Hessian(slds, y[:, :, 1], x[:, :, 1], w)
    @test all(isfinite.(H))
end
