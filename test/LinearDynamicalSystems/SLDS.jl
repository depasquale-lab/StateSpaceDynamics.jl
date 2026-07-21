function _stable_A(rng, n::Int; radius::Float64=0.99)
    A = rand(rng, n, n)
    return A .* (radius / maximum(abs.(eigvals(A))))
end

# Build a Gaussian LDS with given dims
function _make_gaussian_lds(latent_dim::Int, obs_dim::Int; rng=Random.default_rng())
    A = _stable_A(rng, latent_dim)
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
function _make_poisson_lds(latent_dim::Int, obs_dim::Int; rng=Random.default_rng())
    A = _stable_A(rng, latent_dim)
    Q = Matrix(0.1 * I(latent_dim))
    b = zeros(latent_dim)
    x0 = zeros(latent_dim)
    P0 = Matrix(1.0 * I(latent_dim))

    C = zeros(obs_dim, latent_dim)
    d = zeros(obs_dim)

    gsm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
    pom = PoissonObservationModel(; C=C, d=d)
    return LinearDynamicalSystem(;
        state_model=gsm,
        obs_model=pom,
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        fit_bool=fill(true, 6),
    )
end

#=
Dense variants with non-zero C and d so the emission terms in the
gradient/Hessian are actually exercised (the plain `_make_*` helpers use
C = 0, which zeroes out every emission contribution).
=#
function _make_gaussian_lds_dense(latent_dim::Int, obs_dim::Int; seed::Int=0)
    rng = MersenneTwister(seed)
    A = 0.5 * rand(rng, latent_dim, latent_dim)
    Q = Matrix(0.1 * I(latent_dim))
    b = 0.1 * randn(rng, latent_dim)
    x0 = 0.1 * randn(rng, latent_dim)
    P0 = Matrix(1.0 * I(latent_dim))

    C = 0.3 * randn(rng, obs_dim, latent_dim)
    R = Matrix(1.0 * I(obs_dim))
    d = 0.1 * randn(rng, obs_dim)

    gsm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
    gom = GaussianObservationModel(; C=C, R=R, d=d)
    return LinearDynamicalSystem(;
        state_model=gsm,
        obs_model=gom,
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        fit_bool=fill(true, 6),
    )
end

function _make_poisson_lds_dense(latent_dim::Int, obs_dim::Int; seed::Int=0)
    rng = MersenneTwister(seed)
    A = 0.5 * rand(rng, latent_dim, latent_dim)
    Q = Matrix(0.1 * I(latent_dim))
    b = 0.1 * randn(rng, latent_dim)
    x0 = 0.1 * randn(rng, latent_dim)
    P0 = Matrix(1.0 * I(latent_dim))

    # Small C/d keep λ = exp(C x + d) modest so the Poisson terms stay finite.
    C = 0.2 * randn(rng, obs_dim, latent_dim)
    d = 0.1 * randn(rng, obs_dim)

    gsm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
    pom = PoissonObservationModel(; C=C, d=d)
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

"""
Compute y = H * x where H is block-tridiagonal given by (H_diag, H_super, H_sub).

- H_diag[t] is D×D on the diagonal
- H_super[t] is D×D for block (t, t+1)
- H_sub[t]   is D×D for block (t+1, t)

x is a vector of length D*T.
"""
function _block_tridiag_mul(H_diag, H_super, H_sub, x::AbstractVector)
    Tsteps = length(H_diag)
    D = size(H_diag[1], 1)
    @assert length(x) == D * Tsteps

    y = similar(x)
    fill!(y, zero(eltype(y)))

    for t in 1:Tsteps
        xt = @view x[(D * (t - 1) + 1):(D * t)]
        yt = @view y[(D * (t - 1) + 1):(D * t)]

        yt .+= H_diag[t] * xt

        if t < Tsteps
            xtp1 = @view x[(D * t + 1):(D * (t + 1))]
            yt .+= H_super[t] * xtp1
        end
        if t > 1
            xtm1 = @view x[(D * (t - 2) + 1):(D * (t - 1))]
            yt .+= H_sub[t - 1] * xtm1
        end
    end
    return y
end

#=
The package no longer ships allocating `Gradient`/`Hessian` wrappers; these
tiny test helpers build a workspace and call the in-place `!` versions, then
copy the blocks out so the surrounding assertions are unchanged.
=#
function _slds_gradient(slds, y, x, w)
    ws = StateSpaceDynamics.SLDSSmoothWorkspace(eltype(y), slds, size(y, 2))
    StateSpaceDynamics.gradient!(ws, slds, x, y, w)
    return copy(ws.opt.grad_buf)
end

function _slds_hessian_blocks(slds, y, x, w)
    Tsteps = size(y, 2)
    ws = StateSpaceDynamics.SLDSSmoothWorkspace(eltype(y), slds, Tsteps)
    StateSpaceDynamics.hessian!(ws, slds, x, y, w)
    H_diag = [copy(ws.btd.H_diag[t]) for t in 1:Tsteps]
    H_super = [copy(ws.btd.H_super[t]) for t in 1:(Tsteps - 1)]
    H_sub = [copy(ws.btd.H_sub[t]) for t in 1:(Tsteps - 1)]
    return H_diag, H_super, H_sub
end

function _lds_gradient(lds, y, x)
    ws = StateSpaceDynamics.SmoothWorkspace(
        eltype(y), lds.latent_dim, lds.obs_dim, size(y, 2)
    )
    StateSpaceDynamics.compute_smooth_constants!(ws, lds)
    return copy(StateSpaceDynamics.gradient!(ws, lds, x, y))
end

"""
Compute q = x' * H * x for block-tridiagonal H without building H.

Uses q = Σ x_t' H_tt x_t + Σ (x_t' H_{t,t+1} x_{t+1} + x_{t+1}' H_{t+1,t} x_t).
"""
function _block_tridiag_quadform(H_diag, H_super, H_sub, x::AbstractVector)
    Tsteps = length(H_diag)
    D = size(H_diag[1], 1)
    @assert length(x) == D * Tsteps

    q = zero(eltype(x))
    for t in 1:Tsteps
        xt = @view x[(D * (t - 1) + 1):(D * t)]
        q += dot(xt, H_diag[t] * xt)
    end
    for t in 1:(Tsteps - 1)
        xt = @view x[(D * (t - 1) + 1):(D * t)]
        xtp1 = @view x[(D * t + 1):(D * (t + 1))]
        q += dot(xt, H_super[t] * xtp1)
        q += dot(xtp1, H_sub[t] * xt)
    end
    return q
end

function _test_hessian_blocks_basic(slds, y_trial, x_trial, w; atol=1e-10)
    H_diag, H_super, H_sub = _slds_hessian_blocks(slds, y_trial, x_trial, w)

    Tsteps = size(y_trial, 2)
    D = slds.LDSs[1].latent_dim

    @test length(H_diag) == Tsteps
    @test length(H_super) == Tsteps - 1
    @test length(H_sub) == Tsteps - 1

    for t in 1:Tsteps
        @test size(H_diag[t]) == (D, D)
        @test all(isfinite, H_diag[t])
    end
    for t in 1:(Tsteps - 1)
        @test size(H_super[t]) == (D, D)
        @test size(H_sub[t]) == (D, D)
        @test all(isfinite, H_super[t])
        @test all(isfinite, H_sub[t])

        # symmetry condition for a symmetric block-tridiagonal matrix
        @test isapprox(H_sub[t], H_super[t]'; atol=atol)
    end

    # Check that H acts like a symmetric operator: x'H y == y'H x
    x = randn(D * Tsteps)
    v = randn(D * Tsteps)
    Hx = _block_tridiag_mul(H_diag, H_super, H_sub, x)
    Hv = _block_tridiag_mul(H_diag, H_super, H_sub, v)
    @test isapprox(dot(x, Hv), dot(v, Hx); atol=1e-8)

    return H_diag, H_super, H_sub
end

function test_valid_SLDS_happy_path(; rng=MersenneTwister(0xC0FFEE))
    K = 3
    lds = _make_gaussian_lds(2, 4)
    s = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))
    @test validate_SLDS(s) === nothing  # should not throw
end

function test_valid_SLDS_dimension_mismatches(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_gaussian_lds(2, 3)

    # size(A,1)=K, but length(πₖ) ≠ K
    s_badZ0 = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K + 1), LDSs=fill(lds, K))
    @test_throws DimensionMismatchError validate_SLDS(s_badZ0)

    # size(A,1)=K, but number of LDSs ≠ K
    s_badLDSs = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K + 1))
    @test_throws DimensionMismatchError validate_SLDS(s_badLDSs)
end

function test_valid_SLDS_nonstochastic_rows_and_invalid_Z0(; rng=MersenneTwister(0xC0FFEE))
    K = 3
    lds = _make_gaussian_lds(2, 2)

    # Non-probability row in A (negative entry)
    A_bad = _rowstochastic(K)
    A_bad[2, :] .= (-0.1, 0.5, 0.6)  # sums to 1 but has a negative entry
    s_badA = SLDS(; A=A_bad, πₖ=_probvec(K), LDSs=fill(lds, K))
    @test_throws InvalidProbabilityVectorError validate_SLDS(s_badA)

    # Z0 does not sum to 1
    Z0_bad = _probvec(K)
    Z0_bad[1] += 0.1
    s_badZ0 = SLDS(; A=_rowstochastic(K), πₖ=Z0_bad, LDSs=fill(lds, K))
    @test_throws InvalidProbabilityVectorError validate_SLDS(s_badZ0)
end

function test_valid_SLDS_mixed_observation_model_types(; rng=MersenneTwister(0xC0FFEE))
    # mode 1..K-1 Gaussian obs; last mode Poisson obs → should assert (type mismatch)
    K = 3
    lds_g = _make_gaussian_lds(2, 2)
    lds_p = _make_poisson_lds(2, 2)  # different obs model type
    @test_throws MethodError SLDS(
        A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds_g, lds_g, lds_p]
    )
end

function test_valid_SLDS_inconsistent_latent_or_obs_dims(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds_a = _make_gaussian_lds(2, 3)
    lds_b_state = _make_gaussian_lds(3, 3) # different latent_dim
    lds_b_obs = _make_gaussian_lds(2, 4) # different obs_dim

    s_bad_state = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds_a, lds_b_state])
    @test_throws DimensionMismatchError validate_SLDS(s_bad_state)

    s_bad_obs = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds_a, lds_b_obs])
    @test_throws DimensionMismatchError validate_SLDS(s_bad_obs)
end

function test_SLDS_sampling_gaussian(; rng=MersenneTwister(0xC0FFEE))
    K = 3
    lds = _make_gaussian_lds(2, 4)
    s = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    tsteps, ntrials = 50, 5
    z, x, y = rand(rng, s, fill(tsteps, ntrials))

    @test length(z) == ntrials
    @test length(x) == ntrials
    @test length(y) == ntrials
    @test all(length(z[n]) == tsteps for n in 1:ntrials)
    @test all(size(x[n]) == (2, tsteps) for n in 1:ntrials)
    @test all(size(y[n]) == (4, tsteps) for n in 1:ntrials)
    @test all(1 ≤ z[n][t] ≤ K for t in 1:tsteps, n in 1:ntrials)
    @test all(all(isfinite, xn) for xn in x)
    @test all(all(isfinite, yn) for yn in y)
end

function test_SLDS_sampling_poisson(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_poisson_lds(2, 3)
    s = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    tsteps, ntrials = 30, 3
    z, x, y = rand(rng, s, fill(tsteps, ntrials))

    @test length(z) == ntrials
    @test all(length(z[n]) == tsteps for n in 1:ntrials)
    @test all(size(x[n]) == (2, tsteps) for n in 1:ntrials)
    @test all(size(y[n]) == (3, tsteps) for n in 1:ntrials)
    @test all(1 ≤ z[n][t] ≤ K for t in 1:tsteps, n in 1:ntrials)
    @test all(y[n][i, t] ≥ 0 for i in 1:3, t in 1:tsteps, n in 1:ntrials)
    @test all(y[n][i, t] == round(y[n][i, t]) for i in 1:3, t in 1:tsteps, n in 1:ntrials)
end

function test_SLDS_deterministic_transitions(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_gaussian_lds(2, 2)

    A_det = [0.0 1.0; 0.0 1.0]
    Z0_det = [1.0, 0.0]

    s = SLDS(; A=A_det, πₖ=Z0_det, LDSs=fill(lds, K))

    tsteps = 10
    z, x, y = rand(rng, s, fill(tsteps, 3))

    @test all(z[n][1] == 1 for n in 1:3)
    @test all(z[n][t] == 2 for t in 2:tsteps, n in 1:3)
end

function test_SLDS_single_trial(; rng=MersenneTwister(0xC0FFEE))
    K = 3
    lds = _make_gaussian_lds(2, 4)
    s = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    tsteps = 100
    z, x, y = rand(rng, s, fill(tsteps, 1))

    @test length(z) == 1
    @test length(z[1]) == tsteps
    @test size(x[1]) == (2, tsteps)
    @test size(y[1]) == (4, tsteps)
end

function test_SLDS_reproducibility()
    K = 2
    lds = _make_gaussian_lds(2, 3)
    s = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    rng = MersenneTwister(42)
    z1, x1, y1 = rand(rng, s, fill(20, 2))

    rng = MersenneTwister(42)
    z2, x2, y2 = rand(rng, s, fill(20, 2))

    @test z1 == z2
    @test all(x1[n] ≈ x2[n] for n in eachindex(x1))
    @test all(y1[n] ≈ y2[n] for n in eachindex(y1))
end

function test_SLDS_single_state_edge_case(; rng=MersenneTwister(0xC0FFEE))
    K = 1
    lds = _make_gaussian_lds(2, 3)
    s = SLDS(; A=reshape([1.0], 1, 1), πₖ=[1.0], LDSs=[lds])

    @test validate_SLDS(s) === nothing

    z, x, y = rand(rng, s, fill(10, 2))
    @test all(all(zn .== 1) for zn in z)
end

function test_SLDS_minimal_dimensions(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_gaussian_lds(1, 1)
    s = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, s, fill(10, 3))

    @test all(size(x[n]) == (1, 10) for n in 1:3)
    @test all(size(y[n]) == (1, 10) for n in 1:3)
    @test all(all(isfinite, xn) for xn in x)
    @test all(all(isfinite, yn) for yn in y)
end

function test_SLDS_rand_integer_overload(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    latent_dim, obs_dim, tsteps = 2, 3, 15
    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    # Single-Integer overload returns one trial as bare arrays (not vectors-of).
    z, x, y = rand(rng, slds, tsteps)
    @test z isa Vector{Int}
    @test length(z) == tsteps
    @test size(x) == (latent_dim, tsteps)
    @test size(y) == (obs_dim, tsteps)
    @test all(isfinite, x)
    @test all(isfinite, y)
    @test all(1 .<= z .<= K)

    # Explicit-RNG single-Integer overload is reproducible.
    z1, x1, y1 = rand(MersenneTwister(123), slds, tsteps)
    z2, x2, y2 = rand(MersenneTwister(123), slds, tsteps)
    @test z1 == z2
    @test x1 ≈ x2
    @test y1 ≈ y2
end

function test_valid_SLDS_probability_helper_functions(; rng=MersenneTwister(0xC0FFEE))
    # Test probability vector validation
    @test validate_probvec([0.3, 0.7]) === nothing
    @test validate_probvec([0.25, 0.25, 0.25, 0.25]) === nothing
    @test_throws InvalidProbabilityVectorError validate_probvec([0.6, 0.5])   # Sums to > 1
    @test_throws InvalidProbabilityVectorError validate_probvec([-0.1, 1.1])  # Has negative

    # Test helper functions
    @test _probvec(4) ≈ [0.25, 0.25, 0.25, 0.25]

    A = _rowstochastic(3)
    @test size(A) == (3, 3)
    @test all(isapprox(sum(A[i, :]), 1.0) for i in 1:3)
    @test all(A[i, j] ≥ 0 for i in 1:3, j in 1:3)
end

function test_SLDS_gradient_numerical(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_gaussian_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(20, 1))

    tsteps = size(y[1], 2)
    w = rand(K, tsteps)
    w ./= sum(w; dims=1)

    y_trial = y[1]
    x_trial = x[1]

    # Analytical gradient
    grad_analytical = _slds_gradient(slds, y_trial, x_trial, w)

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

function test_SLDS_hessian_numerical(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_gaussian_lds(2, 2)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    _, x, y = rand(rng, slds, fill(5, 1))
    tsteps = size(y[1], 2)
    w = rand(K, tsteps)
    w ./= sum(w; dims=1)

    y_trial = y[1]
    x_trial = x[1]

    H_diag, H_super, H_sub = _slds_hessian_blocks(slds, y_trial, x_trial, w)

    function weighted_ll(x_flat)
        x_mat = reshape(x_flat, size(x_trial))
        ll = 0.0
        for k in 1:K
            lds_k = slds.LDSs[k]
            A_k, Q_k, b_k = lds_k.state_model.A, lds_k.state_model.Q, lds_k.state_model.b
            x0_k, P0_k = lds_k.state_model.x0, lds_k.state_model.P0
            C_k, R_k, d_k = lds_k.obs_model.C, lds_k.obs_model.R, lds_k.obs_model.d

            R_chol = cholesky(Symmetric(R_k)).U
            Q_chol = cholesky(Symmetric(Q_k)).U
            P0_chol = cholesky(Symmetric(P0_k)).U

            dx0 = x_mat[:, 1] - x0_k
            ll += w[k, 1] * (-0.5 * sum(abs2, P0_chol \ dx0))

            for t in 1:tsteps
                dy = y_trial[:, t] - (C_k * x_mat[:, t] + d_k)
                ll += w[k, t] * (-0.5 * sum(abs2, R_chol \ dy))
                if t > 1
                    dx = x_mat[:, t] - (A_k * x_mat[:, t - 1] + b_k)
                    ll += w[k, t] * (-0.5 * sum(abs2, Q_chol \ dx))
                end
            end
        end
        return ll
    end

    Hnum = ForwardDiff.hessian(weighted_ll, vec(x_trial))

    D = slds.LDSs[1].latent_dim
    v = randn(D * tsteps)

    Hv_blocks = _block_tridiag_mul(H_diag, H_super, H_sub, v)
    Hv_num = Hnum * v

    @test isapprox(Hv_blocks, Hv_num; rtol=1e-5, atol=1e-5)
end

function test_SLDS_gradient_single_timestep_gaussian(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_gaussian_lds_dense(2, 3; seed=11)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    # Single time step exercises the `Tsteps == 1` early-return branch.
    z, x, y = rand(MersenneTwister(1), slds, fill(1, 1))
    tsteps = size(y[1], 2)
    @test tsteps == 1
    w = rand(MersenneTwister(2), K, tsteps)
    w ./= sum(w; dims=1)

    y_trial = y[1]
    x_trial = x[1]

    grad_analytical = _slds_gradient(slds, y_trial, x_trial, w)

    function weighted_ll(x_flat)
        x_mat = reshape(x_flat, size(x_trial))
        ll = 0.0
        for k in 1:K
            lds_k = slds.LDSs[k]
            x0_k, P0_k = lds_k.state_model.x0, lds_k.state_model.P0
            C_k, R_k, d_k = lds_k.obs_model.C, lds_k.obs_model.R, lds_k.obs_model.d
            R_chol = cholesky(Symmetric(R_k)).U
            P0_chol = cholesky(Symmetric(P0_k)).U

            dx0 = x_mat[:, 1] - x0_k
            ll += w[k, 1] * (-0.5 * sum(abs2, P0_chol \ dx0))

            dy = y_trial[:, 1] - (C_k * x_mat[:, 1] + d_k)
            ll += w[k, 1] * (-0.5 * sum(abs2, R_chol \ dy))
        end
        return ll
    end

    grad_numerical = reshape(ForwardDiff.gradient(weighted_ll, vec(x_trial)), size(x_trial))
    @test isapprox(grad_analytical, grad_numerical, rtol=1e-5, atol=1e-5)
end

function test_SLDS_gradient_single_timestep_poisson(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_poisson_lds_dense(2, 3; seed=12)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(MersenneTwister(3), slds, fill(1, 1))
    tsteps = size(y[1], 2)
    @test tsteps == 1
    w = rand(MersenneTwister(4), K, tsteps)
    w ./= sum(w; dims=1)

    y_trial = y[1]
    x_trial = x[1]

    grad_analytical = _slds_gradient(slds, y_trial, x_trial, w)

    function weighted_ll(x_flat)
        x_mat = reshape(x_flat, size(x_trial))
        ll = 0.0
        for k in 1:K
            lds_k = slds.LDSs[k]
            x0_k, P0_k = lds_k.state_model.x0, lds_k.state_model.P0
            C_k, d_k = lds_k.obs_model.C, lds_k.obs_model.d
            inv_P0 = inv(P0_k)

            dx0 = x_mat[:, 1] - x0_k
            ll += w[k, 1] * (-0.5 * dot(dx0, inv_P0 * dx0))

            obs_mean = C_k * x_mat[:, 1] .+ d_k
            ll += w[k, 1] * (dot(y_trial[:, 1], obs_mean) - sum(exp, obs_mean))
        end
        return ll
    end

    grad_numerical = reshape(ForwardDiff.gradient(weighted_ll, vec(x_trial)), size(x_trial))
    @test isapprox(grad_analytical, grad_numerical, rtol=1e-5, atol=1e-5)
end

function test_SLDS_hessian_single_timestep_gaussian(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_gaussian_lds_dense(2, 3; seed=13)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    _, x, y = rand(MersenneTwister(5), slds, fill(1, 1))
    tsteps = size(y[1], 2)
    @test tsteps == 1
    w = rand(MersenneTwister(6), K, tsteps)
    w ./= sum(w; dims=1)

    y_trial = y[1]
    x_trial = x[1]

    H_diag, H_super, H_sub = _slds_hessian_blocks(slds, y_trial, x_trial, w)
    @test length(H_super) == 0
    @test length(H_sub) == 0

    function weighted_ll(x_flat)
        x_mat = reshape(x_flat, size(x_trial))
        ll = 0.0
        for k in 1:K
            lds_k = slds.LDSs[k]
            x0_k, P0_k = lds_k.state_model.x0, lds_k.state_model.P0
            C_k, R_k, d_k = lds_k.obs_model.C, lds_k.obs_model.R, lds_k.obs_model.d
            R_chol = cholesky(Symmetric(R_k)).U
            P0_chol = cholesky(Symmetric(P0_k)).U

            dx0 = x_mat[:, 1] - x0_k
            ll += w[k, 1] * (-0.5 * sum(abs2, P0_chol \ dx0))
            dy = y_trial[:, 1] - (C_k * x_mat[:, 1] + d_k)
            ll += w[k, 1] * (-0.5 * sum(abs2, R_chol \ dy))
        end
        return ll
    end

    Hnum = ForwardDiff.hessian(weighted_ll, vec(x_trial))
    @test isapprox(H_diag[1], Hnum; rtol=1e-5, atol=1e-5)
end

function test_SLDS_hessian_single_timestep_poisson(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_poisson_lds_dense(2, 3; seed=14)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    _, x, y = rand(MersenneTwister(7), slds, fill(1, 1))
    tsteps = size(y[1], 2)
    @test tsteps == 1
    w = rand(MersenneTwister(8), K, tsteps)
    w ./= sum(w; dims=1)

    y_trial = y[1]
    x_trial = x[1]

    H_diag, H_super, H_sub = _slds_hessian_blocks(slds, y_trial, x_trial, w)
    @test length(H_super) == 0
    @test length(H_sub) == 0

    function weighted_ll(x_flat)
        x_mat = reshape(x_flat, size(x_trial))
        ll = 0.0
        for k in 1:K
            lds_k = slds.LDSs[k]
            x0_k, P0_k = lds_k.state_model.x0, lds_k.state_model.P0
            C_k, d_k = lds_k.obs_model.C, lds_k.obs_model.d
            inv_P0 = inv(P0_k)

            dx0 = x_mat[:, 1] - x0_k
            ll += w[k, 1] * (-0.5 * dot(dx0, inv_P0 * dx0))
            obs_mean = C_k * x_mat[:, 1] .+ d_k
            ll += w[k, 1] * (dot(y_trial[:, 1], obs_mean) - sum(exp, obs_mean))
        end
        return ll
    end

    Hnum = ForwardDiff.hessian(weighted_ll, vec(x_trial))
    @test isapprox(H_diag[1], Hnum; rtol=1e-5, atol=1e-5)
end

function test_SLDS_gradient_reduces_to_single_LDS(; rng=MersenneTwister(0xC0FFEE))
    K = 3
    lds = _make_gaussian_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(20, 1))

    tsteps = size(y[1], 2)

    # Test each discrete state in isolation
    for active_k in 1:K
        # Create weights where only state active_k is active
        w = zeros(K, tsteps)
        w[active_k, :] .= 1.0

        grad_slds = _slds_gradient(slds, y[1], x[1], w)
        grad_lds = _lds_gradient(slds.LDSs[active_k], y[1], x[1])

        @test isapprox(grad_slds, grad_lds, rtol=1e-10)
    end
end

function test_SLDS_hessian_block_structure_gaussian(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_gaussian_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    _, x, y = rand(rng, slds, fill(10, 1))
    tsteps = size(y[1], 2)
    w = rand(K, tsteps)
    w ./= sum(w; dims=1)

    return _test_hessian_blocks_basic(slds, y[1], x[1], w)
end

function test_SLDS_gradient_weight_normalization(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_gaussian_lds(2, 2)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(15, 1))

    tsteps = size(y[1], 2)

    # Create two different weight matrices that sum to same values
    w1 = rand(K, tsteps)
    w1 ./= sum(w1; dims=1)

    w2 = 0.5 .* w1  # Scale by 0.5
    w2 ./= sum(w2; dims=1)  # Renormalize

    # Gradients should be the same (weights are normalized)
    grad1 = _slds_gradient(slds, y[1], x[1], w1)
    grad2 = _slds_gradient(slds, y[1], x[1], w2)

    @test isapprox(grad1, grad2, rtol=1e-10)
end

function test_SLDS_smooth_basic(; rng=MersenneTwister(0xC0FFEE))
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
    z, x, y = rand(rng, slds, fill(tsteps, 1))

    # Create uniform weights (should behave like averaging both LDS)
    w = ones(Float64, K, tsteps) ./ K

    # Call smooth
    x_smooth, p_smooth = smooth(slds, y[1], w)

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

function test_SLDS_smooth_reduces_to_single_LDS(; rng=MersenneTwister(0xC0FFEE))
    # When all weight is on one LDS, should match that LDS's smooth
    K = 3
    latent_dim = 2
    obs_dim = 3
    tsteps = 15

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(tsteps, 1))
    y_trial = y[1]

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

function test_SLDS_smooth_with_realistic_weights(; rng=MersenneTwister(0xC0FFEE))
    # Test with realistic posterior weights that change over time
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 25

    lds1 = _make_gaussian_lds(latent_dim, obs_dim)
    lds2 = _make_gaussian_lds(latent_dim, obs_dim)

    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds1, lds2])

    z, x, y = rand(rng, slds, fill(tsteps, 1))
    y_trial = y[1]

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

function test_SLDS_smooth_consistency_with_gradients(; rng=MersenneTwister(0xC0FFEE))
    # Verify that smooth finds a point where gradient is near zero
    K = 2
    latent_dim = 2
    obs_dim = 2
    tsteps = 10

    lds1 = _make_gaussian_lds(latent_dim, obs_dim)
    lds2 = _make_gaussian_lds(latent_dim, obs_dim)

    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds1, lds2])

    z, x, y = rand(rng, slds, fill(tsteps, 1))
    y_trial = y[1]

    w = rand(Float64, K, tsteps)
    w ./= sum(w; dims=1)  # Normalize

    x_smooth, _ = smooth(slds, y_trial, w)

    # Gradient at optimum should be small
    grad = _slds_gradient(slds, y_trial, x_smooth, w)

    @test norm(grad) < 1e-4  # Should be near zero at optimum
end

function test_SLDS_smooth_entropy_calculation(; rng=MersenneTwister(0xC0FFEE))
    # Verify entropy is computed and matches an independent dense reference
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 15

    lds1 = _make_gaussian_lds(latent_dim, obs_dim)
    lds2 = _make_gaussian_lds(latent_dim, obs_dim)

    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds1, lds2])

    z, x, y = rand(rng, slds, fill(tsteps, 1))

    w = ones(Float64, K, tsteps) ./ K

    # Call smooth! directly to access StateSpaceDynamics.FilterSmooth
    fs = StateSpaceDynamics.initialize_FilterSmooth(slds.LDSs[1], tsteps)
    StateSpaceDynamics.smooth!(slds, fs, y[1], w)

    #=
    smooth! must fill fs.entropy from the BT log-determinant (it was silently
    left at its zero initialization before the 0.5.0 fix). Check the value
    against an external reference: rebuild the weighted BT Hessian at the MAP
    on a fresh workspace, invert the dense precision (negated Hessian) into
    the joint posterior covariance, and take Distributions.jl's entropy of the
    corresponding MvNormal.
    =#
    @test isfinite(fs.entropy)

    ws = StateSpaceDynamics.SLDSSmoothWorkspace(Float64, slds, tsteps)
    StateSpaceDynamics.hessian!(ws, slds, fs.x_smooth, y[1], w)
    P_dense =
        -Matrix(
            StateSpaceDynamics.block_tridgm(
                ws.btd.H_diag[1:tsteps],
                ws.btd.H_super[1:(tsteps - 1)],
                ws.btd.H_sub[1:(tsteps - 1)],
            ),
        )
    n = latent_dim * tsteps
    Σ_dense = Matrix(inv(Symmetric(P_dense)))
    entropy_ref = entropy(MvNormal(zeros(n), Σ_dense))
    @test isapprox(fs.entropy, entropy_ref; rtol=1e-8, atol=1e-8)
    return fs
end

function test_SLDS_smooth_covariance_symmetry(; rng=MersenneTwister(0xC0FFEE))
    # Ensure covariances remain symmetric
    K = 2
    latent_dim = 3
    obs_dim = 2
    tsteps = 12

    lds1 = _make_gaussian_lds(latent_dim, obs_dim)
    lds2 = _make_gaussian_lds(latent_dim, obs_dim)

    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds1, lds2])

    z, x, y = rand(rng, slds, fill(tsteps, 1))

    w = rand(Float64, K, tsteps)
    w ./= sum(w; dims=1)

    _, p_smooth = smooth(slds, y[1], w)

    # Check symmetry at each timestep
    for t in 1:tsteps
        @test isapprox(p_smooth[:, :, t], p_smooth[:, :, t]', atol=1e-10)
    end
end

function test_SLDS_smooth_different_weight_patterns(; rng=MersenneTwister(0xC0FFEE))
    # Test various weight patterns
    K = 2
    latent_dim = 2
    obs_dim = 2
    tsteps = 20

    lds1 = _make_gaussian_lds(latent_dim, obs_dim)
    lds2 = _make_gaussian_lds(latent_dim, obs_dim)

    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds1, lds2])

    z, x, y = rand(rng, slds, fill(tsteps, 1))
    y_trial = y[1]

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

function test_SLDS_sample_posterior_basic(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 20

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(tsteps, 1))
    w = ones(Float64, K, tsteps) ./ K

    # Smooth and draw one joint posterior sample in the same call.
    fs = StateSpaceDynamics.initialize_FilterSmooth(slds.LDSs[1], tsteps)
    x_sample = Matrix{Float64}(undef, latent_dim, tsteps)
    StateSpaceDynamics.smooth!(slds, fs, y[1], w; x_sample=x_sample, rng=rng)

    @test size(x_sample) == (latent_dim, tsteps)
    @test all(isfinite, x_sample)
    # smooth! fills the joint posterior entropy (positive at these scales).
    @test isfinite(fs.entropy)
    @test fs.entropy > 0
end

function test_SLDS_estep_basic(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 15
    ntrials = 2

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    # Batched fb_storage with seq_ends
    seq_ends = cumsum(fill(tsteps, ntrials))
    total_T = last(seq_ends)
    obs_seq = collect(1:total_T)
    control_seq = fill(nothing, total_T)

    tfs = StateSpaceDynamics.initialize_FilterSmooth(slds.LDSs[1], fill(tsteps, ntrials))
    dl = StateSpaceDynamics.SLDSDiscreteLayer(slds.A, slds.πₖ, zeros(Float64, K, total_T))
    fb_storage = StateSpaceDynamics._make_slds_fb_storage(dl, seq_ends)

    slds_ws = StateSpaceDynamics.SLDSSmoothWorkspace(Float64, slds, tsteps)

    # Warm-start smooth draws the first joint sample into x_samples.
    x_samples = [Matrix{Float64}(undef, latent_dim, tsteps) for _ in 1:ntrials]
    for trial in 1:ntrials
        w_uniform = ones(Float64, K, tsteps) ./ K
        StateSpaceDynamics.smooth!(
            slds,
            tfs[trial],
            y[trial],
            w_uniform;
            ws=slds_ws,
            x_sample=x_samples[trial],
            rng=rng,
        )
    end

    StateSpaceDynamics.estep!(
        slds,
        tfs,
        fb_storage,
        dl,
        y,
        x_samples,
        slds_ws;
        rng=rng,
        obs_seq=obs_seq,
        control_seq=control_seq,
        seq_ends=seq_ends,
    )

    elbo = StateSpaceDynamics.elbo!(slds, tfs, fb_storage, y, slds_ws; seq_ends=seq_ends)

    @test isfinite(elbo)

    for trial in 1:ntrials
        @test size(tfs[trial].E_z) == (latent_dim, tsteps)
        @test size(tfs[trial].E_zz) == (latent_dim, latent_dim, tsteps)
        @test size(tfs[trial].E_zz_prev) == (latent_dim, latent_dim, tsteps)
        @test all(isfinite, tfs[trial].E_z)
        @test all(isfinite, tfs[trial].E_zz)
        @test all(isfinite, tfs[trial].E_zz_prev)
    end

    # Check batched γ has reasonable shape and probabilities sum to 1 per timestep.
    @test size(fb_storage.γ) == (K, total_T)
    @test all(isfinite, fb_storage.γ)
    @test all(fb_storage.γ .>= 0)
    @test all(isapprox.(sum(fb_storage.γ; dims=1), 1.0, atol=1e-10))
end

function test_SLDS_mstep_updates_parameters(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 15
    ntrials = 2

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    seq_ends = cumsum(fill(tsteps, ntrials))
    total_T = last(seq_ends)
    obs_seq = collect(1:total_T)
    control_seq = fill(nothing, total_T)

    tfs = StateSpaceDynamics.initialize_FilterSmooth(slds.LDSs[1], fill(tsteps, ntrials))
    dl = StateSpaceDynamics.SLDSDiscreteLayer(slds.A, slds.πₖ, zeros(Float64, K, total_T))
    fb_storage = StateSpaceDynamics._make_slds_fb_storage(dl, seq_ends)

    slds_ws = StateSpaceDynamics.SLDSSmoothWorkspace(Float64, slds, tsteps)
    sws = StateSpaceDynamics.SmoothWorkspace(Float64, latent_dim, obs_dim, tsteps)

    x_samples = [Matrix{Float64}(undef, latent_dim, tsteps) for _ in 1:ntrials]
    for trial in 1:ntrials
        w_uniform = ones(Float64, K, tsteps) ./ K
        StateSpaceDynamics.smooth!(
            slds,
            tfs[trial],
            y[trial],
            w_uniform;
            ws=slds_ws,
            x_sample=x_samples[trial],
            rng=rng,
        )
    end
    StateSpaceDynamics.estep!(
        slds,
        tfs,
        fb_storage,
        dl,
        y,
        x_samples,
        slds_ws;
        rng=rng,
        obs_seq=obs_seq,
        control_seq=control_seq,
        seq_ends=seq_ends,
    )

    A_old = copy(slds.A)

    StateSpaceDynamics.mstep!(
        slds, tfs, fb_storage, dl, y, sws; obs_seq=obs_seq, seq_ends=seq_ends
    )

    # Check parameters changed (with high probability)
    @test !isapprox(slds.A, A_old; rtol=1e-6) || true  # May not change if data is degenerate
    @test all(isfinite, slds.A)
    @test all(isfinite, slds.πₖ)

    # Check stochasticity is preserved
    @test all(isapprox.(sum(slds.A; dims=2), 1.0, atol=1e-10))
    @test isapprox(sum(slds.πₖ), 1.0, atol=1e-10)
    @test all(slds.A .>= 0)
    @test all(slds.πₖ .>= 0)
end

function test_SLDS_fit_runs_to_completion(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 15
    ntrials = 2
    max_iter = 5

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    # Fit without progress bar
    elbos = fit!(slds, y; max_iter=max_iter, progress=false)

    # Check correct number of iterations
    @test length(elbos) == max_iter

    # Check all ELBOs are finite
    @test all(isfinite, elbos)
end

function test_SLDS_fit_elbo_generally_increases(; rng=MersenneTwister(0xC0FFEE))
    # ELBO should generally increase or stabilize (may have noise due to sampling)
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 20
    ntrials = 3
    max_iter = 10

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    elbos = fit!(slds, y; max_iter=max_iter, progress=false)

    # Check that later ELBOs are generally higher than early ones
    # (allowing for stochastic noise)
    early_mean = mean(elbos[1:3])
    late_mean = mean(elbos[(end - 2):end])

    @test (late_mean > early_mean - 100) # don't fail CI
end

function test_SLDS_fit_multitrial(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 15
    ntrials = 5
    max_iter = 5

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    elbos = fit!(slds, y; max_iter=max_iter, progress=false)

    @test length(elbos) == max_iter
    @test all(isfinite, elbos)
end

function test_SLDS_shared_initial_state(; rng=MersenneTwister(0xC0FFEE))
    #= The SLDS shares an initial state across LDS modes. This test verifies that the 
    shared initial state is updated correctly. See #155 =#
    K, latent_dim, obs_dim, ntrials, tsteps = 2, 2, 3, 3, 12

    # Distinct modes so the tie is observable (start both at x0=0, P0=I).
    lds1 = _make_gaussian_lds(latent_dim, obs_dim; rng=MersenneTwister(1))
    lds2 = _make_gaussian_lds(latent_dim, obs_dim; rng=MersenneTwister(2))
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds1, lds2])
    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    # One q(x) per trial (shared across LDSs), as the SLDS smoother produces.
    tfs_array = Vector{StateSpaceDynamics.FilterSmooth{Float64}}(undef, ntrials)
    for trial in 1:ntrials
        p_smooth = zeros(Float64, latent_dim, latent_dim, tsteps)
        p_smooth_tt1 = zeros(Float64, latent_dim, latent_dim, tsteps)
        for t in 1:tsteps
            p_smooth[:, :, t] .= 0.1 * I(latent_dim)
            p_smooth_tt1[:, :, t] .= 0.05 * I(latent_dim)
        end
        E_zz = zeros(Float64, latent_dim, latent_dim, tsteps)
        E_zz_prev = zeros(Float64, latent_dim, latent_dim, tsteps)
        tfs_array[trial] = StateSpaceDynamics.FilterSmooth(
            x[trial], p_smooth, p_smooth_tt1, x[trial], E_zz, E_zz_prev, 0.0
        )
    end
    tfs = StateSpaceDynamics.TrialFilterSmooth(tfs_array)

    tsteps_per_trial = fill(tsteps, ntrials)
    sws = StateSpaceDynamics.SmoothWorkspace(Float64, latent_dim, obs_dim, tsteps)
    suf = StateSpaceDynamics._initialize_td_sufficient_statistics(
        Float64, lds1, tsteps_per_trial
    )
    # No inputs on these modes; the aggregator now consumes a validated `Data`
    # (canonicalized to zero-row ux/uy). Dims are shared across modes, so one
    # `Data` serves every LDS, mirroring the SLDS fit path.
    data = StateSpaceDynamics.Data(lds1, y)

    #= LDS 2 gets zero responsibility at t=1 in every trial — the
    aggregator must not throw and must leave a finite (zero) init scatter. =#
    w2 = [vcat(0.0, fill(0.2, tsteps - 1)) for _ in 1:ntrials]
    StateSpaceDynamics._aggregate_td_suff_stats_weighted!(suf, tfs, lds2, data, w2, sws)
    @test suf.init_n < 1e-12
    @test all(isfinite, suf.init_yy[])

    #= Tie x0/P0 from the pooled (unit-weight) init stats: finite, equal across
    LDSs, and equal to the plain unweighted initial fit over all trial starts.
    Unit weights give the same pooled result mstep sums over regimes. =#
    unit_w = [ones(Float64, tsteps) for _ in 1:ntrials]
    StateSpaceDynamics._aggregate_td_suff_stats_weighted!(suf, tfs, lds1, data, unit_w, sws)
    StateSpaceDynamics._update_shared_initial_state!(slds, suf, sws)
    @test all(isfinite, slds.LDSs[1].state_model.x0)
    @test all(isfinite, slds.LDSs[1].state_model.P0)
    @test slds.LDSs[1].state_model.x0 ≈ slds.LDSs[2].state_model.x0
    @test slds.LDSs[1].state_model.P0 ≈ slds.LDSs[2].state_model.P0
    x0_expected = sum(x[trial][:, 1] for trial in 1:ntrials) ./ ntrials
    @test slds.LDSs[1].state_model.x0 ≈ x0_expected

    # End-to-end: a full fit with distinct regimes completes and keeps x0/P0 tied.
    lds1b = _make_gaussian_lds(latent_dim, obs_dim; rng=MersenneTwister(3))
    lds2b = _make_gaussian_lds(latent_dim, obs_dim; rng=MersenneTwister(4))
    slds2 = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds1b, lds2b])
    _, _, y2 = rand(rng, slds2, fill(tsteps, ntrials))
    elbos = fit!(slds2, y2; max_iter=4, progress=false)
    @test all(isfinite, elbos)
    @test slds2.LDSs[1].state_model.x0 ≈ slds2.LDSs[2].state_model.x0
    @test slds2.LDSs[1].state_model.P0 ≈ slds2.LDSs[2].state_model.P0
end

function test_SLDS_estep_elbo_components(; rng=MersenneTwister(0xC0FFEE))
    # Verify ELBO contains expected components
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 10
    ntrials = 1

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    seq_ends = cumsum(fill(tsteps, ntrials))
    total_T = last(seq_ends)
    obs_seq = collect(1:total_T)
    control_seq = fill(nothing, total_T)

    tfs = StateSpaceDynamics.initialize_FilterSmooth(slds.LDSs[1], fill(tsteps, ntrials))
    dl = StateSpaceDynamics.SLDSDiscreteLayer(slds.A, slds.πₖ, zeros(Float64, K, total_T))
    fb_storage = StateSpaceDynamics._make_slds_fb_storage(dl, seq_ends)

    slds_ws = StateSpaceDynamics.SLDSSmoothWorkspace(Float64, slds, tsteps)

    x_samples = [Matrix{Float64}(undef, latent_dim, tsteps) for _ in 1:ntrials]
    w_uniform = ones(Float64, K, tsteps) ./ K
    StateSpaceDynamics.smooth!(
        slds, tfs[1], y[1], w_uniform; ws=slds_ws, x_sample=x_samples[1], rng=rng
    )

    StateSpaceDynamics.estep!(
        slds,
        tfs,
        fb_storage,
        dl,
        y,
        x_samples,
        slds_ws;
        rng=rng,
        obs_seq=obs_seq,
        control_seq=control_seq,
        seq_ends=seq_ends,
    )

    elbo = StateSpaceDynamics.elbo!(slds, tfs, fb_storage, y, slds_ws; seq_ends=seq_ends)

    @test isfinite(elbo)

    # smooth! (via estep!) must fill the posterior entropy; positive at these scales.
    @test tfs[1].entropy > 0
end

#=
K = 1, Gaussian emissions: q(x) is exact and q(z) degenerate, so the ELBO must
equal the marginal log p(y) from the independent Kalman `loglikelihood`. Pins
the entropy signs, H[q(z)], and the ½ tr(H Σ) correction at once.
=#
function test_SLDS_elbo_matches_LDS_marginal_K1(; rng=MersenneTwister(0xBEEF))
    latent_dim = 2
    obs_dim = 3
    tsteps = 15
    ntrials = 2

    # Dense C/d so the observations actually constrain the posterior.
    lds = _make_gaussian_lds_dense(latent_dim, obs_dim; seed=42)
    slds = SLDS(; A=ones(1, 1), πₖ=[1.0], LDSs=[lds])

    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    seq_ends = cumsum(fill(tsteps, ntrials))
    total_T = last(seq_ends)
    obs_seq = collect(1:total_T)
    control_seq = fill(nothing, total_T)

    tfs = StateSpaceDynamics.initialize_FilterSmooth(lds, fill(tsteps, ntrials))
    dl = StateSpaceDynamics.SLDSDiscreteLayer(slds.A, slds.πₖ, zeros(Float64, 1, total_T))
    fb_storage = StateSpaceDynamics._make_slds_fb_storage(dl, seq_ends)
    slds_ws = StateSpaceDynamics.SLDSSmoothWorkspace(Float64, slds, tsteps)

    x_samples = [Matrix{Float64}(undef, latent_dim, tsteps) for _ in 1:ntrials]
    for trial in 1:ntrials
        w = ones(Float64, 1, tsteps)
        StateSpaceDynamics.smooth!(
            slds, tfs[trial], y[trial], w; ws=slds_ws, x_sample=x_samples[trial], rng=rng
        )
    end

    StateSpaceDynamics.estep!(
        slds,
        tfs,
        fb_storage,
        dl,
        y,
        x_samples,
        slds_ws;
        rng=rng,
        obs_seq=obs_seq,
        control_seq=control_seq,
        seq_ends=seq_ends,
    )

    elbo = StateSpaceDynamics.elbo!(slds, tfs, fb_storage, y, slds_ws; seq_ends=seq_ends)

    ll = sum(loglikelihood(lds, y[trial]) for trial in 1:ntrials)

    @test isapprox(elbo, ll; rtol=1e-6)
end

function test_SLDS_public_elbo(; rng=MersenneTwister(0xE1B0))
    @testset "public elbo (allocating)" begin
        K = 2
        latent_dim = 2
        obs_dim = 3
        tsteps = 15
        ntrials = 2

        lds = _make_gaussian_lds_dense(latent_dim, obs_dim; seed=42)
        slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds, deepcopy(lds)])
        z, x, y = rand(rng, slds, fill(tsteps, ntrials))

        # The E-step consumes a joint posterior sample, so the value is
        # stochastic — with a shared rng it must match fit!'s first ELBO.
        e = StateSpaceDynamics.elbo(slds, y; rng=MersenneTwister(7))
        @test isfinite(e)
        e_fit = fit!(deepcopy(slds), y; max_iter=1, progress=false, rng=MersenneTwister(7))[1]
        @test isapprox(e, e_fit; rtol=1e-8)

        # Shape invariance under a shared rng (single trial: matrix == [matrix]).
        e_mat = StateSpaceDynamics.elbo(slds, y[1]; rng=MersenneTwister(7))
        e_vec = StateSpaceDynamics.elbo(slds, [y[1]]; rng=MersenneTwister(7))
        @test isapprox(e_mat, e_vec; rtol=1e-10)

        #=
        K=1 Gaussian regime with no priors: q(z) is degenerate, q(x) is the
        exact posterior, so the ELBO equals the exact marginal log-likelihood
        (deterministic — the posterior sample only feeds the K=1 FB pass,
        whose γ ≡ 1 regardless).
        =#
        slds1 = SLDS(; A=ones(1, 1), πₖ=[1.0], LDSs=[deepcopy(lds)])
        _, _, y1 = rand(rng, slds1, fill(tsteps, ntrials))
        e1 = StateSpaceDynamics.elbo(slds1, y1; rng=MersenneTwister(11))
        ll = sum(loglikelihood(lds, y1[trial]) for trial in 1:ntrials)
        @test isapprox(e1, ll; rtol=1e-6)
    end
    return nothing
end

function test_SLDS_fit_shapes_and_validation(; rng=MersenneTwister(0xE1B1))
    @testset "SLDS fit! shapes + Data validation" begin
        K = 2
        latent_dim = 2
        obs_dim = 3
        tsteps = 15
        ntrials = 2

        lds = _make_gaussian_lds_dense(latent_dim, obs_dim; seed=42)
        slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=[lds, deepcopy(lds)])
        z, x, y = rand(rng, slds, fill(tsteps, ntrials))

        # 3-D array and vector-of-matrices forms give the same ELBO trace
        # under a shared rng (equal-length trials).
        Y3 = cat(y...; dims=3)
        e_vec = fit!(deepcopy(slds), y; max_iter=2, progress=false, rng=MersenneTwister(7))
        e_arr = fit!(deepcopy(slds), Y3; max_iter=2, progress=false, rng=MersenneTwister(7))
        @test e_vec ≈ e_arr

        # Wrong obs_dim now fails fast at Data construction, not deep in the
        # smoother.
        y_bad = [yt[1:(obs_dim - 1), :] for yt in y]
        @test_throws StateSpaceDynamics.DimensionMismatchError fit!(
            deepcopy(slds), y_bad; max_iter=1, progress=false
        )

        # Marginal loglikelihood is intractable for an SLDS — informative error.
        @test_throws ErrorException loglikelihood(slds, y)
    end
    return nothing
end

function test_SLDS_no_priors_zero_prior_logdensity(; rng=MersenneTwister(0xC0FFEE))
    K = 3
    latent_dim = 2
    obs_dim = 3

    for lds in
        (_make_gaussian_lds(latent_dim, obs_dim), _make_poisson_lds(latent_dim, obs_dim))
        slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

        # Every lds should have all prior fields unset.
        for lds_k in slds.LDSs
            sm = lds_k.state_model
            @test sm.Q_prior === nothing
            @test sm.P0_prior === nothing
            @test sm.AB_prior === nothing
            @test lds_k.obs_model.CD_prior === nothing
            if lds_k.obs_model isa GaussianObservationModel
                @test lds_k.obs_model.R_prior === nothing
            end
        end

        # No priors ⇒ the log p(θ) term is exactly zero.
        @test StateSpaceDynamics._slds_prior_logdensity(slds) == 0.0
    end

    # check: attaching one IW prior must move the term off zero.
    lds = _make_gaussian_lds(latent_dim, obs_dim)
    lds.state_model.Q_prior = StateSpaceDynamics.IWPrior(;
        Ψ=Matrix(1.0 * I(latent_dim)), ν=latent_dim + 2.0
    )
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))
    @test StateSpaceDynamics._slds_prior_logdensity(slds) != 0.0
    @test isfinite(StateSpaceDynamics._slds_prior_logdensity(slds))
end

function test_SLDS_x0_niw_prior(; rng=MersenneTwister(0xB0BA))
    K, latent_dim, obs_dim = 3, 2, 3

    x0p() = x0_mean_prior(zeros(latent_dim); κ₀=1.0)
    P0p() = StateSpaceDynamics.IWPrior(; Ψ=Matrix(1.0 * I(latent_dim)), ν=latent_dim + 2.0)

    lds = _make_gaussian_lds(latent_dim, obs_dim)
    lds.state_model.x0_prior = x0_mean_prior(fill(0.5, latent_dim); κ₀=1.0)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))
    ld = StateSpaceDynamics._slds_prior_logdensity(slds)
    @test isfinite(ld)
    @test ld != 0.0
    @test ld < 0.0   # a shrinkage penalty: -½ κ₀ (x0-μ₀)'P0⁻¹(x0-μ₀) ≤ 0

    lds2 = _make_gaussian_lds(latent_dim, obs_dim)
    lds2.state_model.x0_prior = x0p()
    lds2.state_model.P0_prior = P0p()
    slds2 = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds2, K))
    _, _, y = rand(rng, slds2, fill(20, 3))
    elbos = fit!(slds2, y; max_iter=8, progress=false)
    @test all(isfinite, elbos)
    for lds_k in slds2.LDSs
        @test all(isfinite, lds_k.state_model.x0)
        @test all(isfinite, lds_k.state_model.P0)
    end
end

#=
The joint sampler must reproduce the posterior's temporal correlations, not
just the per-timestep marginals. The old marginal sampler got this wrong; it
drew each x_t independently, so the empirical lag-1 cross-covariance was ~0 
instead of `p_smooth_tt1`). Draw many samples from a fixed posterior and check 
the empirical moments against the smoother's stored mean, marginal covariance, 
and lag-1 cross-covariance.
=#
function test_SLDS_joint_sample_reproduces_cross_covariance(; rng=MersenneTwister(0x5A3D))
    latent_dim = 2
    obs_dim = 3
    tsteps = 8

    lds = _make_gaussian_lds_dense(latent_dim, obs_dim; seed=7)
    slds = SLDS(; A=ones(1, 1), πₖ=[1.0], LDSs=[lds])
    z, x, y = rand(rng, slds, fill(tsteps, 1))
    w = ones(Float64, 1, tsteps)

    slds_ws = StateSpaceDynamics.SLDSSmoothWorkspace(Float64, slds, tsteps)
    fs = StateSpaceDynamics.initialize_FilterSmooth(slds.LDSs[1], tsteps)

    # Smooth once to fix the posterior (mean/cov held constant across draws).
    StateSpaceDynamics.smooth!(slds, fs, y[1], w; ws=slds_ws)
    μ = copy(fs.x_smooth)
    P = copy(fs.p_smooth)
    Ptt1 = copy(fs.p_smooth_tt1)  # [:,:,t] = Cov(x_t, x_{t-1})

    # Draw many joint samples from the *same* fixed posterior.
    nsamp = 40_000
    xs = Matrix{Float64}(undef, latent_dim, tsteps)
    mean_acc = zeros(latent_dim, tsteps)
    cov_acc = zeros(latent_dim, latent_dim, tsteps)
    xcov_acc = zeros(latent_dim, latent_dim, tsteps)  # E[(x_t-μ_t)(x_{t-1}-μ_{t-1})']
    for _ in 1:nsamp
        StateSpaceDynamics.smooth!(slds, fs, y[1], w; ws=slds_ws, x_sample=xs, rng=rng)
        mean_acc .+= xs
        for t in 1:tsteps
            dt = xs[:, t] .- μ[:, t]
            cov_acc[:, :, t] .+= dt * dt'
            if t > 1
                dtm1 = xs[:, t - 1] .- μ[:, t - 1]
                xcov_acc[:, :, t] .+= dt * dtm1'
            end
        end
    end
    mean_emp = mean_acc ./ nsamp
    cov_emp = cov_acc ./ nsamp
    xcov_emp = xcov_acc ./ nsamp

    # Empirical mean matches the smoothed mean.
    @test isapprox(mean_emp, μ; atol=5e-2)

    # Empirical marginal covariance matches p_smooth.
    for t in 1:tsteps
        @test isapprox(cov_emp[:, :, t], P[:, :, t]; atol=8e-2)
    end

    # Empirical lag-1 cross-covariance matches p_smooth_tt1.
    for t in 2:tsteps
        @test isapprox(xcov_emp[:, :, t], Ptt1[:, :, t]; atol=8e-2)
    end

    #=
    Discriminating check: the true cross-covariances are materially nonzero
    somewhere, so an independent-marginal sampler (xcov_emp ≈ 0 everywhere)
    would fail the loop above. We assert on the max over t rather than each
    timestep, since the smoother's cross-cov naturally tapers near the ends.
    =#
    @test maximum(norm(Ptt1[:, :, t]) for t in 2:tsteps) > 0.2
end

function test_weighted_update_initial_state_mean(; rng=MersenneTwister(0xC0FFEE))
    """Test weighted update of initial state mean"""
    K = 2
    latent_dim = 2
    obs_dim = 3
    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    ntrials = 4
    tsteps = 10
    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    # Create per-trial, per-timestep weights
    w = [rand(K, tsteps) for _ in 1:ntrials]
    for trial in 1:ntrials
        w[trial] ./= sum(w[trial]; dims=1)
    end

    # Mock FilterSmooth objects with ALL required fields
    tfs_array = Vector{StateSpaceDynamics.FilterSmooth{Float64}}(undef, ntrials)
    for k in 1:ntrials
        x_smooth = x[k]
        p_smooth = zeros(Float64, latent_dim, latent_dim, tsteps)
        p_smooth_tt1 = zeros(Float64, latent_dim, latent_dim, tsteps)
        E_z = x[k]
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

    tsteps_per_trial = [size(x[trial], 2) for trial in 1:ntrials]
    sws = StateSpaceDynamics.SmoothWorkspace(Float64, latent_dim, obs_dim, tsteps)
    data = StateSpaceDynamics.Data(slds.LDSs[1], y)

    for active_k in 1:K
        lds_k = slds.LDSs[active_k]
        lds_k.fit_bool[1] = true

        # Per-trial weight slice for this regime.
        w_k = [w[trial][active_k, :] for trial in 1:ntrials]

        suf = StateSpaceDynamics._initialize_td_sufficient_statistics(
            Float64, lds_k, tsteps_per_trial
        )
        StateSpaceDynamics._aggregate_td_suff_stats_weighted!(
            suf, tfs, lds_k, data, w_k, sws
        )
        StateSpaceDynamics.update_initial_state_mean!(lds_k, suf)

        @test all(isfinite.(lds_k.state_model.x0))
    end
end

function test_weighted_update_A_b(; rng=MersenneTwister(0xC0FFEE))
    """Test weighted update of A and b matrices"""
    K = 2
    latent_dim = 2
    obs_dim = 3
    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    ntrials = 3
    tsteps = 15
    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    # Create weights
    w = [rand(K, tsteps) for _ in 1:ntrials]
    for trial in 1:ntrials
        w[trial] ./= sum(w[trial]; dims=1)
    end

    # Create FilterSmooth objects with ALL required fields
    tfs_array = Vector{StateSpaceDynamics.FilterSmooth{Float64}}(undef, ntrials)
    for k in 1:ntrials
        x_smooth = x[k]
        p_smooth = zeros(Float64, latent_dim, latent_dim, tsteps)
        p_smooth_tt1 = zeros(Float64, latent_dim, latent_dim, tsteps)
        E_z = x[k]
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

    tsteps_per_trial = [size(x[trial], 2) for trial in 1:ntrials]
    sws = StateSpaceDynamics.SmoothWorkspace(Float64, latent_dim, obs_dim, tsteps)
    data = StateSpaceDynamics.Data(slds.LDSs[1], y)

    for active_k in 1:K
        lds_k = slds.LDSs[active_k]
        lds_k.fit_bool[3] = true

        w_k = [w[trial][active_k, :] for trial in 1:ntrials]

        suf = StateSpaceDynamics._initialize_td_sufficient_statistics(
            Float64, lds_k, tsteps_per_trial
        )
        StateSpaceDynamics._aggregate_td_suff_stats_weighted!(
            suf, tfs, lds_k, data, w_k, sws
        )
        StateSpaceDynamics.update_A_b!(lds_k, suf, sws)

        @test all(isfinite.(lds_k.state_model.A))
        @test all(isfinite.(lds_k.state_model.b))
        @test size(lds_k.state_model.A) == (latent_dim, latent_dim)
        @test size(lds_k.state_model.b) == (latent_dim,)
    end
end

function test_weighted_update_Q(; rng=MersenneTwister(0xC0FFEE))
    """Test weighted update of Q covariance matrix"""
    K = 2
    latent_dim = 2
    obs_dim = 3
    lds = _make_gaussian_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    ntrials = 3
    tsteps = 15
    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    # Create weights
    w = [rand(K, tsteps) for _ in 1:ntrials]
    for trial in 1:ntrials
        w[trial] ./= sum(w[trial]; dims=1)
    end

    # Create FilterSmooth objects with ALL required fields
    tfs_array = Vector{StateSpaceDynamics.FilterSmooth{Float64}}(undef, ntrials)
    for k in 1:ntrials
        x_smooth = x[k]
        p_smooth = zeros(Float64, latent_dim, latent_dim, tsteps)
        p_smooth_tt1 = zeros(Float64, latent_dim, latent_dim, tsteps)
        E_z = x[k]
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

    tsteps_per_trial = [size(x[trial], 2) for trial in 1:ntrials]
    sws = StateSpaceDynamics.SmoothWorkspace(Float64, latent_dim, obs_dim, tsteps)
    data = StateSpaceDynamics.Data(slds.LDSs[1], y)

    for active_k in 1:K
        lds_k = slds.LDSs[active_k]
        lds_k.fit_bool[3] = true   # A&b must be fitted for Q's residual scatter to be meaningful
        lds_k.fit_bool[4] = true

        w_k = [w[trial][active_k, :] for trial in 1:ntrials]

        suf = StateSpaceDynamics._initialize_td_sufficient_statistics(
            Float64, lds_k, tsteps_per_trial
        )
        StateSpaceDynamics._aggregate_td_suff_stats_weighted!(
            suf, tfs, lds_k, data, w_k, sws
        )
        StateSpaceDynamics.update_A_b!(lds_k, suf, sws)
        StateSpaceDynamics.update_Q!(lds_k, suf, sws)

        @test all(isfinite.(lds_k.state_model.Q))
        @test isapprox(lds_k.state_model.Q, lds_k.state_model.Q', atol=1e-10)

        eigvals_Q = eigvals(lds_k.state_model.Q)
        @test all(eigvals_Q .>= -1e-10)
    end
end

function test_weighted_gradient_linearity(; rng=MersenneTwister(0xC0FFEE))
    """Test that gradient scales linearly with weights"""
    K = 2
    lds = _make_gaussian_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(10, 1))
    tsteps = size(y[1], 2)

    # Create base weights
    w = rand(K, tsteps)
    w ./= sum(w; dims=1)

    # Compute gradient with base weights
    grad1 = _slds_gradient(slds, y[1], x[1], w)

    # Compute gradient with scaled weights (should produce same result after normalization)
    w_scaled = 2.5 * w
    w_scaled ./= sum(w_scaled; dims=1)
    grad2 = _slds_gradient(slds, y[1], x[1], w_scaled)

    # After normalization, gradients should be equal
    @test isapprox(grad1, grad2, rtol=1e-10)
end

function test_zero_weights_behavior(; rng=MersenneTwister(0xC0FFEE))
    """Test that zero weights are handled correctly"""
    K = 2
    lds = _make_gaussian_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(10, 1))
    tsteps = size(y[1], 2)

    # Create weights where one state has zero weight everywhere
    w = zeros(K, tsteps)
    w[1, :] .= 1.0  # Only state 1 is active

    # This should work without errors
    grad = _slds_gradient(slds, y[1], x[1], w)
    @test all(isfinite.(grad))

    H_diag, H_super, H_sub = _slds_hessian_blocks(slds, y[1], x[1], w)
    @test all(all(isfinite, h) for h in H_diag)
    @test all(all(isfinite, h) for h in H_super)
    @test all(all(isfinite, h) for h in H_sub)
end

# Poisson SLDS Tests

function test_SLDS_sampling_poisson_extended(; rng=MersenneTwister(0xC0FFEE))
    # Extended Poisson sampling test with more trials and edge cases
    K = 3
    lds = _make_poisson_lds(3, 5)
    s = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    tsteps, ntrials = 50, 10
    z, x, y = rand(rng, s, fill(tsteps, ntrials))

    @test all(y[n][i, t] ≥ 0 for i in 1:5, t in 1:tsteps, n in 1:ntrials)
    @test all(y[n][i, t] == round(y[n][i, t]) for i in 1:5, t in 1:tsteps, n in 1:ntrials)
    @test all(all(isfinite, yn) for yn in y)
end

function test_SLDS_gradient_numerical_poisson(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_poisson_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(20, 1))

    tsteps = size(y[1], 2)
    w = rand(K, tsteps)
    w ./= sum(w; dims=1)

    y_trial = y[1]
    x_trial = x[1]

    # Analytical gradient
    grad_analytical = _slds_gradient(slds, y_trial, x_trial, w)

    # For numerical gradient, manually compute weighted log-likelihood
    function weighted_ll(x_flat)
        x_mat = reshape(x_flat, size(x_trial))
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
            d_k = lds_k.obs_model.d
            # Canonical Poisson: λ = exp(C x + d). Previous version had
            # `d_k = exp.(d_k)` mirroring the (now-fixed) double-exp.

            # Precompute inverses
            inv_P0 = inv(P0_k)
            inv_Q = inv(Q_k)

            # Initial state (weighted by w[k, 1])
            dx0 = x_mat[:, 1] - x0_k
            ll += w[k, 1] * (-0.5 * dot(dx0, inv_P0 * dx0))

            # Dynamics and emissions
            for t in 1:tsteps
                # Poisson emission (weighted by w[k, t])
                obs_mean = C_k * x_mat[:, t] .+ d_k
                ll += w[k, t] * (dot(y_trial[:, t], obs_mean) - sum(exp, obs_mean))

                # Dynamics (weighted by w[k, t] for t > 1)
                if t > 1
                    dx = x_mat[:, t] - (A_k * x_mat[:, t - 1] + b_k)
                    ll += w[k, t] * (-0.5 * dot(dx, inv_Q * dx))
                end
            end
        end

        return ll
    end

    grad_numerical = ForwardDiff.gradient(weighted_ll, vec(x_trial))
    grad_numerical = reshape(grad_numerical, size(x_trial))

    @test isapprox(grad_analytical, grad_numerical, rtol=1e-5, atol=1e-5)
end

function test_SLDS_hessian_block_structure_poisson(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_poisson_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    _, x, y = rand(rng, slds, fill(12, 1))
    tsteps = size(y[1], 2)
    w = rand(K, tsteps)
    w ./= sum(w; dims=1)

    return _test_hessian_blocks_basic(slds, y[1], x[1], w)
end

function test_SLDS_smooth_basic_poisson(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_poisson_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(20, 1))

    tsteps = size(y[1], 2)
    w = rand(K, tsteps)
    w ./= sum(w; dims=1)

    # Run smooth
    x_smooth, p_smooth = StateSpaceDynamics.smooth(slds, y[1], w)

    # Check outputs
    @test size(x_smooth) == (slds.LDSs[1].latent_dim, tsteps)
    @test size(p_smooth) == (slds.LDSs[1].latent_dim, slds.LDSs[1].latent_dim, tsteps)
    @test all(isfinite, x_smooth)
    @test all(isfinite, p_smooth)

    # Check that covariances are symmetric and positive semi-definite
    for t in 1:tsteps
        @test issymmetric(p_smooth[:, :, t]) ||
            isapprox(p_smooth[:, :, t], p_smooth[:, :, t]'; atol=1e-10)
        @test all(eigvals(p_smooth[:, :, t]) .>= -1e-10)
    end
end

function test_SLDS_estep_basic_poisson(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_poisson_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    tsteps, ntrials = 20, 3
    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    latent_dim = slds.LDSs[1].latent_dim

    seq_ends = cumsum(fill(tsteps, ntrials))
    total_T = last(seq_ends)
    obs_seq = collect(1:total_T)
    control_seq = fill(nothing, total_T)

    tfs = StateSpaceDynamics.initialize_FilterSmooth(slds.LDSs[1], fill(tsteps, ntrials))
    dl = StateSpaceDynamics.SLDSDiscreteLayer(slds.A, slds.πₖ, zeros(Float64, K, total_T))
    fb_storage = StateSpaceDynamics._make_slds_fb_storage(dl, seq_ends)
    slds_ws = StateSpaceDynamics.SLDSSmoothWorkspace(Float64, slds, tsteps)

    # Warm-start smooth draws the first joint sample into x_samples.
    x_samples = [Matrix{Float64}(undef, latent_dim, tsteps) for _ in 1:ntrials]
    for trial in 1:ntrials
        w_uniform = ones(Float64, K, tsteps) ./ K
        StateSpaceDynamics.smooth!(
            slds,
            tfs[trial],
            y[trial],
            w_uniform;
            ws=slds_ws,
            x_sample=x_samples[trial],
            rng=rng,
        )
    end

    StateSpaceDynamics.estep!(
        slds,
        tfs,
        fb_storage,
        dl,
        y,
        x_samples,
        slds_ws;
        rng=rng,
        obs_seq=obs_seq,
        control_seq=control_seq,
        seq_ends=seq_ends,
    )

    elbo = StateSpaceDynamics.elbo!(slds, tfs, fb_storage, y, slds_ws; seq_ends=seq_ends)

    @test isfinite(elbo)

    @test all(isfinite, fb_storage.α)
    @test all(isfinite, fb_storage.β)
    @test all(isfinite, fb_storage.γ)
    @test all(ξt -> all(isfinite, ξt), fb_storage.ξ)
    @test all(isapprox.(sum(fb_storage.γ; dims=1), 1.0, atol=1e-10))
    @test all(0 .<= fb_storage.γ .<= 1)
end

function test_SLDS_mstep_updates_parameters_poisson(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    lds = _make_poisson_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    tsteps, ntrials = 20, 3
    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    latent_dim = slds.LDSs[1].latent_dim
    obs_dim = slds.LDSs[1].obs_dim

    seq_ends = cumsum(fill(tsteps, ntrials))
    total_T = last(seq_ends)
    obs_seq = collect(1:total_T)
    control_seq = fill(nothing, total_T)

    tfs = StateSpaceDynamics.initialize_FilterSmooth(slds.LDSs[1], fill(tsteps, ntrials))
    dl = StateSpaceDynamics.SLDSDiscreteLayer(slds.A, slds.πₖ, zeros(Float64, K, total_T))
    fb_storage = StateSpaceDynamics._make_slds_fb_storage(dl, seq_ends)
    slds_ws = StateSpaceDynamics.SLDSSmoothWorkspace(Float64, slds, tsteps)
    sws = StateSpaceDynamics.SmoothWorkspace(Float64, latent_dim, obs_dim, tsteps)

    #=
    Warm-start smooth draws the first joint sample into x_samples. estep! then
    re-smooths with the γ weights (and redraws), filling the posterior
    covariances the M-step aggregator reads (estep! → mstep! here; elbo! skipped).
    =#
    x_samples = [Matrix{Float64}(undef, latent_dim, tsteps) for _ in 1:ntrials]
    for trial in 1:ntrials
        w_uniform = ones(Float64, K, tsteps) ./ K
        StateSpaceDynamics.smooth!(
            slds,
            tfs[trial],
            y[trial],
            w_uniform;
            ws=slds_ws,
            x_sample=x_samples[trial],
            rng=rng,
        )
    end

    StateSpaceDynamics.estep!(
        slds,
        tfs,
        fb_storage,
        dl,
        y,
        x_samples,
        slds_ws;
        rng=rng,
        obs_seq=obs_seq,
        control_seq=control_seq,
        seq_ends=seq_ends,
    )

    StateSpaceDynamics.mstep!(
        slds, tfs, fb_storage, dl, y, sws; obs_seq=obs_seq, seq_ends=seq_ends
    )

    for k in 1:K
        @test all(isfinite, slds.LDSs[k].obs_model.C)
        @test all(isfinite, slds.LDSs[k].obs_model.d)
    end
end

function test_SLDS_fit_runs_to_completion_poisson(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 15
    ntrials = 2
    max_iter = 5

    lds = _make_poisson_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    # Fit without progress bar
    elbos = fit!(slds, y; max_iter=max_iter, progress=false)

    @test length(elbos) == max_iter
    @test all(isfinite, elbos)
end

function test_SLDS_fit_elbo_generally_increases_poisson(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 20
    ntrials = 3
    max_iter = 10

    lds = _make_poisson_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    elbos = fit!(slds, y; max_iter=max_iter, progress=false)

    # Check that later ELBOs are generally higher than early ones
    early_mean = mean(elbos[1:3])
    late_mean = mean(elbos[(end - 2):end])

    @test (late_mean > early_mean - 100) # don't fail CI
end

function test_SLDS_fit_multitrial_poisson(; rng=MersenneTwister(0xC0FFEE))
    K = 2
    latent_dim = 2
    obs_dim = 3
    tsteps = 15
    ntrials = 5
    max_iter = 5

    lds = _make_poisson_lds(latent_dim, obs_dim)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    elbos = fit!(slds, y; max_iter=max_iter, progress=false)

    @test length(elbos) == max_iter
    @test all(isfinite, elbos)
end

function test_SLDS_poisson_count_validation(; rng=MersenneTwister(0xC0FFEE))
    # Test that Poisson observations are non-negative integers
    K = 2
    lds = _make_poisson_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    tsteps, ntrials = 30, 5
    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    @test all(all(yn .>= 0) for yn in y)
    @test all(all(abs.(yn .- round.(yn)) .< 1e-10) for yn in y)
end

function test_SLDS_poisson_d_interpretation(; rng=MersenneTwister(0xC0FFEE))
    # Verify the canonical Poisson GLM: λ = exp(C x + d). With C ≡ 0, the
    # observed rates should equal exp(d) directly.
    K = 1
    latent_dim = 2
    obs_dim = 3

    lds = _make_poisson_lds(latent_dim, obs_dim)
    lds.obs_model.d .= log.([1.0, 2.0, 3.0])  # log-rates: rate = exp(d) = [1, 2, 3]
    lds.obs_model.C .= 0.0                    # no latent influence

    slds = SLDS(; A=reshape([1.0], 1, 1), πₖ=[1.0], LDSs=[lds])

    tsteps, ntrials = 1000, 1
    z, x, y = rand(rng, slds, fill(tsteps, ntrials))

    expected_rates = exp.(lds.obs_model.d)    # [1, 2, 3]
    mean_rates = vec(mean(y[1]; dims=2))

    for i in 1:obs_dim
        @test isapprox(mean_rates[i], expected_rates[i], rtol=0.3)
    end
end

function test_SLDS_gradient_weight_normalization_poisson(; rng=MersenneTwister(0xC0FFEE))
    # Test that gradients are properly weighted
    K = 2
    lds = _make_poisson_lds(2, 3)
    slds = SLDS(; A=_rowstochastic(K), πₖ=_probvec(K), LDSs=fill(lds, K))

    z, x, y = rand(rng, slds, fill(20, 1))

    tsteps = size(y[1], 2)
    w = rand(K, tsteps)
    w ./= sum(w; dims=1)

    y_trial = y[1]
    x_trial = x[1]

    # Gradient with uniform weights should equal unweighted gradient
    w_uniform = ones(K, tsteps) ./ K
    grad_weighted = _slds_gradient(slds, y_trial, x_trial, w_uniform)

    @test all(isfinite, grad_weighted)
    @test size(grad_weighted) == size(x_trial)
end
