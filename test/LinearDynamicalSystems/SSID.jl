# Tests for the CVA subspace-identification initializer (src/init/SSID.jl).
#
# CVA identifies (A, C) only up to a similarity transform, so recovery tests
# compare invariants (eigenvalues of A, Markov parameters C·Aᵏ·B, reconstructed
# output) rather than raw matrices. The bulk of the suite exercises mechanics:
# shapes, in-place mutation, PD covariances, error paths, dispatch, determinism.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Build a stable Gaussian LDS with a chosen (real) eigenvalue spectrum, random
# observation matrix, small process/observation noise, and a nonzero observation
# bias (to exercise demeaning). Optionally include dynamics/obs input matrices.
function ssid_make_lds(
    ::Type{T}=Float64;
    rng=StableRNG(11),
    eigs=T[0.9, 0.7, 0.4],
    obs::Int=6,
    q::Real=1e-6,
    r::Real=1e-4,
    uin::Int=0,
    vin::Int=0,
) where {T<:Real}
    latent = length(eigs)
    V = randn(rng, T, latent, latent)
    A = Matrix{T}(real(V * Diagonal(T.(eigs)) * inv(V)))
    C = randn(rng, T, obs, latent)
    Q = Matrix{T}(T(q) * I(latent))
    R = Matrix{T}(T(r) * I(obs))
    b = zeros(T, latent)
    d = T(2) .* randn(rng, T, obs)                # nonzero obs bias
    x0 = zeros(T, latent)
    P0 = Matrix{T}(T(0.05) * I(latent))
    B = uin > 0 ? randn(rng, T, latent, uin) : zeros(T, latent, 0)
    D = vin > 0 ? randn(rng, T, obs, vin) : zeros(T, obs, 0)
    sm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0, B=B)
    om = GaussianObservationModel(; C=C, R=R, d=d, D=D)
    return LinearDynamicalSystem(sm, om)
end

# Fresh model + copies of the original parameters, for mutation checks.
function ssid_param_snapshot(lds)
    sm, om = lds.state_model, lds.obs_model
    return (
        A=copy(sm.A),
        Q=copy(sm.Q),
        b=copy(sm.b),
        x0=copy(sm.x0),
        P0=copy(sm.P0),
        B=copy(sm.B),
        C=copy(om.C),
        R=copy(om.R),
        d=copy(om.d),
        D=copy(om.D),
    )
end

# Markov parameters C·Aᵏ·B, k = 0:K (similarity invariant).
function ssid_markov(A, B, C, K)
    out = Vector{Matrix{eltype(A)}}(undef, K + 1)
    M = copy(B)
    for k in 0:K
        out[k + 1] = C * M
        M = A * M
    end
    return out
end

# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------

function test_ssid_shapes_and_mutation()
    lds = ssid_make_lds(; eigs=[0.85, 0.5])
    _, y = StateSpaceDynamics.rand(StableRNG(2), lds, 2000)
    orig = ssid_param_snapshot(lds)

    diag = fit!(lds, y, SSID(; r=12))

    sm, om = lds.state_model, lds.obs_model
    @test size(sm.A) == (lds.latent_dim, lds.latent_dim)
    @test size(sm.Q) == (lds.latent_dim, lds.latent_dim)
    @test size(sm.x0) == (lds.latent_dim,)
    @test size(sm.P0) == (lds.latent_dim, lds.latent_dim)
    @test size(sm.b) == (lds.latent_dim,)
    @test size(om.C) == (lds.obs_dim, lds.latent_dim)
    @test size(om.R) == (lds.obs_dim, lds.obs_dim)
    @test size(om.d) == (lds.obs_dim,)

    # Something actually changed.
    @test sm.A != orig.A
    @test om.C != orig.C
    @test sm.Q != orig.Q

    # Diagnostics.
    @test haskey(diag, :S) && haskey(diag, :fve)
    @test diag.S isa AbstractVector
    @test diag.fve isa Real
end

function test_ssid_fve_and_singular_values()
    lds = ssid_make_lds(; eigs=[0.9, 0.6, 0.3])
    _, y = StateSpaceDynamics.rand(StableRNG(3), lds, 3000)
    diag = fit!(lds, y, SSID(; r=15))
    @test 0 <= diag.fve <= 1 + 1e-8
    @test all(>=(-1e-10), diag.S)
    @test issorted(diag.S; rev=true)
end

function test_ssid_pd_covariances_and_validation()
    lds = ssid_make_lds(; eigs=[0.8, 0.5])
    _, y = StateSpaceDynamics.rand(StableRNG(4), lds, 2500)
    fit!(lds, y, SSID(; r=12))
    for M in (lds.state_model.Q, lds.state_model.P0, lds.obs_model.R)
        @test issymmetric(M)
        @test isposdef(M)
    end
    @test validate_LDS(lds) === nothing
end

function test_ssid_stability_enforced()
    lds = ssid_make_lds(; eigs=[0.95, 0.6, 0.2])
    _, y = StateSpaceDynamics.rand(StableRNG(5), lds, 3000)
    fit!(lds, y, SSID(; r=15, stable=true))
    @test all(abs.(eigvals(lds.state_model.A)) .<= 1 + 1e-6)
end

function test_ssid_weightings_run()
    lds0 = ssid_make_lds(; eigs=[0.85, 0.5])
    _, y = StateSpaceDynamics.rand(StableRNG(6), lds0, 3000)
    for W in (:CVA, :MOESP, :N4SID, :IVM)
        lds = ssid_make_lds(; eigs=[0.85, 0.5])
        diag = fit!(lds, y, SSID(; r=12, W=W))
        @test isposdef(lds.state_model.Q)
        @test isposdef(lds.obs_model.R)
        @test 0 <= diag.fve <= 1 + 1e-8
        @test validate_LDS(lds) === nothing
    end
end

function test_ssid_ridge_vs_backslash()
    lds0 = ssid_make_lds(; eigs=[0.85, 0.5])
    _, y = StateSpaceDynamics.rand(StableRNG(7), lds0, 3000)
    for ridge in (true, false)
        lds = ssid_make_lds(; eigs=[0.85, 0.5])
        fit!(lds, y, SSID(; r=12, ridge=ridge))
        @test validate_LDS(lds) === nothing
    end
end

function test_ssid_reproducibility()
    lds_a = ssid_make_lds(; eigs=[0.85, 0.5, 0.3])
    lds_b = ssid_make_lds(; eigs=[0.85, 0.5, 0.3])
    _, y = StateSpaceDynamics.rand(StableRNG(8), lds_a, 2500)
    da = fit!(lds_a, y, SSID(; r=14))
    db = fit!(lds_b, y, SSID(; r=14))
    @test lds_a.state_model.A == lds_b.state_model.A
    @test lds_a.obs_model.C == lds_b.obs_model.C
    @test lds_a.state_model.Q == lds_b.state_model.Q
    @test lds_a.obs_model.R == lds_b.obs_model.R
    @test lds_a.state_model.x0 == lds_b.state_model.x0
    @test da.fve == db.fve
end

# ---------------------------------------------------------------------------
# Recovery (similarity-invariant)
# ---------------------------------------------------------------------------

function test_ssid_eigenvalue_recovery()
    true_eigs = [0.9, 0.6, 0.3]
    lds = ssid_make_lds(; eigs=true_eigs, obs=8, q=1e-7, r=1e-5)
    _, y = StateSpaceDynamics.rand(StableRNG(9), lds, 6000)
    fit!(lds, y, SSID(; r=20))
    est = sort(real.(eigvals(lds.state_model.A)))
    @test isapprox(est, sort(true_eigs); atol=0.1)
end

function test_ssid_markov_recovery_with_input()
    lds = ssid_make_lds(; eigs=[0.85, 0.5], obs=6, q=1e-7, r=1e-5, uin=2)
    rng = StableRNG(10)
    u = [randn(rng, Float64, 2, 4000)]
    _, y = StateSpaceDynamics.rand(rng, lds, [4000]; latent_inputs=u)
    A0, B0, C0 = lds.state_model.A, lds.state_model.B, lds.obs_model.C
    M_true = ssid_markov(A0, B0, C0, 4)

    fit!(lds, y, SSID(; r=15, zeroD=true); latent_inputs=u)
    M_est = ssid_markov(lds.state_model.A, lds.state_model.B, lds.obs_model.C, 4)

    # Low noise + 4000 samples: the leading Markov parameters should recover.
    for k in 1:3
        @test isapprox(M_true[k], M_est[k]; atol=0.2, rtol=0.2)
    end
end

function test_ssid_input_BD_shapes()
    # zeroD = true: D stays empty.
    lds = ssid_make_lds(; eigs=[0.8, 0.5], uin=2)
    rng = StableRNG(12)
    u = [randn(rng, Float64, 2, 3000)]
    _, y = StateSpaceDynamics.rand(rng, lds, [3000]; latent_inputs=u)
    fit!(lds, y, SSID(; r=12, zeroD=true); latent_inputs=u)
    @test size(lds.state_model.B) == (lds.latent_dim, 2)
    @test size(lds.obs_model.D) == (lds.obs_dim, 0)

    # zeroD = false with obs inputs: D is estimated.
    lds2 = ssid_make_lds(; eigs=[0.8, 0.5], uin=2, vin=1)
    rng2 = StableRNG(13)
    u2 = [randn(rng2, Float64, 2, 3000)]
    v2 = [randn(rng2, Float64, 1, 3000)]
    _, y2 = StateSpaceDynamics.rand(rng2, lds2, [3000]; latent_inputs=u2, obs_inputs=v2)
    fit!(lds2, y2, SSID(; r=12, zeroD=false); latent_inputs=u2, obs_inputs=v2)
    @test size(lds2.state_model.B) == (lds2.latent_dim, 2)
    @test size(lds2.obs_model.D) == (lds2.obs_dim, 1)
end

function test_ssid_bias_reconstructs_mean()
    lds = ssid_make_lds(; eigs=[0.85, 0.5], obs=5)
    _, y = StateSpaceDynamics.rand(StableRNG(14), lds, 3000)
    ymean = vec(sum(y; dims=2)) ./ size(y, 2)
    fit!(lds, y, SSID(; r=12))
    @test isapprox(lds.obs_model.d, ymean; atol=1e-8)
    @test all(iszero, lds.state_model.b)
end

function test_ssid_cross_trial_paths()
    lds0 = ssid_make_lds(; eigs=[0.9, 0.5, 0.3])
    _, y = StateSpaceDynamics.rand(StableRNG(15), lds0, fill(1500, 4))
    for ct in (true, false)
        lds = ssid_make_lds(; eigs=[0.9, 0.5, 0.3])
        diag = fit!(lds, y, SSID(; r=12, cross_trial=ct))
        @test isfinite(diag.fve)
        @test validate_LDS(lds) === nothing
    end
end

function test_ssid_new_init_placeholder()
    lds = ssid_make_lds(; eigs=[0.85, 0.5], uin=2)
    rng = StableRNG(16)
    u = [randn(rng, Float64, 2, 2500)]
    _, y = StateSpaceDynamics.rand(rng, lds, [2500]; latent_inputs=u)
    fit!(lds, y, SSID(; r=12, new_init=true); latent_inputs=u)
    # placeholder x0 = ones(n)/n
    @test isapprox(lds.state_model.x0, fill(1 / lds.latent_dim, lds.latent_dim); atol=1e-10)
    @test validate_LDS(lds) === nothing
end

function test_ssid_init_then_EM_improves()
    lds = ssid_make_lds(; eigs=[0.9, 0.5], obs=5, q=1e-3, r=1e-2)
    _, y = StateSpaceDynamics.rand(StableRNG(17), lds, fill(400, 3))

    # CVA-initialized model.
    lds_ssid = ssid_make_lds(; eigs=[0.9, 0.5], obs=5, q=1e-3, r=1e-2)
    fit!(lds_ssid, y, SSID(; r=12))
    elbos = fit!(lds_ssid, y; max_iter=25, progress=false)
    @test issorted(elbos; rev=false) || all(>=(-1e-6), diff(elbos))
    @test elbos[end] >= elbos[1] - 1e-6
end

# ---------------------------------------------------------------------------
# Internal-kernel unit tests
# ---------------------------------------------------------------------------

function test_ssid_dlyap_identity_and_fallback()
    rng = StableRNG(18)
    A = randn(rng, 4, 4)
    A .*= 0.8 / maximum(abs.(eigvals(A)))
    Q = let M = randn(rng, 4, 4)
        Matrix(M * M' + I)
    end
    P0 = StateSpaceDynamics._dlyap(A, Q; jitter=1e-8, stable=true)
    @test isposdef(P0)
    @test isapprox(A * P0 * A' + Q, P0; atol=1e-6, rtol=1e-6)

    # Fallback path (stable=false) → returns a PD projection of Q.
    Pf = StateSpaceDynamics._dlyap(A, Q; jitter=1e-8, stable=false)
    @test isposdef(Pf)
    @test issymmetric(Pf)
end

function test_ssid_reflectd_stabilizes()
    rng = StableRNG(19)
    A = randn(rng, 3, 3)
    A .*= 1.5 / maximum(abs.(eigvals(A)))     # spectral radius 1.5 (unstable)
    Ar = StateSpaceDynamics._reflectd(A)
    @test maximum(abs.(eigvals(Ar))) <= 1 + 1e-8
    @test eltype(Ar) <: Real
end

function test_ssid_ltisim_matches_recursion()
    rng = StableRNG(20)
    n, p, N = 3, 2, 25
    A = 0.5 .* randn(rng, n, n)
    E = randn(rng, n, 1)
    C = randn(rng, p, n)
    u = randn(rng, 1, N)
    x0 = randn(rng, n)
    Y = StateSpaceDynamics._ltisim(A, E, C, u, x0)
    # reference recursion
    Yref = zeros(p, N)
    x = copy(x0)
    for t in 1:N
        Yref[:, t] = C * x
        x = A * x + E * u[:, t]
    end
    @test isapprox(Y, Yref; atol=1e-12)
end

function test_ssid_findBD_recovers_known_system()
    rng = StableRNG(21)
    n, p, m, N = 3, 4, 2, 600
    A = randn(rng, n, n)
    A .*= 0.7 / maximum(abs.(eigvals(A)))
    C = randn(rng, p, n)
    Btrue = randn(rng, n, m)
    x0true = randn(rng, n)
    U = randn(rng, m, N)
    # Noiseless output of the deterministic system.
    Y = StateSpaceDynamics._ltisim(A, Btrue, C, U, x0true)
    V = zeros(Float64, 0, N)
    ols = (x, yy) -> x \ yy
    B, D, x0 = StateSpaceDynamics._find_BD_ssid(A, C, U, V, Y, true, ols, Float64)
    # Reconstructed output should match (B, x0 identified exactly up to noise).
    Yhat = StateSpaceDynamics._ltisim(A, B, C, U, x0)
    @test isapprox(Yhat, Y; atol=1e-6)
    @test size(D) == (p, 0)
end

# ---------------------------------------------------------------------------
# Error paths & dispatch
# ---------------------------------------------------------------------------

function test_ssid_errors_wrong_model_type()
    # Poisson LDS → fallback method throws.
    latent, obs = 2, 4
    A = Matrix{Float64}(0.5 * I(latent))
    Q = Matrix{Float64}(0.1 * I(latent))
    sm = GaussianStateModel(;
        A=A, Q=Q, b=zeros(latent), x0=zeros(latent), P0=Matrix{Float64}(0.1 * I(latent))
    )
    pom = PoissonObservationModel(; C=randn(obs, latent), d=zeros(obs))
    plds = LinearDynamicalSystem(sm, pom)
    y = [abs.(randn(obs, 50))]
    @test_throws ArgumentError fit!(plds, y, SSID())
end

function test_ssid_errors_bad_inputs()
    lds = ssid_make_lds(; eigs=[0.8, 0.5])
    _, y = StateSpaceDynamics.rand(StableRNG(22), lds, 1000)

    # r smaller than model order + 1.
    @test_throws ArgumentError fit!(ssid_make_lds(; eigs=[0.8, 0.5]), y, SSID(; r=2))

    # insufficient data for the requested horizon.
    short = [randn(StableRNG(23), lds.obs_dim, 8)]
    @test_throws ArgumentError fit!(ssid_make_lds(; eigs=[0.8, 0.5]), short, SSID(; r=20))

    # unknown weighting.
    @test_throws ArgumentError fit!(
        ssid_make_lds(; eigs=[0.8, 0.5]), y, SSID(; r=12, W=:BOGUS)
    )

    # input-dim mismatch (model has no input, but latent_inputs supplied).
    badu = [randn(StableRNG(24), 2, size(y, 2))]
    @test_throws Exception fit!(
        ssid_make_lds(; eigs=[0.8, 0.5]), y, SSID(; r=12); latent_inputs=badu
    )

    # order must equal latent_dim.
    @test_throws ArgumentError fit!(
        ssid_make_lds(; eigs=[0.8, 0.5]), y, SSID(; r=12, order=5)
    )
end

function test_ssid_2d_3d_overloads()
    lds = ssid_make_lds(; eigs=[0.85, 0.5])
    _, ymat = StateSpaceDynamics.rand(StableRNG(25), lds, 2000)   # (obs, T)

    lds2 = ssid_make_lds(; eigs=[0.85, 0.5])
    d_mat = fit!(lds2, ymat, SSID(; r=12))                        # 2D path
    lds3 = ssid_make_lds(; eigs=[0.85, 0.5])
    d_vec = fit!(lds3, [ymat], SSID(; r=12))                      # vector path
    @test lds2.state_model.A == lds3.state_model.A
    @test d_mat.fve == d_vec.fve

    # 3D array path.
    y3 = reshape(ymat, size(ymat, 1), size(ymat, 2), 1)
    lds4 = ssid_make_lds(; eigs=[0.85, 0.5])
    d_3d = fit!(lds4, y3, SSID(; r=12))
    @test d_3d.fve == d_mat.fve

    # 1D / 4D arrays rejected.
    @test_throws ArgumentError fit!(
        ssid_make_lds(; eigs=[0.85, 0.5]), randn(StableRNG(26), 10), SSID()
    )
end

function test_ssid_fit_bool_respected()
    lds = ssid_make_lds(; eigs=[0.85, 0.5])
    _, y = StateSpaceDynamics.rand(StableRNG(27), lds, 2500)
    # Freeze C&d&D (index 5) and R (index 6) via fit_bool, with set_all=false.
    fb = [true, true, true, true, false, false]
    lds_frozen = ssid_make_lds(; eigs=[0.85, 0.5])
    lds_frozen = LinearDynamicalSystem(
        lds_frozen.state_model, lds_frozen.obs_model; fit_bool=fb
    )
    C_orig = copy(lds_frozen.obs_model.C)
    R_orig = copy(lds_frozen.obs_model.R)
    A_orig = copy(lds_frozen.state_model.A)
    fit!(lds_frozen, y, SSID(; r=12, set_all=false))
    @test lds_frozen.obs_model.C == C_orig          # frozen
    @test lds_frozen.obs_model.R == R_orig          # frozen
    @test lds_frozen.state_model.A != A_orig         # updated

    # set_all=true writes everything.
    lds_all = LinearDynamicalSystem(
        ssid_make_lds(; eigs=[0.85, 0.5]).state_model,
        ssid_make_lds(; eigs=[0.85, 0.5]).obs_model;
        fit_bool=fb,
    )
    C_all = copy(lds_all.obs_model.C)
    fit!(lds_all, y, SSID(; r=12, set_all=true))
    @test lds_all.obs_model.C != C_all
end

function test_ssid_type_preservation()
    for T in (Float32, Float64)
        lds = ssid_make_lds(T; eigs=T[0.85, 0.5])
        _, y = StateSpaceDynamics.rand(StableRNG(28), lds, 2500)
        diag = fit!(lds, y, SSID(; r=12))
        @test eltype(lds.state_model.A) === T
        @test eltype(lds.state_model.Q) === T
        @test eltype(lds.obs_model.C) === T
        @test eltype(lds.obs_model.R) === T
        @test eltype(lds.state_model.x0) === T
        @test eltype(diag.S) === T
        @test diag.fve isa T
    end
end
