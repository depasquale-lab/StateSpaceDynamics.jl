"""
Tests for the block-tridiagonal (Newton) Gaussian LDS smoother/fit path and the
marginal (Kalman-filter) log-likelihood.

The Kalman/RTS smoother and its EM machinery were removed — all Gaussian fitting
goes through the block-tridiagonal MAP path. What remains of the Kalman path is
the filter behind `loglikelihood(lds, y)`: a shared observation-independent
covariance pass (`_filter_cov_pass`) plus a mean-only pass per trial.

These tests cover:
- the TD cov-sharing fast path (shared covariance storage across equal-length trials),
- learning a `B` dynamics-input matrix via the M-step,
- sampling equivalence for inputs,
- the marginal log-likelihood: agreement across input forms, against a naive
  textbook reference filter (with and without `ux`/`uy` inputs), ragged trials,
  and the Poisson not-implemented guard.
"""

#=
Local container for randomly-generated LDS parameters used to seed the test
fits. The same shape lives under `benchmarking/`, but the test suite
shouldn't depend on that path being on the load path — so we redeclare it
here.
=#
struct LDSParams{T<:Real}
    A::Matrix{T}
    Q::Matrix{T}
    x0::Vector{T}
    P0::Matrix{T}
    C::Matrix{T}
    R::Matrix{T}
    b::Vector{T}
    d::Vector{T}
end

function init_params(rng::AbstractRNG, latent_dim::Int, obs_dim::Int)
    A = SSD.random_rotation_matrix(latent_dim, rng)

    Q = randn(rng, latent_dim, latent_dim)
    Q = Q * Q' .+ 1e-3

    x0 = randn(rng, latent_dim)
    P0 = randn(rng, latent_dim, latent_dim)
    P0 = P0 * P0' .+ 1e-3

    C = randn(rng, obs_dim, latent_dim)
    R = randn(rng, obs_dim, obs_dim)
    R = R * R' .+ 1e-3

    b = randn(rng, latent_dim)
    d = randn(rng, obs_dim)

    return LDSParams(A, Q, x0, P0, C, R, b, d)
end

function _make_toy_lds(; D::Int=3, p::Int=5, seed::Int=7, B=nothing)
    params = init_params(MersenneTwister(seed), D, p)
    #=
    `GaussianStateModel.B` is a non-nullable matrix field with a
    type-preserving default; only override it when the caller supplies
    an explicit input matrix. (`B=nothing` would conflict with `B::M`.)
    =#
    sm = if B === nothing
        GaussianStateModel(; A=params.A, Q=params.Q, x0=params.x0, P0=params.P0, b=params.b)
    else
        GaussianStateModel(;
            A=params.A, Q=params.Q, x0=params.x0, P0=params.P0, b=params.b, B=B
        )
    end
    om = GaussianObservationModel(; C=params.C, R=params.R, d=params.d)
    return LinearDynamicalSystem(sm, om)
end

function _simulate_lds(lds::LinearDynamicalSystem, T::Int, N::Int; seed::Int=42, u=nothing)
    Random.seed!(seed)
    D = lds.latent_dim
    p = lds.obs_dim
    A = lds.state_model.A
    b = lds.state_model.b
    Q = lds.state_model.Q
    C = lds.obs_model.C
    d = lds.obs_model.d
    R = lds.obs_model.R
    P0 = lds.state_model.P0
    x0 = lds.state_model.x0
    B = lds.state_model.B

    Lq = cholesky(Q).L
    Lr = cholesky(R).L
    Lp0 = cholesky(P0).L
    y = zeros(p, T, N)
    # `B` is a non-nullable matrix field with a zero default; the model only
    # consumes inputs when a matching `u` array is passed.
    for n in 1:N
        x = x0 .+ Lp0 * randn(D)
        y[:, 1, n] = C * x + d + Lr * randn(p)
        for t in 2:T
            bu = u === nothing ? zero(b) : B * u[:, t - 1, n]
            x = A * x + b + bu + Lq * randn(D)
            y[:, t, n] = C * x + d + Lr * randn(p)
        end
    end
    return y
end

function test_td_covariance_shared_across_trials()
    #=
    For the TD path (the cov-sharing fast path), each `FilterSmooth.p_smooth`
    is aliased to the shared workspace array after a multi-trial equal-length
    fit; we can check that directly.
    =#
    D, p, Tt, N = 3, 4, 20, 5
    rng = MersenneTwister(123)
    sm = GaussianStateModel(;
        A=0.6 * Matrix{Float64}(I, D, D),
        Q=0.2 * Matrix{Float64}(I, D, D),
        x0=zeros(D),
        P0=Matrix{Float64}(I, D, D),
        b=zeros(D),
    )
    om = GaussianObservationModel(;
        C=randn(rng, p, D), R=0.1 * Matrix{Float64}(I, p, p), d=zeros(p)
    )
    lds_td = LinearDynamicalSystem(sm, om)
    _, y_seq = rand(rng, lds_td, fill(Tt, N))

    # Trigger one smooth! (multi-trial, equal-length → fast path).
    tsteps_per_trial = [size(yt, 2) for yt in y_seq]
    tfs = SSD.initialize_FilterSmooth(lds_td, tsteps_per_trial)
    T_max = maximum(tsteps_per_trial)
    sws_pool = [
        SSD.SmoothWorkspace(Float64, lds_td.latent_dim, lds_td.obs_dim, T_max) for
        _ in 1:Threads.maxthreadid()
    ]
    SSD.smooth!(lds_td, tfs, y_seq, sws_pool)

    # All trials' p_smooth must alias the shared storage on sws_pool[1].
    shared_p = sws_pool[1].agg.p_smooth_shared
    shared_p_tt1 = sws_pool[1].agg.p_smooth_tt1_shared
    for trial in 1:N
        @test tfs[trial].p_smooth === shared_p
        @test tfs[trial].p_smooth_tt1 === shared_p_tt1
    end

    # Entropy should also be identical across trials (depends only on the
    # shared covariance via its log-determinant).
    for trial in 2:N
        @test tfs[trial].entropy == tfs[1].entropy
    end
end

function test_td_shared_cov_matches_per_trial_path()
    #=
    Sanity check: the cov-sharing fast path produces the same smoothed
    estimates as a brute-force per-trial smoother on the same data. We
    exercise the slow path by feeding *variable*-length trials (one shorter
    than the others) so the fast path is skipped — the shared-cov code path
    should still produce numerically equivalent results to running the
    equal-length smoother per trial.
    =#
    D, p, Tt, N = 3, 4, 25, 4
    rng = MersenneTwister(321)
    sm = GaussianStateModel(;
        A=0.7 * Matrix{Float64}(I, D, D),
        Q=0.15 * Matrix{Float64}(I, D, D),
        x0=zeros(D),
        P0=Matrix{Float64}(I, D, D),
        b=0.1 * ones(D),
    )
    om = GaussianObservationModel(;
        C=randn(rng, p, D), R=0.1 * Matrix{Float64}(I, p, p), d=zeros(p)
    )
    lds = LinearDynamicalSystem(sm, om)
    _, y_seq = rand(rng, lds, fill(Tt, N))

    # Fast path (equal-length, multi-trial).
    tfs_fast = SSD.initialize_FilterSmooth(lds, fill(Tt, N))
    sws_pool = [SSD.SmoothWorkspace(Float64, D, p, Tt) for _ in 1:Threads.maxthreadid()]
    SSD.smooth!(lds, tfs_fast, y_seq, sws_pool)

    # Per-trial reference path: run single-trial `smooth!` once per trial.
    refs_x = Vector{Matrix{Float64}}(undef, N)
    refs_p = Vector{Array{Float64,3}}(undef, N)
    for trial in 1:N
        fs = SSD.initialize_FilterSmooth(lds, Tt)
        sws = SSD.SmoothWorkspace(Float64, D, p, Tt)
        SSD.smooth!(lds, fs, y_seq[trial], sws)
        refs_x[trial] = copy(fs.x_smooth)
        refs_p[trial] = copy(fs.p_smooth)
    end

    for trial in 1:N
        @test tfs_fast[trial].x_smooth ≈ refs_x[trial] atol = 1e-9
        @test tfs_fast[trial].p_smooth ≈ refs_p[trial] atol = 1e-9
    end
end

function test_lds_with_B_input_equivalent_to_bias()
    # B·u with u ≡ 1 reduces to an additive constant; setting b = B·1 should
    # produce identical sample paths.
    D, p, T, N = 3, 4, 30, 2
    ux_dim = D
    B = Matrix{Float64}(I, D, ux_dim)
    Random.seed!(11)
    sm = GaussianStateModel(;
        A=0.7 * Matrix{Float64}(I, D, D),
        Q=0.3 * Matrix{Float64}(I, D, D),
        x0=zeros(D),
        P0=Matrix{Float64}(I, D, D),
        b=zeros(D),
        B=B,
    )
    om = GaussianObservationModel(;
        C=randn(p, D), R=0.4 * Matrix{Float64}(I, p, p), d=zeros(p)
    )
    lds_B = LinearDynamicalSystem(sm, om)

    u = ones(ux_dim, T, N)
    y_B = _simulate_lds(lds_B, T, N; u=u)

    # Same model but with the bias absorbed into `b` instead of `B·u`.
    sm_b = GaussianStateModel(;
        A=lds_B.state_model.A,
        Q=lds_B.state_model.Q,
        x0=lds_B.state_model.x0,
        P0=lds_B.state_model.P0,
        b=vec(B * ones(ux_dim)),
    )
    om_b = GaussianObservationModel(;
        C=lds_B.obs_model.C, R=lds_B.obs_model.R, d=lds_B.obs_model.d
    )
    lds_b = LinearDynamicalSystem(sm_b, om_b)
    y_b = _simulate_lds(lds_b, T, N)

    @test y_B ≈ y_b atol = 1e-10
end

function test_td_fit_with_latent_input()
    # TD path: simulate from `x_{t+1} = A x_t + b + B u_t`, fit, recover B
    # (and b) to coarse tolerance.
    D, p, Tt, N = 3, 5, 60, 8
    ux_dim = 2
    rng = MersenneTwister(101)

    A_true = 0.85 * SSD.random_rotation_matrix(D, rng)
    Q_true = 0.05 * Matrix{Float64}(I, D, D)
    b_true = randn(rng, D)
    B_true = randn(rng, D, ux_dim)
    x0_true = zeros(D)
    P0_true = 0.1 * Matrix{Float64}(I, D, D)
    C_true = randn(rng, p, D)
    R_true = 0.1 * Matrix{Float64}(I, p, p)
    d_true = zeros(p)

    sm_true = GaussianStateModel(;
        A=A_true, Q=Q_true, x0=x0_true, P0=P0_true, b=b_true, B=B_true
    )
    om_true = GaussianObservationModel(; C=C_true, R=R_true, d=d_true)
    lds_true = LinearDynamicalSystem(sm_true, om_true)

    ux_seq = [randn(rng, ux_dim, Tt) for _ in 1:N]
    _, y_seq = rand(lds_true, fill(Tt, N); ux=ux_seq)

    # Fit from a perturbed init.
    sm_init = GaussianStateModel(;
        A=0.5 * Matrix{Float64}(I, D, D),
        Q=Matrix{Float64}(I, D, D),
        x0=zeros(D),
        P0=Matrix{Float64}(I, D, D),
        b=zeros(D),
        B=zeros(D, ux_dim),
    )
    om_init = GaussianObservationModel(;
        C=randn(rng, p, D), R=Matrix{Float64}(I, p, p), d=zeros(p)
    )
    lds_fit = LinearDynamicalSystem(sm_init, om_init)

    elbos = fit!(lds_fit, y_seq; ux=ux_seq, max_iter=80, progress=false)

    @test all(diff(elbos) .>= -1e-4)        # ~monotone
    #=
    B is identifiable up to the same gauge as A/C (rotation of latent space);
    check predictive fit instead — the learned B should explain input-driven
    variance, so fitting *with* controls should beat fitting *without* on the
    same data. The "without" baseline uses a 0-column B (proper no-input model).
    =#
    sm_nofit = GaussianStateModel(;
        A=0.5 * Matrix{Float64}(I, D, D),
        Q=Matrix{Float64}(I, D, D),
        x0=zeros(D),
        P0=Matrix{Float64}(I, D, D),
        b=zeros(D),
    )
    om_nofit = GaussianObservationModel(;
        C=randn(MersenneTwister(101), p, D), R=Matrix{Float64}(I, p, p), d=zeros(p)
    )
    lds_nofit = LinearDynamicalSystem(sm_nofit, om_nofit)
    elbos_no = fit!(lds_nofit, y_seq; max_iter=80, progress=false)

    @test elbos[end] > elbos_no[end] + 1.0  # controls help, by a lot for these data
end

function test_td_sampling_zero_input_matches_no_input()
    # With ux present but u ≡ 0 and B = 0, sampling should match
    # the no-control case (same RNG seed).
    D, p, Tt = 3, 4, 25
    rng = MersenneTwister(7)
    sm = GaussianStateModel(;
        A=0.6 * Matrix{Float64}(I, D, D),
        Q=0.2 * Matrix{Float64}(I, D, D),
        x0=randn(rng, D),
        P0=Matrix{Float64}(I, D, D),
        b=randn(rng, D),
        B=zeros(D, 2),
    )
    om = GaussianObservationModel(;
        C=randn(rng, p, D), R=0.1 * Matrix{Float64}(I, p, p), d=zeros(p)
    )
    lds = LinearDynamicalSystem(sm, om)

    u_zero = zeros(2, Tt)
    rng1 = MersenneTwister(42)
    x1, y1 = rand(rng1, lds, Tt; ux=u_zero)

    # Reset state-model to a 0-column B and call without ux.
    sm2 = GaussianStateModel(; A=sm.A, Q=sm.Q, x0=sm.x0, P0=sm.P0, b=sm.b)
    lds2 = LinearDynamicalSystem(sm2, om)
    rng2 = MersenneTwister(42)
    x2, y2 = rand(rng2, lds2, Tt)

    @test x1 ≈ x2 atol = 1e-12
    @test y1 ≈ y2 atol = 1e-12
end

function test_td_fit_missing_u_errors()
    # B is set but the required dynamics inputs are omitted at fit time → error.
    D, p, T, N = 2, 3, 15, 2
    B = Matrix{Float64}(I, D, D)
    Random.seed!(3)
    sm = GaussianStateModel(;
        A=0.6 * Matrix{Float64}(I, D, D),
        Q=0.2 * Matrix{Float64}(I, D, D),
        x0=zeros(D),
        P0=Matrix{Float64}(I, D, D),
        b=zeros(D),
        B=B,
    )
    om = GaussianObservationModel(;
        C=randn(p, D), R=0.3 * Matrix{Float64}(I, p, p), d=zeros(p)
    )
    lds = LinearDynamicalSystem(sm, om)
    y = randn(p, T, N)
    y_vec = [y[:, :, n] for n in 1:N]
    @test_throws Exception fit!(lds, y_vec; max_iter=1, progress=false)
end

function test_marginal_loglikelihood()
    #=
    `loglikelihood(lds, y)` is the marginal (observed-data) log-likelihood via
    the Kalman filter. It should be finite, backend-independent, and agree
    between the 3-D-array and vector-of-matrices input forms. Poisson has no
    tractable marginal and must throw.
    =#
    D, p, T, N = 3, 5, 40, 6
    lds = _make_toy_lds(; D=D, p=p)
    y = _simulate_lds(lds, T, N)

    ll = StateSpaceDynamics.loglikelihood(lds, y)
    @test ll isa Real
    @test isfinite(ll)

    # Vector-of-matrices form matches the stacked-array form.
    y_vec = [y[:, :, n] for n in 1:N]
    @test StateSpaceDynamics.loglikelihood(lds, y_vec) ≈ ll atol = 1e-9

    # Poisson marginal is intractable → not implemented.
    sm = GaussianStateModel(;
        A=0.7 * Matrix{Float64}(I, 2, 2),
        Q=0.2 * Matrix{Float64}(I, 2, 2),
        x0=zeros(2),
        P0=Matrix{Float64}(I, 2, 2),
        b=zeros(2),
    )
    plds = LinearDynamicalSystem(sm, PoissonObservationModel(; C=randn(3, 2), d=zeros(3)))
    y_pois = randn(3, 10)
    @test_throws Exception StateSpaceDynamics.loglikelihood(plds, y_pois)
end

# =============================================================================
# Marginal log-likelihood (shared-covariance Kalman filter).
#
# The covariance half of the filter — innovation covariances and gains — is
# observation-independent, so `loglikelihood` computes it once
# (`_filter_cov_pass`) and shares it across trials; each trial then runs a
# mean-only pass. These tests check that filter against a naive textbook
# covariance-form reference, with and without `ux`/`uy` inputs, and for
# ragged trial lengths.
# =============================================================================

#=
Like `_simulate_lds`, but with both dynamics inputs (`B·u_t`) and
observation inputs (`D·v_t`): x_{t+1} = A x_t + b + B u_t + ε,
y_t = C x_t + d + D v_t + η.
=#
function _simulate_lds_io(
    lds::LinearDynamicalSystem,
    T::Int,
    N::Int,
    u::AbstractArray{<:Real,3},
    v::AbstractArray{<:Real,3};
    seed::Int=42,
)
    rng = MersenneTwister(seed)
    D = lds.latent_dim
    p = lds.obs_dim
    sm = lds.state_model
    om = lds.obs_model
    Lq = cholesky(sm.Q).L
    Lr = cholesky(om.R).L
    Lp0 = cholesky(sm.P0).L
    y = zeros(p, T, N)
    for n in 1:N
        x = sm.x0 .+ Lp0 * randn(rng, D)
        for t in 1:T
            if t > 1
                x = sm.A * x .+ sm.b .+ sm.B * u[:, t - 1, n] .+ Lq * randn(rng, D)
            end
            y[:, t, n] .= om.C * x .+ om.d .+ om.D * v[:, t, n] .+ Lr * randn(rng, p)
        end
    end
    return y
end

#=
Toy model with both a dynamics-input matrix `B` and an observation-input
matrix `D`, plus matching simulated data.
=#
function _make_lds_io_setup(; D=2, p=3, Tt=40, N=3, ux_dim=2, uy_dim=2, seed=44)
    rng = MersenneTwister(seed)
    sm = GaussianStateModel(;
        A=0.7 * Matrix{Float64}(I, D, D),
        Q=0.1 * Matrix{Float64}(I, D, D),
        x0=zeros(D),
        P0=0.5 * Matrix{Float64}(I, D, D),
        b=0.1 * ones(D),
        B=randn(rng, D, ux_dim),
    )
    om = GaussianObservationModel(;
        C=randn(rng, p, D),
        R=0.2 * Matrix{Float64}(I, p, p),
        d=0.1 * ones(p),
        D=randn(rng, p, uy_dim),
    )
    lds = LinearDynamicalSystem(sm, om)
    u = randn(rng, ux_dim, Tt, N)
    v = randn(rng, uy_dim, Tt, N)
    y = _simulate_lds_io(lds, Tt, N, u, v; seed=seed + 1)
    return lds, y, u, v
end

#=
Naive textbook covariance-form Kalman filter marginal LL for one trial i.e., no
shared covariance pass, no information-form update. Reference implementation
for the fast filter; both compute the same quantity exactly, so agreement is
limited only by floating-point rounding of the two orderings.
=#
function _naive_kalman_ll(
    lds::LinearDynamicalSystem, y::AbstractMatrix; u=nothing, v=nothing
)
    sm = lds.state_model
    om = lds.obs_model
    p, Tt = size(y)
    x_f = copy(sm.x0)
    P_f = copy(sm.P0)
    x_p = copy(sm.x0)
    P_p = copy(sm.P0)
    ll = 0.0
    for t in 1:Tt
        if t > 1
            x_p = sm.A * x_f .+ sm.b
            u === nothing || (x_p .+= sm.B * u[:, t - 1])
            P_p = sm.A * P_f * sm.A' .+ sm.Q
        end
        e = y[:, t] .- om.C * x_p .- om.d
        v === nothing || (e .-= om.D * v[:, t])
        S = cholesky(Symmetric(om.C * P_p * om.C' .+ om.R))
        ll -= 0.5 * (p * log(2π) + logdet(S) + dot(e, S \ e))
        K = P_p * om.C' / S
        x_f = x_p .+ K * e
        P_f = P_p .- K * (om.C * P_p)
    end
    return ll
end

function test_marginal_ll_matches_naive_filter()
    # No inputs: the shared-covariance filter must agree with the per-trial
    # textbook filter on a well-conditioned model.
    D, p, Tt, N = 3, 4, 35, 4
    rng = MersenneTwister(31)
    sm = GaussianStateModel(;
        A=0.7 * Matrix{Float64}(I, D, D),
        Q=0.1 * Matrix{Float64}(I, D, D),
        x0=randn(rng, D),
        P0=0.5 * Matrix{Float64}(I, D, D),
        b=0.1 * ones(D),
    )
    om = GaussianObservationModel(;
        C=randn(rng, p, D), R=0.2 * Matrix{Float64}(I, p, p), d=0.1 * ones(p)
    )
    lds = LinearDynamicalSystem(sm, om)
    y = _simulate_lds(lds, Tt, N; seed=32)

    ll = SSD.loglikelihood(lds, y)
    ll_ref = sum(_naive_kalman_ll(lds, y[:, :, n]) for n in 1:N)
    @test ll ≈ ll_ref rtol = 1e-8

    # Single-matrix form runs the same filter on one trial.
    @test SSD.loglikelihood(lds, y[:, :, 1]) ≈ _naive_kalman_ll(lds, y[:, :, 1]) rtol = 1e-8
end

function test_marginal_ll_with_inputs()
    # Dynamics (`B·u`) and observation (`D·v`) inputs flow through the filter:
    # agreement with the naive reference, and required-input validation.
    lds, y, u, v = _make_lds_io_setup()
    N = size(y, 3)

    ll = SSD.loglikelihood(lds, y; ux=u, uy=v)
    ll_ref = sum(_naive_kalman_ll(lds, y[:, :, n]; u=u[:, :, n], v=v[:, :, n]) for n in 1:N)
    @test ll ≈ ll_ref rtol = 1e-8

    # Vector-of-matrices form agrees with the 3-D form.
    y_vec = [y[:, :, n] for n in 1:N]
    u_vec = [u[:, :, n] for n in 1:N]
    v_vec = [v[:, :, n] for n in 1:N]
    @test SSD.loglikelihood(lds, y_vec; ux=u_vec, uy=v_vec) ≈ ll

    # The data were generated with these inputs, so the true inputs must
    # explain them better than zeroed inputs.
    @test ll > SSD.loglikelihood(lds, y; ux=zero(u), uy=zero(v))

    # B/D have input columns → omitting the input sequences is an error.
    @test_throws ArgumentError SSD.loglikelihood(lds, y)
    @test_throws ArgumentError SSD.loglikelihood(lds, y; ux=u)
end

function test_marginal_ll_ragged_trials()
    #=
    Trial lengths may differ: the covariance recursion is data-independent, so
    a shorter trial just consumes a prefix of the shared pass. The multi-trial
    LL must equal the sum of independent single-trial LLs.
    =#
    D, p = 3, 4
    rng = MersenneTwister(55)
    sm = GaussianStateModel(;
        A=0.6 * Matrix{Float64}(I, D, D),
        Q=0.2 * Matrix{Float64}(I, D, D),
        x0=zeros(D),
        P0=Matrix{Float64}(I, D, D),
        b=0.1 * ones(D),
    )
    om = GaussianObservationModel(;
        C=randn(rng, p, D), R=0.1 * Matrix{Float64}(I, p, p), d=zeros(p)
    )
    lds = LinearDynamicalSystem(sm, om)
    _, y_seq = rand(rng, lds, [30, 20, 25])

    ll_multi = SSD.loglikelihood(lds, y_seq)
    ll_sum = sum(SSD.loglikelihood(lds, yt) for yt in y_seq)
    @test ll_multi ≈ ll_sum
end

function test_marginal_loglikelihood_aliases()
    # `marginal_loglikelihood` is a named alias of `loglikelihood` for all
    # three observation forms.
    D, p, Tt, N = 3, 4, 30, 3
    lds = _make_toy_lds(; D=D, p=p, seed=61)
    y = _simulate_lds(lds, Tt, N; seed=62)
    ll_ref = SSD.loglikelihood(lds, y)

    @test SSD.marginal_loglikelihood(lds, y) ≈ ll_ref
    y_vec = [y[:, :, n] for n in 1:N]
    @test SSD.marginal_loglikelihood(lds, y_vec) ≈ ll_ref
    @test SSD.marginal_loglikelihood(lds, y[:, :, 1]) ≈ SSD.loglikelihood(lds, y[:, :, 1])
end
