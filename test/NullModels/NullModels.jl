function _null_make_data(
    rng::AbstractRNG, obs_dim::Int, tsteps::Int, ntrials::Int; v_dim::Int=0, T::Type=Float64
)
    y = randn(rng, T, obs_dim, tsteps, ntrials)
    ux = randn(rng, T, v_dim, tsteps, ntrials)
    uy = zeros(T, 0, tsteps, ntrials)
    return Data(; y=y, ux=ux, uy=uy)
end

function _null_make_var_data(
    rng::AbstractRNG,
    F::AbstractMatrix{T},
    d_vec::AbstractVector{T},
    R::AbstractMatrix{T},
    μ_0::AbstractVector{T},
    R_0::AbstractMatrix{T},
    tsteps::Int,
    ntrials::Int,
) where {T<:Real}
    obs_dim = length(d_vec)
    y = zeros(T, obs_dim, tsteps, ntrials)
    R_0_chol = cholesky(Symmetric(R_0)).L
    R_chol = cholesky(Symmetric(R)).L
    for n in 1:ntrials
        y[:, 1, n] .= μ_0 .+ R_0_chol * randn(rng, T, obs_dim)
        for t in 2:tsteps
            y[:, t, n] .= F * y[:, t - 1, n] .+ d_vec .+ R_chol * randn(rng, T, obs_dim)
        end
    end
    ux = zeros(T, 0, tsteps, ntrials)
    uy = zeros(T, 0, tsteps, ntrials)
    return Data(; y=y, ux=ux, uy=uy)
end

# Closed-form multivariate-normal LL on a stacked data matrix. Used by both
# intercept and test-LL identity tests.
function _mvn_stacked_ll(Y, μ, R)
    obs_dim, n = size(Y)
    Yc = Y .- μ
    R_inv_Yc = R \ Yc
    return -0.5 * (n * obs_dim * log(2π) + n * logdet(R) + sum(Yc .* R_inv_Yc))
end

# ----- Intercept-only model ------------------------------------------------

function test_null_intercept_matches_mvnormal_loglik(rng=MersenneTwister(0xC0FFEE))
    obs_dim, tsteps, ntrials = 3, 20, 5
    data = _null_make_data(rng, obs_dim, tsteps, ntrials)

    res = test_null(data)

    # Closed-form MLE: d = mean over (t, n), R = empirical residual scatter / n.
    Y = reshape(data.y, obs_dim, tsteps * ntrials)
    n = size(Y, 2)
    d_hat = vec(mean(Y; dims=2))
    Yc = Y .- d_hat
    R_hat = (Yc * Yc') ./ n

    ref_ll = _mvn_stacked_ll(Y, d_hat, R_hat)

    @test res.intercept.train_ll ≈ ref_ll atol = 1e-8 rtol = 1e-8
    @test res.intercept.params.d ≈ d_hat atol = 1e-10
    @test res.intercept.params.R ≈ R_hat atol = 1e-10
    @test res.intercept.test_ll === nothing
end

# ----- Plug-in test LL identity --------------------------------------------

function test_null_test_ll_matches_plugin_gaussian(rng=MersenneTwister(0xC0FFEE))
    obs_dim, tsteps, ntrials = 2, 15, 4
    rng = MersenneTwister(0xC0FFEE)
    train = _null_make_data(rng, obs_dim, tsteps, ntrials)
    test = _null_make_data(rng, obs_dim, 12, 3)

    res = test_null(train; test_data=test)

    d_hat = res.intercept.params.d
    R_hat = res.intercept.params.R
    Y_te = reshape(test.y, obs_dim, 12 * 3)
    ref = _mvn_stacked_ll(Y_te, d_hat, R_hat)
    @test res.intercept.test_ll ≈ ref atol = 1e-8 rtol = 1e-8
end

function test_null_test_ll_nothing_when_no_test_data(rng=MersenneTwister(0xC0FFEE))
    data = _null_make_data(rng, 2, 10, 3)
    res = test_null(data)
    @test res.intercept.test_ll === nothing
    @test res.inputs.test_ll === nothing
    @test res.var.test_ll === nothing
    @test res.var_inputs.test_ll === nothing
end

# ----- Collapse identities --------------------------------------------------

function test_null_inputs_collapses_to_intercept_when_no_inputs(
    rng=MersenneTwister(0xC0FFEE)
)
    obs_dim, tsteps, ntrials = 3, 10, 6
    data = _null_make_data(rng, obs_dim, tsteps, ntrials; v_dim=0)

    res = test_null(data)

    # With v_dim = 0 the inputs model has the same regressors as the intercept
    # model, and the VAR+inputs model matches the VAR-only model.
    @test res.inputs.train_ll ≈ res.intercept.train_ll atol = 1e-10
    @test res.var_inputs.train_ll ≈ res.var.train_ll atol = 1e-10
end

# ----- Signal recovery + LL ordering ---------------------------------------

function test_null_inputs_helps_when_signal_present(rng=MersenneTwister(0xC0FFEE))
    T = Float64
    obs_dim, tsteps, ntrials = 4, 40, 10
    v_dim = 2

    D_true = randn(rng, T, obs_dim, v_dim)
    d_true = randn(rng, T, obs_dim)
    v = randn(rng, T, v_dim, tsteps, ntrials)
    R = Matrix{T}(0.1 * I, obs_dim, obs_dim)
    R_chol = cholesky(Symmetric(R)).L
    y = zeros(T, obs_dim, tsteps, ntrials)
    for n in 1:ntrials, t in 1:tsteps
        y[:, t, n] .= d_true .+ D_true * v[:, t, n] .+ R_chol * randn(rng, T, obs_dim)
    end
    data = Data(; y=y, ux=v, uy=zeros(T, 0, tsteps, ntrials))

    res = test_null(data)

    # The inputs model should fit strictly better than intercept-only, and
    # var_inputs should be at least as good as var-only when inputs matter.
    @test res.inputs.train_ll > res.intercept.train_ll
    @test res.var_inputs.train_ll > res.var.train_ll
    @test res.inputs.params.D ≈ D_true rtol = 0.15
    @test res.inputs.params.d ≈ d_true rtol = 0.2
    # Shape sanity on returned parameters.
    @test size(res.inputs.params.D) == (obs_dim, v_dim)
    @test size(res.var_inputs.params.D) == (obs_dim, v_dim)
    @test size(res.var_inputs.params.F) == (obs_dim, obs_dim)
end

# ----- VAR(1) parameter recovery -------------------------------------------

function test_null_var_recovers_true_F_on_var_data(rng=MersenneTwister(0xC0FFEE))
    T = Float64
    obs_dim, tsteps, ntrials = 3, 1000, 100

    F_true =
        T(0.7) * Matrix{T}(I, obs_dim, obs_dim) .+
        T(0.05) .* randn(rng, T, obs_dim, obs_dim)
    d_true = zeros(T, obs_dim)
    R_true = Matrix{T}(0.05 * I, obs_dim, obs_dim)
    μ_0 = zeros(T, obs_dim)
    R_0 = Matrix{T}(0.2 * I, obs_dim, obs_dim)
    R0_prior = IWPrior(; Ψ=Matrix{T}(0.1 * I, obs_dim, obs_dim), ν=T(obs_dim + 2))

    data = _null_make_var_data(rng, F_true, d_true, R_true, μ_0, R_0, tsteps, ntrials)
    res = test_null(data; R0_prior=R0_prior)

    @test res.var.params.F ≈ F_true atol = 0.05
    @test res.var.params.R ≈ R_true atol = 0.02
    @test length(res.var.params.μ_0) == obs_dim
    @test size(res.var.params.R_0) == (obs_dim, obs_dim)
    @test length(res.var.params.d) == obs_dim
    # VAR model should beat intercept-only on truly autocorrelated data.
    @test res.var.train_ll > res.intercept.train_ll
end

# ----- IW-prior LL shift identity ------------------------------------------

function test_null_R_prior_shifts_LL_by_iw_logprior_term(rng=MersenneTwister(0xC0FFEE))
    T = Float64
    obs_dim, tsteps, ntrials = 3, 20, 5
    data = _null_make_data(rng, obs_dim, tsteps, ntrials)

    res_no = test_null(data)
    Ψ = Matrix{T}(0.1 * I, obs_dim, obs_dim)
    ν = T(obs_dim + 3)
    R_prior = IWPrior(; Ψ=Ψ, ν=ν)
    res_with = test_null(data; R_prior=R_prior)

    function data_ll(W, R)
        Y = reshape(data.y, obs_dim, tsteps * ntrials)
        n = size(Y, 2)
        return _mvn_stacked_ll(Y, W, R)
    end

    W_no = reshape(res_no.intercept.params.d, obs_dim, 1)
    W_with = reshape(res_with.intercept.params.d, obs_dim, 1)
    @test res_no.intercept.train_ll ≈ data_ll(W_no, res_no.intercept.params.R) atol = 1e-8
    delta = res_with.intercept.train_ll - data_ll(W_with, res_with.intercept.params.R)
    # The remaining piece must be exactly the IW log-prior term (no MN prior).
    R_with = res_with.intercept.params.R
    expected = -0.5 * ((ν + obs_dim + 1) * logdet(R_with) + tr(R_with \ Ψ))
    @test delta ≈ expected atol = 1e-8
end

# ----- MN-prior contribution -----------------------------------------------

function test_null_mn_prior_shifts_LL_by_mn_logprior_term(rng=MersenneTwister(0xC0FFEE))
    T = Float64
    obs_dim, tsteps, ntrials = 3, 25, 4
    v_dim = 2
    data = _null_make_data(rng, obs_dim, tsteps, ntrials; v_dim=v_dim)

    # Shrink [d D] toward zero — the identity: train_ll(with MN prior) - data_ll
    # = mn_logprior_term(W, R, prior) + iw_logprior_term(R, R_prior)  (=0 here).
    M₀ = zeros(T, obs_dim, 1 + v_dim)
    Λ = Matrix{T}(0.5 * I, 1 + v_dim, 1 + v_dim)
    prior = MNPrior(; M₀=M₀, Λ=Λ)

    res = test_null(data; inputs_W_prior=prior)

    # Reconstruct W = [d D] and recompute the data-only LL by hand.
    W = hcat(res.inputs.params.d, res.inputs.params.D)
    R = res.inputs.params.R
    Y = reshape(data.y, obs_dim, tsteps * ntrials)
    n = size(Y, 2)
    v_flat = reshape(data.ux, v_dim, n)
    X = vcat(ones(T, 1, n), v_flat)
    E = Y .- W * X
    data_ll_ref = -0.5 * (n * obs_dim * log(2π) + n * logdet(R) + tr(R \ (E * E')))

    Wm = W .- M₀
    expected_shift = -0.5 * tr(R \ (Wm * Λ * Wm'))
    @test res.inputs.train_ll - data_ll_ref ≈ expected_shift atol = 1e-8

    # Strong shrinkage should also observably pull the MAP toward M₀ = 0.
    strong_prior = MNPrior(; M₀=M₀, Λ=Matrix{T}(1e6 * I, 1 + v_dim, 1 + v_dim))
    res_strong = test_null(data; inputs_W_prior=strong_prior)
    W_strong = hcat(res_strong.inputs.params.d, res_strong.inputs.params.D)
    @test norm(W_strong) < norm(W)
end

# ----- Input override + validation -----------------------------------------

function test_null_inputs_override_uses_supplied_array(rng=MersenneTwister(0xC0FFEE))
    T = Float64
    obs_dim, tsteps, ntrials = 2, 12, 5
    v_dim = 3

    data = _null_make_data(rng, obs_dim, tsteps, ntrials; v_dim=v_dim)
    # `train_data.ux` has v_dim = 3; supply a 1-column override.
    v_override_train = randn(rng, T, 1, tsteps, ntrials)
    # Test with matching test override too.
    test_data_ = _null_make_data(rng, obs_dim, tsteps, ntrials; v_dim=v_dim)
    v_override_test = randn(rng, T, 1, tsteps, ntrials)
    res = test_null(
        data;
        test_data=test_data_,
        train_inputs=v_override_train,
        test_inputs=v_override_test,
    )

    @test size(res.inputs.params.D, 2) == 1
    @test size(res.var_inputs.params.D, 2) == 1
    @test res.inputs.test_ll isa Float64 && isfinite(res.inputs.test_ll)
    @test res.var_inputs.test_ll isa Float64 && isfinite(res.var_inputs.test_ll)
end

function test_null_test_inputs_default_from_test_data(rng=MersenneTwister(0xC0FFEE))
    # When test_inputs is not supplied, defaults to test_data.ux — differs
    # from what happens if we explicitly pass zeros.
    train = _null_make_data(rng, 2, 12, 4; v_dim=1)
    test = _null_make_data(rng, 2, 10, 3; v_dim=1)
    res = test_null(train; test_data=test)
    @test res.inputs.test_ll isa Float64 && isfinite(res.inputs.test_ll)
    @test res.var_inputs.test_ll isa Float64 && isfinite(res.var_inputs.test_ll)
end

# ----- Error paths ---------------------------------------------------------

function test_null_var_requires_tsteps_ge_2(rng=MersenneTwister(0xC0FFEE))
    data = _null_make_data(rng, 2, 1, 4)
    @test_throws ArgumentError test_null(data)
end

function test_null_input_shape_mismatch_throws(rng=MersenneTwister(0xC0FFEE))
    T = Float64
    obs_dim, tsteps, ntrials = 2, 8, 3
    data = _null_make_data(rng, obs_dim, tsteps, ntrials)
    bad_inputs = randn(rng, T, 2, tsteps + 1, ntrials)  # wrong tsteps
    @test_throws SSD.DimensionMismatchError test_null(data; train_inputs=bad_inputs)
end

function test_null_test_inputs_shape_mismatch_throws(rng=MersenneTwister(0xC0FFEE))
    T = Float64
    train = _null_make_data(rng, 2, 10, 3)
    test = _null_make_data(rng, 2, 10, 3)
    bad = randn(rng, T, 1, 10, 4)  # ntrials mismatch
    @test_throws SSD.DimensionMismatchError test_null(
        train; test_data=test, test_inputs=bad
    )
end

function test_null_test_data_obs_dim_mismatch_throws(rng=MersenneTwister(0xC0FFEE))
    train = _null_make_data(rng, 3, 10, 4)
    test = _null_make_data(rng, 2, 10, 4)
    @test_throws SSD.DimensionMismatchError test_null(train; test_data=test)
end

# ----- Capacity ordering ---------------------------------------------------

function test_null_capacity_ordering_on_var_data(rng=MersenneTwister(0xC0FFEE))
    T = Float64
    obs_dim, tsteps, ntrials = 3, 100, 6

    F_true = T(0.6) * Matrix{T}(I, obs_dim, obs_dim)
    d_true = T(0.5) * ones(T, obs_dim)
    R_true = Matrix{T}(0.1 * I, obs_dim, obs_dim)
    μ_0 = zeros(T, obs_dim)
    R_0 = Matrix{T}(0.2 * I, obs_dim, obs_dim)

    data = _null_make_var_data(rng, F_true, d_true, R_true, μ_0, R_0, tsteps, ntrials)
    R0_prior = IWPrior(; Ψ=Matrix{T}(0.1 * I, obs_dim, obs_dim), ν=T(obs_dim + 2))
    res = test_null(data; R0_prior=R0_prior)

    @test res.var.train_ll > res.intercept.train_ll
    @test res.var_inputs.train_ll ≈ res.var.train_ll atol = 1e-10
    @test res.inputs.train_ll ≈ res.intercept.train_ll atol = 1e-10
end

# ----- All-priors path finite ---------------------------------------------

function test_null_all_priors_active_returns_finite_lls(rng=MersenneTwister(0xC0FFEE))
    # Exercise the branch where every prior kwarg is set (covers each
    # `mn_logprior_term`/`iw_logprior_term` addition path in `_null_train_ll`
    # and the MN residual contribution in `_null_fit_regression`).
    T = Float64
    obs_dim, tsteps, ntrials = 3, 20, 5
    v_dim = 2
    train = _null_make_data(rng, obs_dim, tsteps, ntrials; v_dim=v_dim)
    test = _null_make_data(rng, obs_dim, 10, 3; v_dim=v_dim)

    R_prior = IWPrior(; Ψ=Matrix{T}(0.1 * I, obs_dim, obs_dim), ν=T(obs_dim + 2))
    R0_prior = IWPrior(; Ψ=Matrix{T}(0.2 * I, obs_dim, obs_dim), ν=T(obs_dim + 2))
    intercept_prior = MNPrior(; M₀=zeros(T, obs_dim, 1), Λ=Matrix{T}(0.5 * I, 1, 1))
    inputs_prior = MNPrior(;
        M₀=zeros(T, obs_dim, 1 + v_dim), Λ=Matrix{T}(0.5 * I, 1 + v_dim, 1 + v_dim)
    )
    var_prior = MNPrior(;
        M₀=zeros(T, obs_dim, obs_dim + 1), Λ=Matrix{T}(0.5 * I, obs_dim + 1, obs_dim + 1)
    )
    var_inputs_prior = MNPrior(;
        M₀=zeros(T, obs_dim, obs_dim + 1 + v_dim),
        Λ=Matrix{T}(0.5 * I, obs_dim + 1 + v_dim, obs_dim + 1 + v_dim),
    )

    res = test_null(
        train;
        test_data=test,
        intercept_W_prior=intercept_prior,
        inputs_W_prior=inputs_prior,
        var_W_prior=var_prior,
        var_inputs_W_prior=var_inputs_prior,
        R_prior=R_prior,
        R0_prior=R0_prior,
    )
    for name in (:intercept, :inputs, :var, :var_inputs)
        entry = getproperty(res, name)
        @test isfinite(entry.train_ll)
        @test isfinite(entry.test_ll)
    end
end

# ===========================================================================
# Cox–Snell R² (`compute_R2`) against the null baselines.
# ===========================================================================

# Build a Gaussian LDS with oscillatory dynamics (complex eigenvalues) and
# optional dynamics (`ux`) / observation (`uy`) inputs.
function _r2_make_lds(
    rng::AbstractRNG;
    latent_dim::Int=2,
    obs_dim::Int=4,
    ux_dim::Int=0,
    uy_dim::Int=0,
    decay::Float64=0.9,
    R_prior=nothing,
)
    T = Float64
    A = T(decay) * random_rotation_matrix(latent_dim, rng)
    Q = Matrix{T}(0.05 * I, latent_dim, latent_dim)
    b = zeros(T, latent_dim)
    x0 = zeros(T, latent_dim)
    P0 = Matrix{T}(0.3 * I, latent_dim, latent_dim)
    B = ux_dim > 0 ? randn(rng, T, latent_dim, ux_dim) : zeros(T, latent_dim, 0)
    C = randn(rng, T, obs_dim, latent_dim)
    R = Matrix{T}(0.3 * I, obs_dim, obs_dim)
    d = randn(rng, T, obs_dim)
    Dm = uy_dim > 0 ? randn(rng, T, obs_dim, uy_dim) : zeros(T, obs_dim, 0)
    sm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0, B=B)
    om = GaussianObservationModel(; C=C, R=R, d=d, D=Dm, R_prior=R_prior)
    return LinearDynamicalSystem(sm, om)
end

# Simulate a no-input LDS into a `Data` struct (obs_dim × tsteps × ntrials).
function _r2_sim_data(rng::AbstractRNG, lds, tsteps::Int, ntrials::Int)
    _, y_seq = rand(rng, lds, fill(tsteps, ntrials))
    y = cat(y_seq...; dims=3)
    T = eltype(y)
    return Data(; y=y, ux=zeros(T, 0, tsteps, ntrials), uy=zeros(T, 0, tsteps, ntrials))
end

_r2_rowstochastic(K) = (M=rand(K, K); M ./ sum(M; dims=2))
_r2_probvec(K) = (v=rand(K); v ./ sum(v))

# The inputs-aware marginal LL must reduce to `loglikelihood(lds, y)` when the
# model carries no inputs.
function test_compute_R2_marginal_matches_loglikelihood_no_inputs(rng=MersenneTwister(11))
    lds = _r2_make_lds(rng; latent_dim=2, obs_dim=4)
    data = _r2_sim_data(rng, lds, 40, 6)
    ll_aware = SSD.loglikelihood(lds, data)
    ll_base = loglikelihood(lds, data.y)
    @test ll_aware ≈ ll_base atol = 1e-8 rtol = 1e-8
end

# The inputs-aware marginal LL must honour `D·uy`: subtracting the obs-input
# mean shift and scoring with a D-free model reproduces it exactly, and it
# differs from the input-blind `loglikelihood(lds, y)`.
function test_compute_R2_marginal_accounts_for_obs_inputs(rng=MersenneTwister(12))
    T = Float64
    latent_dim, obs_dim, uy_dim = 2, 4, 2
    tsteps, ntrials = 30, 5
    lds = _r2_make_lds(rng; latent_dim=latent_dim, obs_dim=obs_dim, uy_dim=uy_dim)

    uy_seq = [randn(rng, T, uy_dim, tsteps) for _ in 1:ntrials]
    _, y_seq = rand(rng, lds, fill(tsteps, ntrials); obs_inputs=uy_seq)
    y = cat(y_seq...; dims=3)
    uy = cat(uy_seq...; dims=3)
    data = Data(; y=y, ux=zeros(T, 0, tsteps, ntrials), uy=uy)

    ll_aware = SSD.loglikelihood(lds, data)

    Dm = lds.obs_model.D
    y_shift = similar(y)
    for n in 1:ntrials, t in 1:tsteps
        @views y_shift[:, t, n] .= y[:, t, n] .- Dm * uy[:, t, n]
    end
    lds_noD = LinearDynamicalSystem(
        lds.state_model,
        GaussianObservationModel(; C=lds.obs_model.C, R=lds.obs_model.R, d=lds.obs_model.d),
    )
    ll_ref = loglikelihood(lds_noD, y_shift)
    @test ll_aware ≈ ll_ref atol = 1e-8 rtol = 1e-8
    @test !isapprox(ll_aware, loglikelihood(lds, y); rtol=1e-3)
end

# Returned R² equals the closed-form Cox–Snell value built from the reported
# log-likelihoods, and `n = obs_dim · tsteps · ntrials`.
function test_compute_R2_formula_and_fields(rng=MersenneTwister(13))
    obs_dim, tsteps, ntrials = 4, 30, 6
    lds = _r2_make_lds(rng; latent_dim=2, obs_dim=obs_dim)
    data = _r2_sim_data(rng, lds, tsteps, ntrials)
    res = compute_R2(lds, data)

    n_expected = obs_dim * tsteps * ntrials
    lds_ll = SSD.loglikelihood(lds, data)
    for name in (:intercept, :inputs, :var, :var_inputs)
        e = getproperty(res, name)
        @test e.n == n_expected
        @test e.lds_ll ≈ lds_ll atol = 1e-10
        @test e.R2 ≈ 1 - exp((2 / n_expected) * (e.null_ll - lds_ll)) atol = 1e-12
        @test isfinite(e.R2)
    end
end

# Ground-truth LDS (the data-generating parameters) out-predicts every null
# baseline on held-out data.
function test_compute_R2_ground_truth_beats_null_heldout(rng=MersenneTwister(14))
    T = Float64
    latent_dim, obs_dim = 2, 6
    tsteps, ntrials = 60, 30

    # Oscillatory (complex-eigenvalue) latent dynamics with sustaining process
    # noise: an obs-space VAR(1) cannot capture this, so the true LDS wins.
    A = T(0.95) * random_rotation_matrix(latent_dim, rng)
    Q = Matrix{T}(0.1 * I, latent_dim, latent_dim)
    b = zeros(T, latent_dim)
    x0 = zeros(T, latent_dim)
    P0 = Matrix{T}(0.5 * I, latent_dim, latent_dim)
    C = randn(rng, T, obs_dim, latent_dim)
    R = Matrix{T}(0.3 * I, obs_dim, obs_dim)
    d = T(0.5) .* randn(rng, T, obs_dim)
    sm = GaussianStateModel(; A=A, Q=Q, b=b, x0=x0, P0=P0)
    om = GaussianObservationModel(; C=C, R=R, d=d)
    lds = LinearDynamicalSystem(sm, om)

    train = _r2_sim_data(rng, lds, tsteps, ntrials)
    test = _r2_sim_data(rng, lds, tsteps, ntrials)
    res = compute_R2(lds, train, test)

    for name in (:intercept, :inputs, :var, :var_inputs)
        e = getproperty(res, name)
        @test e.test_R2 > 0
        @test e.train_R2 > 0
        @test e.test_R2 < 1
    end
end

# Single-`data` R² equals the split form's `train_R2` when the same data is the
# training split (both fit the null on that split and score it in-sample).
function test_compute_R2_single_matches_split_train(rng=MersenneTwister(15))
    lds = _r2_make_lds(rng; latent_dim=2, obs_dim=4)
    train = _r2_sim_data(rng, lds, 40, 8)
    test = _r2_sim_data(rng, lds, 40, 8)
    single = compute_R2(lds, train)
    split = compute_R2(lds, train, test)
    for name in (:intercept, :inputs, :var, :var_inputs)
        @test getproperty(single, name).R2 ≈ getproperty(split, name).train_R2 atol = 1e-12
    end
end

# `null_inputs` selects the input channel fed to the input-bearing baselines:
# the LDS LL is unchanged, but the null side differs between "ux" and "uy".
function test_compute_R2_null_inputs_ux_vs_uy(rng=MersenneTwister(16))
    T = Float64
    latent_dim, obs_dim, ux_dim, uy_dim = 2, 4, 2, 3
    tsteps, ntrials = 40, 8
    lds = _r2_make_lds(
        rng; latent_dim=latent_dim, obs_dim=obs_dim, ux_dim=ux_dim, uy_dim=uy_dim
    )
    ux_seq = [randn(rng, T, ux_dim, tsteps) for _ in 1:ntrials]
    uy_seq = [randn(rng, T, uy_dim, tsteps) for _ in 1:ntrials]
    _, y_seq = rand(
        rng, lds, fill(tsteps, ntrials); latent_inputs=ux_seq, obs_inputs=uy_seq
    )
    data = Data(;
        y=cat(y_seq...; dims=3), ux=cat(ux_seq...; dims=3), uy=cat(uy_seq...; dims=3)
    )

    res_ux = compute_R2(lds, data; null_inputs="ux")
    res_uy = compute_R2(lds, data; null_inputs="uy")

    # Same LDS marginal LL regardless of which channel the null consumes.
    @test res_ux.inputs.lds_ll ≈ res_uy.inputs.lds_ll atol = 1e-10
    # The intercept baseline ignores inputs, so it is identical across channels.
    @test res_ux.intercept.null_ll ≈ res_uy.intercept.null_ll atol = 1e-10
    # The input-bearing baselines see different regressors → different fit.
    @test res_ux.inputs.null_ll != res_uy.inputs.null_ll
    @test res_ux.var_inputs.null_ll != res_uy.var_inputs.null_ll
    for r in (res_ux, res_uy), name in (:intercept, :inputs, :var, :var_inputs)
        @test isfinite(getproperty(r, name).R2)
    end

    @test_throws ArgumentError compute_R2(lds, data; null_inputs="uz")
end

# The null models inherit the LDS's observation-noise prior by default; passing
# it explicitly matches, and dropping it changes the null fit (hence null_ll).
function test_compute_R2_forwards_R_prior(rng=MersenneTwister(17))
    T = Float64
    obs_dim, tsteps, ntrials = 4, 40, 8
    Rp = IWPrior(; Ψ=Matrix{T}(0.5 * I, obs_dim, obs_dim), ν=T(obs_dim + 5))
    lds = _r2_make_lds(rng; latent_dim=2, obs_dim=obs_dim, R_prior=Rp)
    data = _r2_sim_data(rng, lds, tsteps, ntrials)

    res_default = compute_R2(lds, data)
    res_explicit = compute_R2(lds, data; R_prior=Rp)
    res_none = compute_R2(lds, data; R_prior=nothing)
    for name in (:intercept, :inputs, :var, :var_inputs)
        @test getproperty(res_default, name).null_ll ≈
            getproperty(res_explicit, name).null_ll atol = 1e-12
        @test getproperty(res_default, name).null_ll != getproperty(res_none, name).null_ll
    end
end

# `compute_R2` is defined only for the Gaussian LDS: Poisson observations and
# the SLDS type fall through to a `MethodError`.
function test_compute_R2_rejects_plds_and_slds(rng=MersenneTwister(18))
    T = Float64
    latent_dim, obs_dim = 2, 3
    tsteps, ntrials = 10, 4
    data = Data(;
        y=randn(rng, T, obs_dim, tsteps, ntrials),
        ux=zeros(T, 0, tsteps, ntrials),
        uy=zeros(T, 0, tsteps, ntrials),
    )
    sm = GaussianStateModel(;
        A=T(0.5) * Matrix{T}(I, latent_dim, latent_dim),
        Q=Matrix{T}(0.1 * I, latent_dim, latent_dim),
        b=zeros(T, latent_dim),
        x0=zeros(T, latent_dim),
        P0=Matrix{T}(I, latent_dim, latent_dim),
    )
    plds = LinearDynamicalSystem(
        sm,
        PoissonObservationModel(;
            C=randn(rng, T, obs_dim, latent_dim), d=zeros(T, obs_dim)
        ),
    )
    @test_throws MethodError compute_R2(plds, data)

    glds = LinearDynamicalSystem(
        sm,
        GaussianObservationModel(;
            C=randn(rng, T, obs_dim, latent_dim),
            R=Matrix{T}(I, obs_dim, obs_dim),
            d=zeros(T, obs_dim),
        ),
    )
    slds = SLDS(; A=_r2_rowstochastic(2), πₖ=_r2_probvec(2), LDSs=fill(glds, 2))
    @test_throws MethodError compute_R2(slds, data)
end
