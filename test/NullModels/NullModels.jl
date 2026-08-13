#=============================================================================
Tests for the latent-free baselines (`AffineNullModel`) and the StatsAPI model
comparison methods built on them (`nobs`, `nullloglikelihood`, `r2`).
=============================================================================#

# Per-trial observation matrices, the shape family the baselines canonicalize to.
function _null_make_y(rng::AbstractRNG, obs_dim, tsteps_per_trial; T::Type=Float64)
    return [randn(rng, T, obs_dim, Ti) for Ti in tsteps_per_trial]
end

function _null_make_inputs(rng::AbstractRNG, v_dim, tsteps_per_trial; T::Type=Float64)
    return [randn(rng, T, v_dim, Ti) for Ti in tsteps_per_trial]
end

# Simulate a VAR(1) process with a known (F, d, R, μ₀, R₀).
function _null_make_var_y(
    rng::AbstractRNG,
    F::AbstractMatrix{T},
    d::AbstractVector{T},
    R::AbstractMatrix{T},
    μ₀::AbstractVector{T},
    R₀::AbstractMatrix{T},
    tsteps::Int,
    ntrials::Int,
) where {T<:Real}
    obs_dim = length(d)
    R₀_chol = cholesky(Symmetric(R₀)).L
    R_chol = cholesky(Symmetric(R)).L
    y = Vector{Matrix{T}}(undef, ntrials)
    for n in 1:ntrials
        yn = zeros(T, obs_dim, tsteps)
        yn[:, 1] .= μ₀ .+ R₀_chol * randn(rng, T, obs_dim)
        for t in 2:tsteps
            yn[:, t] .= F * yn[:, t - 1] .+ d .+ R_chol * randn(rng, T, obs_dim)
        end
        y[n] = yn
    end
    return y
end

# Closed-form Gaussian log-density of `Y` under `y ~ N(W X, R)`.
function _null_ref_ll(Y, X, W, R)
    obs_dim, n = size(Y)
    E = Y .- W * X
    return -0.5 * (n * obs_dim * log(2π) + n * logdet(R) + tr(R \ (E * E')))
end

_null_stack(y) = reduce(hcat, y)

#=
Baseline fitting and scoring
=#

function test_null_intercept_matches_mvnormal_loglik(rng=MersenneTwister(0xC0FFEE))
    obs_dim, tsteps, ntrials = 3, 20, 5
    y = _null_make_y(rng, obs_dim, fill(tsteps, ntrials))

    null = AffineNullModel{Float64}(:intercept, obs_dim)
    fit!(null, y)

    # Closed-form MLE: d = mean over (t, n), R = residual scatter / n.
    Y = _null_stack(y)
    n = size(Y, 2)
    d_hat = vec(mean(Y; dims=2))
    Yc = Y .- d_hat
    R_hat = (Yc * Yc') ./ n

    @test null.d ≈ d_hat atol = 1e-10
    @test null.R ≈ R_hat atol = 1e-10
    @test isempty(null.F)
    @test size(null.D) == (obs_dim, 0)

    ref = _null_ref_ll(Y, ones(1, n), reshape(d_hat, obs_dim, 1), R_hat)
    @test loglikelihood(null, y) ≈ ref atol = 1e-8
end

# Scoring data other than the fit data is the plug-in (held-out) log-likelihood.
function test_null_heldout_ll_matches_plugin_gaussian(rng=MersenneTwister(1))
    obs_dim, tsteps, ntrials = 3, 15, 4
    y_train = _null_make_y(rng, obs_dim, fill(tsteps, ntrials))
    y_test = _null_make_y(rng, obs_dim, fill(tsteps, ntrials))

    null = AffineNullModel{Float64}(:intercept, obs_dim)
    fit!(null, y_train)

    Y_te = _null_stack(y_test)
    ref = _null_ref_ll(
        Y_te, ones(1, size(Y_te, 2)), reshape(null.d, obs_dim, 1), null.R
    )
    @test loglikelihood(null, y_test) ≈ ref atol = 1e-8
end

# Collapse identities
function test_null_inputs_collapses_to_intercept_when_no_inputs(rng=MersenneTwister(2))
    obs_dim, tsteps, ntrials = 3, 20, 4
    y = _null_make_y(rng, obs_dim, fill(tsteps, ntrials))
    zero_inputs = [zeros(0, tsteps) for _ in 1:ntrials]

    intercept = fit!(AffineNullModel{Float64}(:intercept, obs_dim), y)
    inputs = AffineNullModel{Float64}(:inputs, obs_dim; input_dim=0)
    fit!(inputs, y; inputs=zero_inputs)

    @test inputs.d ≈ intercept.d atol = 1e-10
    @test inputs.R ≈ intercept.R atol = 1e-10
    @test loglikelihood(inputs, y; inputs=zero_inputs) ≈ loglikelihood(intercept, y) atol =
        1e-8
end

# Signal recovery + LL ordering
function test_null_inputs_recovers_signal(rng=MersenneTwister(3))
    T = Float64
    obs_dim, v_dim, tsteps, ntrials = 3, 2, 40, 6
    D_true = randn(rng, T, obs_dim, v_dim)
    d_true = randn(rng, T, obs_dim)

    v = _null_make_inputs(rng, v_dim, fill(tsteps, ntrials))
    y = [d_true .+ D_true * vn .+ 0.1 .* randn(rng, T, obs_dim, tsteps) for vn in v]

    null = AffineNullModel{T}(:inputs, obs_dim; input_dim=v_dim)
    fit!(null, y; inputs=v)

    @test null.D ≈ D_true atol = 5e-2
    @test null.d ≈ d_true atol = 5e-2

    intercept = fit!(AffineNullModel{T}(:intercept, obs_dim), y)
    @test loglikelihood(null, y; inputs=v) > loglikelihood(intercept, y)
end

# VAR(1) parameter recovery
function test_null_var_recovers_true_F(rng=MersenneTwister(4))
    T = Float64
    obs_dim, tsteps, ntrials = 2, 200, 8
    F_true = T[0.6 0.2; -0.1 0.5]
    d_true = T[0.3, -0.2]
    R_true = Matrix{T}(0.05 * I, obs_dim, obs_dim)
    μ₀_true = T[1.0, -1.0]
    R₀_true = Matrix{T}(0.2 * I, obs_dim, obs_dim)

    y = _null_make_var_y(rng, F_true, d_true, R_true, μ₀_true, R₀_true, tsteps, ntrials)
    null = fit!(AffineNullModel{T}(:var, obs_dim), y)

    @test null.F ≈ F_true atol = 5e-2
    @test null.d ≈ d_true atol = 5e-2
    @test null.R ≈ R_true atol = 2e-2
    @test length(null.μ₀) == obs_dim
    @test size(null.R₀) == (obs_dim, obs_dim)

    # A VAR baseline must beat intercept-only on truly autocorrelated data.
    intercept = fit!(AffineNullModel{T}(:intercept, obs_dim), y)
    @test loglikelihood(null, y) > loglikelihood(intercept, y)
end

# Adding capacity cannot lower the in-sample plug-in log-likelihood.
function test_null_capacity_ordering(rng=MersenneTwister(5))
    T = Float64
    obs_dim, v_dim, tsteps, ntrials = 3, 2, 60, 5
    F_true = Matrix{T}(0.5 * I, obs_dim, obs_dim)
    y = _null_make_var_y(
        rng,
        F_true,
        zeros(T, obs_dim),
        Matrix{T}(0.1 * I, obs_dim, obs_dim),
        zeros(T, obs_dim),
        Matrix{T}(0.3 * I, obs_dim, obs_dim),
        tsteps,
        ntrials,
    )
    v = _null_make_inputs(rng, v_dim, fill(tsteps, ntrials))

    ll = Dict{Symbol,Float64}()
    for baseline in (:intercept, :inputs, :var, :var_inputs)
        null = AffineNullModel{T}(baseline, obs_dim; input_dim=v_dim)
        if null.input_dim > 0
            fit!(null, y; inputs=v)
            ll[baseline] = loglikelihood(null, y; inputs=v)
        else
            fit!(null, y)
            ll[baseline] = loglikelihood(null, y)
        end
    end

    @test ll[:inputs] >= ll[:intercept] - 1e-8
    @test ll[:var] >= ll[:intercept] - 1e-8
    @test ll[:var_inputs] >= ll[:var] - 1e-8
end

#=
Input alignment
=#

# `input_shift=1` scores `v_{t-1}`, `input_shift=0` scores `v_t`; a lagged input
# predating the trial is zero, mirroring the LDS's input-free `x_1`.
function test_null_input_shift_alignment(rng=MersenneTwister(6))
    T = Float64
    obs_dim, v_dim, tsteps, ntrials = 2, 2, 30, 4
    y = _null_make_y(rng, obs_dim, fill(tsteps, ntrials))
    v = _null_make_inputs(rng, v_dim, fill(tsteps, ntrials))

    lagged = AffineNullModel{T}(:inputs, obs_dim; input_dim=v_dim, input_shift=1)
    fit!(lagged, y; inputs=v)

    # Shifting the inputs by hand and fitting contemporaneously must agree.
    v_shifted = [hcat(zeros(T, v_dim, 1), vn[:, 1:(tsteps - 1)]) for vn in v]
    contemp = AffineNullModel{T}(:inputs, obs_dim; input_dim=v_dim, input_shift=0)
    fit!(contemp, y; inputs=v_shifted)

    @test lagged.D ≈ contemp.D atol = 1e-10
    @test lagged.d ≈ contemp.d atol = 1e-10
    @test loglikelihood(lagged, y; inputs=v) ≈
        loglikelihood(contemp, y; inputs=v_shifted) atol = 1e-8

    # The two alignments genuinely differ on the same inputs.
    plain = AffineNullModel{T}(:inputs, obs_dim; input_dim=v_dim, input_shift=0)
    fit!(plain, y; inputs=v)
    @test !isapprox(lagged.D, plain.D; rtol=1e-6)

    @test SSD._shifted_inputs(v[1], Val(1))[:, 1] == zeros(T, v_dim)
    @test SSD._shifted_inputs(v[1], Val(1))[:, 2:end] == v[1][:, 1:(end - 1)]
    @test SSD._shifted_inputs(v[1], Val(0)) === v[1]
end

#=
Ragged trials and rank-deficient initial covariance
=#

function test_null_ragged_trials(rng=MersenneTwister(7))
    T = Float64
    obs_dim = 2
    lengths = [12, 5, 20]
    y = _null_make_y(rng, obs_dim, lengths)

    intercept = fit!(AffineNullModel{T}(:intercept, obs_dim), y)
    @test isfinite(loglikelihood(intercept, y))
    @test nobs(intercept, y) == obs_dim * sum(lengths)

    var_null = fit!(AffineNullModel{T}(:var, obs_dim), y)
    @test isfinite(loglikelihood(var_null, y))

    # A length-1 trial informs only (μ₀, R₀), so it drops out of the VAR design.
    y_with_singleton = vcat(y, [randn(rng, T, obs_dim, 1)])
    var_singleton = fit!(AffineNullModel{T}(:var, obs_dim), y_with_singleton)
    @test isfinite(loglikelihood(var_singleton, y_with_singleton))
    @test size(var_singleton.F) == (obs_dim, obs_dim)
end

# With ntrials ≤ obs_dim and no `R₀_prior` the initial scatter is singular, so
# `R₀` falls back to `R` instead of throwing a `PosDefException`.
function test_null_single_trial_var_falls_back_to_R(rng=MersenneTwister(8))
    T = Float64
    obs_dim, tsteps = 3, 50
    y = _null_make_y(rng, obs_dim, [tsteps])

    null = fit!(AffineNullModel{T}(:var, obs_dim), y)
    @test null.R₀ ≈ null.R atol = 1e-12
    @test isfinite(loglikelihood(null, y))

    # An `R₀_prior` keeps the initial covariance estimable, so no fallback.
    prior = IWPrior(; Ψ=Matrix{T}(0.5 * I, obs_dim, obs_dim), ν=T(obs_dim + 3))
    with_prior = fit!(AffineNullModel{T}(:var, obs_dim; R₀_prior=prior), y)
    @test !isapprox(with_prior.R₀, with_prior.R; rtol=1e-6)
    @test isfinite(loglikelihood(with_prior, y))
end

#=
Prior contributions to the MAP objective
=#

# IW-prior LL shift identity
function test_null_R_prior_shifts_logmap_by_iw_term(rng=MersenneTwister(9))
    T = Float64
    obs_dim, tsteps, ntrials = 3, 20, 5
    y = _null_make_y(rng, obs_dim, fill(tsteps, ntrials))

    Ψ = Matrix{T}(0.1 * I, obs_dim, obs_dim)
    ν = T(obs_dim + 3)
    R_prior = IWPrior(; Ψ=Ψ, ν=ν)

    plain = fit!(AffineNullModel{T}(:intercept, obs_dim), y)
    @test SSD._null_logmap(plain, y) ≈ loglikelihood(plain, y) atol = 1e-10

    with = fit!(AffineNullModel{T}(:intercept, obs_dim; R_prior=R_prior), y)
    delta = SSD._null_logmap(with, y) - loglikelihood(with, y)
    expected = -0.5 * ((ν + obs_dim + 1) * logdet(with.R) + tr(with.R \ Ψ))
    @test delta ≈ expected atol = 1e-8
end

function test_null_mn_prior_shifts_logmap_by_mn_term(rng=MersenneTwister(10))
    T = Float64
    obs_dim, v_dim, tsteps, ntrials = 3, 2, 25, 4
    y = _null_make_y(rng, obs_dim, fill(tsteps, ntrials))
    v = _null_make_inputs(rng, v_dim, fill(tsteps, ntrials))

    M₀ = zeros(T, obs_dim, 1 + v_dim)
    Λ = Matrix{T}(0.5 * I, 1 + v_dim, 1 + v_dim)
    prior = MNPrior(; M₀=M₀, Λ=Λ)

    null = AffineNullModel{T}(:inputs, obs_dim; input_dim=v_dim, W_prior=prior)
    fit!(null, y; inputs=v)

    W = hcat(null.d, null.D)
    expected = -0.5 * tr(null.R \ ((W .- M₀) * Λ * (W .- M₀)'))
    delta = SSD._null_logmap(null, y; inputs=v) - loglikelihood(null, y; inputs=v)
    @test delta ≈ expected atol = 1e-8

    # Strong shrinkage pulls the MAP toward M₀ = 0.
    strong = AffineNullModel{T}(
        :inputs,
        obs_dim;
        input_dim=v_dim,
        W_prior=MNPrior(; M₀=M₀, Λ=Matrix{T}(1e6 * I, 1 + v_dim, 1 + v_dim)),
    )
    fit!(strong, y; inputs=v)
    @test norm(hcat(strong.d, strong.D)) < norm(W)
end

# All-priors path finite
function test_null_all_priors_active_returns_finite_lls(rng=MersenneTwister(11))
    T = Float64
    obs_dim, v_dim, tsteps, ntrials = 3, 2, 30, 6
    y = _null_make_var_y(
        rng,
        Matrix{T}(0.4 * I, obs_dim, obs_dim),
        zeros(T, obs_dim),
        Matrix{T}(0.1 * I, obs_dim, obs_dim),
        zeros(T, obs_dim),
        Matrix{T}(0.3 * I, obs_dim, obs_dim),
        tsteps,
        ntrials,
    )
    v = _null_make_inputs(rng, v_dim, fill(tsteps, ntrials))

    nregressors = obs_dim + 1 + v_dim
    null = AffineNullModel{T}(
        :var_inputs,
        obs_dim;
        input_dim=v_dim,
        W_prior=MNPrior(;
            M₀=zeros(T, obs_dim, nregressors),
            Λ=Matrix{T}(0.25 * I, nregressors, nregressors),
        ),
        R_prior=IWPrior(; Ψ=Matrix{T}(0.2 * I, obs_dim, obs_dim), ν=T(obs_dim + 4)),
        R₀_prior=IWPrior(; Ψ=Matrix{T}(0.4 * I, obs_dim, obs_dim), ν=T(obs_dim + 4)),
    )
    fit!(null, y; inputs=v)

    @test isfinite(loglikelihood(null, y; inputs=v))
    @test isfinite(SSD._null_logmap(null, y; inputs=v))
    @test isposdef(Symmetric(null.R))
    @test isposdef(Symmetric(null.R₀))
end

#=
Error paths
=#

function test_null_construction_validates_arguments()
    T = Float64
    @test_throws ArgumentError AffineNullModel{T}(0)
    @test_throws ArgumentError AffineNullModel{T}(2; input_dim=-1)
    @test_throws ArgumentError AffineNullModel{T}(2; input_shift=2)
    @test_throws ArgumentError AffineNullModel{T}(:nonsense, 2)

    # `input_dim` is forced to 0 for the input-free baselines.
    @test AffineNullModel{T}(:intercept, 2; input_dim=3).input_dim == 0
    @test AffineNullModel{T}(:var, 2; input_dim=3).input_dim == 0
    @test AffineNullModel{T}(:inputs, 2; input_dim=3).input_dim == 3
end

function test_null_input_shape_mismatch_throws(rng=MersenneTwister(12))
    T = Float64
    obs_dim, v_dim, tsteps, ntrials = 2, 3, 10, 4
    y = _null_make_y(rng, obs_dim, fill(tsteps, ntrials))
    null = AffineNullModel{T}(:inputs, obs_dim; input_dim=v_dim)

    # A baseline with an input block requires inputs.
    @test_throws ArgumentError fit!(null, y)

    wrong_rows = _null_make_inputs(rng, v_dim + 1, fill(tsteps, ntrials))
    @test_throws DimensionMismatchError fit!(null, y; inputs=wrong_rows)

    wrong_tsteps = _null_make_inputs(rng, v_dim, fill(tsteps + 1, ntrials))
    @test_throws DimensionMismatchError fit!(null, y; inputs=wrong_tsteps)

    wrong_ntrials = _null_make_inputs(rng, v_dim, fill(tsteps, ntrials - 1))
    @test_throws DimensionMismatchError fit!(null, y; inputs=wrong_ntrials)
end

function test_null_var_requires_tsteps_ge_2(rng=MersenneTwister(13))
    T = Float64
    obs_dim = 2
    y = [randn(rng, T, obs_dim, 1) for _ in 1:3]
    @test_throws ArgumentError fit!(AffineNullModel{T}(:var, obs_dim), y)

    # The no-lag baselines are happy with length-1 trials.
    @test isfinite(loglikelihood(fit!(AffineNullModel{T}(:intercept, obs_dim), y), y))
end

#=
StatsAPI model comparison against a fitted LDS
=#

# Gaussian LDS with oscillatory dynamics and optional ux / uy inputs.
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

_r2_rowstochastic(K) = (M = rand(K, K); M ./ sum(M; dims=2))
_r2_probvec(K) = (v = rand(K); v ./ sum(v))

function test_nobs_counts_scalar_observations(rng=MersenneTwister(14))
    obs_dim = 4
    lds = _r2_make_lds(rng; latent_dim=2, obs_dim=obs_dim)
    _, y = rand(rng, lds, [30, 12, 25])

    @test nobs(lds, y) == obs_dim * (30 + 12 + 25)

    _, y_rect = rand(rng, lds, fill(20, 5))
    @test nobs(lds, y_rect) == obs_dim * 20 * 5
    @test nobs(lds, cat(y_rect...; dims=3)) == obs_dim * 20 * 5
    @test nobs(lds, y_rect[1]) == obs_dim * 20
end

function test_nullloglikelihood_matches_intercept_baseline(rng=MersenneTwister(15))
    obs_dim = 4
    lds = _r2_make_lds(rng; latent_dim=2, obs_dim=obs_dim)
    _, y = rand(rng, lds, fill(30, 5))

    null = fit!(AffineNullModel{Float64}(:intercept, obs_dim), y)
    @test nullloglikelihood(lds, y) ≈ loglikelihood(null, y) atol = 1e-10
end

# The returned R² equals the closed-form value built from the two LLs and `nobs`.
function test_r2_cox_snell_formula(rng=MersenneTwister(16))
    obs_dim, tsteps, ntrials = 4, 30, 6
    lds = _r2_make_lds(rng; latent_dim=2, obs_dim=obs_dim)
    _, y = rand(rng, lds, fill(tsteps, ntrials))

    n = nobs(lds, y)
    @test n == obs_dim * tsteps * ntrials

    ll = loglikelihood(lds, y)
    ll₀ = nullloglikelihood(lds, y)
    @test r2(lds, y) ≈ 1 - exp(2 * (ll₀ - ll) / n) atol = 1e-12
    @test r2(lds, y, :CoxSnell) ≈ r2(lds, y) atol = 1e-12
end

function test_r2_variants(rng=MersenneTwister(17))
    obs_dim, tsteps, ntrials = 4, 30, 6
    lds = _r2_make_lds(rng; latent_dim=2, obs_dim=obs_dim)
    _, y = rand(rng, lds, fill(tsteps, ntrials))

    n = nobs(lds, y)
    ll = loglikelihood(lds, y)
    ll₀ = nullloglikelihood(lds, y)

    cox_snell = 1 - exp(2 * (ll₀ - ll) / n)
    @test r2(lds, y, :McFadden) ≈ 1 - ll / ll₀ atol = 1e-12
    @test r2(lds, y, :CoxSnell) ≈ cox_snell atol = 1e-12
    @test r2(lds, y, :Nagelkerke) ≈ cox_snell / (1 - exp(2 * ll₀ / n)) atol = 1e-12

    # Nagelkerke rescales Cox–Snell to a larger value on the same fit.
    @test r2(lds, y, :Nagelkerke) > r2(lds, y, :CoxSnell)
    @test_throws ArgumentError r2(lds, y, :nonsense)
end

function test_r2_baseline_selection(rng=MersenneTwister(18))
    T = Float64
    latent_dim, obs_dim, ux_dim = 2, 4, 2
    tsteps, ntrials = 40, 6
    lds = _r2_make_lds(rng; latent_dim=latent_dim, obs_dim=obs_dim, ux_dim=ux_dim)
    ux = [randn(rng, T, ux_dim, tsteps) for _ in 1:ntrials]
    _, y = rand(rng, lds, fill(tsteps, ntrials); ux=ux)

    values = Dict(
        baseline => r2(lds, y; ux=ux, null=baseline) for
        baseline in (:intercept, :inputs, :var, :var_inputs)
    )
    for (_, value) in values
        @test isfinite(value)
    end

    # A richer baseline is harder to beat, so R² cannot increase with capacity.
    @test values[:inputs] <= values[:intercept] + 1e-8
    @test values[:var] <= values[:intercept] + 1e-8
    @test values[:var_inputs] <= values[:var] + 1e-8

    @test_throws ArgumentError r2(lds, y; ux=ux, null=:nonsense)
end

# The two input channels are distinct: `:ux` feeds the baseline the dynamics
# inputs (lagged), `:uy` the observation inputs (contemporaneous).
function test_r2_null_inputs_ux_vs_uy(rng=MersenneTwister(19))
    T = Float64
    latent_dim, obs_dim = 2, 4
    ux_dim, uy_dim = 2, 3
    tsteps, ntrials = 40, 6
    lds = _r2_make_lds(
        rng; latent_dim=latent_dim, obs_dim=obs_dim, ux_dim=ux_dim, uy_dim=uy_dim
    )
    ux = [randn(rng, T, ux_dim, tsteps) for _ in 1:ntrials]
    uy = [randn(rng, T, uy_dim, tsteps) for _ in 1:ntrials]
    _, y = rand(rng, lds, fill(tsteps, ntrials); ux=ux, uy=uy)

    r2_ux = r2(lds, y; ux=ux, uy=uy, null=:inputs, null_inputs=:ux)
    r2_uy = r2(lds, y; ux=ux, uy=uy, null=:inputs, null_inputs=:uy)
    @test isfinite(r2_ux) && isfinite(r2_uy)
    @test r2_ux != r2_uy

    # The `:intercept` baseline has no input block, so the channel is inert.
    @test r2(lds, y; ux=ux, uy=uy, null=:intercept, null_inputs=:ux) ≈
        r2(lds, y; ux=ux, uy=uy, null=:intercept, null_inputs=:uy) atol = 1e-12

    @test_throws ArgumentError r2(lds, y; ux=ux, uy=uy, null_inputs=:nope)
end

# The baseline inherits the LDS's observation-noise prior by default.
function test_r2_forwards_R_prior(rng=MersenneTwister(20))
    T = Float64
    obs_dim, tsteps, ntrials = 4, 40, 8
    Rp = IWPrior(; Ψ=Matrix{T}(0.5 * I, obs_dim, obs_dim), ν=T(obs_dim + 5))
    lds = _r2_make_lds(rng; latent_dim=2, obs_dim=obs_dim, R_prior=Rp)
    _, y = rand(rng, lds, fill(tsteps, ntrials))

    @test r2(lds, y) ≈ r2(lds, y; R_prior=Rp) atol = 1e-12
    @test r2(lds, y) != r2(lds, y; R_prior=nothing)
    @test nullloglikelihood(lds, y) ≈ nullloglikelihood(lds, y; R_prior=Rp) atol = 1e-12
end

# The data-generating LDS out-predicts every baseline on held-out data. Held-out
# R² is the documented two-call recipe: fit the baseline on train, score test.
function test_r2_ground_truth_beats_null_heldout(rng=MersenneTwister(21))
    T = Float64
    latent_dim, obs_dim = 2, 6
    tsteps, ntrials = 60, 10
    lds = _r2_make_lds(rng; latent_dim=latent_dim, obs_dim=obs_dim)

    _, y_train = rand(rng, lds, fill(tsteps, ntrials))
    _, y_test = rand(rng, lds, fill(tsteps, ntrials))

    ll_test = loglikelihood(lds, y_test)
    n_test = nobs(lds, y_test)

    for baseline in (:intercept, :var)
        null = fit!(AffineNullModel{T}(baseline, obs_dim), y_train)
        ll₀ = loglikelihood(null, y_test)
        r2_heldout = 1 - exp(2 * (ll₀ - ll_test) / n_test)
        @test ll_test > ll₀
        @test r2_heldout > 0
    end
end

# `r2` / `nullloglikelihood` are defined only for the Gaussian LDS: Poisson
# observations and the SLDS type fall through to a `MethodError`.
function test_r2_rejects_plds_and_slds(rng=MersenneTwister(22))
    T = Float64
    latent_dim, obs_dim = 2, 3
    y = [randn(rng, T, obs_dim, 10) for _ in 1:4]
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
    @test_throws MethodError r2(plds, y)
    @test_throws MethodError nullloglikelihood(plds, y)

    glds = LinearDynamicalSystem(
        sm,
        GaussianObservationModel(;
            C=randn(rng, T, obs_dim, latent_dim),
            R=Matrix{T}(I, obs_dim, obs_dim),
            d=zeros(T, obs_dim),
        ),
    )
    slds = SLDS(; A=_r2_rowstochastic(2), πₖ=_r2_probvec(2), LDSs=fill(glds, 2))
    @test_throws MethodError r2(slds, y)
end
