# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Construct one basis of each concrete type, all with the same number of
# functions, suitable for the generic tests that work over any basis.
function _example_bases(K::Int)
    return [
        BSpline(K; order=4),
        Fourier(K),
        RaisedCosineLinear(K),
        RaisedCosineLog(K),
        Polynomial(K),
    ]
end

# Build a Data struct with pre-allocated `u` (or `d`) sized for P predictors
# × K basis functions × tsteps × ntrials. Pass `P=0` for an empty
# `trial_pred` (apply! then uses the default single all-ones predictor, so
# the target field is sized as if `P=1`).
function _make_data(
    ::Type{T}, K::Int, tsteps::Int, ntrials::Int, P::Int; field::Symbol=:u
) where {T<:Real}
    P_eff = max(P, 1)
    y = randn(MersenneTwister(0), T, 1, tsteps, ntrials)
    u = if field == :u
        zeros(T, P_eff * K, tsteps, ntrials)
    else
        zeros(T, 1, tsteps, ntrials)
    end
    d = if field == :d
        zeros(T, P_eff * K, tsteps, ntrials)
    else
        zeros(T, 1, tsteps, ntrials)
    end
    trial_pred = P == 0 ? Matrix{T}(undef, 0, 0) : ones(T, ntrials, P)
    return Data(; y=y, u=u, d=d, trial_pred=trial_pred)
end

# ---------------------------------------------------------------------------
# Generic apply!/get_penalty
# ---------------------------------------------------------------------------

function test_apply_shape_and_kron_structure()
    T = Float64
    tsteps, ntrials, K, P = 30, 4, 6, 3
    rng = MersenneTwister(123)
    trial_pred = randn(rng, T, ntrials, P)

    for basis in _example_bases(K)
        y = randn(rng, T, 2, tsteps, ntrials)
        u = zeros(T, P * K, tsteps, ntrials)
        d = zeros(T, 1, tsteps, ntrials)
        data = Data(; y=y, u=u, d=d, trial_pred=trial_pred)

        apply!(data, basis; target=:u)
        @test size(data.u) == (P * K, tsteps, ntrials)
        @test eltype(data.u) === T

        # Per-predictor block within a single trial = trial_pred[n,p] × B'.
        coeff_ref = trial_pred[1, 1]
        @assert abs(coeff_ref) > 1e-6
        Bt_recovered = data.u[1:K, :, 1] ./ coeff_ref
        for n in 1:ntrials, p in 1:P
            rows = ((p - 1) * K + 1):(p * K)
            @test data.u[rows, :, n] ≈ trial_pred[n, p] .* Bt_recovered atol = 1e-10
        end
    end
end

function test_apply_default_trial_pred_broadcasts_across_trials()
    T = Float64
    tsteps, ntrials, K = 25, 5, 5
    for basis in _example_bases(K)
        data = _make_data(T, K, tsteps, ntrials, 0)  # empty trial_pred
        apply!(data, basis)
        for n in 2:ntrials
            @test data.u[:, :, n] ≈ data.u[:, :, 1] atol = 1e-14
        end
    end
end

function test_apply_target_d()
    T = Float64
    tsteps, ntrials, K, P = 20, 3, 4, 2
    basis = Fourier(K)
    data = _make_data(T, K, tsteps, ntrials, P; field=:d)
    @test size(data.d) == (P * K, tsteps, ntrials)
    apply!(data, basis; target=:d)
    @test !all(iszero, data.d)
    @test all(iszero, data.u)  # u was 1-row, untouched
end

function test_apply_size_mismatch_throws()
    T = Float64
    tsteps, ntrials, K, P = 20, 2, 4, 2
    basis = BSpline(K)
    y = randn(T, 1, tsteps, ntrials)
    # P*K = 8, but allocate 5 rows.
    u = zeros(T, 5, tsteps, ntrials)
    d = zeros(T, 1, tsteps, ntrials)
    data = Data(; y=y, u=u, d=d, trial_pred=ones(T, ntrials, P))
    @test_throws DimensionMismatch apply!(data, basis)
end

function test_apply_trial_pred_row_mismatch_throws()
    T = Float64
    tsteps, ntrials, K = 20, 4, 4
    basis = BSpline(K)
    y = randn(T, 1, tsteps, ntrials)
    u = zeros(T, K, tsteps, ntrials)
    d = zeros(T, 1, tsteps, ntrials)
    data = Data(; y=y, u=u, d=d, trial_pred=ones(T, 3, 1))  # 3 ≠ 4
    @test_throws DimensionMismatch apply!(data, basis)
end

function test_apply_invalid_target_throws()
    T = Float64
    data = _make_data(T, 4, 20, 2, 1)
    @test_throws ArgumentError apply!(data, BSpline(4); target=:bogus)
end

function test_get_penalty_data_convenience_overload()
    T = Float64
    tsteps, ntrials, K, P = 30, 2, 5, 3
    basis = BSpline(K)
    data = _make_data(T, K, tsteps, ntrials, P)

    Ω_direct = get_penalty(basis, tsteps; P=P, eltype=T)
    Ω_data = get_penalty(data, basis)
    @test size(Ω_data) == (P * K, P * K)
    @test eltype(Ω_data) === T
    @test Ω_data ≈ Ω_direct
end

function test_get_penalty_kron_block_structure()
    T = Float64
    tsteps, K, P = 30, 6, 4
    basis = BSpline(K)
    Ω = get_penalty(basis, tsteps; P=P, eltype=T)
    Ω_single = get_penalty(basis, tsteps; P=1, eltype=T)

    @test size(Ω) == (P * K, P * K)
    @test size(Ω_single) == (K, K)
    for i in 1:P, j in 1:P
        rows = ((i - 1) * K + 1):(i * K)
        cols = ((j - 1) * K + 1):(j * K)
        block = Ω[rows, cols]
        if i == j
            @test block ≈ Ω_single atol = 1e-12
        else
            @test all(iszero, block)
        end
    end
end

# ---------------------------------------------------------------------------
# BSpline specifics
# ---------------------------------------------------------------------------

function test_bspline_partition_of_unity()
    T = Float64
    tsteps, K = 25, 6
    basis = BSpline(K)
    B = evaluate_basis(basis, collect(T(1):T(tsteps)))
    row_sums = vec(sum(B; dims=2))
    @test all(abs.(row_sums .- one(T)) .< 1e-10)
end

function test_bspline_manual_knots()
    T = Float64
    tsteps, K, order = 40, 8, 4
    knots = collect(range(T(1), T(tsteps); length=K - order + 2))
    basis = BSpline(K; order=order, knots=knots)
    B = evaluate_basis(basis, collect(T(1):T(tsteps)))
    @test size(B) == (tsteps, K)
    row_sums = vec(sum(B; dims=2))
    @test all(abs.(row_sums .- one(T)) .< 1e-10)
end

function test_bspline_manual_knots_wrong_length_throws()
    bad_knots = collect(1.0:5.0)  # need K-order+2 = 4; pass 5
    basis = BSpline(6; order=4, knots=bad_knots)
    @test_throws ArgumentError evaluate_basis(basis, collect(1.0:20.0))
end

function test_bspline_invalid_args()
    @test_throws ArgumentError BSpline(3; order=4)        # num_bases < order
    @test_throws ArgumentError BSpline(4; order=0)        # order < 1
    @test_throws ArgumentError BSpline(4; knots=:bogus)   # bad knots symbol
end

# ---------------------------------------------------------------------------
# Fourier specifics
# ---------------------------------------------------------------------------

function test_fourier_basis_values()
    T = Float64
    tsteps, K = 32, 5
    basis = Fourier(K)
    ts = collect(T(1):T(tsteps))
    Φ = evaluate_basis(basis, ts)
    @test size(Φ) == (tsteps, K)

    # Column 1: DC, all ones.
    @test all(Φ[:, 1] .≈ one(T))

    # Column 2/3 = cos/sin at freq 1; column 4/5 = cos/sin at freq 2.
    Tp = T(tsteps)
    @test Φ[:, 2] ≈ cos.(T(2π) .* ts ./ Tp)
    @test Φ[:, 3] ≈ sin.(T(2π) .* ts ./ Tp)
    @test Φ[:, 4] ≈ cos.(T(2π) .* T(2) .* ts ./ Tp)
    @test Φ[:, 5] ≈ sin.(T(2π) .* T(2) .* ts ./ Tp)
end

function test_fourier_analytic_penalty_dc_nullspace()
    T = Float64
    tsteps, K = 50, 7
    basis = Fourier(K)
    Ω = get_penalty(basis, tsteps; eltype=T, use_analytic=true)

    @test issymmetric(Ω)
    @test isdiag(Ω)
    # DC entry is zero.
    @test Ω[1, 1] == zero(T)
    # Frequency 1 entries (cos, sin) are equal and positive.
    @test Ω[2, 2] > 0
    @test Ω[2, 2] ≈ Ω[3, 3] atol = 1e-14
    # Frequency 2 > frequency 1 (higher curvature for higher freq).
    @test Ω[4, 4] > Ω[2, 2]
end

function test_fourier_analytic_vs_discrete_ordering()
    # The analytic-vs-discrete penalties differ by a basis-dependent scale,
    # but both should rank-order the diagonal the same way (higher freq →
    # larger penalty).
    T = Float64
    tsteps, K = 60, 7
    basis = Fourier(K)
    Ω_an = get_penalty(basis, tsteps; eltype=T, use_analytic=true)
    Ω_di = get_penalty(basis, tsteps; eltype=T, use_analytic=false)

    diag_an = diag(Ω_an)
    diag_di = diag(Ω_di)
    @test diag_an[1] == 0
    @test all(diag_an[2:end] .> 0)
    @test all(diag_di[2:end] .> 0)
    # Both penalties should rank cos₂ above cos₁.
    @test diag_an[4] > diag_an[2]
    @test diag_di[4] > diag_di[2]
end

function test_fourier_period_kwarg()
    T = Float64
    tsteps, K = 40, 3
    period = 100.0  # explicit override
    basis = Fourier(K; period=period)
    Φ = evaluate_basis(basis, collect(T(1):T(tsteps)))
    @test Φ[:, 2] ≈ cos.(T(2π) .* collect(T(1):T(tsteps)) ./ T(period))
end

# ---------------------------------------------------------------------------
# RaisedCosine specifics
# ---------------------------------------------------------------------------

function test_raised_cosine_linear_bumps()
    T = Float64
    tsteps, K = 50, 6
    basis = RaisedCosineLinear(K; width_factor=2.0)
    ts = collect(T(1):T(tsteps))
    Φ = evaluate_basis(basis, ts)

    @test size(Φ) == (tsteps, K)
    @test all(Φ .>= 0)
    @test all(Φ .<= 1.0 + 1e-12)

    # At each centre c_k, basis k attains the maximum value 1.
    centers = range(T(1), T(tsteps); length=K)
    for k in 1:K
        c = centers[k]
        # The nearest grid point to c should give Φ near 1 for basis k.
        i = argmin(abs.(ts .- c))
        @test Φ[i, k] > 0.99
    end
end

function test_raised_cosine_log_bumps()
    T = Float64
    tsteps, K = 50, 5
    basis = RaisedCosineLog(K; width_factor=2.0, offset=1.0)
    ts = collect(T(1):T(tsteps))
    Φ = evaluate_basis(basis, ts)

    @test size(Φ) == (tsteps, K)
    @test all(Φ .>= 0)
    @test all(Φ .<= 1.0 + 1e-12)

    # Log spacing → bumps narrower near the start, wider near the end.
    # First basis has a narrower support (in t) than the last basis.
    support_first = count(Φ[:, 1] .> 1e-6)
    support_last = count(Φ[:, end] .> 1e-6)
    @test support_first < support_last
end

function test_raised_cosine_invalid_args()
    @test_throws ArgumentError RaisedCosineLinear(1)
    @test_throws ArgumentError RaisedCosineLinear(4; width_factor=-1.0)
    @test_throws ArgumentError RaisedCosineLog(1)
    @test_throws ArgumentError RaisedCosineLog(4; offset=0.0)
end

# ---------------------------------------------------------------------------
# Polynomial specifics
# ---------------------------------------------------------------------------

function test_polynomial_values()
    T = Float64
    tsteps, K = 20, 4
    basis = Polynomial(K)
    ts = collect(T(1):T(tsteps))
    Φ = evaluate_basis(basis, ts)

    @test size(Φ) == (tsteps, K)
    @test all(Φ[:, 1] .≈ one(T))
    x = (ts .- one(T)) ./ T(tsteps - 1)
    @test Φ[:, 2] ≈ x
    @test Φ[:, 3] ≈ x .^ 2
    @test Φ[:, 4] ≈ x .^ 3
end

function test_polynomial_invalid_args()
    @test_throws ArgumentError Polynomial(0)
end

# ---------------------------------------------------------------------------
# Curvature penalty properties
# ---------------------------------------------------------------------------

function test_curvature_penalty_symmetric_psd_all_bases()
    T = Float64
    tsteps, K = 40, 6
    for basis in _example_bases(K)
        Ω = get_penalty(basis, tsteps; P=1, eltype=T)
        @test issymmetric(Ω)
        eigs = eigvals(Symmetric(Ω))
        @test all(eigs .>= -1e-8)
    end
end

function test_curvature_penalty_nullspace_bspline()
    # B-spline basis has partition of unity → all-ones coefficient vector
    # gives a constant function in time, which is in the curvature nullspace.
    T = Float64
    tsteps, K = 40, 7
    Ω = get_penalty(BSpline(K), tsteps; eltype=T)
    @test maximum(abs.(Ω * ones(T, K))) < 1e-8
end

function test_curvature_penalty_nullspace_polynomial()
    # Polynomial basis: constant (e_1) and linear (e_2) are both in nullspace.
    T = Float64
    tsteps, K = 40, 5
    Ω = get_penalty(Polynomial(K), tsteps; eltype=T)
    e1 = zeros(T, K)
    e1[1] = one(T)
    e2 = zeros(T, K)
    e2[2] = one(T)
    @test maximum(abs.(Ω * e1)) < 1e-8
    @test maximum(abs.(Ω * e2)) < 1e-8
end

# ---------------------------------------------------------------------------
# Type preservation
# ---------------------------------------------------------------------------

function test_float32_type_preservation_all_bases()
    T = Float32
    tsteps, ntrials, K, P = 25, 2, 4, 2
    trial_pred = randn(MersenneTwister(7), T, ntrials, P)
    for basis in _example_bases(K)
        y = randn(MersenneTwister(8), T, 1, tsteps, ntrials)
        u = zeros(T, P * K, tsteps, ntrials)
        d = zeros(T, 1, tsteps, ntrials)
        data = Data(; y=y, u=u, d=d, trial_pred=trial_pred)
        apply!(data, basis)
        Ω = get_penalty(data, basis)
        @test eltype(data.u) === T
        @test eltype(Ω) === T
    end
end
