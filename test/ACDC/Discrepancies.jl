# Tests for ACDC discrepancy measures

function test_sample_categorical_deterministic()
    # Deterministic case: all probability on one component
    @test ACDC._sample_categorical([1.0, 0.0, 0.0]) == 1
    @test ACDC._sample_categorical([0.0, 1.0, 0.0]) == 2
    @test ACDC._sample_categorical([0.0, 0.0, 1.0]) == 3
end

function test_sample_categorical_valid_indices()
    Random.seed!(42)
    p = [0.2, 0.3, 0.5]
    for _ in 1:100
        idx = ACDC._sample_categorical(p)
        @test idx >= 1 && idx <= 3
    end
end

function test_sample_categorical_frequencies()
    Random.seed!(42)
    p = [0.2, 0.3, 0.5]
    counts = zeros(Int, 3)
    n_samples = 10000
    for _ in 1:n_samples
        counts[ACDC._sample_categorical(p)] += 1
    end
    freqs = counts ./ n_samples
    @test isapprox(freqs[1], 0.2; atol=0.03)
    @test isapprox(freqs[2], 0.3; atol=0.03)
    @test isapprox(freqs[3], 0.5; atol=0.03)
end

function test_normal_cdf_known_values()
    @test isapprox(ACDC._normal_cdf(0.0), 0.5; atol=1e-10)
    @test isapprox(ACDC._normal_cdf(-Inf), 0.0; atol=1e-10)
    @test isapprox(ACDC._normal_cdf(Inf), 1.0; atol=1e-10)
end

function test_normal_cdf_symmetry()
    @test isapprox(ACDC._normal_cdf(1.0) + ACDC._normal_cdf(-1.0), 1.0; atol=1e-10)
    @test isapprox(ACDC._normal_cdf(2.0) + ACDC._normal_cdf(-2.0), 1.0; atol=1e-10)
end

function test_normal_cdf_monotonicity()
    @test ACDC._normal_cdf(-2.0) < ACDC._normal_cdf(-1.0) < ACDC._normal_cdf(0.0)
    @test ACDC._normal_cdf(0.0) < ACDC._normal_cdf(1.0) < ACDC._normal_cdf(2.0)
end

function test_normal_cdf_bounds()
    for x in [-3.0, -1.0, 0.0, 1.0, 3.0]
        val = ACDC._normal_cdf(x)
        @test val >= 0.0 && val <= 1.0
    end
end

function test_normal_cdf_type_stability()
    @test ACDC._normal_cdf(0.0f0) isa Float32
    @test ACDC._normal_cdf(0.0) isa Float64
end

function test_poisson_cdf_randomized_bounds()
    Random.seed!(42)
    for λ in [0.1, 1.0, 5.0, 10.0]
        for x in [0, 1, 2, 5, 10]
            for _ in 1:10
                val = ACDC._poisson_cdf_randomized(x, λ)
                @test val >= 0.0 && val <= 1.0
            end
        end
    end
end

function test_poisson_cdf_randomized_zero_rate()
    Random.seed!(42)
    vals = [ACDC._poisson_cdf_randomized(0, 0.0) for _ in 1:100]
    @test all(0.0 .<= vals .<= 1.0)
end

function test_poisson_cdf_randomized_uniformity()
    Random.seed!(42)
    λ = 3.0
    n_samples = 5000
    uniform_samples = Float64[]
    for _ in 1:n_samples
        x = rand(Poisson(λ))
        push!(uniform_samples, ACDC._poisson_cdf_randomized(x, λ))
    end
    sorted = sort(uniform_samples)
    ks_stat = maximum(abs.(sorted .- collect(1:n_samples) ./ n_samples))
    @test ks_stat < 0.02
end

function test_bernoulli_cdf_randomized_bounds()
    Random.seed!(42)
    for p in [0.1, 0.5, 0.9]
        for x in [0, 1]
            for _ in 1:10
                val = ACDC._bernoulli_cdf_randomized(Float64(x), p)
                @test val >= 0.0 && val <= 1.0
            end
        end
    end
end

function test_bernoulli_cdf_randomized_ranges()
    Random.seed!(42)
    p = 0.7
    vals_0 = [ACDC._bernoulli_cdf_randomized(0.0, p) for _ in 1:100]
    @test all(0.0 .<= vals_0 .<= (1 - p + 0.01))

    vals_1 = [ACDC._bernoulli_cdf_randomized(1.0, p) for _ in 1:100]
    @test all((1 - p - 0.01) .<= vals_1 .<= 1.0)
end

function test_bernoulli_cdf_randomized_uniformity()
    Random.seed!(42)
    p = 0.4
    n_samples = 5000
    uniform_samples = Float64[]
    for _ in 1:n_samples
        x = rand() < p ? 1.0 : 0.0
        push!(uniform_samples, ACDC._bernoulli_cdf_randomized(x, p))
    end
    sorted = sort(uniform_samples)
    ks_stat = maximum(abs.(sorted .- collect(1:n_samples) ./ n_samples))
    @test ks_stat < 0.02
end

function test_deconvolve_gaussian_sum_constraint()
    Random.seed!(42)
    x_sum = 5.0
    μs = [1.0, 2.0, 1.5]
    σs = [0.5, 0.5, 0.5]

    for _ in 1:100
        ys = ACDC._deconvolve_gaussian_sum(x_sum, μs, σs)
        @test isapprox(sum(ys), x_sum; atol=1e-10)
    end
end

function test_deconvolve_gaussian_sum_scalar_sigma()
    Random.seed!(42)
    x_sum = 5.0
    μs = [1.0, 2.0, 1.5]
    ys = ACDC._deconvolve_gaussian_sum(x_sum, μs, 0.5)
    @test isapprox(sum(ys), x_sum; atol=1e-10)
end

function test_deconvolve_gaussian_sum_length()
    @test length(ACDC._deconvolve_gaussian_sum(5.0, [1.0, 2.0], [0.5, 0.5])) == 2
    @test length(ACDC._deconvolve_gaussian_sum(5.0, [1.0, 2.0, 3.0, 4.0], 0.5)) == 4
end

function test_KS_discrepancy_uniform_small()
    Random.seed!(42)
    N = 2000
    D = 3
    uniform_samples = rand(D, N)
    ks = KSDiscrepancy()
    ks_val = compute_discrepancy(ks, uniform_samples)
    @test ks_val >= 0.0
    @test ks_val < 0.05
end

function test_KS_discrepancy_1D()
    Random.seed!(42)
    uniform_1d = rand(1, 1000)
    ks = KSDiscrepancy()
    @test compute_discrepancy(ks, uniform_1d) < 0.05

    # Concentrated at 0.5 should have large KS
    concentrated = fill(0.5, 1, 1000) .+ randn(1, 1000) .* 0.01
    concentrated = clamp.(concentrated, 0.0, 1.0)
    @test compute_discrepancy(ks, concentrated) > 0.4
end

function test_Wasserstein_discrepancy_uniform_small()
    Random.seed!(42)
    N = 2000
    D = 3
    uniform_samples = rand(D, N)
    wass = WassersteinDiscrepancy()
    wass_val = compute_discrepancy(wass, uniform_samples)
    @test wass_val >= 0.0
    @test wass_val < 0.1
end

function test_Wasserstein_discrepancy_1D()
    Random.seed!(42)
    uniform_1d = rand(1, 1000)
    wass1 = WassersteinDiscrepancy(; p=1)
    @test compute_discrepancy(wass1, uniform_1d) < 0.05

    wass2 = WassersteinDiscrepancy(; p=2)
    @test compute_discrepancy(wass2, uniform_1d) < 0.05
end

function test_SquaredError_discrepancy_uniform_small()
    Random.seed!(42)
    N = 2000
    D = 3
    uniform_samples = rand(D, N)
    se = SquaredErrorDiscrepancy()
    se_val = compute_discrepancy(se, uniform_samples)
    @test se_val >= 0.0
    @test se_val < 0.01
end

function test_MMD_discrepancy_uniform_small()
    Random.seed!(42)
    N = 2000
    D = 3
    uniform_samples = rand(D, N)
    mmd = MMDDiscrepancy()
    mmd_val = compute_discrepancy(mmd, uniform_samples)
    @test mmd_val >= -0.01  # Can be slightly negative due to unbiased estimator
    @test mmd_val < 0.05
end

function test_MMD_discrepancy_block_strategy()
    Random.seed!(42)
    N = 12000
    D = 2
    samples = rand(D, N)
    mmd = MMDDiscrepancy(; block_size=5000)
    val = compute_discrepancy(mmd, samples)
    @test val >= -0.01
    @test val < 0.05
end

function test_KL_discrepancy_uniform_small()
    Random.seed!(42)
    N = 2000
    D = 3
    uniform_samples = rand(D, N)
    kl = KLDiscrepancy()
    kl_val = compute_discrepancy(kl, uniform_samples)
    @test kl_val >= 0.0
    @test kl_val < 0.5
end

function test_discrepancies_nonuniform_larger()
    Random.seed!(42)
    N = 2000
    D = 2

    non_uniform = rand(Beta(2, 5), D, N)
    uniform_samples = rand(D, N)

    kl = KLDiscrepancy()
    @test compute_discrepancy(kl, non_uniform) > compute_discrepancy(kl, uniform_samples)

    ks = KSDiscrepancy()
    @test compute_discrepancy(ks, non_uniform) > compute_discrepancy(ks, uniform_samples)

    wass = WassersteinDiscrepancy()
    @test compute_discrepancy(wass, non_uniform) >
        compute_discrepancy(wass, uniform_samples)

    se = SquaredErrorDiscrepancy()
    @test compute_discrepancy(se, non_uniform) > compute_discrepancy(se, uniform_samples)

    mmd = MMDDiscrepancy()
    @test compute_discrepancy(mmd, non_uniform) > compute_discrepancy(mmd, uniform_samples)
end

function test_discrepancy_type_stability()
    samples_f32 = rand(Float32, 2, 100)
    samples_f64 = rand(Float64, 2, 100)

    mmd32 = MMDDiscrepancy{Float32}()
    mmd64 = MMDDiscrepancy{Float64}()
    @test compute_discrepancy(mmd32, samples_f32) isa Float32
    @test compute_discrepancy(mmd64, samples_f64) isa Float64

    ks32 = KSDiscrepancy{Float32}()
    ks64 = KSDiscrepancy{Float64}()
    @test compute_discrepancy(ks32, samples_f32) isa Float32
    @test compute_discrepancy(ks64, samples_f64) isa Float64
end

function test_discrepancy_single_dimension()
    Random.seed!(42)
    samples_1d = rand(1, 500)

    @test compute_discrepancy(KSDiscrepancy(), samples_1d) >= 0
    @test compute_discrepancy(WassersteinDiscrepancy(), samples_1d) >= 0
    @test compute_discrepancy(MMDDiscrepancy(), samples_1d) >= -0.01
    @test compute_discrepancy(SquaredErrorDiscrepancy(), samples_1d) >= 0
end

function test_discrepancy_high_dimension()
    Random.seed!(42)
    D = 50
    N = 200
    samples = rand(D, N)

    @test compute_discrepancy(MMDDiscrepancy(), samples) isa Float64
    @test compute_discrepancy(KSDiscrepancy(), samples) isa Float64
    @test compute_discrepancy(SquaredErrorDiscrepancy(), samples) isa Float64
end

function test_KL_discrepancy_small_sample()
    kl = KLDiscrepancy(; k_neighbors=5)
    small_samples = rand(2, 4)  # N < k + 1
    @test compute_discrepancy(kl, small_samples) == Inf
end
