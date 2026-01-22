# Tests for ACDC loss function and model selection

function test_acdc_loss_basic()
    discs = [0.1, 0.2, 0.3]
    usage = [0.3, 0.4, 0.3]
    result = ACDCResult(3, discs, usage)

    # ρ = 0: all discrepancies contribute
    @test isapprox(acdc_loss(result, 0.0), 0.1 + 0.2 + 0.3; atol=1e-10)

    # ρ = 0.15: only 0.2 and 0.3 contribute
    @test isapprox(acdc_loss(result, 0.15), (0.2 - 0.15) + (0.3 - 0.15); atol=1e-10)

    # ρ = 0.25: only 0.3 contributes
    @test isapprox(acdc_loss(result, 0.25), 0.3 - 0.25; atol=1e-10)

    # ρ >= max(discs): loss is 0
    @test isapprox(acdc_loss(result, 0.3), 0.0; atol=1e-10)
    @test isapprox(acdc_loss(result, 0.5), 0.0; atol=1e-10)
end

function test_acdc_loss_nonnegative()
    discs = [0.1, 0.2, 0.3]
    usage = [0.3, 0.4, 0.3]
    result = ACDCResult(3, discs, usage)

    for ρ in 0.0:0.05:0.5
        @test acdc_loss(result, ρ) >= 0.0
    end
end

function test_acdc_loss_single_component()
    result = ACDCResult(1, [0.1], [1.0])
    @test acdc_loss(result, 0.0) == 0.1
    @test acdc_loss(result, 0.2) == 0.0
end

function test_acdc_select_high_rho()
    result_K2 = ACDCResult(2, [0.5, 0.5], [0.5, 0.5])
    result_K3 = ACDCResult(3, [0.1, 0.1, 0.1], [0.33, 0.33, 0.34])
    result_K4 = ACDCResult(4, [0.08, 0.08, 0.08, 0.08], [0.25, 0.25, 0.25, 0.25])

    results = [result_K2, result_K3, result_K4]

    # At high ρ, prefer smaller K (all losses are 0)
    @test acdc_select(results, 1.0) == 2
end

function test_acdc_select_low_rho()
    result_K2 = ACDCResult(2, [0.5, 0.5], [0.5, 0.5])
    result_K3 = ACDCResult(3, [0.1, 0.1, 0.1], [0.33, 0.33, 0.34])
    result_K4 = ACDCResult(4, [0.08, 0.08, 0.08, 0.08], [0.25, 0.25, 0.25, 0.25])

    results = [result_K2, result_K3, result_K4]

    # At low ρ, prefer K with lower loss
    @test acdc_select(results, 0.05) in [3, 4]
end

function test_acdc_select_tie_breaking()
    result_K2 = ACDCResult(2, [0.5, 0.5], [0.5, 0.5])
    result_K3 = ACDCResult(3, [0.1, 0.1, 0.1], [0.33, 0.33, 0.34])
    result_K4 = ACDCResult(4, [0.08, 0.08, 0.08, 0.08], [0.25, 0.25, 0.25, 0.25])

    results = [result_K2, result_K3, result_K4]

    # At ρ = 0.2, K=3 and K=4 have 0 loss, prefer smaller
    @test acdc_select(results, 0.2) == 3
end

function test_get_critical_rho_values()
    result_K2 = ACDCResult(2, [0.1, 0.3], [0.5, 0.5])
    result_K3 = ACDCResult(3, [0.2, 0.3, 0.4], [0.33, 0.33, 0.34])

    results = [result_K2, result_K3]
    critical = get_critical_rho_values(results)

    # Should contain all unique discrepancy values, sorted
    @test critical == [0.1, 0.2, 0.3, 0.4]
    @test length(critical) == 4
end

function test_StochasticDriverResult_construction()
    ε = [rand(2, 100) for _ in 1:3]  # D=2, K=3
    usage = [0.3, 0.4, 0.3]
    result = StochasticDriverResult(ε, usage)
    @test result.ε_pools === ε
    @test result.usage === usage
end

function test_StochasticDriverResult_invalid_usage()
    ε = [rand(2, 100) for _ in 1:3]
    @test_throws AssertionError StochasticDriverResult(ε, [0.5, 0.5])
end

function test_ACDCResult_construction()
    result = ACDCResult(3, [0.1, 0.2, 0.3], [0.3, 0.4, 0.3])
    @test result.K == 3
    @test result.component_discrepancies == [0.1, 0.2, 0.3]
    @test result.component_usage == [0.3, 0.4, 0.3]
end

function test_ACDCResult_invalid_discrepancy_length()
    @test_throws AssertionError ACDCResult(3, [0.1, 0.2], [0.3, 0.4, 0.3])
end

function test_ACDCResult_invalid_usage_length()
    @test_throws AssertionError ACDCResult(3, [0.1, 0.2, 0.3], [0.5, 0.5])
end

function test_driver_uniformity_correct_model()
    Random.seed!(42)

    D = 5
    K = 3
    N = 1000
    S = 10

    uniform_drivers = [rand(D, N * S) for _ in 1:K]
    usage = [0.3, 0.4, 0.3]
    result = StochasticDriverResult(uniform_drivers, usage)

    mmd = MMDDiscrepancy()
    for k in 1:K
        disc = compute_discrepancy(mmd, uniform_drivers[k])
        @test disc < 0.05
    end
end

function test_driver_nonuniformity_misspecified_model()
    Random.seed!(42)

    D = 5
    K = 3
    N = 1000
    S = 10

    misspecified_drivers = [rand(Beta(2, 5), D, N * S) for _ in 1:K]
    usage = [0.3, 0.4, 0.3]
    result = StochasticDriverResult(misspecified_drivers, usage)

    mmd = MMDDiscrepancy()
    for k in 1:K
        disc = compute_discrepancy(mmd, misspecified_drivers[k])
        @test disc > 0.01
    end
end
