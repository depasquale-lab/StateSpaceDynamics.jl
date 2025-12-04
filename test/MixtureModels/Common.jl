# Common test utilities for Mixture Models

"""
    test_rand_dimensions(model, n, expected_shape)

Test that rand produces correct dimensions for mixture models.
"""
function test_rand_dimensions_gmm(gmm::GaussianMixtureModel, n::Int)
    samples = rand(gmm, n)
    D = size(gmm.μₖ, 1)
    @test size(samples) == (D, n)
    @test eltype(samples) <: Real
end

function test_rand_dimensions_pmm(pmm::PoissonMixtureModel, n::Int)
    samples = rand(pmm, n)
    @test length(samples) == n
    @test eltype(samples) <: Integer
    @test all(>=(0), samples)  # Poisson samples are non-negative
end

"""
    test_estep_probabilities(model, data)

Test that E-step produces valid probability assignments.
"""
function test_estep_probabilities_common(γ::AbstractMatrix, k::Int, n::Int)
    @test size(γ) == (k, n)
    # Each column (sample) should sum to 1
    for i in 1:n
        @test isapprox(sum(γ[:, i]), 1.0; atol=1e-6)
    end
    # All probabilities should be in [0, 1]
    @test all(0 .<= γ .<= 1)
end

"""
    test_mstep_maintains_validity(model, data)

Test that M-step maintains model validity.
"""
function test_mstep_maintains_validity_gmm(gmm::GaussianMixtureModel, data::AbstractMatrix)
    k = gmm.k
    D = size(data, 1)

    γ = StateSpaceDynamics.estep(gmm, data)
    StateSpaceDynamics.mstep!(gmm, data, γ)

    # Check πₖ sums to 1
    @test isapprox(sum(gmm.πₖ), 1.0; atol=1e-6)
    # Check all πₖ are in [0, 1]
    @test all(0 .<= gmm.πₖ .<= 1)
    # Check dimensions maintained
    @test size(gmm.μₖ) == (D, k)
    @test length(gmm.Σₖ) == k
    # Check covariances are symmetric and positive definite
    for Σ in gmm.Σₖ
        @test ishermitian(Σ)
        @test isposdef(Σ)
    end
end

function test_mstep_maintains_validity_pmm(pmm::PoissonMixtureModel, data::AbstractMatrix)
    k = pmm.k

    γ = StateSpaceDynamics.estep(pmm, data)
    StateSpaceDynamics.mstep!(pmm, data, γ)

    # Check πₖ sums to 1
    @test isapprox(sum(pmm.πₖ), 1.0; atol=1e-6)
    # Check all πₖ are in [0, 1]
    @test all(0 .<= pmm.πₖ .<= 1)
    # Check all λₖ are positive
    @test all(pmm.λₖ .> 0)
    # Check dimensions maintained
    @test length(pmm.λₖ) == k
    @test length(pmm.πₖ) == k
end

"""
    test_fit_convergence(model, data)

Test that fit! converges and improves likelihood.
"""
function test_fit_convergence_common(lls::Vector{Float64})
    @test length(lls) > 0
    # Likelihood should generally increase (allowing small numerical errors)
    for i in 2:length(lls)
        @test lls[i] >= lls[i - 1] - 1e-6
    end
end
