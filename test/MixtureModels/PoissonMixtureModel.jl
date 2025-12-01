# Include common test utilities
include("Common.jl")

function test_PoissonMixtureModel_properties(pmm::PoissonMixtureModel, k::Int)
    @test pmm.k == k
    @test length(pmm.λₖ) == k
    @test length(pmm.πₖ) == k
    @test isapprox(sum(pmm.πₖ), 1.0; atol=1e-6)
end

function testPoissonMixtureModel_EStep(
    pmm::PoissonMixtureModel, data::Union{Matrix{Int},Vector{Int}}
)
    data = isa(data, Vector) ? reshape(data, 1, :) : data
    k = pmm.k
    γ = StateSpaceDynamics.estep(pmm, data)
    @test size(γ) == (k, size(data, 2))
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(γ; dims=1))
    return test_PoissonMixtureModel_properties(pmm, k)
end

function testPoissonMixtureModel_MStep(
    pmm::PoissonMixtureModel, data::Union{Matrix{Int},Vector{Int}}
)
    data = isa(data, Vector) ? reshape(data, 1, :) : data
    γ = StateSpaceDynamics.estep(pmm, data)
    StateSpaceDynamics.mstep!(pmm, data, γ)
    return test_PoissonMixtureModel_properties(pmm, pmm.k)
end

function testPoissonMixtureModel_fit(
    pmm::PoissonMixtureModel, data::Union{Matrix{Int},Vector{Int}}
)
    data = isa(data, Vector) ? reshape(data, 1, :) : data
    fit!(pmm, data; maxiter=10, tol=1e-3)
    return test_PoissonMixtureModel_properties(pmm, pmm.k)
end

function test_loglikelihood_pmm(
    pmm::PoissonMixtureModel, data::Union{Matrix{Int},Vector{Int}}
)
    data = isa(data, Vector) ? reshape(data, 1, :) : data
    ll = StateSpaceDynamics.loglikelihood(pmm, data)
    @test isa(ll, Float64)
    pmm.λₖ = vec(kmeanspp_initialization(Float64.(data), pmm.k))
    prev_ll = -Inf
    for i in 1:10
        fit!(pmm, data; maxiter=1, tol=1e-3, initialize_kmeans=false)
        curr_ll = StateSpaceDynamics.loglikelihood(pmm, data)
        @test curr_ll >= prev_ll - 1e-6
        prev_ll = curr_ll
    end
end

function test_rand_sampling_pmm()
    # Test sampling from PMM
    pmm = PoissonMixtureModel(3)
    pmm.λₖ = [2.0, 5.0, 10.0]
    pmm.πₖ = [0.3, 0.5, 0.2]

    test_rand_dimensions_pmm(pmm, 100)

    # Test that samples are non-negative integers
    samples = rand(pmm, 1000)
    @test all(>=(0), samples)
    @test eltype(samples) <: Integer
end

function test_estep_probabilities_pmm()
    # Test that E-step produces valid probabilities
    pmm = PoissonMixtureModel(3)
    data = rand(0:20, 1, 50)  # Random Poisson-like data

    γ = StateSpaceDynamics.estep(pmm, data)
    return test_estep_probabilities_common(γ, pmm.k, size(data, 2))
end

function test_mstep_validity_pmm()
    # Test that M-step maintains model validity
    pmm = PoissonMixtureModel(3)
    data = rand(0:20, 1, 100)

    return test_mstep_maintains_validity_pmm(pmm, data)
end

function test_fit_with_kmeans_init_pmm()
    # Test fitting with K-means initialization
    data = rand(0:20, 1, 100)
    pmm = PoissonMixtureModel(3)

    γ, lls = fit!(pmm, data; maxiter=20, tol=1e-4, initialize_kmeans=true)

    @test size(γ) == (3, 100)
    test_fit_convergence_common(lls)
    return test_PoissonMixtureModel_properties(pmm, 3)
end

function test_vector_input_handling_pmm()
    # Test that vector inputs are handled correctly
    pmm = PoissonMixtureModel(2)
    data_vec = rand(0:20, 50)
    data_mat = reshape(data_vec, 1, :)

    # Test estep
    γ_vec = StateSpaceDynamics.estep(pmm, data_vec)
    γ_mat = StateSpaceDynamics.estep(pmm, data_mat)
    @test γ_vec ≈ γ_mat

    # Test loglikelihood
    ll_vec = StateSpaceDynamics.loglikelihood(pmm, data_vec)
    ll_mat = StateSpaceDynamics.loglikelihood(pmm, data_mat)
    @test ll_vec ≈ ll_mat

    # Test mstep
    pmm1 = PoissonMixtureModel(2)
    pmm2 = PoissonMixtureModel(2)
    StateSpaceDynamics.mstep!(pmm1, data_vec, γ_vec)
    StateSpaceDynamics.mstep!(pmm2, data_mat, γ_mat)
    @test pmm1.λₖ ≈ pmm2.λₖ
    @test pmm1.πₖ ≈ pmm2.πₖ
end

