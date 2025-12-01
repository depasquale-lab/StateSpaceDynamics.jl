# Include common test utilities
include("Common.jl")

function test_GaussianMixtureModel_properties(gmm::GaussianMixtureModel, k::Int, D::Int)
    @test gmm.k == k
    @test size(gmm.μₖ) == (D, k)
    @test length(gmm.Σₖ) == k
    for Σ in gmm.Σₖ
        @test size(Σ) == (D, D)
        @test ishermitian(Σ)
    end
    @test length(gmm.πₖ) == k
    @test isapprox(sum(gmm.πₖ), 1.0; atol=1e-6)
end

function testGaussianMixtureModel_EStep(
    gmm::GaussianMixtureModel, data::Union{Matrix{Float64},Vector{Float64}}
)
    data = isa(data, Vector) ? reshape(data, :, 1) : data
    D, N = size(data)
    k = gmm.k
    class_probabilities = StateSpaceDynamics.estep(gmm, data)
    @test size(class_probabilities) == (k, N)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(class_probabilities; dims=1))
    return test_GaussianMixtureModel_properties(gmm, k, D)
end

function testGaussianMixtureModel_MStep(
    gmm::GaussianMixtureModel, data::Union{Matrix{Float64},Vector{Float64}}
)
    data = isa(data, Vector) ? reshape(data, :, 1) : data
    D, _ = size(data)
    k = gmm.k
    γ = StateSpaceDynamics.estep(gmm, data)
    StateSpaceDynamics.mstep!(gmm, data, γ)
    return test_GaussianMixtureModel_properties(gmm, k, D)
end

function testGaussianMixtureModel_fit(
    gmm::GaussianMixtureModel, data::Union{Matrix{Float64},Vector{Float64}}
)
    data = isa(data, Vector) ? reshape(data, :, 1) : data
    D, _ = size(data)
    k = gmm.k
    fit!(gmm, data; maxiter=10, tol=1e-3)
    return test_GaussianMixtureModel_properties(gmm, k, D)
end

function test_loglikelihood(
    gmm::GaussianMixtureModel, data::Union{Matrix{Float64},Vector{Float64}}
)
    data = isa(data, Vector) ? reshape(data, :, 1) : data
    ll = StateSpaceDynamics.loglikelihood(gmm, data)
    @test isa(ll, Float64)
    @test ll < 0.0 || isapprox(ll, 0.0; atol=1e-6)

    gmm.μₖ = kmeanspp_initialization(data, gmm.k)
    prev_ll = -Inf
    for i in 1:10
        fit!(gmm, data; maxiter=1, tol=1e-3, initialize_kmeans=false)
        curr_ll = StateSpaceDynamics.loglikelihood(gmm, data)
        @test curr_ll >= prev_ll - 1e-6
        prev_ll = curr_ll
    end
end

function test_rand_sampling()
    # Test sampling from GMM
    gmm = GaussianMixtureModel(3, 2)
    gmm.μₖ = randn(2, 3)
    gmm.Σₖ = [Matrix{Float64}(I, 2, 2) for _ in 1:3]
    gmm.πₖ = [0.3, 0.5, 0.2]

    test_rand_dimensions_gmm(gmm, 100)

    # Test that samples look reasonable
    samples = rand(gmm, 1000)
    @test size(samples, 1) == 2  # Dimension
    @test size(samples, 2) == 1000  # Number of samples
end

function test_estep_probabilities()
    # Test that E-step produces valid probabilities
    gmm = GaussianMixtureModel(3, 2)
    data = randn(2, 50)

    γ = StateSpaceDynamics.estep(gmm, data)
    return test_estep_probabilities_common(γ, gmm.k, size(data, 2))
end

function test_mstep_validity()
    # Test that M-step maintains model validity
    gmm = GaussianMixtureModel(3, 2)
    data = randn(2, 100)

    return test_mstep_maintains_validity_gmm(gmm, data)
end

function test_fit_with_kmeans_init()
    # Test fitting with K-means initialization
    data = randn(2, 100)
    gmm = GaussianMixtureModel(3, 2)

    γ, lls = fit!(gmm, data; maxiter=20, tol=1e-4, initialize_kmeans=true)

    @test size(γ) == (3, 100)
    test_fit_convergence_common(lls)
    return test_GaussianMixtureModel_properties(gmm, 3, 2)
end

function test_vector_input_handling()
    # Test that vector inputs are handled correctly
    gmm = GaussianMixtureModel(2, 1)
    data_vec = randn(50)
    data_mat = reshape(data_vec, :, 1)

    # Test estep
    γ_vec = StateSpaceDynamics.estep(gmm, data_vec)
    γ_mat = StateSpaceDynamics.estep(gmm, data_mat)
    @test γ_vec ≈ γ_mat

    # Test loglikelihood
    ll_vec = StateSpaceDynamics.loglikelihood(gmm, data_vec)
    ll_mat = StateSpaceDynamics.loglikelihood(gmm, data_mat)
    @test ll_vec ≈ ll_mat

    # Test mstep
    gmm1 = GaussianMixtureModel(2, 1)
    gmm2 = GaussianMixtureModel(2, 1)
    StateSpaceDynamics.mstep!(gmm1, data_vec, γ_vec)
    StateSpaceDynamics.mstep!(gmm2, data_mat, γ_mat)
    @test gmm1.μₖ ≈ gmm2.μₖ
    @test gmm1.πₖ ≈ gmm2.πₖ
end

