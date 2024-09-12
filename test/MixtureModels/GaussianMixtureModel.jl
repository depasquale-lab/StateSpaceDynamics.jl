# Test general properties of GaussianMixtureModel
function test_GaussianMixtureModel_properties(
    gmm::GaussianMixtureModel, k::Int, data_dim::Int
)
    @test gmm.k == k
    @test size(gmm.μₖ) == (k, data_dim)

    for Σ in gmm.Σₖ
        @test size(Σ) == (data_dim, data_dim)
        @test ishermitian(Σ)
    end

    @test length(gmm.πₖ) == k
    @test sum(gmm.πₖ) ≈ 1.0
end

function testGaussianMixtureModel_EStep(
    gmm::GaussianMixtureModel, data::Union{Matrix{Float64},Vector{Float64}}
)
    k::Int = gmm.k
    data_dim::Int = size(data, 2)

    # Run E_Step
    class_probabilities = StateSpaceDynamics.E_Step(gmm, data)
    # Check dimensions
    @test size(class_probabilities) == (size(data, 1), k)
    # Check if the row sums are close to 1 (since they represent probabilities)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(class_probabilities; dims=2))

    return test_GaussianMixtureModel_properties(gmm, k, data_dim)
end

function testGaussianMixtureModel_MStep(
    gmm::GaussianMixtureModel, data::Union{Matrix{Float64},Vector{Float64}}
)
    k::Int = gmm.k
    data_dim::Int = size(data, 2)

    class_probabilities = StateSpaceDynamics.E_Step(gmm, data)

    # Run MStep
    StateSpaceDynamics.M_Step!(gmm, data, class_probabilities)

    return test_GaussianMixtureModel_properties(gmm, k, data_dim)
end

function testGaussianMixtureModel_fit(
    gmm::GaussianMixtureModel, data::Union{Matrix{Float64},Vector{Float64}}
)
    k::Int = gmm.k
    data_dim::Int = size(data, 2)

    # Run fit!
    fit!(gmm, data; maxiter=10, tol=1e-3)

    return test_GaussianMixtureModel_properties(gmm, k, data_dim)
end

function test_log_likelihood(
    gmm::GaussianMixtureModel, data::Union{Matrix{Float64},Vector{Float64}}
)

    # Calculate log-likelihood
    ll = log_likelihood(gmm, data)

    # Check if log-likelihood is a scalar
    @test size(ll) == ()

    # Log-likelihood should be a negative float
    @test ll < 0.0

    # Log-likelihood should monotonically increase with iterations (when using exact EM)

    #repeatedly applying fit! without initializtion, so first initialize means
    # Initialize k means of gmm
    gmm.μₖ = permutedims(kmeanspp_initialization(data, gmm.k))

    ll_prev = -Inf
    for i in 1:10
        fit!(gmm, data; maxiter=1, tol=1e-3, initialize_kmeans=false)
        ll = log_likelihood(gmm, data)
        @test ll > ll_prev || isapprox(ll, ll_prev; atol=1e-6)
        ll_prev = ll
    end
end
