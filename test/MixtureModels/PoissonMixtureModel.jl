# Test general properties of PoissonMixtureModel
function test_PoissonMixtureModel_properties(pmm::PoissonMixtureModel, k::Int)
    @test pmm.k == k
    @test length(pmm.λₖ) == k
    @test length(pmm.πₖ) == k
    @test sum(pmm.πₖ) ≈ 1.0
end

function testPoissonMixtureModel_EStep(
    pmm::PoissonMixtureModel, data::Union{Matrix{Int},Vector{Int}}
)
    k::Int = pmm.k

    # Run EStep
    class_probabilities = StateSpaceDynamics.E_Step(pmm, data)
    # Check dimensions
    @test size(class_probabilities) == (size(data, 1), k)
    # Check if the row sums are close to 1 (since they represent probabilities)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(class_probabilities; dims=2))

    return test_PoissonMixtureModel_properties(pmm, k)
end

function testPoissonMixtureModel_MStep(
    pmm::PoissonMixtureModel, data::Union{Matrix{Int},Vector{Int}}
)
    k::Int = pmm.k

    class_probabilities = StateSpaceDynamics.E_Step(pmm, data)

    # Run MStep
    StateSpaceDynamics.M_Step!(pmm, data, class_probabilities)

    return test_PoissonMixtureModel_properties(pmm, k)
end

function testPoissonMixtureModel_fit(
    pmm::PoissonMixtureModel, data::Union{Matrix{Int},Vector{Int}}
)
    k::Int = pmm.k

    # Run fit!
    fit!(pmm, data; maxiter=10, tol=1e-3)

    return test_PoissonMixtureModel_properties(pmm, k)
end

function test_log_likelihood(pmm::PoissonMixtureModel, data::Union{Matrix{Int},Vector{Int}})
    # Calculate log-likelihood
    ll = log_likelihood(pmm, data)

    # Check if log-likelihood is a scalar
    @test size(ll) == ()

    # Log-likelihood should not necessarily be negative for Poisson models

    # Initialize λₖ with kmeans_init
    λₖ_matrix = permutedims(kmeanspp_initialization(Float64.(data), pmm.k))
    pmm.λₖ = vec(λₖ_matrix)

    # Log-likelihood should monotonically increase with iterations (when using exact EM)
    ll_prev = -Inf
    for i in 1:10
        fit!(pmm, data; maxiter=1, tol=1e-3, initialize_kmeans=false)
        ll = log_likelihood(pmm, data)
        @test ll > ll_prev || isapprox(ll, ll_prev; atol=1e-6)
        ll_prev = ll
    end
end
