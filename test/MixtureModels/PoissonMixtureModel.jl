function test_PoissonMixtureModel_properties(pmm::PoissonMixtureModel, k::Int)
    @test pmm.k == k
    @test length(pmm.λₖ) == k
    @test length(pmm.πₖ) == k
    @test isapprox(sum(pmm.πₖ), 1.0; atol=1e-6)
end

function testPoissonMixtureModel_EStep(pmm::PoissonMixtureModel, data::Union{Matrix{Int}, Vector{Int}})
    data = isa(data, Vector) ? reshape(data, 1, :) : data
    k = pmm.k
    γ = StateSpaceDynamics.estep(pmm, data)
    @test size(γ) == (k, size(data, 2))
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(γ; dims=1))
    test_PoissonMixtureModel_properties(pmm, k)
end

function testPoissonMixtureModel_MStep(pmm::PoissonMixtureModel, data::Union{Matrix{Int}, Vector{Int}})
    data = isa(data, Vector) ? reshape(data, 1, :) : data
    γ = StateSpaceDynamics.estep(pmm, data)
    StateSpaceDynamics.mstep!(pmm, data, γ)
    test_PoissonMixtureModel_properties(pmm, pmm.k)
end

function testPoissonMixtureModel_fit(pmm::PoissonMixtureModel, data::Union{Matrix{Int}, Vector{Int}})
    data = isa(data, Vector) ? reshape(data, 1, :) : data
    fit!(pmm, data; maxiter=10, tol=1e-3)
    test_PoissonMixtureModel_properties(pmm, pmm.k)
end

function test_loglikelihood_pmm(pmm::PoissonMixtureModel, data::Union{Matrix{Int}, Vector{Int}})
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