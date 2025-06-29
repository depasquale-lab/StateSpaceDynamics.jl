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

function testGaussianMixtureModel_EStep(gmm::GaussianMixtureModel, data::Union{Matrix{Float64}, Vector{Float64}})
    data = isa(data, Vector) ? reshape(data, :, 1) : data
    D, N = size(data)
    k = gmm.k
    class_probabilities = StateSpaceDynamics.estep(gmm, data)
    @test size(class_probabilities) == (k, N)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(class_probabilities; dims=1))
    test_GaussianMixtureModel_properties(gmm, k, D)
end

function testGaussianMixtureModel_MStep(gmm::GaussianMixtureModel, data::Union{Matrix{Float64}, Vector{Float64}})
    data = isa(data, Vector) ? reshape(data, :, 1) : data
    D, _ = size(data)
    k = gmm.k
    γ = StateSpaceDynamics.estep(gmm, data)
    StateSpaceDynamics.mstep!(gmm, data, γ)
    test_GaussianMixtureModel_properties(gmm, k, D)
end

function testGaussianMixtureModel_fit(gmm::GaussianMixtureModel, data::Union{Matrix{Float64}, Vector{Float64}})
    data = isa(data, Vector) ? reshape(data, :, 1) : data
    D, _ = size(data)
    k = gmm.k
    fit!(gmm, data; maxiter=10, tol=1e-3)
    test_GaussianMixtureModel_properties(gmm, k, D)
end

function test_loglikelihood(gmm::GaussianMixtureModel, data::Union{Matrix{Float64}, Vector{Float64}})
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