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


function testGaussianMixtureModel_Dispatch()
    @testset "GaussianMixtureModel Dispatch Tests" begin
        # Build a simple 1D GMM with known parameters
        k = 2
        data_dim = 1
        base_gmm = GaussianMixtureModel(k, data_dim)
        base_gmm.μₖ .= reshape([0.0, 5.0], 2, 1)
        base_gmm.Σₖ = [Matrix{Float64}(I, 1, 1) for _ in 1:k]
        base_gmm.πₖ .= [0.5, 0.5]

        # Sample N points from base_gmm (Float64 matrix (N×1))
        N = 1000
        raw_samples = StateSpaceDynamics.sample(base_gmm, N)
        @test size(raw_samples) == (N, 1)
        @test eltype(raw_samples) == Float64

        data_f64 = raw_samples

        # Convert to Int and Float32
        data_int = Int.(round.(data_f64))
        data_f32 = Float32.(data_f64)
        @test eltype(data_int) == Int
        @test eltype(data_f32) == Float32

        # (A) fit! on Int data
        gmm_int = GaussianMixtureModel(k, data_dim)
        ll_int = fit!(
            gmm_int,
            data_int;
            maxiter=100,
            tol=1e-4,
            initialize_kmeans=true
        )
        @test eltype(ll_int) == Float64
        @test eltype(gmm_int.μₖ)   == Float64
        @test all(eltype.(gmm_int.Σₖ) .== Float64) 
        @test eltype(gmm_int.πₖ)  == Float64

        # (B) fit! on Float32 data
        gmm_f32 = GaussianMixtureModel(k, data_dim)
        ll_f32 = fit!(
            gmm_f32,
            data_f32;
            maxiter=100,
            tol=1e-4,
            initialize_kmeans=false
        )
        @test eltype(ll_f32) == Float64
        @test eltype(gmm_f32.μₖ)   == Float64
        @test all(eltype.(gmm_f32.Σₖ) .== Float64)
        @test eltype(gmm_f32.πₖ)  == Float64

        # (C) log_likelihood dispatch on Int/Float32
        ll_f64_ref = log_likelihood(gmm_f32, data_f64)
        ll_int2    = log_likelihood(gmm_f32, data_int)
        ll_f322    = log_likelihood(gmm_f32, data_f32)

        @test isa(ll_f64_ref, Float64)
        @test isa(ll_int2, Float64)
        @test isa(ll_f322, Float64)
    end
end