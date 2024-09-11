function test_PPCA_with_params()
    # Set parameters
    W = randn(3, 2)
    σ² = 0.5
    # create "data"
    X = randn(100, 3)
    μ = mean(X; dims=1)
    # create PPCA object
    ppca = ProbabilisticPCA(W, σ², μ, 2, 3, Matrix{Float64}(undef, 0, 0))
    # Check if parameters are set correctly
    @test ppca.W === W
    @test ppca.σ² === σ²
    @test ppca.μ === μ
    @test ppca.D === 3
    @test ppca.k === 2
    @test isempty(ppca.z)
end

function test_PPCA_without_params()
    # create ppca object
    ppca = ProbabilisticPCA(; k=2, D=3)
    # Check if parameters are set correctly
    @test size(ppca.W) == (3, 2)
    @test ppca.σ² > 0
    @test isempty(ppca.μ)
    @test ppca.D == 3
    @test ppca.k == 2
    @test isempty(ppca.z)
end

function test_PPCA_E_and_M_Step()
    # create ppca object
    ppca = ProbabilisticPCA(; k=2, D=3)
    # create data
    X = randn(100, 3)
    # assign μ, normally fit! does this
    μ = mean(X; dims=1)
    ppca.μ = μ
    # run E-step
    E_z, E_zz = StateSpaceDynamics.E_Step(ppca, X)
    # check dimensions
    @test size(E_z) == (100, 2)
    @test size(E_zz) == (100, 2, 2)
    # run M-step, but first save the old parameters
    W_old = ppca.W
    σ²_old = ppca.σ²
    StateSpaceDynamics.M_Step!(ppca, X, E_z, E_zz)
    # check if the parameters are updated
    @test ppca.W !== W_old
    @test ppca.σ² !== σ²_old
    @test ppca.μ === μ
end

function test_PPCA_fit()
    # create ppca object
    ppca = ProbabilisticPCA(; k=2, D=3)
    # create data
    X = randn(100, 3)
    ppca.μ = mean(X; dims=1)
    # fit the model
    ll = fit!(ppca, X)
    # check if the parameters are updated
    @test ppca.σ² > 0
    @test size(ppca.W) == (3, 2)
    @test size(ppca.μ) == (1, 3)
    @test size(ppca.z) == (100, 2)
    # check loglikelihood only increases
    @test all(diff(ll) .> 0)
    # also check that the loglikelihood is a scalar
    ll = StateSpaceDynamics.loglikelihood(ppca, X)
    @test size(ll) == ()
end
