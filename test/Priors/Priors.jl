function test_iw_prior_helpers()
    rng = MersenneTwister(1)
    d = 3
    # SPD scale matrix Ψ and a SPD sufficient-stat matrix S.
    G = randn(rng, d, d)
    Ψ = Matrix(Symmetric(G * G' + d * I))
    H = randn(rng, d, d)
    S = Matrix(Symmetric(H * H' + I))
    ν = 7.0
    n = 12.0

    # iw_map: closed-form posterior mode (Ψ + S) / (ν + n + d + 1).
    expected = (Ψ .+ S) ./ (ν + n + d + 1.0)
    @test StateSpaceDynamics.iw_map(Ψ, ν, S, n, d) ≈ expected

    # iw_logprior_term: -½((ν + d + 1) log|Σ| + tr(Ψ Σ⁻¹)).
    K = randn(rng, d, d)
    Σ = Matrix(Symmetric(K * K' + d * I))
    prior = StateSpaceDynamics.IWPrior(; Ψ=Ψ, ν=ν)
    ref = -0.5 * ((ν + d + 1.0) * logdet(Σ) + tr(Ψ * inv(Σ)))
    @test StateSpaceDynamics.iw_logprior_term(Σ, prior) ≈ ref
end

function test_mn_prior_helpers()
    rng = MersenneTwister(2)
    k, p = 2, 3   # W is k×p; regression has p inputs, k outputs.

    L = randn(rng, p, p)
    Λ = Matrix(Symmetric(L * L' + p * I))     # column precision (SPD, p×p)
    M₀ = randn(rng, k, p)                       # prior mean (k×p)
    D = randn(rng, p, p)
    XX = Matrix(Symmetric(D * D' + p * I))     # design covariance (SPD, p×p)
    XY = randn(rng, p, k)                       # cross term (p×k)

    prior = StateSpaceDynamics.MNPrior(; M₀=M₀, Λ=Λ)

    # mn_map with a prior: W = (XYᵀ + M₀ Λ)(XX + Λ)⁻¹.
    W = StateSpaceDynamics.mn_map(XX, XY, prior)
    @test size(W) == (k, p)
    expected = (XY' + M₀ * Λ) * inv(XX + Λ)
    @test W ≈ expected

    # mn_map with no prior reduces to OLS: W = XYᵀ XX⁻¹.
    W_ols = StateSpaceDynamics.mn_map(XX, XY, nothing)
    @test W_ols ≈ XY' * inv(XX)

    # Shrinkage: inflating Λ pulls the MAP toward M₀.
    strong = StateSpaceDynamics.MNPrior(; M₀=M₀, Λ=1e6 .* Λ)
    W_strong = StateSpaceDynamics.mn_map(XX, XY, strong)
    @test norm(W_strong - M₀) < norm(W - M₀)

    # mn_logprior_term: -½ tr(Σ⁻¹ (W - M₀) Λ (W - M₀)').
    E = randn(rng, k, k)
    Σ = Matrix(Symmetric(E * E' + k * I))      # row covariance (k×k)
    Wm = W - M₀
    ref = -0.5 * tr(inv(Σ) * (Wm * Λ * Wm'))
    @test StateSpaceDynamics.mn_logprior_term(W, Σ, prior) ≈ ref

    # No prior contributes nothing to the objective.
    @test StateSpaceDynamics.mn_logprior_term(W, Σ, nothing) == 0.0
end
