"""
    ProbabilisticPCA{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}

Probabilistic Principal Component Analysis (PPCA) model. Observations
`x ∈ ℝ^D` are generated from a `k`-dimensional latent factor `z` with
isotropic Gaussian noise:

```math
z ∼ N(0, I_k), \\qquad x \\mid z ∼ N(μ + W z, σ² I_D)
```

so that marginally `x ∼ N(μ, W Wᵀ + σ² I)`. As `σ² → 0`, PPCA recovers
classical PCA. Construct with `ProbabilisticPCA(W, σ², μ)`; `k` and `D` are
inferred from the size of `W`.

# Fields
- `W::M`: Loading matrix (`D × k`). Identifiable only up to an orthogonal
    rotation of its columns.
- `σ²::T`: Isotropic observation noise variance.
- `μ::V`: Observation mean (length `D`).
- `k::Int`: Latent dimension.
- `D::Int`: Observation dimension.
- `z::M`: Posterior means of the latent factors from the most recent E-step
    (`k × N`; empty until [`fit!`](@ref) is called).
"""
mutable struct ProbabilisticPCA{T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    W::M
    σ²::T
    μ::V
    k::Int
    D::Int
    z::M

    function ProbabilisticPCA(
        W::AbstractMatrix{T}, σ²::T, μ::AbstractVector{T}
    ) where {T<:Real}
        D, k = size(W)
        z = Matrix{T}(undef, k, 0)  # placeholder, filled after E-step

        return new{T,typeof(W),typeof(μ)}(W, σ², μ, k, D, z)
    end
end

function Base.show(io::IO, ppca::ProbabilisticPCA; gap="")
    println(io, gap, "Probabilistic PCA Model:")
    println(io, gap, "------------------------")
    println(io, gap, " size(W) = ($(size(ppca.W,1)), $(size(ppca.W,2)))")
    println(io, gap, " size(z) = ($(size(ppca.z,1)), $(size(ppca.z,2)))")
    println(io, gap, "      σ² = $(round(ppca.σ², digits=2))")

    if length(ppca.μ) > 6
        println(io, gap, " size(μ) = ($(length(ppca.μ)),)")
    else
        println(io, gap, "      μ  = $(round.(ppca.μ, digits=2))")
    end

    return nothing
end

function estep(ppca::ProbabilisticPCA, X::AbstractMatrix{T}) where {T<:Real}
    D, N = size(X)
    E_z = zeros(T, ppca.k, N)
    E_zz = zeros(T, ppca.k, ppca.k, N)
    M = ppca.W' * ppca.W + ppca.σ² * I(ppca.k)
    M_inv = inv(M)

    @views for i in 1:N
        E_z[:, i] .= M_inv * ppca.W' * (X[:, i] - ppca.μ)
        E_zz[:, :, i] .= ppca.σ² * M_inv + E_z[:, i] * E_z[:, i]'
    end

    return E_z, E_zz
end

function mstep!(
    ppca::ProbabilisticPCA, X::AbstractMatrix{T}, E_z::Matrix{T}, E_zz::Array{T,3}
) where {T<:Real}
    D, N = size(X)

    # W update: W_new = [Σₙ (xₙ - μ) E[zₙ]ᵀ][Σₙ E[zₙ zₙᵀ]]⁻¹ (Tipping–Bishop).
    W_num = zeros(T, D, ppca.k)
    for i in 1:N
        x_centered = @view(X[:, i]) .- ppca.μ
        W_num .+= x_centered * @view(E_z[:, i])'
    end
    W_new = W_num / sum(E_zz; dims=3)[:, :, 1]

    # σ² update must use the updated W_new, not the old W;
    # hence the second pass. Using the pre-update W breaks EM monotonicity.
    WW_new = W_new' * W_new
    σ²_sum = zero(T)
    for i in 1:N
        x_centered = @view(X[:, i]) .- ppca.μ
        ez = @view(E_z[:, i])
        ezz = @view(E_zz[:, :, i])
        σ²_sum +=
            sum(abs2, x_centered) - 2 * dot(ez, W_new' * x_centered) + tr(ezz * WW_new)
    end

    ppca.z = E_z
    ppca.W = W_new
    ppca.σ² = σ²_sum / (N * D)

    return nothing
end

"""
    loglikelihood(ppca::ProbabilisticPCA, X::AbstractMatrix)

Marginal log-likelihood `∑_n log p(x_n)` of the data `X` (`D × N`, one
observation per column) under the PPCA model, using the closed-form marginal
`x ∼ N(μ, W Wᵀ + σ² I)`.
"""
function loglikelihood(ppca::ProbabilisticPCA, X::AbstractMatrix{T}) where {T<:Real}
    D, N = size(X)
    @assert D == ppca.D "Dimension mismatch: X has $D features, model expects $(ppca.D)"
    C = ppca.W * ppca.W' + ppca.σ² * I(D)
    X_centered = X .- ppca.μ
    S = (X_centered * X_centered') / N

    try
        C_chol = cholesky(Symmetric(C))
        log_det_C = 2sum(log, diag(C_chol.U))
        trace_term = tr(C_chol \ S)

        return -(N / 2) * (D * log(2π) + log_det_C + trace_term)
    catch e
        @warn "Covariance matrix is not positive definite" e
        return -Inf
    end
end

"""
    Random.rand([rng::AbstractRNG,] ppca::ProbabilisticPCA, n::Int)

Sample `n` observations from the PPCA generative model. Returns a tuple
`(X, z)` where `X` is `D × n` and `z` holds the sampled latent factors
(`k × n`).
"""
function Random.rand(rng::AbstractRNG, ppca::ProbabilisticPCA, n::Int)
    z = rand(rng, MvNormal(zeros(ppca.k), I), n)  # k × n
    ε = rand(rng, MvNormal(zeros(ppca.D), Diagonal(fill(ppca.σ², ppca.D))), n)  # D × n
    μ = reshape(ppca.μ, ppca.D, 1)

    return ppca.W * z .+ μ .+ ε, z
end

function Random.rand(ppca::ProbabilisticPCA, n::Int)
    return rand(Random.default_rng(), ppca, n)
end

"""
    fit!(ppca::ProbabilisticPCA, X::AbstractMatrix, max_iters::Int=100, tol::Float64=1e-6)

Fit the PPCA model to data `X` (`D × N`, one observation per column) with EM.
`μ` is set directly to the sample mean (its exact ML estimate); `W` and `σ²`
are then updated iteratively until the log-likelihood improves by less than
`tol` or `max_iters` is reached.

Returns the vector of marginal log-likelihood values, one per iteration
(non-decreasing).
"""
function fit!(
    ppca::ProbabilisticPCA, X::AbstractMatrix{T}, max_iters::Int=100, tol::Float64=1e-6
) where {T<:Real}
    # The ML estimate of μ for PPCA is exactly the sample mean, independent of
    # W/σ² and of the EM iterations, so set it directly (mstep! never updates μ).
    ppca.μ .= vec(mean(X; dims=2))

    lls = Float64[]
    prev_ll = -Inf
    prog = Progress(max_iters; desc="Fitting Probabilistic PCA...")

    for iter in 1:max_iters
        E_z, E_zz = estep(ppca, X)
        mstep!(ppca, X, E_z, E_zz)
        ll = loglikelihood(ppca, X)
        push!(lls, ll)
        next!(prog)

        if abs(ll - prev_ll) < tol
            finish!(prog)
            return lls
        end

        prev_ll = ll
    end

    finish!(prog)

    return lls
end
