# Public API
export ProbabilisticPCA, loglikelihood, fit!

mutable struct ProbabilisticPCA{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
    W::M
    σ²::T
    μ::V
    k::Int
    D::Int
    z::M

    function ProbabilisticPCA(W::AbstractMatrix{T}, σ²::T, μ::AbstractVector{T}) where {T<:Real}
        D, k = size(W)
        z = Matrix{T}(undef, k, 0)  # placeholder, filled after E-step
        new{T, typeof(W), typeof(μ)}(W, σ², μ, k, D, z)
    end
end

function Base.show(io::IO, ppca::ProbabilisticPCA; gap = "")
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
end

function estep(ppca::ProbabilisticPCA, X::Matrix{T}) where {T<:Real}
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

function mstep!(ppca::ProbabilisticPCA, X::Matrix{T}, E_z::Matrix{T}, E_zz::Array{T, 3}) where {T<:Real}
    D, N = size(X)
    W_new = zeros(T, D, ppca.k)
    σ²_sum = zero(T)
    WW = ppca.W' * ppca.W
    for i in 1:N
        x_centered = X[:, i] .- ppca.μ
        ez = @view(E_z[:, i])
        ezz = @view(E_zz[:, :, i])
        W_new .+= x_centered * ez'
        σ²_sum += sum(x_centered .^ 2) - 2 * dot(ez, ppca.W' * x_centered) + tr(ezz * WW)
    end
    ppca.z = E_z
    ppca.W = W_new * inv(sum(E_zz, dims=3)[:, :, 1])
    ppca.σ² = σ²_sum / (N * D)
end

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

function Random.rand(rng::AbstractRNG, ppca::ProbabilisticPCA, n::Int)
    z = rand(rng, MvNormal(zeros(ppca.k), I), n)  # k × n
    ε = rand(rng, MvNormal(zeros(ppca.D), ppca.σ² * I), n)  # D × n
    μ = reshape(ppca.μ, ppca.D, 1)
    return ppca.W * z .+ μ .+ ε, z
end

function Random.rand(ppca::ProbabilisticPCA, n::Int)
    rand(Random.default_rng(), ppca, n)
end

function fit!(ppca::ProbabilisticPCA, X::AbstractMatrix{T}, max_iters::Int=100, tol::Float64=1e-6) where {T<:Real}
    if all(iszero, ppca.μ)
        ppca.μ .= vec(mean(X; dims=2))
    end
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
