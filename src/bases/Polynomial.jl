"""
    Polynomial(num_bases::Int) <: AbstractInputBasis

Monomial polynomial basis on the time window normalised to `[0, 1]`:

    φ_k(t) = ((t - first(ts)) / (last(ts) - first(ts)))^(k - 1)
             for k = 1, …, num_bases.

The first basis function is the constant `1` and the second is the
normalised linear ramp; both lie in the nullspace of the curvature
penalty. For `num_bases ≳ 10` the monomial basis becomes ill-conditioned
— consider [`BSpline`](@ref) or [`Fourier`](@ref) for smoother fits at
high `num_bases`.
"""
struct Polynomial <: AbstractInputBasis
    num_bases::Int
end

function Polynomial(num_bases::Integer)
    num_bases >= 1 || throw(ArgumentError("num_bases ($num_bases) must be >= 1."))
    return Polynomial(Int(num_bases))
end

n_bases(b::Polynomial) = b.num_bases

function evaluate_basis(b::Polynomial, ts::AbstractVector{T}) where {T<:Real}
    K = b.num_bases
    M = length(ts)
    tmin = T(first(ts))
    tmax = T(last(ts))
    span = tmax - tmin
    span = iszero(span) ? one(T) : span

    Φ = Matrix{T}(undef, M, K)
    @inbounds for i in 1:M
        x = (T(ts[i]) - tmin) / span
        Φ[i, 1] = one(T)
        for k in 2:K
            Φ[i, k] = Φ[i, k - 1] * x
        end
    end
    return Φ
end
