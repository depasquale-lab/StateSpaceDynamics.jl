"""
    BSpline(num_bases::Int; order::Int=4, knots=:auto) <: AbstractInputBasis

B-spline basis of `num_bases` functions with the given polynomial `order`
(default `4` = cubic).

# Knot placement
- `knots = :auto` (default): breakpoints are placed by
  `BSplines.averagebasis` (de Boor 1978, p. 219) on `num_bases`
  equally-spaced sites in `[first(ts), last(ts)]`, resolved at evaluation
  time. This yields a basis whose Schoenberg–Whitney conditions are
  satisfied at the sites.
- `knots::AbstractVector{<:Real}`: a sorted vector of breakpoints. The
  resulting basis must have exactly `num_bases` functions, i.e.
  `length(knots) == num_bases - order + 2`.
"""
struct BSpline{K} <: AbstractInputBasis
    num_bases::Int
    order::Int
    knots::K
end

function BSpline(
    num_bases::Integer;
    order::Integer=4,
    knots::Union{Symbol,AbstractVector{<:Real}}=:auto,
)
    num_bases >= order ||
        throw(ArgumentError("num_bases ($num_bases) must be >= order ($order)."))
    order >= 1 || throw(ArgumentError("order ($order) must be >= 1."))
    knots_resolved = if knots isa AbstractVector
        collect(Float64.(knots))
    elseif knots === :auto
        :auto
    else
        throw(
            ArgumentError(
                "knots must be :auto or an AbstractVector, got $(typeof(knots)).",
            ),
        )
    end
    return BSpline{typeof(knots_resolved)}(Int(num_bases), Int(order), knots_resolved)
end

n_bases(b::BSpline) = b.num_bases

function evaluate_basis(b::BSpline, ts::AbstractVector{T}) where {T<:Real}
    K = b.num_bases
    basis = if b.knots === :auto
        data_sites = collect(range(T(first(ts)), T(last(ts)); length=K))
        BSplines.averagebasis(b.order, data_sites)
    else
        BSplines.BSplineBasis(b.order, collect(T.(b.knots)))
    end
    length(basis) == K || throw(
        ArgumentError(
            "constructed basis has length $(length(basis)) but num_bases=$K. " *
            "With explicit knots, length(knots) must equal num_bases - order + 2.",
        ),
    )
    B_raw = BSplines.basismatrix(basis, ts)
    return eltype(B_raw) === T ? B_raw : convert(Matrix{T}, B_raw)
end
