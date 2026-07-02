abstract type AbstractRaisedCosineBasis <: AbstractInputBasis end

"""
    RaisedCosineLinear(num_bases::Int; width_factor::Real=2.0) <: AbstractInputBasis

Raised-cosine bumps with centres equally spaced on
`[first(ts), last(ts)]`. With spacing `dc = (last - first) / (K - 1)`,
the half-width is `w = width_factor · dc`:

    φ_k(t) = 0.5 · (1 + cos((t - c_k) · π / w))   for |t - c_k| ≤ w,
             0                                       otherwise.

The default `width_factor = 2.0` makes adjacent bumps overlap at
half-amplitude (Pillow / nemos convention). Requires `num_bases ≥ 2`.
"""
struct RaisedCosineLinear <: AbstractRaisedCosineBasis
    num_bases::Int
    width_factor::Float64
end

function RaisedCosineLinear(num_bases::Integer; width_factor::Real=2.0)
    num_bases >= 2 || throw(
        ArgumentError("RaisedCosineLinear requires num_bases >= 2, got $num_bases.")
    )
    width_factor > 0 ||
        throw(ArgumentError("width_factor must be > 0, got $width_factor."))
    return RaisedCosineLinear(Int(num_bases), Float64(width_factor))
end

"""
    RaisedCosineLog(num_bases::Int; width_factor::Real=2.0, offset::Real=1.0) <: AbstractInputBasis

Like [`RaisedCosineLinear`](@ref), but centres are equally spaced in
`log(t + offset)` space rather than in `t`. Bumps are narrow near the
start of the window and broad near the end — useful for spike-history
filters and other responses whose dynamics evolve on a log time scale
(Pillow 2005, nemos `RaisedCosineLog`). `offset > 0` keeps `log` finite
when `t = 0`. Requires `num_bases ≥ 2`.
"""
struct RaisedCosineLog <: AbstractRaisedCosineBasis
    num_bases::Int
    width_factor::Float64
    offset::Float64
end

function RaisedCosineLog(
    num_bases::Integer; width_factor::Real=2.0, offset::Real=1.0
)
    num_bases >= 2 ||
        throw(ArgumentError("RaisedCosineLog requires num_bases >= 2, got $num_bases."))
    width_factor > 0 ||
        throw(ArgumentError("width_factor must be > 0, got $width_factor."))
    offset > 0 || throw(ArgumentError("offset must be > 0, got $offset."))
    return RaisedCosineLog(Int(num_bases), Float64(width_factor), Float64(offset))
end

n_bases(b::AbstractRaisedCosineBasis) = b.num_bases

_rc_transform(::RaisedCosineLinear, ts::AbstractVector{T}) where {T<:Real} = T.(ts)
function _rc_transform(b::RaisedCosineLog, ts::AbstractVector{T}) where {T<:Real}
    return log.(T.(ts) .+ T(b.offset))
end

function evaluate_basis(
    b::AbstractRaisedCosineBasis, ts::AbstractVector{T}
) where {T<:Real}
    K = b.num_bases
    M = length(ts)
    gt = _rc_transform(b, ts)
    gmin, gmax = first(gt), last(gt)
    centers = collect(range(gmin, gmax; length=K))
    dc = (gmax - gmin) / T(K - 1)
    w = T(b.width_factor) * dc
    π_T = T(π)

    Φ = zeros(T, M, K)
    @inbounds for k in 1:K
        c = centers[k]
        for i in 1:M
            arg = (gt[i] - c) * π_T / w
            if -π_T <= arg <= π_T
                Φ[i, k] = T(0.5) * (one(T) + cos(arg))
            end
        end
    end
    return Φ
end
