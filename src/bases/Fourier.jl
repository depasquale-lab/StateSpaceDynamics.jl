"""
    Fourier(num_bases::Int; period=nothing) <: AbstractInputBasis

Fourier basis of `num_bases` functions. The first function is the DC term
(constant `1`), followed by alternating cosine/sine pairs at increasing
integer multiples of the fundamental frequency `2π / period`:

    φ_1(t)       = 1                              (DC, freq = 0)
    φ_{2k}(t)    = cos(2π · k · t / period)
    φ_{2k+1}(t)  = sin(2π · k · t / period)       for k = 1, 2, …

For odd `num_bases = 2m+1` the basis includes the full `[cos(k), sin(k)]`
pair up through `k = m`. For even `num_bases = 2m`, the last function is
an extra `cos((m)·…)` (no matching `sin`).

`period === nothing` (default) → `period = last(ts) - first(ts) + 1` at
evaluation time, which equals `tsteps` for the standard `1:tsteps` grid.
"""
struct Fourier <: AbstractInputBasis
    num_bases::Int
    period::Union{Nothing,Float64}
end

function Fourier(num_bases::Integer; period::Union{Nothing,Real}=nothing)
    num_bases >= 1 || throw(ArgumentError("num_bases ($num_bases) must be >= 1."))
    if period !== nothing
        period > 0 || throw(ArgumentError("period must be > 0, got $period."))
    end
    return Fourier(Int(num_bases), period === nothing ? nothing : Float64(period))
end

n_bases(b::Fourier) = b.num_bases

# Frequency index per basis function: [0, 1, 1, 2, 2, 3, 3, …].
_fourier_freqs(N::Int) = Int[j == 1 ? 0 : div(j, 2) for j in 1:N]

function _resolve_period(b::Fourier, ts::AbstractVector{T}) where {T<:Real}
    return b.period === nothing ? T(last(ts) - first(ts) + 1) : T(b.period)
end

function evaluate_basis(b::Fourier, ts::AbstractVector{T}) where {T<:Real}
    N = b.num_bases
    M = length(ts)
    Tp = _resolve_period(b, ts)
    ω = T(2π) / Tp

    Φ = Matrix{T}(undef, M, N)
    @inbounds for i in 1:M
        Φ[i, 1] = one(T)
    end
    @inbounds for j in 2:N
        k = div(j, 2)
        is_cos = iseven(j)
        for i in 1:M
            arg = ω * T(k) * T(ts[i])
            Φ[i, j] = is_cos ? cos(arg) : sin(arg)
        end
    end
    return Φ
end

"""
    get_penalty(
        basis::Fourier,
        tsteps::Integer;
        P::Int=1,
        eltype::Type=Float64,
        use_analytic::Bool=true,
        n_grid::Int=max(20*tsteps, 200),
    ) -> Matrix

Fourier-specialised curvature penalty.

When `use_analytic = true` (default), returns the closed-form diagonal
`kron(I_P, Diagonal((2π · freq / Tp).^4))` with `freq = [0, 1, 1, 2, 2, …]`
and `Tp` equal to `basis.period` (or `tsteps` when auto). The DC entry is
exactly zero, so the constant direction lies in the penalty's nullspace.

When `use_analytic = false`, falls back to the generic finite-difference
curvature penalty on `n_grid` points.
"""
function get_penalty(
    basis::Fourier,
    tsteps::Integer;
    P::Int=1,
    eltype::Type=Float64,
    use_analytic::Bool=true,
    n_grid::Int=max(20 * Int(tsteps), 200),
)
    if !use_analytic
        return _generic_curvature_penalty(basis, Int(tsteps), P, eltype, n_grid)
    end
    T = eltype
    Tp = basis.period === nothing ? T(tsteps) : T(basis.period)
    freqs = _fourier_freqs(basis.num_bases)
    diag_entries = T[(T(2π) * T(f) / Tp)^4 for f in freqs]
    Ωk = Matrix{T}(Diagonal(diag_entries))
    return kron(Matrix{T}(I, P, P), Ωk)
end
