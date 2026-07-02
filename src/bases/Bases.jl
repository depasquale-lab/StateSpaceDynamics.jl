"""
    AbstractInputBasis

Abstract supertype for time-varying input bases. A concrete subtype
`B <: AbstractInputBasis` describes a set of basis functions
`{φ_1, …, φ_K}` over the time domain `1:tsteps`, and is used to construct
the `(P*K, tsteps, ntrials)` input array stored in `data.u` (dynamics) or
`data.d` (observation).

# Required interface

Each concrete subtype must implement two methods:

- [`n_bases`](@ref): `n_bases(b) -> Int` — number of basis functions `K`.
- [`evaluate_basis`](@ref): `evaluate_basis(b, ts) -> Matrix` — returns a
  `(length(ts) × K)` matrix `Φ` with `Φ[i, k] = φ_k(ts[i])`. The element
  type of the returned matrix should match `eltype(ts)`.

The generic [`apply!`](@ref) and [`get_penalty`](@ref) methods are written
in terms of those two primitives. Concrete bases may override
`get_penalty` to provide an analytic closed form (see [`Fourier`](@ref)).
"""
abstract type AbstractInputBasis end

"""
    n_bases(b::AbstractInputBasis) -> Int

Number of basis functions in `b`. Every concrete `AbstractInputBasis` must
implement this method.
"""
function n_bases end

"""
    evaluate_basis(b::AbstractInputBasis, ts::AbstractVector{<:Real}) -> Matrix

Evaluate the basis at the points `ts`. Returns a `(length(ts), n_bases(b))`
matrix `Φ` with `Φ[i, k] = φ_k(ts[i])`. Every concrete `AbstractInputBasis`
must implement this method.
"""
function evaluate_basis end

"""
    apply!(data::Data{T}, basis::AbstractInputBasis; target::Symbol=:u) where {T<:Real}

Construct the time-varying input array
`kron(data.trial_pred[n, :], B')` per trial, where `B` is the
`(tsteps × n_bases(basis))` basis matrix obtained by evaluating `basis` at
the integer timesteps `1:tsteps`, and write it in place into `data.u`
(when `target=:u`) or `data.d` (when `target=:d`) via `copyto!`.

When `data.trial_pred` is empty, a single all-ones predictor is used so
the per-trial input is just `B'`.

The caller must pre-allocate `data.<target>` with shape
`(P*K, tsteps, ntrials)` where `P = size(data.trial_pred, 2)` (or `1` when
empty) and `K = n_bases(basis)`. A `DimensionMismatch` is thrown otherwise.

Returns `nothing`.
"""
function apply!(
    data::Data{T}, basis::AbstractInputBasis; target::Symbol=:u
) where {T<:Real}
    target in (:u, :d) ||
        throw(ArgumentError("target must be :u or :d, got $(repr(target))."))

    tsteps = size(data.y, 2)
    ntrials = size(data.y, 3)

    trial_pred = if isempty(data.trial_pred)
        ones(T, ntrials, 1)
    else
        size(data.trial_pred, 1) == ntrials || throw(
            DimensionMismatch(
                "data.trial_pred has $(size(data.trial_pred, 1)) rows but data.y " *
                "has $ntrials trials. trial_pred must be shape (ntrials, npredictors).",
            ),
        )
        data.trial_pred
    end
    P = size(trial_pred, 2)
    K = n_bases(basis)

    target_arr = getfield(data, target)
    size(target_arr) == (P * K, tsteps, ntrials) || throw(
        DimensionMismatch(
            "data.$target has shape $(size(target_arr)) but inputs require " *
            "$((P * K, tsteps, ntrials)). Pre-allocate data.$target before calling.",
        ),
    )

    ts = collect(T(1):T(tsteps))
    B_raw = evaluate_basis(basis, ts)
    B = eltype(B_raw) === T ? B_raw : convert(Matrix{T}, B_raw)
    Bt = transpose(B)

    @inbounds for n in 1:ntrials
        for p in 1:P
            row_start = (p - 1) * K + 1
            row_end = p * K
            coeff = trial_pred[n, p]
            @views target_arr[row_start:row_end, :, n] .= coeff .* Bt
        end
    end
    return nothing
end

"""
    get_penalty(
        basis::AbstractInputBasis,
        tsteps::Integer;
        P::Int=1,
        eltype::Type{T}=Float64,
        n_grid::Int=max(20 * tsteps, 200),
    ) -> Matrix{T}

Time-domain curvature penalty `kron(I_P, Ω_K)` for `basis` on the window
`[1, tsteps]`. `Ω_K ≈ ∫ φ''(τ) φ''(τ)ᵀ dτ` is estimated on a fine grid of
`n_grid` points via a centred second difference:

```
Φ  = evaluate_basis(basis, range(1, tsteps; length=n_grid))
d² = diff(diff(Φ; dims=1); dims=1) ./ Δτ²
Ω_K = Δτ · (d²)ᵀ d²
```

The penalty captures roughness in **time** (independent of how the basis
indexes its coefficients), and so applies uniformly across bases — in
particular it remains well-behaved for unequally-spaced bases such as the
B-spline knot averaging.

Concrete bases may specialise this method to return an analytic form
instead; see [`get_penalty(::Fourier, ...)`](@ref).
"""
function get_penalty(
    basis::AbstractInputBasis,
    tsteps::Integer;
    P::Int=1,
    eltype::Type=Float64,
    n_grid::Int=max(20 * Int(tsteps), 200),
)
    return _generic_curvature_penalty(basis, Int(tsteps), P, eltype, n_grid)
end

"""
    get_penalty(data::Data{T}, basis::AbstractInputBasis; kwargs...) where {T} -> Matrix{T}

Convenience overload that reads `tsteps` from `data.y`, `P` from
`data.trial_pred` (or `1` when empty), and `eltype` from `T`. All keyword
arguments are forwarded to the underlying `get_penalty(basis, tsteps; …)`
method, so basis-specific keywords (e.g. `use_analytic=true` for
[`Fourier`](@ref)) work transparently.
"""
function get_penalty(
    data::Data{T}, basis::AbstractInputBasis; kwargs...
) where {T<:Real}
    tsteps = size(data.y, 2)
    P = isempty(data.trial_pred) ? 1 : size(data.trial_pred, 2)
    return get_penalty(basis, tsteps; P=P, eltype=T, kwargs...)
end

function _generic_curvature_penalty(
    basis::AbstractInputBasis, tsteps::Int, P::Int, ::Type{T}, n_grid::Int
) where {T<:Real}
    n_grid >= 3 ||
        throw(ArgumentError("n_grid ($n_grid) must be >= 3 for a 2nd difference."))
    τ = collect(range(T(1), T(tsteps); length=n_grid))
    Δτ = (T(tsteps) - one(T)) / T(n_grid - 1)
    Φ_raw = evaluate_basis(basis, τ)
    Φ = eltype(Φ_raw) === T ? Φ_raw : convert(Matrix{T}, Φ_raw)
    d2 = diff(diff(Φ; dims=1); dims=1) ./ (Δτ * Δτ)
    Ωk = Δτ .* (transpose(d2) * d2)
    return kron(Matrix{T}(I, P, P), Ωk)
end
