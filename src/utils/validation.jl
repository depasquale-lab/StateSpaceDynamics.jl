"""
    DimensionMismatchError <: Exception

Custom exception for dimension mismatches in model parameters.

# Fields
- `parameter::String`: Name of the parameter with incorrect dimensions
- `expected::Union{Int,Tuple{Vararg{Int}}}`: Expected dimension(s)
- `got::Union{Int,Tuple{Vararg{Int}}}`: Actual dimension(s)
"""
struct DimensionMismatchError <: Exception
    parameter::String
    expected::Union{Int,Tuple{Vararg{Int}}}
    got::Union{Int,Tuple{Vararg{Int}}}
end

function Base.showerror(io::IO, e::DimensionMismatchError)
    print(io, "DimensionMismatchError: ")
    return print(io, "$(e.parameter) has dimensions $(e.got), expected $(e.expected)")
end

"""
    NotPositiveDefiniteError <: Exception

Custom exception for matrices that should be positive definite but aren't.

# Fields
- `matrix_name::String`: Name of the matrix
- `min_eigenvalue::Float64`: Minimum eigenvalue found
"""
struct NotPositiveDefiniteError <: Exception
    matrix_name::String
    min_eigenvalue::Float64
end

function Base.showerror(io::IO, e::NotPositiveDefiniteError)
    print(io, "NotPositiveDefiniteError: ")
    print(io, "$(e.matrix_name) is not positive definite ")
    print(io, "(minimum eigenvalue: $(e.min_eigenvalue)). ")
    return print(io, "Consider adding regularization or checking for numerical issues.")
end

"""
    NotSymmetricError <: Exception

Custom exception for matrices that should be symmetric but aren't.

# Fields
- `matrix_name::String`: Name of the matrix
- `max_asymmetry::Float64`: Maximum asymmetry measure
"""
struct NotSymmetricError <: Exception
    matrix_name::String
    max_asymmetry::Float64
end

function Base.showerror(io::IO, e::NotSymmetricError)
    print(io, "NotSymmetricError: ")
    print(io, "$(e.matrix_name) is not symmetric ")
    return print(io, "(max asymmetry: $(e.max_asymmetry))")
end

"""
    InvalidProbabilityVectorError <: Exception

Custom exception for invalid probability vectors.

# Fields
- `vector_name::String`: Name of the probability vector
- `sum_value::Float64`: Sum of the vector
- `has_negative::Bool`: Whether the vector contains negative values
- `has_greater_than_one::Bool`: Whether the vector contains values > 1.0
"""
struct InvalidProbabilityVectorError <: Exception
    vector_name::String
    sum_value::Float64
    has_negative::Bool
    has_greater_than_one::Bool
end

function Base.showerror(io::IO, e::InvalidProbabilityVectorError)
    print(io, "InvalidProbabilityVectorError: ")
    print(io, "$(e.vector_name) is not a valid probability vector. ")
    if !isapprox(e.sum_value, 1.0; atol=1e-10)
        print(io, "Sum is $(e.sum_value), not 1.0. ")
    end
    if e.has_negative
        print(io, "Contains negative values. ")
    end
    if e.has_greater_than_one
        print(io, "Contains values > 1.0.")
    end
end

"""
    NumericalStabilityError <: Exception

Custom exception for numerical stability issues.

# Fields
- `parameter::String`: Name of the parameter
- `issue::String`: Description of the numerical issue
"""
struct NumericalStabilityError <: Exception
    parameter::String
    issue::String
end

function Base.showerror(io::IO, e::NumericalStabilityError)
    print(io, "NumericalStabilityError: ")
    return print(io, "$(e.parameter) - $(e.issue)")
end

"""
    _validate_state_model(state_model::GaussianStateModel{T}, latent_dim::Int) where T

Validate GaussianStateModel parameters. Throws exceptions on validation failure.

# Throws
- `DimensionMismatchError`: If dimensions don't match expected values
- `NotSymmetricError`: If covariance matrices aren't symmetric
- `NotPositiveDefiniteError`: If covariance matrices aren't positive definite
"""
function _validate_state_model(
    state_model::GaussianStateModel{T}, latent_dim::Int
) where {T}
    # Check A matrix
    if size(state_model.A) != (latent_dim, latent_dim)
        throw(
            DimensionMismatchError(
                "A matrix", (latent_dim, latent_dim), size(state_model.A)
            ),
        )
    end

    # Check optional B matrix (dynamics input)
    if size(state_model.B, 1) != latent_dim
        throw(DimensionMismatchError("B matrix rows", latent_dim, size(state_model.B, 1)))
    end

    # Check Q matrix (process noise covariance)
    if size(state_model.Q) != (latent_dim, latent_dim)
        throw(
            DimensionMismatchError(
                "Q matrix", (latent_dim, latent_dim), size(state_model.Q)
            ),
        )
    end

    if !issymmetric(state_model.Q)
        max_asym = maximum(abs.(state_model.Q - state_model.Q'))
        throw(NotSymmetricError("Q matrix", max_asym))
    end

    if !isposdef(state_model.Q)
        min_eval = minimum(eigvals(state_model.Q))
        throw(NotPositiveDefiniteError("Q matrix", min_eval))
    end

    # Check bias vector b
    if length(state_model.b) != latent_dim
        throw(DimensionMismatchError("bias vector b", latent_dim, length(state_model.b)))
    end

    # Check initial state x0
    if length(state_model.x0) != latent_dim
        throw(
            DimensionMismatchError("initial state x0", latent_dim, length(state_model.x0))
        )
    end

    # Check P0 matrix (initial covariance)
    if size(state_model.P0) != (latent_dim, latent_dim)
        throw(
            DimensionMismatchError(
                "P0 matrix", (latent_dim, latent_dim), size(state_model.P0)
            ),
        )
    end

    if !issymmetric(state_model.P0)
        max_asym = maximum(abs.(state_model.P0 - state_model.P0'))
        throw(NotSymmetricError("P0 matrix", max_asym))
    end

    if !isposdef(state_model.P0)
        min_eval = minimum(eigvals(state_model.P0))
        throw(NotPositiveDefiniteError("P0 matrix", min_eval))
    end

    return nothing
end

"""
    _validate_obs_model(obs_model::GaussianObservationModel{T}, obs_dim::Int, latent_dim::Int) where T

Validate GaussianObservationModel parameters. Throws exceptions on validation failure.

# Throws
- `DimensionMismatchError`: If dimensions don't match expected values
- `NotSymmetricError`: If R matrix isn't symmetric
- `NotPositiveDefiniteError`: If R matrix isn't positive definite
"""
function _validate_obs_model(
    obs_model::GaussianObservationModel{T}, obs_dim::Int, latent_dim::Int
) where {T}
    # Check C matrix
    if size(obs_model.C) != (obs_dim, latent_dim)
        throw(DimensionMismatchError("C matrix", (obs_dim, latent_dim), size(obs_model.C)))
    end

    # Check R matrix (observation noise covariance)
    if size(obs_model.R) != (obs_dim, obs_dim)
        throw(DimensionMismatchError("R matrix", (obs_dim, obs_dim), size(obs_model.R)))
    end

    # TODO: check D matrix

    if !issymmetric(obs_model.R)
        max_asym = maximum(abs.(obs_model.R - obs_model.R'))
        throw(NotSymmetricError("R matrix", max_asym))
    end

    if !isposdef(obs_model.R)
        min_eval = minimum(eigvals(obs_model.R))
        throw(NotPositiveDefiniteError("R matrix", min_eval))
    end

    # Check bias vector d
    if length(obs_model.d) != obs_dim
        throw(DimensionMismatchError("observation bias d", obs_dim, length(obs_model.d)))
    end

    return nothing
end

"""
    _validate_obs_model(obs_model::PoissonObservationModel{T}, obs_dim::Int, latent_dim::Int) where T

Validate PoissonObservationModel parameters. Throws exceptions on validation failure.

# Throws
- `DimensionMismatchError`: If dimensions don't match expected values
- `NumericalStabilityError`: If `d` values are extremely large/small
"""
function _validate_obs_model(
    obs_model::PoissonObservationModel{T}, obs_dim::Int, latent_dim::Int
) where {T}
    # Check C matrix
    if size(obs_model.C) != (obs_dim, latent_dim)
        throw(DimensionMismatchError("C matrix", (obs_dim, latent_dim), size(obs_model.C)))
    end

    # Check d vector
    if length(obs_model.d) != obs_dim
        throw(DimensionMismatchError("d vector", obs_dim, length(obs_model.d)))
    end

    # Check that d values are reasonable. `d` enters the linear predictor as
    # `λ = exp(C x + d)`; |d| above ~50 risks exp overflow/underflow once Cx
    # is added on top.
    if any(x -> abs(x) > 50, obs_model.d)  # exp(50) ≈ 5e21, exp(-50) ≈ 2e-22
        max_val = maximum(abs.(obs_model.d))
        throw(
            NumericalStabilityError(
                "d vector",
                "contains extremely large/small values (max |d| = $max_val), may cause numerical overflow/underflow",
            ),
        )
    end

    return nothing
end

"""
    validate_LDS(lds::LinearDynamicalSystem{T,S,O}) where {T,S,O}

Validate that all parameters in a LinearDynamicalSystem are dimensionally consistent
and mathematically valid. Throws descriptive exceptions on validation failure.

# Checks performed
- Matrix dimensions are consistent
- Covariance matrices are positive definite and symmetric
- fit_bool has correct length for the observation model type
- Stored dimensions match dimensions inferred from matrices

# Throws
- `DimensionMismatchError`: If dimensions don't match
- `NotPositiveDefiniteError`: If covariance matrices aren't positive definite
- `NotSymmetricError`: If covariance matrices aren't symmetric
- `NumericalStabilityError`: If numerical issues are detected

# Examples
```julia
# This will throw DimensionMismatchError if invalid
validate_LDS(my_lds)

# Can be caught for custom handling
try
    validate_LDS(my_lds)
    println("LDS is valid!")
catch e
    if e isa DimensionMismatchError
        println("Dimension error: ", e)
    end
end
```
"""
function validate_LDS(lds::LinearDynamicalSystem{T,S,O}) where {T,S,O}
    # Check state model dimensions and properties
    _validate_state_model(lds.state_model, lds.latent_dim)

    # Check observation model dimensions and properties
    _validate_obs_model(lds.obs_model, lds.obs_dim, lds.latent_dim)

    # Check fit_bool length. Gaussian path (BTD and Kalman) uses length 6 —
    # the regression M-step fits A&b&B and C&d&D jointly, so flag layout is
    # the same across backends.
    expected_fit_length = lds.obs_model isa PoissonObservationModel ? 5 : 6
    if length(lds.fit_bool) != expected_fit_length
        throw(DimensionMismatchError("fit_bool", expected_fit_length, length(lds.fit_bool)))
    end

    # Check consistency between inferred and stored dimensions
    inferred_latent = size(lds.state_model.A, 1)
    inferred_obs = size(lds.obs_model.C, 1)

    if lds.latent_dim != inferred_latent
        throw(
            DimensionMismatchError(
                "latent_dim (stored vs inferred from A)", inferred_latent, lds.latent_dim
            ),
        )
    end

    if lds.obs_dim != inferred_obs
        throw(
            DimensionMismatchError(
                "obs_dim (stored vs inferred from C)", inferred_obs, lds.obs_dim
            ),
        )
    end

    return nothing
end

"""
    validate_SLDS(slds::SLDS)

Validate SLDS structure. Throws descriptive exceptions on validation failure.

# Checks performed
- Dimensions of A match the length of πₖ and the number of LDSs
- Rows of A and πₖ are valid probability vectors
- Each LDS has the same state dimension and observation dimension
- Each individual LDS is valid

# Throws
- `DimensionMismatchError`: If dimensions are inconsistent
- `InvalidProbabilityVectorError`: If probability vectors are invalid
- Other exceptions from `validate_LDS` for individual LDS validation

# Examples
```julia
# This will throw if invalid
validate_SLDS(my_slds)

# Can be caught for custom handling
try
    validate_SLDS(my_slds)
catch e
    if e isa InvalidProbabilityVectorError
        println("Probability vector error: ", e)
    end
end
```
"""
function validate_SLDS(slds::SLDS)
    k = size(slds.A, 1)
    D = length(slds.πₖ)
    lds_count = length(slds.LDSs)

    # Checks for HMM components
    if k != D
        throw(DimensionMismatchError("size(A, 1) vs length(πₖ)", D, k))
    end

    if k != lds_count
        throw(DimensionMismatchError("size(A, 1) vs number of LDSs", lds_count, k))
    end

    # Validate transition matrix rows
    for i in 1:k
        row = slds.A[i, :]
        sum_val = sum(row)
        has_neg = any(x -> x < 0, row)
        has_gt1 = any(x -> x > 1, row)

        if !isapprox(sum_val, 1.0; atol=1e-10) || has_neg || has_gt1
            throw(InvalidProbabilityVectorError("A[$i, :]", sum_val, has_neg, has_gt1))
        end
    end

    # Validate initial distribution
    sum_val = sum(slds.πₖ)
    has_neg = any(x -> x < 0, slds.πₖ)
    has_gt1 = any(x -> x > 1, slds.πₖ)

    if !isapprox(sum_val, 1.0; atol=1e-10) || has_neg || has_gt1
        throw(InvalidProbabilityVectorError("πₖ", sum_val, has_neg, has_gt1))
    end

    # Checks for LDS models
    latent_dim = slds.LDSs[1].latent_dim
    obs_dim = slds.LDSs[1].obs_dim

    for (i, lds) in enumerate(slds.LDSs)
        if lds.latent_dim != latent_dim
            throw(DimensionMismatchError("LDS[$i].latent_dim", latent_dim, lds.latent_dim))
        end

        if lds.obs_dim != obs_dim
            throw(DimensionMismatchError("LDS[$i].obs_dim", obs_dim, lds.obs_dim))
        end

        # This will throw if invalid
        validate_LDS(lds)
    end

    return nothing
end

"""
    validate_probvec(v::AbstractVector{T}; name::String="vector") where {T<:Real}

Validate that a vector is a valid probability vector (sums to 1, all non-negative, all ≤ 1).
Throws `InvalidProbabilityVectorError` if validation fails.

# Arguments
- `v`: The vector to validate
- `name`: Optional name for the vector (used in error messages)

# Examples
```julia
v1 = [0.3, 0.5, 0.2]
validate_probvec(v1)  # No error

v2 = [0.3, 0.5, 0.3]
validate_probvec(v2)  # Throws InvalidProbabilityVectorError
```
"""
function validate_probvec(v::AbstractVector{T}; name::String="vector") where {T<:Real}
    sum_val = sum(v)
    has_neg = any(x -> x < 0, v)
    has_gt1 = any(x -> x > 1, v)

    if !isapprox(sum_val, one(T); atol=1e-10) || has_neg || has_gt1
        throw(InvalidProbabilityVectorError(name, sum_val, has_neg, has_gt1))
    end

    return nothing
end

# ============================================================================
# input-sequence normalization helpers. The public `latent_inputs`/`obs_inputs`
# kwargs accept either `nothing` (no inputs — must match a zero-column `B`/`D`)
# or per-trial matrices. Internally every sampler/smoother/M-step expects an
# `AbstractMatrix{T}` of shape `(ux_dim, T_i)` (possibly `0 × T_i`), so these
# helpers validate the supplied sequences and canonicalize on the way in.
# ============================================================================

function _check_latent_inputs(
    cs::Nothing, expected_dim::Int, tsteps::Int, name::AbstractString, ::Type{T}
) where {T}
    expected_dim == 0 || throw(
        ArgumentError(
            "$(name)=nothing is only valid when the corresponding input matrix is " *
            "zero-column; got expected_dim=$(expected_dim). Pass a $(expected_dim)×T " *
            "matrix or shrink the input matrix.",
        ),
    )
    return zeros(T, 0, tsteps)
end

function _check_latent_inputs(
    cs::AbstractMatrix{T}, expected_dim::Int, tsteps::Int, name::AbstractString, ::Type{T}
) where {T<:Real}
    size(cs, 1) == expected_dim || throw(
        DimensionMismatchError(
            "$(name) rows vs input-matrix cols", expected_dim, size(cs, 1)
        ),
    )
    size(cs, 2) == tsteps ||
        throw(DimensionMismatchError("$(name) tsteps", tsteps, size(cs, 2)))
    return cs
end

@inline function _check_obs_inputs(
    cs, expected_dim::Int, tsteps::Int, ::GaussianObservationModel{T}
) where {T}
    return _check_latent_inputs(cs, expected_dim, tsteps, "obs_inputs", T)
end

@inline function _check_obs_inputs(
    cs::Nothing, expected_dim::Int, tsteps::Int, ::PoissonObservationModel{T}
) where {T}
    expected_dim == 0 || error(
        "Poisson observation model does not support obs_inputs (expected_dim must be 0)"
    )
    return zeros(T, 0, tsteps)
end

function _normalize_multitrial_latent_inputs(
    cs::Nothing, expected_dim::Int, tsteps_per_trial, ::Type{T}, name::AbstractString
) where {T<:Real}
    expected_dim == 0 || throw(
        ArgumentError(
            "$(name)=nothing is only valid when expected_dim == 0; got $(expected_dim)"
        ),
    )
    return [zeros(T, 0, Int(Ti)) for Ti in tsteps_per_trial]
end

function _normalize_multitrial_latent_inputs(
    cs::AbstractVector{<:AbstractMatrix{T}},
    expected_dim::Int,
    tsteps_per_trial,
    ::Type{T},
    name::AbstractString,
) where {T<:Real}
    length(cs) == length(tsteps_per_trial) || throw(
        DimensionMismatchError("$(name) ntrials", length(tsteps_per_trial), length(cs))
    )
    for (i, ci) in enumerate(cs)
        size(ci, 1) == expected_dim ||
            throw(DimensionMismatchError("$(name)[$i] rows", expected_dim, size(ci, 1)))
        size(ci, 2) == Int(tsteps_per_trial[i]) || throw(
            DimensionMismatchError(
                "$(name)[$i] tsteps", Int(tsteps_per_trial[i]), size(ci, 2)
            ),
        )
    end
    return cs
end

@inline function _normalize_multitrial_obs_inputs(
    cs, expected_dim::Int, tsteps_per_trial, ::Type{T}, ::GaussianObservationModel
) where {T<:Real}
    return _normalize_multitrial_latent_inputs(
        cs, expected_dim, tsteps_per_trial, T, "obs_inputs"
    )
end

@inline function _normalize_multitrial_obs_inputs(
    cs::Nothing, expected_dim::Int, tsteps_per_trial, ::Type{T}, ::PoissonObservationModel
) where {T<:Real}
    expected_dim == 0 || error(
        "Poisson observation model does not support obs_inputs (expected_dim must be 0)"
    )
    return [zeros(T, 0, Int(Ti)) for Ti in tsteps_per_trial]
end
