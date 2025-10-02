export isvalid_SLDS, isvalid_LDS, isvalid_probvec

"""
    isvalid_LDS(lds::LinearDynamicalSystem{T,S,O}) where {T,S,O}

Validate that all parameters in a LinearDynamicalSystem are dimensionally consistent
and mathematically valid.

# Checks performed:
- Matrix dimensions are consistent
- Covariance matrices are positive definite
- fit_bool has correct length for the observation model type
- All parameters have consistent data types
"""
function isvalid_LDS(lds::LinearDynamicalSystem{T,S,O}) where {T,S,O}
    try
        # Check state model dimensions and properties
        if !_isvalid_state_model(lds.state_model, lds.latent_dim)
            return false
        end

        # Check observation model dimensions and properties
        if !_isvalid_obs_model(lds.obs_model, lds.obs_dim, lds.latent_dim)
            return false
        end

        # Check fit_bool length
        expected_fit_length = lds.obs_model isa PoissonObservationModel ? 5 : 6
        if length(lds.fit_bool) != expected_fit_length
            @warn "fit_bool has length $(length(lds.fit_bool)), expected $expected_fit_length"
            return false
        end

        # Check consistency between inferred and stored dimensions
        inferred_latent = size(lds.state_model.A, 1)
        inferred_obs = size(lds.obs_model.C, 1)

        if lds.latent_dim != inferred_latent
            @warn "Stored latent_dim ($(lds.latent_dim)) ≠ inferred from A matrix ($inferred_latent)"
            return false
        end

        if lds.obs_dim != inferred_obs
            @warn "Stored obs_dim ($(lds.obs_dim)) ≠ inferred from C matrix ($inferred_obs)"
            return false
        end

        return true

    catch e
        @warn "Error during LDS validation: $e"
        return false
    end
end

"""
    _isvalid_state_model(state_model::GaussianStateModel{T}, latent_dim::Int) where T

Validate GaussianStateModel parameters.
"""
function _isvalid_state_model(state_model::GaussianStateModel{T}, latent_dim::Int) where {T}
    # Check A matrix
    if size(state_model.A) != (latent_dim, latent_dim)
        @warn "A matrix size $(size(state_model.A)) ≠ expected ($latent_dim, $latent_dim)"
        return false
    end

    # Check Q matrix (process noise covariance)
    if size(state_model.Q) != (latent_dim, latent_dim)
        @warn "Q matrix size $(size(state_model.Q)) ≠ expected ($latent_dim, $latent_dim)"
        return false
    end

    if !issymmetric(state_model.Q)
        @warn "Q matrix is not symmetric"
        return false
    end

    if !isposdef(state_model.Q)
        @warn "Q matrix is not positive definite"
        return false
    end

    # Check bias vector b
    if length(state_model.b) != latent_dim
        @warn "Bias vector b length $(length(state_model.b)) ≠ expected $latent_dim"
        return false
    end

    # Check initial state x0
    if length(state_model.x0) != latent_dim
        @warn "Initial state x0 length $(length(state_model.x0)) ≠ expected $latent_dim"
        return false
    end

    # Check P0 matrix (initial covariance)
    if size(state_model.P0) != (latent_dim, latent_dim)
        @warn "P0 matrix size $(size(state_model.P0)) ≠ expected ($latent_dim, $latent_dim)"
        return false
    end

    if !issymmetric(state_model.P0)
        @warn "P0 matrix is not symmetric"
        return false
    end

    if !isposdef(state_model.P0)
        @warn "P0 matrix is not positive definite"
        return false
    end

    return true
end

"""
    _isvalid_obs_model(obs_model::GaussianObservationModel{T}, obs_dim::Int, latent_dim::Int) where T

Validate GaussianObservationModel parameters.
"""
function _isvalid_obs_model(
    obs_model::GaussianObservationModel{T}, obs_dim::Int, latent_dim::Int
) where {T}
    # Check C matrix
    if size(obs_model.C) != (obs_dim, latent_dim)
        @warn "C matrix size $(size(obs_model.C)) ≠ expected ($obs_dim, $latent_dim)"
        return false
    end

    # Check R matrix (observation noise covariance)
    if size(obs_model.R) != (obs_dim, obs_dim)
        @warn "R matrix size $(size(obs_model.R)) ≠ expected ($obs_dim, $obs_dim)"
        return false
    end

    if !issymmetric(obs_model.R)
        @warn "R matrix is not symmetric"
        return false
    end

    if !isposdef(obs_model.R)
        @warn "R matrix is not positive definite"
        return false
    end

    # Check bias vector d
    if length(obs_model.d) != obs_dim
        @warn "Observation bias d length $(length(obs_model.d)) ≠ expected $obs_dim"
        return false
    end

    return true
end

"""
    _isvalid_obs_model(obs_model::PoissonObservationModel{T}, obs_dim::Int, latent_dim::Int) where T

Validate PoissonObservationModel parameters.
"""
function _isvalid_obs_model(
    obs_model::PoissonObservationModel{T}, obs_dim::Int, latent_dim::Int
) where {T}
    # Check C matrix
    if size(obs_model.C) != (obs_dim, latent_dim)
        @warn "C matrix size $(size(obs_model.C)) ≠ expected ($obs_dim, $latent_dim)"
        return false
    end

    # Check log_d vector
    if length(obs_model.log_d) != obs_dim
        @warn "log_d vector length $(length(obs_model.log_d)) ≠ expected $obs_dim"
        return false
    end

    # Check that log_d values are reasonable (not extremely large/small)
    if any(x -> abs(x) > 50, obs_model.log_d)  # exp(50) ≈ 5e21, exp(-50) ≈ 2e-22
        @warn "Some log_d values are extremely large/small, may cause numerical issues"
        return false
    end

    return true
end

"""
    isvalid_probvec(v::AbstractVector{T}) where {T<:Real}

Check if vector is a valid probability vector (sums to 1, all non-negative).
"""
function isvalid_probvec(v::AbstractVector{T}) where {T<:Real}
    return isapprox(sum(v), one(T); atol=1e-10) &&
           all(x -> x ≥ zero(T), v) &&
           all(x -> x ≤ one(T), v)
end

"""
    isvalid_SLDS(slds::SLDS)

Check if SLDS structure is valid under the following criteria:
    - Dims of A match the length of Z₀ and the number of LDSs
    - Rows of A and Z₀ sum to 1
    - Each LDS has the same state dimension and observation dimension
 """
function isvalid_SLDS(slds::SLDS)
    k = size(slds.A, 1)
    D = length(slds.Z₀)
    lds_count = length(slds.LDSs)

    # Checks for HMM components
    @assert k == D "Dimension mismatch: size(A, 1) must equal length(Z₀)"
    @assert k == lds_count "Dimension mismatch: size(A, 1) must equal number of LDSs"

    for i in 1:k
        @assert isprobvec(slds.A[i, :]) "Row $i of A is not a valid probability vector"
    end

    @assert isprobvec(slds.Z₀) "Z₀ is not a valid probability vector"

    # Checks for LDS models
    latent_dim = slds.LDSs[1].latent_dim
    obs_dim = slds.LDSs[1].obs_dim
    for (i, lds) in enumerate(slds.LDSs)
        @assert lds.latent_dim == latent_dim "LDS $i has inconsistent latent_dim"
        @assert lds.obs_dim == obs_dim "LDS $i has inconsistent obs_dim"
        @assert isvalid_LDS(lds) "LDS $i is not valid"
    end
    return true
end

