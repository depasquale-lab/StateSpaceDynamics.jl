export SwitchingLinearDynamicalSystem, fit!, loglikelihood, sample, variational_expectation

"""
Switching Linear Dynamical System
"""
mutable struct SwitchingLinearDynamicalSystem
    A::Matrix{<:Real}                 # Transition matrix for mode switching
    B::Vector{LinearDynamicalSystem}  # Vector of Linear Dynamical System models
    πₖ::Vector{Float64}               # Initial state distribution
    K::Int                            # Number of modes
end


"""
Generate synthetic data with switching LDS models
"""
function sample(slds, T::Int)
    state_dim = slds.B[1].latent_dim
    obs_dim = slds.B[1].obs_dim
    K = slds.K

    x = zeros(state_dim, T)  # Latent states
    y = zeros(obs_dim, T)   # Observations
    z = zeros(Int, T)       # Mode sequence

    # Sample initial mode
    z[1] = rand(Categorical(slds.πₖ / sum(slds.πₖ)))
    #z[1] = sample(1:K, Weights(slds.πₖ))
    x[:, 1] = rand(MvNormal(zeros(state_dim), slds.B[z[1]].state_model.Q))
    y[:, 1] = rand(MvNormal(slds.B[z[1]].obs_model.C * x[:, 1], slds.B[z[1]].obs_model.R))

    for t in 2:T
        # Sample mode based on transition probabilities
        z[t] = rand(Categorical(slds.A[z[t-1], :] ./ sum(slds.A[z[t-1], :])))
        #z[t] = sample(1:K, Weights(slds.A[z[t - 1], :]))
        # Update latent state and observation
        x[:, t] = rand(MvNormal(slds.B[z[t]].state_model.A * x[:, t-1], slds.B[z[t]].state_model.Q))
        y[:, t] = rand(MvNormal(slds.B[z[t]].obs_model.C * x[:, t], slds.B[z[t]].obs_model.R))
    end

    return x, y, z
    
end


"""
Initialize a Switching Linear Dynamical System with random parameters.
"""
function initialize_slds(;K::Int=2, d::Int=2, p::Int=10, seed::Int=42)
    Random.seed!(seed)

    A = rand(K, K)
    A ./= sum(A, dims=2) # Normalize rows to sum to 1

    πₖ = rand(K)
    πₖ ./= sum(πₖ) # Normalize to sum to 1

    # set up the state parameters
    A2 = 0.95 * [cos(0.25) -sin(0.25); sin(0.25) cos(0.25)] 
    Q = Matrix(0.1 * I(d))

    x0 = [0.0; 0.0]
    P0 = Matrix(0.1 * I(d))

    # set up the observation parameters
    C = randn(p, d)
    R = Matrix(0.5 * I(p))

    B = [LinearDynamicalSystem(
        GaussianStateModel(A2, Q, x0, P0),
        GaussianObservationModel(C, R),
        d, p, fill(true, 6  )) for _ in 1:K]

    return SwitchingLinearDynamicalSystem(A, B, πₖ, K)

end

function weighted_loglikelihood(
    x::AbstractMatrix{T}, lds::LinearDynamicalSystem{S,O}, y::AbstractMatrix{U},
    h::Vector{Float64}  # h[t] responsibilities for state m
) where {T<:Real, U<:Real, S<:GaussianStateModel{<:Real}, O<:GaussianObservationModel{<:Real}}

    T_steps = size(y, 2)
    A, Q, x0, P0 = lds.state_model.A, lds.state_model.Q, lds.state_model.x0, lds.state_model.P0
    C, R = lds.obs_model.C, lds.obs_model.R 

    # Pre-compute Cholesky factors for efficiency
    R_chol = cholesky(Symmetric(R)).U
    Q_chol = cholesky(Symmetric(Q)).U
    P0_chol = cholesky(Symmetric(P0)).U

    # Initial state contribution
    dx0 = view(x, :, 1) - x0
    # Replace dx0' * inv_P0 * dx0 with equivalent using Cholesky
    ll = h[1] * sum(abs2, P0_chol \ dx0)

    # Initialize temporary variables
    temp_dx = zeros(T, size(x, 1))
    temp_dy = zeros(promote_type(T, U), size(y, 1))

    # Create temporaries with the same element type as x
    temp_dx = zeros(T, size(x, 1))
    temp_dy = zeros(promote_type(T, U), size(y, 1))

    @inbounds for t in 1:T_steps
        if t > 1
            mul!(temp_dx, A, view(x, :, t-1), -1.0, false)
            temp_dx .+= view(x, :, t)
            # Replace temp_dx' * inv_Q * temp_dx
            ll += h[t] * sum(abs2, Q_chol \ temp_dx)
        end
        mul!(temp_dy, C, view(x, :, t), -1.0, false)
        temp_dy .+= view(y, :, t)
        # Replace temp_dy' * inv_R * temp_dy
        ll += h[t] * sum(abs2, R_chol \ temp_dy)
    end
    return -0.5 * ll
end