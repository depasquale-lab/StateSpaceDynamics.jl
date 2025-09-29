export SLDS
@kwdef struct SLDS{T<:Real,
                   S<:AbstractStateModel,
                   O<:AbstractObservationModel,
                   TM<:AbstractMatrix{T},
                   ISV<:AbstractVector{T}} 
    A::TM                                  
    Z₀::ISV                                 
    LDSs::Vector{LinearDynamicalSystem{T,S,O}} 
end

"""
    rand(rng::AbstractRNG, slds::SLDS{T,S,O}; tsteps::Int, ntrials::Int=1) where {T<:Real, S<:AbstractStateModel, O<:AbstractObservationModel}

Sample from a Switching Linear Dynamical System (SLDS). Returns a tuple `(z, x, y)` where:
- `z` is a matrix of discrete states of size `(tsteps, ntrials)`
- `x` is a 3D array of continuous latent states of size `(latent_dim, tsteps, ntrials)`
- `y` is a 3D array of observations of size `(obs_dim, tsteps, ntrials)`
"""
function Random.rand(
    rng::AbstractRNG, 
    slds::SLDS{T,S,O}; 
    tsteps::Int, 
    ntrials::Int = 1
) where {T<:Real, S<:AbstractStateModel, O<:AbstractObservationModel}
    
    K = length(slds.LDSs)  # Number of discrete states
    latent_dim = slds.LDSs[1].latent_dim
    obs_dim = slds.LDSs[1].obs_dim
    
    # Pre-allocate outputs
    z = Array{Int,2}(undef, tsteps, ntrials)        # Discrete states
    x = Array{T,3}(undef, latent_dim, tsteps, ntrials)  # Continuous states  
    y = Array{T,3}(undef, obs_dim, tsteps, ntrials)     # Observations
    
    # Pre-extract parameters for all LDS models (avoid repeated extraction)
    state_params = [_extract_state_params(lds.state_model) for lds in slds.LDSs]
    obs_params = [_extract_obs_params(lds.obs_model) for lds in slds.LDSs]
    
    # Sample each trial
    for trial in 1:ntrials
        _sample_slds_trial!(rng, 
                           view(z, :, trial), 
                           view(x, :, :, trial), 
                           view(y, :, :, trial),
                           slds.A, slds.Z₀, state_params, obs_params, 
                           slds.LDSs[1].obs_model)  # Use first for type dispatch
    end
    
    return z, x, y
end

# Core SLDS trial sampling logic
function _sample_slds_trial!(rng, z_trial, x_trial, y_trial, A, Z₀, state_params, obs_params, obs_model_type)
    tsteps = length(z_trial)
    K = size(A, 1)
    
    # Sample discrete state sequence using forward sampling
    z_trial[1] = rand(rng, Categorical(Z₀))
    for t in 2:tsteps
        z_trial[t] = rand(rng, Categorical(A[z_trial[t-1], :]))
    end
    
    # Sample continuous states and observations given discrete sequence
    _sample_continuous_given_discrete!(rng, x_trial, y_trial, z_trial, 
                                      state_params, obs_params, obs_model_type)
end

# Sample continuous dynamics given discrete state sequence
function _sample_continuous_given_discrete!(rng, x_trial, y_trial, z_trial, 
                                           state_params, obs_params, obs_model_type::GaussianObservationModel)
    tsteps = length(z_trial)
    
    # Initial state from the selected LDS
    k1 = z_trial[1]
    x_trial[:, 1] = rand(rng, MvNormal(state_params[k1].x0, state_params[k1].P0))
    y_trial[:, 1] = rand(rng, MvNormal(obs_params[k1].C * x_trial[:, 1] + obs_params[k1].d, 
                                      obs_params[k1].R))
    
    # Subsequent states - switch dynamics based on discrete state
    for t in 2:tsteps
        k_prev, k_curr = z_trial[t-1], z_trial[t]
        
        # Continuous state follows previous discrete state's dynamics
        x_trial[:, t] = rand(rng, MvNormal(state_params[k_prev].A * x_trial[:, t-1] + state_params[k_prev].b, 
                                          state_params[k_prev].Q))
        
        # Observation follows current discrete state's model
        y_trial[:, t] = rand(rng, MvNormal(obs_params[k_curr].C * x_trial[:, t] + obs_params[k_curr].d, 
                                          obs_params[k_curr].R))
    end
end

function _sample_continuous_given_discrete!(rng, x_trial, y_trial, z_trial, 
                                           state_params, obs_params, obs_model_type::PoissonObservationModel)
    tsteps = length(z_trial)
    
    # Initial state
    k1 = z_trial[1]
    x_trial[:, 1] = rand(rng, MvNormal(state_params[k1].x0, state_params[k1].P0))
    y_trial[:, 1] = rand.(rng, Poisson.(exp.(obs_params[k1].C * x_trial[:, 1] + obs_params[k1].d)))
    
    # Subsequent states
    for t in 2:tsteps
        k_prev, k_curr = z_trial[t-1], z_trial[t]
        
        x_trial[:, t] = rand(rng, MvNormal(state_params[k_prev].A * x_trial[:, t-1] + state_params[k_prev].b, 
                                          state_params[k_prev].Q))
        
        y_trial[:, t] = rand.(rng, Poisson.(exp.(obs_params[k_curr].C * x_trial[:, t] + obs_params[k_curr].d)))
    end
end

# Convenience method without explicit RNG
Random.rand(slds::SLDS; kwargs...) = rand(Random.default_rng(), slds; kwargs...)


