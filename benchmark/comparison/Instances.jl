export LDSInstance

@kwdef struct LDSInstance
    latent_dim::Int
    obs_dim::Int
    num_trials::Int
    seq_length::Int
end
