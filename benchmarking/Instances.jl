export HMMInstance, LDSInstance

# Define struct for storing hmm benchmarking params
@kwdef struct HMMInstance
    num_states::Int
    num_trials::Int
    seq_length::Int
    emission_dim::Int
end

@kwdef struct LDSInstance
    latent_dim::Int
    obs_dim::Int
    num_trials::Int
    seq_length::Int
end