export HMMInstance

# Define struct for storing glmhmm benchmarking params
@kwdef struct HMMInstance
    num_states::Int
    num_trials::Int
    seq_length::Int
    input_dim::Int
    output_dim::Int
end

@kwdef struct LDSInstance
    latent_dim::Int
    obs_dim::Int
    num_trials::Int
    seq_length::Int
end