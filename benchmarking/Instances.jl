export Instance

# Define struct for storing glmhmm benchmarking params
@kwdef struct Instance
    num_states::Int
    num_trials::Int
    seq_length::Int
    input_dim::Int
    output_dim::Int
end