export SSDImplem, build_model

struct SSDImplem <: Implementation end
Base.string(::SSDImplem) = "StateSpaceDynamics.jl"

function build_model(::SSDImplem, instance::Instance, params::Params)
    # Read in instance and params
    (; num_states, num_trials, seq_length, input_dim, output_dim) = instance
    (; πₖ, A, β) = params

    # Build the Bernoulli regression emission models
    B = Vector{StateSpaceDynamics.BernoulliRegressionEmission}(undef, num_states)

    for i=1:size(B,1)
        B[i] = BernoulliRegressionEmission(; input_dim=input_dim, output_dim=output_dim, include_intercept=false, β=β[i], λ=0.0)
    end

    # Build the HMM
    return StateSpaceDynamics.HiddenMarkovModel(K=num_states, A=A, πₖ=πₖ, B=B)

end
