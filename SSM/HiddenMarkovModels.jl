
# HMM Definition
struct HMM{EM <: EmissionModel}
    A::Matrix{Float64}  # State Transition Matrix
    B::EM               # Emission Model
    π::Vector{Float64}  # Initial State Distribution
end

function forward()
    #TODO Implement
end

function backward()
    #TODO Implement
end

function baumWelch!()
    #TODO Implement
end

function fit!()
    #TODO Implement the viterbi algorithm.
end