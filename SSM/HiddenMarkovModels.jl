
# HMM Definition
struct HMM{EM <: EmissionsModel}
    A::Matrix{Float64}  # State Transition Matrix
    B::EM               # Emission Model
    πₖ ::Vector{Float64}  # Initial State Distribution
    D::Int              # Observation Dimension
end

function HMM(data::Matrix{Float64}, k_states::Int=2, emissions::String="Gaussian")
    N, D = size(data)
    # Initialize A
    A = rand(k_states, k_states)
    A = A ./ sum(A, dims=2)  # normalize rows to ensure they are valid probabilities
    # Initialize π
    πₖ = rand(k_states)
    πₖ = πₖ ./ sum(πₖ)          # normalize to ensure it's a valid probability vector
    # Initialize Emission Model
    if emissions == "Gaussian"
        # Randomly sample k_states observations from data
        sample_means = data[sample(1:N, k_states, replace=false), :]
        # Using identity matrix for covariance
        sample_covs = [Matrix(I, D, D) for _ in 1:k_states]
        B = [GaussianEmission(sample_means[i, :], sample_covs[i]) for i in 1:k_states]
    else
        throw(ErrorException("$emissions is not a supported emissions model, please choose one of the supported models."))
    end
    return HMM(A, B, πₖ, D)
end


function forward(hmm::HMM, data::matrix{Float64})
    T, _ = size(data)
    # Initialize an α-vector 
    α = zeros(hmm.k_states, T)
    # Calculate α₁
    for k in 1:K
        α[k, 1] = hmm.πₖ[k] * hmm.B.likelihood(data[1, :])
    end
    # Now perform the rest of the forward algorithm for t=2 to T
    for t in 2:T
        for j in 1:K
            sum_alpha_a = 0.0
            for i in 1:K
                sum_alpha_a += α[i, t-1] * hmm.A[i, j]
            end
            α[j, t] = sum_alpha_a * hmm.B[j].likelihood(data[t, :])
        end
    end

    return α
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