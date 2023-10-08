export HMM
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


function forward(hmm::HMM, data::Matrix{Float64})
    T, _ = size(data)
    K = size(hmm.A, 1)  # Number of states

    # Initialize an α-matrix 
    α = zeros(Float64, K, T)

    # Calculate α₁
    for k in 1:K
        α[k, 1] = hmm.πₖ[k] * hmm.B[k].likelihood(data[1, :])
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

function backward(hmm::HMM, data::Matrix{Float64})
    T, _ = size(data)
    K = size(hmm.A, 1)

    # Initialize a β matrix
    β = zeros(Float64, K, T)
    
    # Set last β equal to 1 for all states
    β[:, T] .= 1

    # Calculate β
    for t in T-1:-1:1
        for i in 1:K
            sum_beta_b = 0.0
            for j in 1:K
                sum_beta_b += hmm.A[i, j] * hmm.B[j].likelihood(data[t+1, :]) * β[j, t+1]
            end
            β[i, t] = sum_beta_b
        end
    end

    return β
end


function baumWelch!(hmm::HMM,  data::Matrix{Float64}, max_iters::Int=100)
    T, _ = size(data)
    K = size(hmm.A, 1)
    # EM via BaumWelch algorithm
    for iter in 1:max_iters
        α = forward(hmm, data)
        β = backward(hmm, data)
        # Calculate proabilities according to Bayes rule, i.e. E-Step
        γ = α .* β
        γ ./= sum(γ, dims=1)
        # Now we calculate ξ values
        ξ = zeros(Float64, K, K, T-1)
        for t in 1:T-1
            # Dummy denominator variable so after ξ calculatiosn we can just divide as opposed to recalculating anything
            denominator = 0.0
            for i in 1:K
                for j in 1:K
                    ξ[i, j, t] = α[i, t] * hmm.A[i, j] * likelihood(hmm.B[j], data[t+1, :]) * β[j, t+1]
                    denominator += ξ[i, j, t]
                end
            end
            ξ[:, :, t] ./= denominator
        end
        # M-Step; update our parameters based on E-Step
        # Update initial state probabilities
        hmm.πₖ .= γ[:,1]
        # Update transition probabilities
        for i in 1:K
            for j in 1:K
                hmm.A[i,j] = sum(ξ[i,j,:]) / sum(γ[i,1:T-1])
            end
        end
        for k in 1:K
            hmm.B[k] = updateEmissionModel!(hmm.B[k], data, γ[k,:])
        end
    end
end

function fit!()
    #TODO Implement the viterbi algorithm.
end