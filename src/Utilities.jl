export kmeanspp_initialization, kmeans_clustering, Autoregression, fit!, loglikelihood

"""Euclidean Distance for two points"""
function euclidean_distance(a::AbstractVector{Float64}, b::AbstractVector{Float64})
    return sqrt(sum((a .- b).^2))
end

"""KMeans++ Initialization"""
function kmeanspp_initialization(data::Matrix{Float64}, k_means::Int)
    N, D = size(data)
    centroids = zeros(D, k_means)
    rand_idx = rand(1:N)
    centroids[:, 1] = data[rand_idx, :]
    for k in 2:k_means
        dists = zeros(N)
        for i in 1:N
            dists[i] = minimum([euclidean_distance(data[i, :], centroids[:, j]) for j in 1:(k-1)])
        end
        probs = dists .^ 2
        probs ./= sum(probs)
        next_idx = StatsBase.sample(1:N, Weights(probs))
        centroids[:, k] = data[next_idx, :]
    end
    return centroids
end

"""K++ Initialization for Vector input"""
function kmeanspp_initialization(data::Vector{Float64}, k_means::Int)
    # reshape data
    data = reshape(data, length(data), 1)
    return kmeanspp_initialization(data, k_means)
end

"""KMeans Clustering Initialization"""
function kmeans_clustering(data::Matrix{Float64}, k_means::Int, max_iters::Int=100, tol::Float64=1e-6)
    N, _ = size(data)
    centroids = kmeanspp_initialization(data, k_means)  # Assuming you have this function defined
    labels = zeros(Int, N)
    for iter in 1:max_iters
        # Assign each data point to the nearest cluster
        labels .= [argmin([euclidean_distance(x, c) for c in eachcol(centroids)]) for x in eachrow(data)]
        # Cache old centroids for convergence check
        old_centroids = centroids
        # Update the centroids
        new_centroids = zeros(size(centroids))
        for k in 1:k_means
            points_in_cluster = data[labels .== k, :]
            if isempty(points_in_cluster)
                # If a cluster has no points, reinitialize its centroid
                new_centroids[:, k] = data[rand(1:N), :]
            else
                new_centroids[:, k] = mean(points_in_cluster, dims=1)
            end
        end
        centroids .= new_centroids
        # Check for convergence
        if maximum([euclidean_distance(centroids[:, i], old_centroids[:, i]) for i in 1:k_means]) < tol
            # println("Converged after $iter iterations")
            break
        end
    end
    return centroids, labels
end

"""KMeans Clustering Initialization for Vector input"""
function kmeans_clustering(data::Vector{Float64}, k_means::Int, max_iters::Int=100, tol::Float64=1e-6)
    # reshape data
    data = reshape(data, length(data), 1)
    return kmeans_clustering(data, k_means, max_iters, tol)
end


"""Probabilistic Principal Component Analysis"""
struct PPCA
    W::Matrix{Float64}
    σ2::Vector{Float64}
    μ::Vector{Float64}
    k::Int
    D::Int
end

"""PPCA Constructor"""
function PPCA(X::Matrix{Float64}, k::Int)
    _, D = size(X)
    # Initialize parameters
    W = randn(D, k)
    σ2 = ones(D)
    μ = mean(X, dims=1)[:]
    return PPCA(W, σ2, μ, k, D)
end

"""
Autoregressive Model
"""
mutable struct Autoregression{T<:Real}
    X::AbstractArray # Data
    p::Int # order of the autoregressive model
    β::Vector{T} # Coefficients for the autoregressive model
    σ²::T # Variance of the noise
end

"""
Constructor for the autoregressive model.
"""
function Autoregression(X::Vector{T}, p::Int) where T<:Real
    # Initialize parameters
    β = zeros(p + 1)
    σ² = 1.0
    return Autoregression(X, p, β, σ²)
end

"""
Create lagged matrix for autoregressive model.
"""
function create_lagged_matrix(data::Vector{Float64}, p::Int)
    T = length(data)
    backward_lagged = zeros(Float64, T - p, p + 1)
    # create the backward lagged matrix
    for i in 1:p
        backward_lagged[:, i] = data[i:T-p+i-1] # i think this is right
    end
    # add a constant
    backward_lagged[:, end] = ones(T - p)
    # create the forward lagged matrix
    forward_lagged = data[p+1:end, :]
    return forward_lagged, backward_lagged
end

"""
Log-Likelihood of the autoregressive model.
"""
function loglikelihood(model::Autoregression)
    forward_lagged, backward_lagged = create_lagged_matrix(model.X, model.p)
    residuals = Vector(forward_lagged[:, 1]) - (backward_lagged * β)
    ll = logpdf(Normal(0, sqrt(model.σ²)), residuals)
    return ll
end

"""
Update σ² for the autoregressive model.
"""
function update_σ²!(model::Autoregression, residuals::Vector{Float64})
    n = length(residuals)
    model.σ² = sum(residuals.^2) / n
end

"""
Fit the model using MLE and Autodifferentiation.
"""
function fit!(model::Autoregression)
    # create lagged matrix
    forward_lagged, backward_lagged = create_lagged_matrix(model.X, model.p)
    # define loss function i.e. log-likelihood
    function loss(β)
        predictions = backward_lagged * β
        sse = sum((forward_lagged[:, 1] - predictions).^2)
        return sse
    end
    # optimize using BFGS
    result = Optim.optimize(β -> loss(β), model.β, BFGS(), autodiff=:forward)
    # update parameters
    model.β = result.minimizer
    residuals = forward_lagged[:, 1] - backward_lagged * model.β
    update_σ²!(model, residuals)
end


