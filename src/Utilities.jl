export kmeanspp_initialization, kmeans_clustering

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

""" Factor Analysis and by special case PPCA"""
struct FactorAnalysis
    W::Matrix{Float64}
    σ2::Vector{Float64}
    μ::Vector{Float64}
    k::Int
    D::Int
end