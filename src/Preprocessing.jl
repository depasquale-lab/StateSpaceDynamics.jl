export ProbabilisticPCA

"""Probabilistic Principal Component Analysis"""
mutable struct ProbabilisticPCA
    W::Matrix{Float64}
    σ²::Float64
    μ::Vector{Float64}
    k::Int
    D::Int
    z::Matrix{Float64}
end

"""ProbabilisticPCA Constructor"""
function ProbabilisticPCA(X::Matrix{Float64}, k::Int)
    _, D = size(X)
    # Initialize parameters
    W = randn(D, k) / sqrt(D)
    σ² = 1.0
    μ = vec(mean(X, dims=1))
    z = zeros(size(X, 1), k)
    return ProbabilisticPCA(W, σ², μ, k, D, z)
end

"""
E-Step for Probabilistic PCA

Args:
    ppca: ProbabilisticPCA object
    X: Data matrix
"""
function E_step(ppca::ProbabilisticPCA, X::Matrix{Float64})
    # get dims
    N, D = size(X)
    # preallocate E_zz and E_zz
    E_z = zeros(N, ppca.k)
    E_zz = zeros(N, ppca.k, ppca.k)
    # calculate M
    M = ppca.W' * ppca.W + (ppca.σ² * I(ppca.k))
    M_inv = cholesky(M).U \ (cholesky(M).L \ I)
    # calculate E_z and E_zz
    for i in 1:N
        E_z[i, :] = M_inv * ppca.W' * (X[i, :] - ppca.μ)
        E_zz[i, :, :] = (ppca.σ² * M_inv) + (E_z[i, :] * E_z[i, :]')
    end
    return E_z, E_zz
end

"""
M-Step for Probabilistic PCA.

Args:
    ppca: ProbabilisticPCA object
    X: Data matrix
    E_z: E_z matrix from E-Step
    E_zz: E_zz matrix from E-Step
"""
function M_step!(ppca::ProbabilisticPCA, X::Matrix{Float64}, E_z::AbstractArray, E_zz::AbstractArray)
    # get dims
    N, D = size(X)
    # update W and σ²
    running_sum_W = zeros(D, ppca.k)
    running_sum_σ² = 0.0
    WW = ppca.W' * ppca.W
    for i in 1:N
        running_sum_W += (X[i, :] - ppca.μ) * E_z[i, :]'
        running_sum_σ² +=sum((X[i, :] - ppca.μ).^2) - (2 * E_z[i, :]' * ppca.W' * (X[i, :] - ppca.μ)) + tr(E_zz[i, :, :] * WW)
    end
    ppca.z = E_z
    ppca.W = running_sum_W * pinv(sum(E_zz, dims=1)[1, :, :])
    ppca.σ² = running_sum_σ² / (N*D)
end

function loglikelihood(ppca::ProbabilisticPCA, X::Matrix{Float64})
    # get dims
    N, D = size(X)
    # calculate C and S
    C = ppca.W * ppca.W' + (ppca.σ² * I(D))
    S = sum([X[i, :] * X[i, :]' for i in 1:size(X, 1)]) / N
    # calculate log-likelihood
    ll = -(N/2) * (D * log(2*π) +logdet(C) + tr(pinv(C) * S))
    return ll
end

"""
Fits the ProbabilisticPCA model to the data using the EM algorithm for ProbabilisticPCA as discussed in Bishop's Pattern Recognition and Machine Learning.
See Chapter 12.2.2 for more details.

Args:
    ppca: ProbabilisticPCA object
    X: Data matrix
    max_iters: Maximum number of iterations
    tol: Tolerance for convergence of the reconstruction error
"""
function fit!(ppca::ProbabilisticPCA, X::Matrix{Float64}, max_iters::Int=100, tol::Float64=1e-6)
    prev_ll = -Inf  # Initialize to negative infinity
    for i in 1:max_iters
        # E-Step
        E_z, E_zz = E_step(ppca, X)
        # M-Step
        M_step!(ppca, X, E_z, E_zz)
        # Check for convergence
        ll = loglikelihood(ppca, X)
        println("Log-Likelihood at iter $i: $ll")
        if abs(ll - prev_ll) < tol
            # println("Converged after $i iterations")
            break
        end
        prev_ll = ll
    end
end

"""
    mutable struct PoissonPCA end

A probabilsitic (PCA?) model for Poisson distributed data.

# Fields
- `C::Matrix{<:Real}`: The loading matrix
- `b::Vector{<:Real}`: The bias vector
- `latent_dim::Int64`: The latent dimensionality
- `obs_dim::Int64`: The observation dimensionality
"""
mutable struct PoissonPCA 
    C::Matrix{<:Real}
    b::Vector{<:Real}
    latent_dim::Int64
    obs_dim::Int64
end

"""
    PoissonPCA(;C::Matrix{<:Real}=Matrix{Float64}(undef, 0, 0), b::Vector{<:Real}=Vector{Float64}(undef, 0), latent_dim::Int64, obs_dim::Int64)

Constructor for PoissonPCA

# Arguments
- `C::Matrix{<:Real}`: The loading matrix
- `b::Vector{<:Real}`: The bias vector
- `latent_dim::Int64`: The latent dimensionality
- `obs_dim::Int64`: The observation dimensionality
"""
function PoissonPCA(;C::Matrix{<:Real}=Matrix{Float64}(undef, 0, 0), b::Vector{<:Real}=Vector{Float64}(undef, 0), latent_dim::Int64, obs_dim::Int64)
    # initialize C
    if isempty(C)
        C = randn(obs_dim, latent_dim)
    end
    # initialize b
    if isempty(b)
        b = rand(obs_dim)
    end
    return PoissonPCA(C, b, latent_dim, obs_dim)
end

"""
    gradient!(grad::Vector{<:Real}, x::Vector{<:Real}, model::PoissonPCA, y::Matrix{<:Real})
    
Calculate the gradient of the Poisson PCA model. Designed to be passed to LBFGS in Optim.jl.

# Arguments
- `grad::Vector{<:Real}`: The gradient vector
- `x::Vector{<:Real}`: The latent variables
- `model::PoissonPCA`: The Poisson PCA model
- `y::Matrix{<:Real}`: The observed data
"""
function gradient!(grad::Vector{<:Real}, x::Vector{<:Real}, model::PoissonPCA, y::Matrix{<:Real})
    # reshape X
    x = reshape(x, size(y, 1), model.latent_dim)
    # pre-allocate a gradient array
    gradient = zeros(size(y, 1), model.latent_dim)
    # calculate the gradient
    for i in 1:size(x, 1)
        gradient[i, :] = (y[i, :]' * model.C)' - (model.C' * exp.(model.C * x[i, :] + model.b))
    end
    # shape gradient back to a vector
    grad .= -vec(gradient)
end

"""
    loglikelihood_(model::PoissonPCA, x::Matrix{<:Real}, y::Matrix{<:Real})

Calculate the log-likelihood of the Poisson PCA model.

# Arguments
- `model::PoissonPCA`: The Poisson PCA model
- `x::Matrix{<:Real}`: The latent variables
- `y::Matrix{<:Real}`: The observed data
"""
function loglikelihood_(model::PoissonPCA, x::Matrix{<:Real}, y::Matrix{<:Real})
    T = size(y, 1)
    ll = 0.0
    for t in 1:T
        ll += sum(y[t, :] .* (model.C * x[t, :] + model.b)) - sum(exp.(model.C * x[t, :] + model.b))
    end
    return ll
end

"""
    E_Step(model::PoissonPCA, y::Matrix{<:Real})

E-Step for Poisson PCA. Computes a MAP estimate of the latent variables by optimiizing the log-likelihood w.r.t. the latent variables.

# Arguments
- `model::PoissonPCA`: The Poisson PCA model
- `y::Matrix{<:Real}`: The observed data
"""
function E_Step(model::PoissonPCA, y::Matrix{<:Real})
    # create an objective function to minimize
    function obj(x)
        x = reshape(x, size(y, 1), model.latent_dim)
        return -loglikelihood_(model, x, y)
    end
    # create a wrapper for gradient
    function grad!(g, x)
        gradient!(g, x, model, y)
    end
    # optimize the objective function
    res = optimize(obj, grad!, vec(rand(size(y, 1), model.latent_dim)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=1000), autodiff=:forward)
    return reshape(res.minimizer, size(y, 1), model.latent_dim)
end

"""
    M_Step!(model::PoissonPCA, x::Matrix{<:Real}, y::Matrix{<:Real})

M-Step for Poisson PCA

# Arguments
- `model::PoissonPCA`: The Poisson PCA model
- `x::Matrix{<:Real}`: The latent variables
- `y::Matrix{<:Real}`: The observed data
"""
function M_Step!(model::PoissonPCA, x::Matrix{<:Real}, y::Matrix{<:Real})
    # define objective function to minimize
    function obj(params)
        C = reshape(params[1:model.latent_dim * model.obs_dim], model.obs_dim, model.latent_dim)
        b = params[model.latent_dim * model.obs_dim + 1:end]
        return -loglikelihood_(PoissonPCA(C, b, model.latent_dim, model.obs_dim), x, y)
    end
    # optimize the objective function
    res = optimize(obj, vcat(reshape(model.C, model.latent_dim * model.obs_dim), model.b), LBFGS(), Optim.Options(g_tol=1e-6, iterations=1000), autodiff=:forward)
    model.C = reshape(res.minimizer[1:model.latent_dim * model.obs_dim], model.obs_dim, model.latent_dim)
    model.b = res.minimizer[model.latent_dim * model.obs_dim + 1:end]
end

"""
    fit!(model::PoissonPCA, y::Matrix{<:Real}, max_iters::Int=100, tol=1e-6)

Fit the Poisson PCA model to the data.

# Arguments
- `model::PoissonPCA`: The Poisson PCA model
- `y::Matrix{<:Real}`: The observed data
- `max_iters::Int=100`: The maximum number of iterations
- `tol::Float64=1e-6`: The tolerance for convergence
"""
function fit!(model::PoissonPCA, y::Matrix{<:Real}, max_iters::Int=100, tol=1e-6)
    # set up first ll
    ll_prev = -Inf
    # Go!
    prog = Progress(max_iters; desc="Fitting Poisson PCA...")
    for i in 1:max_iters
        x = E_Step(model, y)
        M_Step!(model, x, y)
        #println("Log-Likelihood at iter $i: $(loglikelihood_(model, x, y))")
        next!(prog)
        # check for convergence
        ll = loglikelihood_(model, x, y)
        if abs(ll - ll_prev) < tol
            #println("Converged after $i iterations")
            finish!(prog)
            break
        end
    end
end
