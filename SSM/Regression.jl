export RegressionModel, Link

"""Model Class for GLM's, especially for SSM's."""
abstract type RegressionModel end


"""Definition of Gaussian Regression Model."""
struct GaussianRegression <: RegressionModel
    X::Matrix{Float64}
    y::Vector{Float64}
    β::Vector{Float64}
    link::Link
end

"""Link Functions"""
abstract type Link end

"""Identity Link"""
struct IdentityLink <: Link end

"""Log Link"""
struct LogLink <: Link end

"""Logit Link"""
struct LogitLink <: Link end

"""Probit Link"""
struct ProbitLink <: Link end


