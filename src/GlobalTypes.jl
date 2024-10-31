"""
The purpose of this file is to provide a common place for all global types to be defined. This is to avoid circular dependencies between files.
"""

export RegressionModel, MixtureModel, EmissionsModel, AbstractHMM, DynamicalSystem, EmissionModel

# Create abstract types here 
"""
Abstract type for Mixture Models. I.e. GMM's, etc.
"""

abstract type MixtureModel end

"""
Abstract type for Regression Models. I.e. GaussianRegression, BernoulliRegression, etc.
"""

abstract type RegressionModel  end

"""
Abstract type for HMMs 
"""

abstract type AbstractHMM end

"""
Abstract type for Dynamical Systems. I.e. LDS, etc.
"""

abstract type DynamicalSystem end

"""
Base type hierarchy for emission models.
Each emission model must implement:
- sample()
- loglikelihood()
- fit!()
"""
abstract type EmissionModel end
abstract type RegressionEmission <: EmissionModel end