"""
The purpose of this file is to provide a common place for all global types to be defined. This is to avoid circular dependencies between files.
"""

export MixtureModel, EmissionsModel, AbstractHMM, DynamicalSystem


# Create abstract types here 

"""
Abstract type for Mixture Models. I.e. GMM's, etc.
"""

abstract type MixtureModel end

"""
Abstract type for Emissions Models. I.e. Poisson, Gaussian, etc.
"""

abstract type EmissionsModel end

"""
Abstract type for Markov Models. I.e. HMM's, Markov Regressions, etc.
"""
abstract type AbstractHMM end

"""
Abstract type for Dynamical Systems. I.e. LDS, etc.
"""
abstract type DynamicalSystem end