using Documenter
using StateSpaceDynamics
using Literate
using Random

# Set up the documentation environment
DocMeta.setdocmeta!(StateSpaceDynamics, :DocTestSetup, :(using StateSpaceDynamics, Random); recursive=true)

# Convert Gaussian LDS example
Literate.markdown(
    joinpath(@__DIR__, "examples", "GaussianLDS.jl"),
    joinpath(@__DIR__, "src", "tutorials");
    name = "gaussian_latent_dynamics_example",
    documenter = true
)

# Convert Poisson LDS example
Literate.markdown(
    joinpath(@__DIR__, "examples", "PoissonLDS.jl"),
    joinpath(@__DIR__, "src", "tutorials");
    name = "poisson_latent_dynamics_example",
    documenter = true
)

# Convert Poisson LDS example
Literate.markdown(
    joinpath(@__DIR__, "examples", "HMM.jl"),
    joinpath(@__DIR__, "src", "tutorials");
    name = "hidden_markov_model_example",
    documenter = true
)

# Convert Poisson LDS example
Literate.markdown(
    joinpath(@__DIR__, "examples", "Gaussian_GLM_HMM.jl"),
    joinpath(@__DIR__, "src", "tutorials");
    name = "gaussian_glm_hmm_example",
    documenter = true
)

# Convert Gaussian Mixture model example
Literate.markdown(
    joinpath(@__DIR__, "examples", "GaussianMixtureModel.jl"),
    joinpath(@__DIR__, "src", "tutorials");
    name = "gaussian_mixture_model_example",
    documenter = true
)

# Convert Poisson Mixture model example
Literate.markdown(
    joinpath(@__DIR__, "examples", "PoissonMixtureModel.jl"),
    joinpath(@__DIR__, "src", "tutorials");
    name = "poisson_mixture_model_example",
    documenter = true
)

# Generate the documentation site
makedocs(;
    modules=[StateSpaceDynamics],
    authors="Ryan Senne",
    sitename="StateSpaceDynamics.jl",
    format = Documenter.HTML(
    prettyurls = get(ENV, "CI", "false") == "true",
    repolink = "https://github.com/depasquale-lab/StateSpaceDynamics.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Models" => [
            "Linear Dynamical Systems" => "LinearDynamicalSystems.md",
            "Hidden Markov Models" => "HiddenMarkovModels.md",
            "Switching Linear Dynamical Systems" => "SLDS.md",
            "EmissionModels" => "EmissionModels.md",
            "Mixture Models" => "MixtureModels.md",
        ],
        "Tutorials" => [
            "Gaussian LDS Example" => "tutorials/gaussian_latent_dynamics_example.md",
            "Poisson LDS Example" => "tutorials/poisson_latent_dynamics_example.md",
            "Hidden Markov Model Example" => "tutorials/hidden_markov_model_example.md",
            "Gaussian GLM-GMM Example" => "tutorials/gaussian_glm_hmm_example.md",
            "Gaussian Mixture Model Example" => "tutorials/gaussian_mixture_model_example.md",
            "Poisson Mixture Model Example" => "tutorials/poisson_mixture_model_example.md",
        ],
        "Miscellaneous" => "Misc.md",
    ],
    checkdocs = :exports,
    warnonly = true
)

# Deploy the documentation
deploydocs(; repo="github.com/depasquale-lab/StateSpaceDynamics.jl", devbranch="docs_dev_")
