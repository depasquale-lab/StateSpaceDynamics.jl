using Documenter
using StateSpaceDynamics
using Random

# Set up the documentation environment
DocMeta.setdocmeta!(StateSpaceDynamics, :DocTestSetup, :(using StateSpaceDynamics, Random); recursive=true)

makedocs(;
    modules=[StateSpaceDynamics],
    authors="Ryan Senne",
    sitename="StateSpaceDynamics.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
    ),
    remotes=nothing,
    pages=[
        "Home" => "index.md",
        "Models" => [
            "Linear Dynamical Systems" => "LinearDynamicalSystems.md",
            "Hidden Markov Models" => "HiddenMarkovModels.md",
            "Switching Linear Dynamical Systems" => "SLDS.md",
            "EmissionModels" => "EmissionModels.md",
            "Mixture Models" => "MixtureModels.md",
        ],
        "Miscellaneous" => "Misc.md",
    ],
    checkdocs = :exports,  # Only check exported functions
    warnonly = true # Do not fail on warnings
)

deploydocs(; repo="github.com/depasquale-lab/StateSpaceDynamics.jl", devbranch="docs_dev_")