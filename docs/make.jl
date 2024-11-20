using Documenter
using StateSpaceDynamics


DocMeta.setdocmeta!(StateSpaceDynamics, :DocTestSetup, :(using StateSpaceDynamics); recursive=true)

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
        "Getting Started" => "getting_started.md",
        "Models" => [
            "EmissionModels" => "EmissionModels.md",
            "Hidden Markov Models" => "HiddenMarkovModels.md",
            "Mixture Models" => "MixtureModels.md",
            "Linear Dynamical Systems" => "LinearDynamicalSystems.md",
        ],
        "Miscellaneous" => "Misc.md",
    ]
)
