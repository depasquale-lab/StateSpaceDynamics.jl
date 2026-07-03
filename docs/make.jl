using Documenter
using StateSpaceDynamics
using Literate
using Random

# Set up the documentation environment
DocMeta.setdocmeta!(StateSpaceDynamics, :DocTestSetup, :(using StateSpaceDynamics, Random); recursive=true)

# Define tutorial configurations
tutorials = [
    ("QuickStart.jl", "quick_start_example"),
    ("GaussianLDS.jl", "gaussian_latent_dynamics_example"),
    ("PoissonLDS.jl", "poisson_latent_dynamics_example"),
    ("LDSModelSelection.jl", "lds_model_selection_example"),
    ("LDSIdentifiability.jl", "lds_identifiability_example"),
    ("SLDS.jl", "switching_linear_dynamical_system_example"),
    ("ProbabilisticPCA.jl", "Probabilistic_PCA_example"),
]

# Convert all Julia examples to Markdown tutorials
println("Converting tutorial examples...")
for (source_file, output_name) in tutorials
    println("  Converting $source_file -> $output_name.md")
    Literate.markdown(
        joinpath(@__DIR__, "examples", source_file),
        joinpath(@__DIR__, "src", "tutorials");
        name = output_name,
        documenter = true
    )
end

# Generate the documentation site
println("Building documentation...")
makedocs(;
    modules=[StateSpaceDynamics],
    authors="Ryan Senne",
    sitename="StateSpaceDynamics.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        repolink = "https://github.com/depasquale-lab/StateSpaceDynamics.jl",
        assets = ["assets/custom.css"],
    ),
    pages=[
        "Home" => "index.md",
        "Models" => [
            "Linear Dynamical Systems" => "LinearDynamicalSystems.md",
            "Switching Linear Dynamical Systems" => "SLDS.md",
        ],
        "Tutorials" => [
            "Quick Start" => "tutorials/quick_start_example.md",
            "Gaussian LDS Example" => "tutorials/gaussian_latent_dynamics_example.md",
            "Poisson LDS Example" => "tutorials/poisson_latent_dynamics_example.md",
            "LDS Model Selection Example" => "tutorials/lds_model_selection_example.md",
            "Non-Identifiability in LDS Models" => "tutorials/lds_identifiability_example.md",
            "Probabilistic PCA Example" => "tutorials/Probabilistic_PCA_example.md",
            "Switching Linear Dynamical System Example" => "tutorials/switching_linear_dynamical_system_example.md",
        ]
    ],
    checkdocs = :exports,
    doctest=true,
    doctestfilters = [r"docs/src/tutorials/.*\.md"],
    warnonly = true
)

# Deploy the documentation
println("Deploying documentation...")
deploydocs(;
    repo="github.com/depasquale-lab/StateSpaceDynamics.jl",
    devbranch="main"
)
