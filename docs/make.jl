using Documenter
using StateSpaceDynamics
using Literate
using Random

# Set up the documentation environment
DocMeta.setdocmeta!(StateSpaceDynamics, :DocTestSetup, :(using StateSpaceDynamics, Random); recursive=true)

# Define tutorial configurations
tutorials = [
    ("GaussianLDS.jl", "gaussian_latent_dynamics_example"),
    ("PoissonLDS.jl", "poisson_latent_dynamics_example"),
    ("SLDS.jl", "switching_linear_dynamical_system_example"),
    ("HMM.jl", "hidden_markov_model_example"),
    ("HMM_ModelSelection.jl", "hmm_model_selection_example"),
    ("Gaussian_GLM_HMM.jl", "gaussian_glm_hmm_example"),
    ("GaussianMixtureModel.jl", "gaussian_mixture_model_example"),
    ("PoissonMixtureModel.jl", "poisson_mixture_model_example"),
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
    ),
    pages=[
        "Home" => "index.md",
        "Models" => [
            "Linear Dynamical Systems" => "LinearDynamicalSystems.md",
            "Hidden Markov Models" => "HiddenMarkovModels.md",
            "Switching Linear Dynamical Systems" => "SLDS.md",
            "EmissionModels" => "EmissionModels.md",
            "Mixture Models" => "MixtureModels.md"
        ],
        "Tutorials" => [
            "Gaussian LDS Example" => "tutorials/gaussian_latent_dynamics_example.md",
            "Poisson LDS Example" => "tutorials/poisson_latent_dynamics_example.md",
            "Hidden Markov Model Example" => "tutorials/hidden_markov_model_example.md",
            "HMM Model Selection" => "tutorials/hmm_model_selection_example.md",
            "Gaussian GLM-HMM Example" => "tutorials/gaussian_glm_hmm_example.md",
            "Gaussian Mixture Model Example" => "tutorials/gaussian_mixture_model_example.md",
            "Poisson Mixture Model Example" => "tutorials/poisson_mixture_model_example.md",
            "Probabilistic PCA Example" => "tutorials/Probabilistic_PCA_example.md",
            "Switching Linear Dynamical System Example" => "tutorials/switching_linear_dynamical_system_example.md",
        ],
        "Miscellaneous" => "Misc.md",
    ],
    checkdocs = :exports,
    warnonly = true
)

# Deploy the documentation
println("Deploying documentation...")
deploydocs(; 
    repo="github.com/depasquale-lab/StateSpaceDynamics.jl", 
    devbranch="docs_dev_"
)