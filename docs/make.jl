using Documenter
using Revise
using SSM

root = joinpath(dirname(pathof(SSM)), "..", "docs")

println("working dir: ", pwd())

entr(["./docs/src"], [SSM]) do
    makedocs(
        root = root, 
        sitename="SSM Julia",
        pages=[
            "Home" => "index.md",
            "getting_started.md",
            "using_models.md",
            "Models" => [
                "mixture_models.md"
                ],
            "training_algorithms.md"
            ])
end