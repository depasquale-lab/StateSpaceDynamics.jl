using Documenter
using Revise
using SSM

root = joinpath(dirname(pathof(SSM)), "..", "docs")

entr([], [SSM]) do
    makedocs(root = root, sitename="SSM Julia")
end