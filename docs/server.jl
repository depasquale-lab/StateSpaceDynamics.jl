using LiveServer

# Set the directory to "build"; assumes server.jl and the build directory are in the same folder
serve(; dir=joinpath(@__DIR__, "build"))

#Why we host a server to view the build file: https://documenter.juliadocs.org/stable/man/guide/#Package-Guide
# ^^ Refer to Note in blue at the BOTTOM of "Building an Empty Document" section.
