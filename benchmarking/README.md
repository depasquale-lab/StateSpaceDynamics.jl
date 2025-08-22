# Benchmarking Code Instructions

## Setting up the environment

To set up the benchmarking environment (assuming you have a cloned/forked version of StateSpaceDynamics.jl) do the following:

1. `cd` into the `benchmarking` directory.
2. Start the Julia REPL.
3. Type `]` to open the Pkg manager then type `activate .` to activate the benchmarking environment.
4. Type `instantiate` to install all necessary packages. 
5. We also recommend that you type `dev your/local/path/to/StateSpaceDynamics.jl` so that you ensure your local version is being benchmarked if developing the package.

## Running the code

To run the benchmarking code please do the following:

1. `cd` into the StateSpaceDynamics.jl directory (i.e., one level above the benchmarking folder).
2. Run `julia benchmarking/run_benchmark.jl` to run the benchmark code.
3. Results will be stored in `benchmarking/results`.