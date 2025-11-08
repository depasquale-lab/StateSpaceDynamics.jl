function extract_julia_code(markdown_file::String)
    """
    Extract Julia code from julia code blocks in a markdown file.
    Returns the code as a string, ready to be evaluated.
    """
    content = read(markdown_file, String)
    # Julia regex pattern
    pattern = r"```julia\s*\n([\s\S]*?)\n```"
    matches = collect(eachmatch(pattern, content))
        
    if isempty(matches)
        error("No julia code blocks found in $markdown_file")
    end
    
    # Find the first substantial code block (not installation-related)
    for (i, match) in enumerate(matches)
        code_block = match.captures[1]
        
        
        # Skip installation blocks
        if occursin("Pkg", code_block) || 
           occursin("add StateSpaceDynamics", code_block) ||
           occursin("test()", code_block)
            continue
        end
        
        # Look for blocks that define models
        if occursin("LinearDynamicalSystem", code_block)
            return code_block
        end

        # If it's a substantial block (not just a single line), use it
        if length(split(code_block, '\n')) > 3
            return code_block
        end
    end
    
    error("No suitable julia code blocks found in $markdown_file")
end

function run_index_example()
    """
    Parse and execute the jldoctest from docs/src/index.md
    """
    # Try to find the index.md file
    current_dir = @__DIR__
    possible_paths = [
        joinpath(current_dir, "..", "..", "docs", "src", "index.md"),
        joinpath(current_dir, "..", "docs", "src", "index.md"),
        joinpath(current_dir, "docs", "src", "index.md")
    ]
    
    index_file = nothing
    for path in possible_paths
        if isfile(path)
            index_file = path
            break
        end
    end
    
    if isnothing(index_file)
        error("Could not find index.md file. Tried: $(join(possible_paths, ", "))")
    end
        
    # Extract the Julia code
    code = extract_julia_code(index_file)

    # Execute the code in a module to avoid namespace pollution
    test_module = Module()
    
    # Import required packages into the test module
    Core.eval(test_module, :(using StateSpaceDynamics, LinearAlgebra))
    
    # Use include_string which is designed for this purpose
    # Create a string that includes the code and returns the lds object
    code_with_return = """
    $code
    lds
    """
    
    result = Base.include_string(test_module, code_with_return)
    return result
end

@testset "Index.md Documentation Example" begin
    @testset "Extract and Run Julia Code Block" begin
        # Parse and execute the index.md example
        lds = run_index_example()
        
        # Test that the example ran successfully
        @test lds isa LinearDynamicalSystem
        @test lds.latent_dim == 3
        @test lds.obs_dim == 10
        
        # Test model structure
        @test size(lds.state_model.A) == (3, 3)
        @test size(lds.obs_model.C) == (10, 3)
        @test size(lds.state_model.Q) == (3, 3)
        @test size(lds.obs_model.R) == (10, 10)
        
        # Test that matrices have expected properties
        @test all(diag(lds.state_model.A) .≈ 0.95)  # Diagonal A matrix
        @test all(lds.obs_model.C .== 1.0)         # All ones C matrix
        @test all(diag(lds.state_model.Q) .≈ 0.01)  # Diagonal Q matrix  
        @test all(diag(lds.obs_model.R) .≈ 0.5)     # Diagonal R matrix
        @test all(lds.state_model.x0 .== 0.0)       # Zero initial state
        @test all(diag(lds.state_model.P0) .≈ 0.1)  # Initial covariance
        
    end
end