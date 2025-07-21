using Random
using LinearAlgebra
using StateSpaceDynamics
using Distributions
using ProfileView
using BenchmarkTools
using Profile

function benchmark_emission(emission::RegressionEmission, Φ, Y, w)

    println("\n--- Benchmarking create_optimization ---")
    bench_create = @benchmark begin
        StateSpaceDynamics.create_optimization($emission, $Φ, $Y, $w)
    end samples=5

    println("create_optimization: time=$(median(bench_create).time), memory=$(bench_create.memory), allocs=$(bench_create.allocs)")

    opt = StateSpaceDynamics.create_optimization(emission, Φ, Y, w)
    β = vec(emission.β)
    G = similar(β)

    println("\n--- Benchmarking objective ---")
    bench_obj = @benchmark begin
        StateSpaceDynamics.objective($opt, $β)
    end samples=5

    println("objective: time=$(median(bench_obj).time), memory=$(bench_obj.memory), allocs=$(bench_obj.allocs)")

    println("\n--- Benchmarking objective_gradient! ---")
    bench_grad = @benchmark begin
        StateSpaceDynamics.objective_gradient!($G, $opt, $β)
    end samples=5

    println("gradient: time=$(median(bench_grad).time), memory=$(bench_grad.memory), allocs=$(bench_grad.allocs)")

    println("\n--- Benchmark loglikelihood ---")
    bench_ll = @benchmark begin
        StateSpaceDynamics.loglikelihood($emission, $Φ, $Y, $w)
    end samples=5

    println("gradient: time=$(median(bench_ll).time), memory=$(bench_ll.memory), allocs=$(bench_ll.allocs)")

    # println("\n--- Allocation profile with @profview_allocs (gradient) ---")
    # @profview_allocs for _ in 1:100
    #     StateSpaceDynamics.objective_gradient!(G, opt, β)
    # end

    # println("\n--- Allocation profile with @profview_allocs (objective) ---")
    # @profview_allocs for _ in 1:100
    #     StateSpaceDynamics.objective(opt, β)
    # end
end

Random.seed!(42)
n = 30_000
input_dim = 10
output_dim = 2

Φ = randn(input_dim, n)         
Y = randn(output_dim, n)
w = ones(n) 

λ = 3.0    
Y_poisson = rand.(Poisson(λ), output_dim, n)      

gaussian = GaussianRegressionEmission(
    input_dim=input_dim,
    output_dim=output_dim,
    include_intercept=true,
    β=randn(input_dim + 1, output_dim),    
    Σ=Matrix{Float64}(I, output_dim, output_dim), 
    λ=0.0
)

bernoulli = BernoulliRegressionEmission(
    input_dim=input_dim,
    output_dim=output_dim,
    include_intercept=true,
    β=randn(input_dim + 1, output_dim), 
    λ=0.0
)

poisson = PoissonRegressionEmission(
    input_dim=input_dim,
    output_dim=output_dim,
    include_intercept=true,
    β=randn(input_dim + 1, output_dim), 
    λ=0.0
)

println("\n=======================")
println("Benchmarks on Gaussian Emission")
println("=======================")
benchmark_emission(gaussian, Φ, Y, w)
    
println("\n=======================")
println("Benchmarks on Bernoulli Emission")
println("=======================")
benchmark_emission(bernoulli, Φ, Y, w)

println("\n=======================")
println("Benchmarks on Poisson Emission")
println("=======================")
benchmark_emission(poisson, Φ, Y_poisson, w)
