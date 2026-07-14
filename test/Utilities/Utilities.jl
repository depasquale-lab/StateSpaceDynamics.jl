function test_block_tridgm()
    # Test with minimal block sizes
    super = [rand(1, 1) for i in 1:1]
    sub = [rand(1, 1) for i in 1:1]
    main = [rand(1, 1) for i in 1:2]
    A = block_tridgm(main, super, sub)
    @test size(A) == (2, 2)
    @test A[1, 1] == main[1][1, 1]
    @test A[2, 2] == main[2][1, 1]
    @test A[1, 2] == super[1][1, 1]
    @test A[2, 1] == sub[1][1, 1]

    # Test with 2x2 blocks and a larger matrix
    super = [rand(2, 2) for i in 1:9]
    sub = [rand(2, 2) for i in 1:9]
    main = [rand(2, 2) for i in 1:10]
    A = block_tridgm(main, super, sub)
    @test size(A) == (20, 20)

    # Check some blocks in the matrix
    for i in 1:10
        @test A[(2i - 1):(2i), (2i - 1):(2i)] == main[i]
        if i < 10
            @test A[(2i - 1):(2i), (2i + 1):(2i + 2)] == super[i]
            @test A[(2i + 1):(2i + 2), (2i - 1):(2i)] == sub[i]
        end
    end

    # Test with integer blocks
    super = [rand(Int, 2, 2) for i in 1:9]
    sub = [rand(Int, 2, 2) for i in 1:9]
    main = [rand(Int, 2, 2) for i in 1:10]
    A = block_tridgm(main, super, sub)
    @test size(A) == (20, 20)
    for i in 1:10
        @test A[(2i - 1):(2i), (2i - 1):(2i)] == main[i]
        if i < 10
            @test A[(2i - 1):(2i), (2i + 1):(2i + 2)] == super[i]
            @test A[(2i + 1):(2i + 2), (2i - 1):(2i)] == sub[i]
        end
    end
end

function test_gaussian_entropy()
    n = 3
    A = sprandn(n, n, 0.6)
    Λ = Symmetric(A' * A) + 1e-8I

    F = cholesky(Λ)
    Σ = Symmetric(Matrix(F \ Matrix{Float64}(I, n, n)))

    gaus_entropy_dist = entropy(MvNormal(zeros(n), Σ))
    gauss_entropy_ssd = gaussian_entropy(-Λ)
    @test isapprox(gaus_entropy_dist, gauss_entropy_ssd; atol=1e-6)
end

# Build a random, well-conditioned block tridiagonal matrix and return both
# (A, B, C) blocks plus the dense reconstruction H so tests can compare against inv(H).
function _random_block_tridiag(::Type{T}, block_size::Int, n::Int, rng) where {T<:Real}
    A = Matrix{T}[randn(rng, T, block_size, block_size) for _ in 1:(n - 1)]
    B = Matrix{T}[
        T(5.0) * Matrix{T}(I, block_size, block_size) +
        randn(rng, T, block_size, block_size) for _ in 1:n
    ]
    C = Matrix{T}[randn(rng, T, block_size, block_size) for _ in 1:(n - 1)]
    H = Matrix(block_tridgm(B, C, A))
    return A, B, C, H
end

function test_block_tridiagonal_inverse_mutating()
    rng = MersenneTwister(42)

    for T in (Float64, Float32)
        atol = T === Float32 ? 1e-4 : 1e-8
        for block_size in (1, 2, 3), n in (1, 2, 3, 4)
            A, B, C, H = _random_block_tridiag(T, block_size, n, rng)
            Hinv = inv(H)

            ws = StateSpaceDynamics.BlockTridiagonalWorkspace(T, block_size, n)
            p_smooth = zeros(T, block_size, block_size, n)
            p_smooth_tt1 = zeros(T, block_size, block_size, n)

            StateSpaceDynamics.block_tridiagonal_inverse!(
                p_smooth, p_smooth_tt1, A, B, C, ws
            )

            for i in 1:n
                rows = ((i - 1) * block_size + 1):(i * block_size)
                @test isapprox(p_smooth[:, :, i], Hinv[rows, rows]; atol=atol, rtol=0)
            end
            for i in 2:n
                rows_i = ((i - 1) * block_size + 1):(i * block_size)
                rows_im1 = ((i - 2) * block_size + 1):((i - 1) * block_size)
                @test isapprox(
                    p_smooth_tt1[:, :, i], Hinv[rows_i, rows_im1]; atol=atol, rtol=0
                )
            end
        end
    end
end

#=
Build a random SPD block tridiagonal matrix by forming `H = L Lᵀ` for a
block-bidiagonal `L`. This matches the precondition of
`block_tridiagonal_inverse_logdet!`, which factors the SPD Schur
complements via Cholesky.
=#
function _random_spd_block_tridiag(::Type{T}, block_size::Int, n::Int, rng) where {T<:Real}
    # Block-bidiagonal L: diagonal L_diag and lower off-diagonal L_off.
    L_diag = Matrix{T}[
        T(2.0) * Matrix{T}(I, block_size, block_size) +
        T(0.3) * randn(rng, T, block_size, block_size) for _ in 1:n
    ]
    L_off = Matrix{T}[T(0.3) * randn(rng, T, block_size, block_size) for _ in 1:(n - 1)]
    # Assemble L (lower block-bidiagonal) densely, then H = L Lᵀ.
    Ldense = zeros(T, n * block_size, n * block_size)
    for i in 1:n
        rows = ((i - 1) * block_size + 1):(i * block_size)
        Ldense[rows, rows] = L_diag[i]
        if i < n
            rows_next = (i * block_size + 1):((i + 1) * block_size)
            Ldense[rows_next, rows] = L_off[i]
        end
    end
    H = Ldense * Ldense'
    H = (H + H') / 2  # exact symmetry
    # Extract the block tridiagonal pieces.
    A = Matrix{T}[
        copy(
            H[
                (i * block_size + 1):((i + 1) * block_size),
                ((i - 1) * block_size + 1):(i * block_size),
            ],
        ) for i in 1:(n - 1)
    ]
    B = Matrix{T}[
        copy(
            H[
                ((i - 1) * block_size + 1):(i * block_size),
                ((i - 1) * block_size + 1):(i * block_size),
            ],
        ) for i in 1:n
    ]
    C = Matrix{T}[copy(transpose(A[i])) for i in 1:(n - 1)]
    return A, B, C, H
end

function test_block_tridiagonal_inverse_logdet()
    rng = MersenneTwister(7)
    T = Float64
    block_size = 3
    n = 5

    A, B, C, H = _random_spd_block_tridiag(T, block_size, n, rng)
    expected_logdet = logdet(H)

    ws = StateSpaceDynamics.BlockTridiagonalWorkspace(T, block_size, n)
    p_smooth = zeros(T, block_size, block_size, n)
    p_smooth_tt1 = zeros(T, block_size, block_size, n)

    logdet_val = StateSpaceDynamics.block_tridiagonal_inverse_logdet!(
        p_smooth, p_smooth_tt1, A, B, C, ws
    )

    @test isapprox(logdet_val, expected_logdet; atol=1e-8, rtol=0)

    Hinv = inv(H)
    for i in 1:n
        rows = ((i - 1) * block_size + 1):(i * block_size)
        @test isapprox(p_smooth[:, :, i], Hinv[rows, rows]; atol=1e-8, rtol=0)
    end
end

function test_block_tridiagonal_solve()
    rng = MersenneTwister(11)
    T = Float64
    block_size = 2
    n = 4

    A, B, C, H = _random_block_tridiag(T, block_size, n, rng)
    b = randn(rng, T, n * block_size)
    expected = H \ b

    ws = StateSpaceDynamics.BlockTridiagonalWorkspace(T, block_size, n)
    x = zeros(T, n * block_size)
    StateSpaceDynamics.block_tridiagonal_solve!(x, A, B, C, b, ws)

    @test isapprox(x, expected; atol=1e-8, rtol=0)
end

function test_block_tridiagonal_solve_spd()
    #=
    `block_tridiagonal_solve_spd!` routes through the packed LAPACK `pbsv`
    fast path when `bs ≤ 8` and BlasFloat, else falls back to the general
    block-Thomas solve. Both branches must match a dense `H \ b`.
    =#
    rng = MersenneTwister(2024)
    for T in (Float64, Float32)
        atol = T === Float32 ? 1e-3 : 1e-8
        # bs = 3 → pbsv fast path; bs = 9 → generic fallback branch.
        for (block_size, n) in ((3, 5), (9, 2))
            A, B, C, H = _random_spd_block_tridiag(T, block_size, n, rng)
            b = randn(rng, T, n * block_size)
            expected = H \ b

            ws = StateSpaceDynamics.BlockTridiagonalWorkspace(T, block_size, n)
            x = zeros(T, n * block_size)
            StateSpaceDynamics.block_tridiagonal_solve_spd!(x, A, B, C, b, ws)
            @test isapprox(x, expected; atol=atol, rtol=0)
        end
    end
end

function test_valid_Σ()
    # SPD + symmetric → valid.
    Σ = [2.0 0.5; 0.5 1.0]
    @test valid_Σ(Σ)
    @test valid_Σ(Matrix{Float64}(I, 4, 4))

    # Symmetric but indefinite (negative eigenvalue) → invalid.
    @test !valid_Σ([1.0 2.0; 2.0 1.0])

    # Non-symmetric → invalid even if positive on the diagonal.
    @test !valid_Σ([2.0 0.3; 0.7 1.0])
end

function test_tol_PD_type_preservation()
    @testset "tol_PD preserves eltype across Float32/Float64 (A11)" begin
        for T in (Float32, Float64)
            A = Symmetric(T[2 0.5; 0.5 1])
            P = StateSpaceDynamics.tol_PD(A)                 # default tol
            @test P isa PDMat{T,Matrix{T}}
            @test eltype(P) === T
            @test isposdef(Matrix(P))

            P2 = StateSpaceDynamics.tol_PD(A; tol=1e-5)
            @test eltype(P2) === T
        end

        # Eigen-floor actually lifts a near-singular direction to tol·λ_max.
        Asing = Symmetric([1.0 0.0; 0.0 1e-14])
        Pf = StateSpaceDynamics.tol_PD(Asing; tol=1e-6)
        @test minimum(eigen(Matrix(Pf)).values) ≈ 1e-6 rtol = 1e-6
    end
    return nothing
end

function test_info_update()
    #=
    `info_update!(cache, P0, CiRC)` returns `inv(inv(P0) + CiRC)` as a PDMat,
    exploiting P0's cached Cholesky. Check against a dense reference and that
    the result is genuinely PD. Also exercise the in-place variant.
    =#
    rng = MersenneTwister(99)
    for n in (1, 2, 5)
        G = randn(rng, n, n)
        P0_mat = Symmetric(G * G' + n * I)
        H = randn(rng, n, n)
        CiRC_mat = Symmetric(H * H' + I)

        P0 = PDMat(Matrix(P0_mat))
        CiRC = PDMat(Matrix(CiRC_mat))
        expected = inv(inv(Matrix(P0_mat)) + Matrix(CiRC_mat))

        # Allocating-into-cache variant.
        cache = StateSpaceDynamics.CovUpdateCache(n)
        P = StateSpaceDynamics.info_update!(cache, P0, CiRC)
        @test isapprox(Matrix(P), expected; atol=1e-8, rtol=0)
        @test isposdef(P.mat)
        # The returned PDMat's own Cholesky must reconstruct its `mat`.
        @test isapprox(Matrix(P.chol), P.mat; atol=1e-8, rtol=0)

        # In-place variant writing into an existing PDMat (upper Cholesky).
        P_dest = PDMat(Matrix{Float64}(I, n, n))
        scratch = Matrix{Float64}(undef, n, n)
        ret = StateSpaceDynamics.info_update!(P_dest, scratch, P0, CiRC)
        @test ret === P_dest
        @test isapprox(Matrix(P_dest), expected; atol=1e-8, rtol=0)
        @test isposdef(P_dest.mat)
    end

    # Typed and default-eltype cache constructors size their buffers correctly.
    c32 = StateSpaceDynamics.CovUpdateCache{Float32}(3)
    @test size(c32.M) == (3, 3)
    @test eltype(c32.M) == Float32
    @test eltype(StateSpaceDynamics.CovUpdateCache(2).M) == Float64
end
