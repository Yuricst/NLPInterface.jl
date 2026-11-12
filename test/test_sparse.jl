# Sparse Jacobians (src/sparsity.jl): the pattern, its coloring, the sparse
# evaluator, sparsity detection, and the `jac_sparsity` solver keyword.

using Random
using SparseArrays
using SparseConnectivityTracer: TracerSparsityDetector, TracerLocalSparsityDetector

const JP = NLPInterface.JacobianPattern

# Dense matrix holding the sparse evaluator's entries.
densify(p, Jnz) = Matrix(sparse(p.rows, p.cols, Jnz, p.nrows, p.ncols))

# Columns of the same color must not share a row.
function valid_coloring(p::NLPInterface.JacobianPattern)
    for c in 1:p.ncolors
        seen = falses(p.nrows)
        for k in p.color_entries[c]
            seen[p.rows[k]] && return false
            seen[p.rows[k]] = true
        end
    end
    return true
end

# Sparse Jacobian at `x` with `pattern`, and the number of `fitness` calls.
function sparse_jac(fitness, pattern, x, n_ceq, n_cineq, deriv; r_valid=false, kw...)
    calls = Ref(0)
    f = ξ -> (calls[] += 1; fitness(ξ))
    p = JP(pattern, 1 + n_ceq + n_cineq, length(x))
    ev = NLPInterface.SparseResidualEvaluator(f, p, n_ceq, n_cineq, deriv; kw...)
    r = r_valid ? NLPInterface.packed_residual(fitness, x, n_ceq, n_cineq) : fill(NaN, p.nrows)
    Jnz = fill(NaN, nnz(p))
    NLPInterface.residual_value_and_jacobian!(ev, r, Jnz, x, r_valid)
    return r, densify(p, Jnz), calls[], p
end

@testset "JacobianPattern" begin
    S = sparse([1, 2, 2, 3], [1, 1, 3, 2], true, 3, 3)
    p = JP(S, 3, 3)
    @test (p.rows, p.cols) == ([1, 2, 3, 2], [1, 1, 2, 3])   # column-major
    @test p.obj_entries == [1]
    @test p.con_entries == [2, 3, 4]
    @test valid_coloring(p)
    # Every input form gives the same pattern.
    for A in (Matrix(S), BitMatrix(Matrix(S)), Float64.(Matrix(S)), Int.(Matrix(S)))
        q = JP(A, 3, 3)
        @test (q.rows, q.cols) == (p.rows, p.cols)
    end
    # Stored zeros of a sparse matrix are structural.
    Z = sparse([1, 2], [1, 2], [1.0, 0.0], 2, 2)
    @test JP(Z, 2, 2).rows == [1, 2]
    @test JP(Matrix(Z), 2, 2).rows == [1]
    # Size and element type are checked.
    @test_throws ArgumentError JP(S, 3, 4)
    @test_throws ArgumentError JP(S', 4, 3)
    @test_throws ArgumentError JP(Any["a" 1; 0 0], 2, 2)
    @test_throws ArgumentError JP(Any[missing 1; 0 0], 2, 2)
    # Empty pattern, empty columns.
    e = JP(spzeros(Bool, 2, 3), 2, 3)
    @test nnz(e) == 0 && e.ncolors == 0 && e.colors == [0, 0, 0]
    q = JP(sparse([1, 2], [1, 3], true, 2, 3), 2, 3)
    @test q.colors[2] == 0
    @test q.ncolors == 1   # columns 1 and 3 share no row
end

@testset "column coloring" begin
    # Dense pattern: every column needs its own color.
    @test JP(trues(4, 6), 4, 6).ncolors == 6
    # Diagonal: one color.
    @test JP(spdiagm(0 => trues(8)), 8, 8).ncolors == 1
    # Band of half-width 1 (tridiagonal): 3 colors.
    T = spdiagm(-1 => trues(9), 0 => trues(10), 1 => trues(9))
    @test JP(T, 10, 10).ncolors == 3
    # Random patterns: always a valid coloring, every column with an entry colored.
    for seed in 1:50
        m, n = rand(1:15), rand(1:15)
        S = sprand(Xoshiro(seed), Bool, m, n, rand(Xoshiro(seed + 100)) * 0.5)
        p = JP(S, m, n)
        @test valid_coloring(p)
        nonempty = [!isempty(nzrange(S, j)) for j in 1:n]
        @test all((p.colors .> 0) .== nonempty)
        @test sort(unique(p.colors[nonempty])) == 1:p.ncolors
        @test sum(length, p.color_entries; init=0) == nnz(S)
    end
end

@testset "sparse evaluator matches the dense Jacobian" begin
    P = transcription_problem(12)
    x = P.x0 .+ 0.01 .* sin.(1:P.n)
    Jref = NLPInterface.jacobian_residual(P.fitness, x, P.n_ceq, P.n_cineq, :forwarddiff)
    rref = NLPInterface.packed_residual(P.fitness, x, P.n_ceq, P.n_cineq)
    for deriv in (:forwarddiff, :finitediff), r_valid in (false, true)
        r, J, calls, p = sparse_jac(P.fitness, P.pattern, x, P.n_ceq, P.n_cineq, deriv; r_valid=r_valid)
        @test r == rref
        if deriv === :forwarddiff
            @test J == Jref
            @test calls == cld(p.ncolors, NLPInterface.SPARSE_CHUNK_MAX)
        else
            @test J ≈ Jref rtol = 1e-6 atol = 1e-6
            @test calls == p.ncolors + !r_valid
        end
        @test p.ncolors < P.n ÷ 4
    end

    # Every test problem, with the pattern of its dense Jacobian (a superset of
    # the true one is fine: extra entries come out as zero).
    for (name, q) in problems
        nr, n = 1 + q.n_ceq + q.n_cineq, length(q.x0)
        x = clamp.(q.x0 .+ 0.1, q.x_lb, q.x_ub)
        Jref = NLPInterface.jacobian_residual(q.fitness, x, q.n_ceq, q.n_cineq, :forwarddiff)
        for pattern in (Jref .!= 0, trues(nr, n))
            for deriv in (:forwarddiff, :finitediff)
                _, J, _, _ = sparse_jac(q.fitness, pattern, x, q.n_ceq, q.n_cineq, deriv; x_lb=q.x_lb, x_ub=q.x_ub)
                @test isapprox(J, Jref; rtol=deriv === :forwarddiff ? 1e-12 : 1e-5, atol=deriv === :forwarddiff ? 0 : 1e-5)
            end
        end
    end

    # More colors than one Dual chunk: several passes, all entries filled.
    n = 30
    f = x -> (sum(abs2, x), [x[i] * x[i + 1] for i in 1:(n - 1)], nothing)
    x = collect(range(0.1, 1.0; length=n))
    Jref = NLPInterface.jacobian_residual(f, x, n - 1, 0, :forwarddiff)
    _, J, calls, p = sparse_jac(f, Jref .!= 0, x, n - 1, 0, :forwarddiff)
    @test p.ncolors == n   # the objective row is dense
    @test calls == cld(n, NLPInterface.SPARSE_CHUNK_MAX)
    @test J == Jref

    # Finite differences step backward at the upper bound, as the dense evaluator does.
    lb, ub = zeros(n), copy(x)
    Jd = NLPInterface.jacobian_residual(f, x, n - 1, 0, :forwarddiff)
    seen = Vector{Float64}[]
    g = ξ -> (push!(seen, copy(ξ)); f(ξ))
    _, J, _, _ = sparse_jac(g, Jref .!= 0, x, n - 1, 0, :finitediff; x_lb=lb, x_ub=ub)
    @test all(ξ -> all(lb .<= ξ .<= ub), seen)
    @test J ≈ Jd rtol = 1e-5

    # Empty pattern: no derivative work, the value is still computed.
    f0 = x -> (1.0, [2.0], nothing)
    for deriv in (:forwarddiff, :finitediff)
        r, J, calls, _ = sparse_jac(f0, spzeros(Bool, 2, 3), zeros(3), 1, 0, deriv)
        @test r == [1.0, 2.0] && calls == 1 && iszero(J)
    end
end

@testset "detect_jac_sparsity" begin
    P = transcription_problem(10)
    S = detect_jac_sparsity(P.fitness, P.x0, P.n_ceq, P.n_cineq)
    @test S isa SparseMatrixCSC{Bool}
    @test S == P.pattern
    @test detect_jac_sparsity(P.fitness, P.x0, P.n_ceq, P.n_cineq; detector=TracerSparsityDetector()) == S
    # A branch on the value of x fails with the global detector, and the error
    # says what to do; the local detector gives the pattern at x0.
    fb = x -> (x[1] > 0 ? x[1]^2 : x[2]^2, [x[1] - x[3]], nothing)
    err = try
        detect_jac_sparsity(fb, [1.0, 2.0, 3.0], 1, 0)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("TracerLocalSparsityDetector", sprint(showerror, err))
    @test detect_jac_sparsity(fb, [1.0, 2.0, 3.0], 1, 0; detector=TracerLocalSparsityDetector()) ==
          sparse([1, 2, 2], [1, 1, 3], true, 2, 3)
    @test_throws ArgumentError detect_jac_sparsity(fb, [1.0, 2.0, 3.0], 1, 0; detector=:tracer)
    # Wrong constraint counts are reported as usual.
    @test_throws DimensionMismatch detect_jac_sparsity(P.fitness, P.x0, P.n_ceq - 1, P.n_cineq)
end
