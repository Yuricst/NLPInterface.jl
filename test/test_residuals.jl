# Packing and differentiating the fitness output (src/residuals.jl).

@testset "constraint counts must match fitness output" begin
    f = x -> (sum(abs2, x), [x[1] + x[2] - 1.0], [x[1] - 2.0, x[2] - 2.0])
    x = [0.3, 0.4]
    @test NLPInterface.packed_residual(f, x, 1, 2) ≈ [0.25, -0.3, -1.7, -1.6]
    # equality block declared absent but returned: must not be dropped silently
    @test_throws DimensionMismatch NLPInterface.packed_residual(f, x, 0, 2)
    # one value returned for two declared rows: must not be broadcast
    @test_throws DimensionMismatch NLPInterface.packed_residual(f, x, 2, 2)
    @test_throws DimensionMismatch NLPInterface.packed_residual(f, x, 1, 1)
    @test_throws DimensionMismatch NLPInterface.packed_residual(f, x, 1, 3)
    # scalar block is accepted for a count of one, `nothing` for a count of zero
    @test_throws ArgumentError NLPInterface.packed_residual(x -> (1.0, missing, nothing), x, 0, 0)
    @test_throws ArgumentError NLPInterface.packed_residual(x -> (1.0, "ab", nothing), x, 2, 0)
    @test NLPInterface.packed_residual(x -> (1.0, (2.0, 3.0), reshape([4.0], 1, 1)), x, 2, 1) == [1.0, 2.0, 3.0, 4.0]
    g = x -> (x[1]^2, x[1] - 1.0, nothing)
    @test NLPInterface.packed_residual(g, x, 1, 0) ≈ [0.09, -0.7]
    @test_throws DimensionMismatch NLPInterface.packed_residual(g, x, 2, 0)
    @test_throws DimensionMismatch NLPInterface.packed_residual(g, x, 1, 1)
    # Dual inputs (ForwardDiff path) are checked as well
    @test_throws DimensionMismatch NLPInterface.jacobian_residual(f, x, 0, 2, :forwarddiff)
    @test_throws DimensionMismatch NLPInterface.jacobian_residual(f, x, 0, 2, :finitediff)

    lb, ub = [-5.0, -5.0], [5.0, 5.0]
    @test_throws DimensionMismatch solve_ipopt(f, [0.0, 0.0], lb, ub, 0, 2; print_level=0)
end

@testset "constraint block types" begin
    x = [0.3, 0.4]
    pr(ceq, n) = NLPInterface.packed_residual(x -> (1.0, ceq, nothing), x, n, 0)
    @test_throws ArgumentError pr(['a'], 1)               # not character codes
    @test_throws ArgumentError pr(('a',), 1)
    @test_throws ArgumentError pr('a', 1)
    @test_throws ArgumentError pr([1.0, missing], 2)      # not a MethodError from convert
    @test_throws ArgumentError pr(Any[1.0, "b"], 2)
    @test_throws ArgumentError pr(:a, 1)                  # not MethodError: length
    @test_throws ArgumentError pr(Some(1.0), 1)
    @test_throws ArgumentError pr(1.0 + 2.0im, 1)
    @test_throws ArgumentError pr([1.0 + 2.0im], 1)
    @test pr(Any[1.0, 2], 2) == [1.0, 1.0, 2.0]
    @test pr(Union{Missing,Float64}[1.0, 2.0], 2) == [1.0, 1.0, 2.0]
    @test pr((v for v in (1.0, 2.0)), 2) == [1.0, 1.0, 2.0]
    @test pr(true, 1) == [1.0, 1.0]
    # iterables without a known length are collected (and iterated once)
    @test pr((v for v in Iterators.filter(>(0), [1.0, -1.0, 2.0])), 2) == [1.0, 1.0, 2.0]
    @test_throws DimensionMismatch pr((v for v in Iterators.filter(>(0), [1.0, -1.0])), 2)
    # unordered collections are rejected
    @test_throws ArgumentError pr(Set([1.0, 2.0]), 2)
    @test_throws ArgumentError pr(Dict(1 => 2.0), 1)
    @test_throws ArgumentError pr(Iterators.repeated(1.0), 1)  # infinite
    @test pr(NoLengthIter(), 2) == [1.0, 0.0, 1.0]
    @test_throws ArgumentError NLPInterface.packed_residual(x -> ([1.0], nothing, nothing), x, 0, 0)
    @test_throws ArgumentError NLPInterface.packed_residual(x -> (nothing, nothing, nothing), x, 0, 0)
    # the Dual (ForwardDiff) path checks entries too
    @test_throws ArgumentError NLPInterface.jacobian_residual(x -> (x[1], ['a'], nothing), x, 1, 0, :forwarddiff)
end

@testset "fitness evaluation counts: ResidualEvaluator" begin
    p = problems["rosenbrock"]
    n = length(p.x0)

    # ResidualEvaluator: value and Jacobian in one pass.
    x = [0.3, 0.4]
    J_ref = NLPInterface.jacobian_residual(p.fitness, x, p.n_ceq, p.n_cineq, :forwarddiff)
    r_ref = NLPInterface.packed_residual(p.fitness, x, p.n_ceq, p.n_cineq)
    for deriv in (:forwarddiff, :finitediff), r_valid in (false, true)
        g, plain, dual = counted(p.fitness)
        ev = NLPInterface.ResidualEvaluator(g, n, p.n_ceq, p.n_cineq, deriv)
        r = r_valid ? copy(r_ref) : fill(NaN, length(r_ref))
        J = fill(NaN, size(J_ref))
        NLPInterface.residual_value_and_jacobian!(ev, r, J, x, r_valid)
        @test r == r_ref
        @test J ≈ J_ref rtol = (deriv === :forwarddiff ? 1e-12 : 1e-6)
        if deriv === :forwarddiff
            @test isempty(plain)   # the value comes from the Dual pass
            @test length(dual) == 1
        else
            @test length(plain) == n + !r_valid
            @test isempty(dual)
        end
    end
end
