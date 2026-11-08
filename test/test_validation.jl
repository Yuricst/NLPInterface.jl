# check_solver_inputs (src/validation.jl), directly and through the solvers.

@testset "input validation" begin
    f = x -> (sum(abs2, x), nothing, nothing)
    bad = [
        (Float64[], Float64[], Float64[]),        # empty x0
        ([0.0, 0.0], [NaN, -1.0], [1.0, 1.0]),    # NaN bound
        ([0.0, 0.0], [-1.0, -1.0], [1.0, NaN]),
        ([0.0, 0.0], [0.5, -1.0], [0.0, 1.0]),    # x_lb > x_ub
        ([0.0, 0.0], [Inf, -1.0], [Inf, 1.0]),    # lower bound +Inf
        ([0.0, 0.0], [-Inf, -1.0], [-Inf, 1.0]),  # upper bound -Inf
        ([NaN, 0.0], [-1.0, -1.0], [1.0, 1.0]),   # x0 not finite
        ([Inf, 0.0], [-1.0, -1.0], [1.0, 1.0]),
        ([0.0], [-1.0, -1.0], [1.0, 1.0]),        # length mismatch
    ]
    for (x0, lb, ub) in bad
        @test_throws ArgumentError NLPInterface.check_solver_inputs(x0, lb, ub, 0, 0, :forwarddiff)
        @test_throws ArgumentError solve_ipopt(f, x0, lb, ub, 0, 0; print_level=0)
        if has_snopt()
            @test_throws ArgumentError solve_snopt(f, x0, lb, ub, 0, 0)
        end
    end
    for (x0, lb, ub) in (([missing, 0.0], [-1.0, -1.0], [1.0, 1.0]),
                         ([0.0, 0.0], [nothing, -1.0], [1.0, 1.0]),
                         ([0.0, 0.0], [-1.0, -1.0], Any[1.0, "1"]),
                         ([big"1e400", 0.0], [-Inf, -1.0], [Inf, 1.0]),  # Inf as Float64
                         ([0.0, 0.0], [big"1e400", -1.0], [Inf, 1.0]))   # +Inf lower bound
        @test_throws ArgumentError NLPInterface.check_solver_inputs(x0, lb, ub, 0, 0, :forwarddiff)
    end
    err = try
        NLPInterface.check_solver_inputs([0.0, 0.0], [-1.0, 2.0], [1.0, 1.0], 0, 0, :forwarddiff)
    catch e
        e
    end
    @test occursin("x_lb[2]", err.msg)
    @test_throws ArgumentError NLPInterface.check_solver_inputs([0.0], [-1.0], [1.0], -1, 0, :forwarddiff)
    @test_throws ArgumentError NLPInterface.check_solver_inputs([0.0], [-1.0], [1.0], 0, 0, :symbolic)
    # infinite bounds, equal bounds and x0 outside the bounds are fine
    @test NLPInterface.check_solver_inputs([5.0, 0.0], [-Inf, 0.0], [Inf, 0.0], 0, 0, :finitediff) === nothing
    xopt, _, info = solve_ipopt(x -> ((x[1] - 1)^2 + x[2]^2, nothing, nothing),
                                [3.0, 0.0], [-Inf, 0.5], [Inf, 0.5], 0, 0; print_level=0)
    @test info === :Solve_Succeeded
    @test xopt ≈ [1.0, 0.5] atol = 1e-6
end
