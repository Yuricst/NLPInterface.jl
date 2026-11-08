# solve_ipopt (src/ipopt.jl): options, exceptions, evaluation counts.

@testset "Ipopt options" begin
    p = problems["rosenbrock"]
    args = (p.fitness, p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq)
    # no Hessian callback: any other approximation is an error, not ignored
    @test_throws ArgumentError solve_ipopt(args...; print_level=0, hessian_approximation="exact")
    @test_throws ArgumentError solve_ipopt(args...; print_level=0, hessian_approximation=:exact)
    for val in ("limited-memory", "LIMITED-MEMORY", Symbol("limited-memory"))
        _, _, info = solve_ipopt(args...; print_level=0, hessian_approximation=val)
        @test info === :Solve_Succeeded
    end
    # integer values for real options are converted; integer options stay integers
    @test "tol" in NLPInterface.IPOPT_REAL_OPTIONS
    @test !("max_iter" in NLPInterface.IPOPT_REAL_OPTIONS)
    _, _, info = solve_ipopt(args...; print_level=0, tol=1, constr_viol_tol=1, acceptable_tol=1)
    @test info in (:Solve_Succeeded, :Solved_To_Acceptable_Level)
    _, _, info = solve_ipopt(args...; print_level=0, max_iter=2)
    @test info === :Maximum_Iterations_Exceeded
    @test_throws ErrorException solve_ipopt(args...; print_level=0, no_such_option=1)
end

@testset "Ipopt exceptions in fitness" begin
    p = problems["rosenbrock"]
    args = (p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq)
    # The exception is caught inside the callback (it never unwinds through
    # Ipopt's C++ frames), Ipopt is stopped, and the exception is rethrown.
    for deriv in (:forwarddiff, :finitediff),
        err in (ErrorException("boom"), DomainError(1.0), InterruptException(), DimensionMismatch("dm"))
        for nth in (1, 5, 12)
            ncalls = Ref(0)
            nafter = Ref(0)
            f = function (x)
                ncalls[] += 1
                ncalls[] > nth && (nafter[] += 1)
                ncalls[] == nth && throw(err)
                return p.fitness(x)
            end
            @test_throws typeof(err) solve_ipopt(f, args...; deriv=deriv, print_level=0)
            @test nafter[] == 0
        end
    end
    # the original backtrace is logged at debug level
    f_err = x -> error("from fitness")
    @test_logs (:debug, r"backtrace") min_level = Base.CoreLogging.Debug match_mode = :any begin
        @test_throws ErrorException solve_ipopt(f_err, args...; print_level=0)
    end
    # later solves are unaffected
    xopt, fopt, info = solve_ipopt(p.fitness, args...; print_level=0)
    @test info === :Solve_Succeeded
    @test fopt ≈ p.fopt atol = 1e-5
end

@testset "fitness evaluation counts: Ipopt" begin
    p = problems["rosenbrock"]

    # Ipopt: the cache answers repeated requests at the same point, so two
    # consecutive calls are never at the same `x`, except a plain call followed
    # by the Dual call for its Jacobian. Trial points that only need f or g do
    # not get a Jacobian. (Ipopt may come back to an earlier point after
    # visiting another one, e.g. after a second-order correction, and that
    # point is evaluated again, so `allunique` would depend on the
    # floating-point path of the line search and differs between platforms.)
    for deriv in (:forwarddiff, :finitediff)
        g, plain, dual, events = counted(p.fitness)
        xopt, _, info = solve_ipopt(g, p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq;
                                    deriv=deriv, print_level=0)
        @test info === :Solve_Succeeded
        @test all(2:length(events)) do i
            (dual1, x1), (dual2, x2) = events[i - 1], events[i]
            x1 != x2 || (!dual1 && dual2)
        end
        if deriv === :forwarddiff
            @test length(dual) < length(plain)
        end
    end
end
