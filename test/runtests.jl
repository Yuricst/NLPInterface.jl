using NLPInterface
using ForwardDiff
using Test

include("problems.jl")

@testset "NLPInterface" begin
    @testset "$name" for name in sort!(collect(keys(problems)))
        p = problems[name]

        @testset "solve_ipopt deriv=$deriv" for deriv in (:forwarddiff, :finitediff)
            ipopt_tols = if deriv === :forwarddiff
                (tol=1e-6, constr_viol_tol=1e-8)
            else
                NamedTuple()
            end
            x0_before = copy(p.x0)
            xopt, fopt, info = solve_ipopt(
                p.fitness,
                p.x0,
                p.x_lb,
                p.x_ub,
                p.n_ceq,
                p.n_cineq;
                deriv=deriv,
                print_level=0,
                ipopt_tols...,
            )
            @test p.x0 == x0_before
            @test info in (:Solve_Succeeded, :Solved_To_Acceptable_Level)
            @test xopt ≈ p.xopt atol = 1e-4
            @test fopt ≈ p.fopt atol = 1e-5
            _, ceq, cineq = p.fitness(xopt)
            @test all(abs.(ceq) .<= 1e-6)
            @test all(cineq .<= 1e-6)
        end

        if has_snopt()
            @testset "solve_snopt deriv=$deriv" for deriv in (:forwarddiff, :finitediff)
                snopt_tols = if deriv === :forwarddiff
                    (Major_feasibility_tolerance=1e-8, Major_optimality_tolerance=1e-6)
                else
                    NamedTuple()
                end
                x0_before = copy(p.x0)
                xopt, fopt, info = solve_snopt(
                    p.fitness,
                    p.x0,
                    p.x_lb,
                    p.x_ub,
                    p.n_ceq,
                    p.n_cineq;
                    deriv=deriv,
                    Major_print_level=0,
                    snopt_tols...,
                )
                @test p.x0 == x0_before
                @test info in (:Solve_Succeeded, :Solved_To_Acceptable_Level)
                @test xopt ≈ p.xopt atol = 1e-4
                @test fopt ≈ p.fopt atol = 1e-5
                _, ceq, cineq = p.fitness(xopt)
                @test all(abs.(ceq) .<= 1e-6)
                @test all(cineq .<= 1e-6)
            end
        end
    end

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
        if has_snopt()
            @test_throws DimensionMismatch solve_snopt(f, [0.0, 0.0], lb, ub, 0, 2; Major_print_level=0)
            # Default output settings (no files) must work, including after a solve
            # that aborted in the user callback. With /dev/null as both outputs,
            # these returned :Invalid_Problem_Definition.
            p = problems["rosenbrock"]
            for _ in 1:2
                _, _, info = solve_snopt(p.fitness, p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq)
                @test info === :Solve_Succeeded
            end
        end
    end

    if has_snopt()
        @testset "SNOPT output files" begin
            p = problems["rosenbrock"]
            mktempdir() do dir
                pf = joinpath(dir, "print.out")
                sf = joinpath(dir, "summary.out")
                @test_throws ArgumentError solve_snopt(
                    p.fitness, p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq;
                    Print_file=pf, Summary_file=pf,
                )
                # Two names for one file are rejected before SNOPT opens anything,
                # and the file stays usable afterwards.
                cd(dir) do
                    pairs = [("p.out", abspath("p.out")), ("p.out", "./p.out"),
                             ("p.out", "p.out "), (" screen", " screen")]
                    if !Sys.iswindows()  # symlinks need extra privileges on Windows
                        touch("real.out")
                        symlink("real.out", "link.out")
                        mkdir("d")
                        symlink("d", "dl")
                        push!(pairs, ("real.out", "link.out"), ("d/q.out", "dl/q.out"))
                    end
                    for (a, b) in pairs
                        @test_throws ArgumentError solve_snopt(
                            p.fitness, p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq;
                            Print_file=a, Summary_file=b,
                        )
                    end
                    _, _, info = solve_snopt(
                        p.fitness, p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq;
                        Print_file="p.out",
                    )
                    @test info === :Solve_Succeeded
                    # leading blanks are kept by SNOPT: " p.out" and "p.out" differ
                    _, _, info = solve_snopt(
                        p.fitness, p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq;
                        Print_file=" p.out", Summary_file="p.out",
                    )
                    @test info === :Solve_Succeeded
                end
                @test_throws ArgumentError solve_snopt(
                    p.fitness, p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq;
                    Print_file=joinpath(dir, "no_such_dir", "p.out"),
                )
                @test_throws ArgumentError solve_snopt(
                    p.fitness, p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq;
                    Print_file=dir,
                )
                _, _, info = solve_snopt(
                    p.fitness, p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq;
                    Print_file=pf, Summary_file=sf,
                )
                @test info === :Solve_Succeeded
                @test filesize(pf) > 0
                @test filesize(sf) > 0
            end
        end
    end

    @testset "fitness evaluation counts" begin
        # Wrap `fitness` to record every point it is called at, split into plain
        # (Float64) and Dual (ForwardDiff) calls. `events` keeps the call order as
        # `(is_dual, x)` pairs.
        function counted(f)
            plain = Vector{Vector{Float64}}()
            dual = Vector{Vector{Float64}}()
            events = Tuple{Bool,Vector{Float64}}[]
            g = function (x)
                if eltype(x) <: ForwardDiff.Dual
                    push!(dual, ForwardDiff.value.(x))
                    push!(events, (true, dual[end]))
                else
                    push!(plain, collect(Float64, x))
                    push!(events, (false, plain[end]))
                end
                return f(x)
            end
            return g, plain, dual, events
        end
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

        if has_snopt()
            # SNOPT with ForwardDiff: no plain call at points that get a Jacobian,
            # except the last one, which evaluates `fopt` at `xopt`.
            g, plain, dual = counted(p.fitness)
            xopt, fopt, info = solve_snopt(g, p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq)
            @test info === :Solve_Succeeded
            @test plain[end] == xopt
            @test fopt == p.fitness(xopt)[1]
            @test isempty(intersect(plain[1:(end - 1)], dual))
        end
    end

    if has_snopt()
        @testset "SNOPT exceptions in fitness" begin
            p = problems["rosenbrock"]
            args = (p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq)

            # A transient DomainError: SNOPT shortens the step and converges, and
            # the error is not rethrown afterwards.
            for deriv in (:forwarddiff, :finitediff)
                ncalls = Ref(0)
                f = function (x)
                    ncalls[] += 1
                    ncalls[] == 5 && throw(DomainError(x, "transient"))
                    return p.fitness(x)
                end
                xopt, fopt, info = solve_snopt(f, args...; deriv=deriv)
                @test ncalls[] > 5
                @test info === :Solve_Succeeded
                @test xopt ≈ p.xopt atol = 1e-4
                @test fopt ≈ p.fopt atol = 1e-5
            end

            # DomainError at x0: SNOPT gives up, fopt is NaN (not the initial 0.0).
            xopt, fopt, info = @test_logs (:warn, r"undefined") match_mode = :any solve_snopt(
                x -> throw(DomainError(x)), args...,
            )
            @test info === :User_Supplied_Function_Undefined
            @test isnan(fopt)
            @test xopt == p.x0

            # Any other error (and Ctrl-C) stops SNOPT at once and is rethrown.
            for err in (ErrorException("boom"), InterruptException(), ArgumentError("bad"))
                ncalls = Ref(0)
                nafter = Ref(0)
                f = function (x)
                    ncalls[] += 1
                    ncalls[] > 5 && (nafter[] += 1)
                    ncalls[] == 5 && throw(err)
                    return p.fitness(x)
                end
                @test_throws typeof(err) solve_snopt(f, args...; deriv=:finitediff)
                @test nafter[] == 0
            end

            # Later solves are unaffected.
            xopt, fopt, info = solve_snopt(p.fitness, args...)
            @test info === :Solve_Succeeded
            @test fopt ≈ p.fopt atol = 1e-5
        end

        @testset "SNOPT fopt and fallback path" begin
            p = problems["rosenbrock"]
            args = (p.fitness, p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq)
            # SNOPT's derivative check can leave F at a perturbed point; fopt
            # must still be the objective at xopt.
            for deriv in (:forwarddiff, :finitediff)
                xopt, fopt, info = solve_snopt(args...; deriv=deriv, Iterations_limit=1)
                @test info === :Maximum_Iterations_Exceeded
                @test fopt == p.fitness(xopt)[1]
            end

            mktempdir() do dir
                # "stdout" is a file name, as f_sninitx treats it
                cd(dir) do
                    _, _, info = solve_snopt(args...; Summary_file="stdout")
                    @test info === :Solve_Succeeded
                    @test filesize("stdout") > 0
                end
                if !Sys.iswindows() && ccall(:geteuid, Cuint, ()) != 0
                    ro = joinpath(dir, "ro")
                    mkdir(ro)
                    chmod(ro, 0o500)
                    try
                        # used to return :Numerical_Difficulties without a message
                        @test_throws ArgumentError solve_snopt(args...; Print_file=joinpath(ro, "p.out"))
                    finally
                        chmod(ro, 0o700)
                    end
                end

                # The sninit_ fallback (libraries without f_sninitx) opens the
                # files with SNOPT's Fortran helpers.
                if NLPInterface.libsnopt7_has_sninitx && NLPInterface.snopt_fallback_opens_files()
                    try
                        NLPInterface.eval(:(libsnopt7_has_sninitx = false))
                        pf = joinpath(dir, "fb_print.out")
                        sf = joinpath(dir, "fb_summ.out")
                        for _ in 1:2
                            xopt, fopt, info = solve_snopt(args...; Print_file=pf, Summary_file=sf)
                            @test info === :Solve_Succeeded
                            @test xopt ≈ p.xopt atol = 1e-4
                            @test occursin("S N O P T", read(pf, String))
                            @test filesize(sf) > 0
                        end
                    finally
                        NLPInterface.eval(:(libsnopt7_has_sninitx = true))
                    end
                end
            end
        end

        @testset "nested solve_snopt" begin
            inner_args = ([0.0, 0.0], [-10.0, -10.0], [10.0, 10.0], 0, 0)
            inner = x -> ((x[1] - 2)^2 + (x[2] - 3)^2, nothing, nothing)
            # Calling solve_snopt from fitness used to clear the outer callback,
            # so the outer solve silently returned garbage. It now throws.
            outer = function (x)
                solve_snopt(inner, inner_args...)
                return inner(x)
            end
            err = try
                solve_snopt(outer, inner_args...)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("nested", sprint(showerror, err))

            if isdefined(Base, :ScopedValues)
                # From a task spawned in fitness: used to deadlock on the lock.
                outer_spawn = function (x)
                    fetch(Threads.@spawn solve_snopt(inner, inner_args...))
                    return inner(x)
                end
                @test_throws TaskFailedException solve_snopt(outer_spawn, inner_args...)

                # A task spawned in fitness that runs after the solve has ended
                # is not nested and may solve.
                go = Channel{Nothing}(1)
                worker = Ref{Task}()
                outer_worker = function (x)
                    if !isassigned(worker)
                        worker[] = Threads.@spawn (take!(go); solve_snopt(inner, inner_args...))
                    end
                    return inner(x)
                end
                _, _, info = solve_snopt(outer_worker, inner_args...)
                @test info === :Solve_Succeeded
                put!(go, nothing)
                xw, _, infow = fetch(worker[])
                @test infow === :Solve_Succeeded
                @test xw ≈ [2.0, 3.0] atol = 1e-6
            end

            # Neither leaves global state behind.
            xopt, fopt, info = solve_snopt(inner, inner_args...)
            @test info === :Solve_Succeeded
            @test xopt ≈ [2.0, 3.0] atol = 1e-6
            @test fopt ≈ 0.0 atol = 1e-10
        end
    end

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
                             ([big"1e400", 0.0], [-Inf, -1.0], [Inf, 1.0]))  # Inf as Float64
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
        @test_throws ArgumentError NLPInterface.packed_residual(x -> ([1.0], nothing, nothing), x, 0, 0)
        @test_throws ArgumentError NLPInterface.packed_residual(x -> (nothing, nothing, nothing), x, 0, 0)
        # the Dual (ForwardDiff) path checks entries too
        @test_throws ArgumentError NLPInterface.jacobian_residual(x -> (x[1], ['a'], nothing), x, 1, 0, :forwarddiff)
    end

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

    @testset "SNOPT helpers" begin
        # DomainErrors from tasks spawned in fitness count as "undefined here"
        de = DomainError(1.0)
        wait_failed(e) = try
            wait(Threads.@spawn throw(e)); nothing
        catch err
            err
        end
        @test NLPInterface.snopt_callback_status(de) == -1
        @test NLPInterface.snopt_callback_status(wait_failed(de)) == -1
        @test NLPInterface.snopt_callback_status(CompositeException([wait_failed(de), de])) == -1
        @test NLPInterface.snopt_callback_status(CapturedException(de, [])) == -1
        @test NLPInterface.snopt_callback_status(wait_failed(ErrorException("x"))) == -2
        @test NLPInterface.snopt_callback_status(CompositeException([de, ErrorException("x")])) == -2
        @test NLPInterface.snopt_callback_status(CompositeException([])) == -2
        @test NLPInterface.snopt_callback_status(InterruptException()) == -2

        # only the exact name "screen" is the console
        @test NLPInterface._snopt_is_console_name("screen")
        @test NLPInterface._snopt_is_console_name("screen  ")
        for name in ("stdout", "-", "Screen", " screen", "")
            @test !NLPInterface._snopt_is_console_name(name)
        end
        @test !NLPInterface._snopt_console_summary(1, "stdout")
        @test NLPInterface._snopt_console_summary(1, "")

        # library search: no empty entries, no development-only directory
        sep = Sys.iswindows() ? ';' : ':'
        var = Sys.iswindows() ? "PATH" : "LD_LIBRARY_PATH"
        dirs = withenv(var => join(["", "/opt/a", "", "/opt/b", ""], sep)) do
            NLPInterface.snopt_search_dirs()
        end
        @test !any(isempty, dirs)
        @test "/opt/a" in dirs && "/opt/b" in dirs
        @test !any(d -> occursin("__references", d), dirs)

        # output file checks (pure Julia, run before SNOPT opens anything)
        mktempdir() do dir
            check = NLPInterface.check_snopt_output_files
            @test check(joinpath(dir, "p.out"), joinpath(dir, "s.out")) === nothing
            @test !ispath(joinpath(dir, "p.out"))  # the probe leaves nothing behind
            write(joinpath(dir, "keep.out"), "data")
            @test check(joinpath(dir, "keep.out"), "") === nothing
            @test read(joinpath(dir, "keep.out"), String) == "data"
            @test check("screen", "") === nothing
            Sys.iswindows() || @test check("/dev/null", "") === nothing
            @test_throws ArgumentError check(joinpath(dir, "p.out"), joinpath(dir, "p.out"))
            @test_throws ArgumentError check(joinpath(dir, "nodir", "p.out"), "")
            @test_throws ArgumentError check(dir, "")
            if !Sys.iswindows()  # symlinks need extra privileges on Windows
                # a dangling symlink is probed through its target and left intact
                lnk = joinpath(dir, "dangling.out")
                tgt = joinpath(dir, "target.out")
                symlink(tgt, lnk)
                @test check(lnk, "") === nothing
                @test islink(lnk) && !ispath(tgt)
                loop = joinpath(dir, "loop.out")
                symlink(loop, loop)
                @test_throws ArgumentError check(loop, "")
            end
            if !Sys.iswindows() && ccall(:geteuid, Cuint, ()) != 0
                ro = joinpath(dir, "ro")
                mkdir(ro)
                chmod(ro, 0o500)
                try
                    @test_throws ArgumentError check(joinpath(ro, "p.out"), "")
                    @test_throws ArgumentError check("", joinpath(ro, "s.out"))
                finally
                    chmod(ro, 0o700)
                end
            end
            insensitive = NLPInterface._snopt_dir_is_case_insensitive(dir)
            touch(joinpath(dir, "Probe.out"))
            @test isfile(joinpath(dir, "probe.out")) == insensitive
            @test !any(startswith("NLPInterfaceCaseProbe"), readdir(dir))
            same = NLPInterface._snopt_same_output_file(joinpath(dir, "a.out"), joinpath(dir, "A.out"))
            @test same == (insensitive && (Sys.isapple() || Sys.iswindows()))
        end
    end

    # The SNOPT ccall plumbing against a mock library (test/mock_snopt), so that
    # it runs in CI without a licensed SNOPT. Needs a C compiler; CI sets
    # NLPINTERFACE_REQUIRE_MOCK_SNOPT=1 where one must be available. On Windows
    # (DLL build and loading untested) it runs only when that variable is set.
    require_mock = get(ENV, "NLPINTERFACE_REQUIRE_MOCK_SNOPT", "") == "1"
    @testset "mock SNOPT library" begin
        include(joinpath(@__DIR__, "mock_snopt", "build.jl"))
        mktempdir() do root
            libs = Sys.iswindows() && !require_mock ? nothing : build_mock_snopt(root)
            if libs === nothing
                if require_mock
                    error("NLPINTERFACE_REQUIRE_MOCK_SNOPT=1 but the mock SNOPT library could not be built")
                end
                @test_skip "mock SNOPT library not built"
            else
                script = joinpath(@__DIR__, "mock_snopt", "runtests_mock.jl")
                for variant in (:sninitx, :fallback)
                    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $script $variant`
                    cmd = addenv(cmd, "SNOPT_LIB" => libs[variant])
                    @test success(pipeline(cmd; stdout=stdout, stderr=stderr))
                end
            end
        end
    end

    @testset "SNOPT status map" begin
        success = (:Solve_Succeeded, :Solved_To_Acceptable_Level)
        @test NLPInterface.SNOPT_RETURN_STATUS[1] === :Solve_Succeeded
        @test NLPInterface.SNOPT_RETURN_STATUS[3] === :Solved_To_Acceptable_Level
        @test NLPInterface.SNOPT_RETURN_STATUS[2] === :Feasible_Point_Found
        # codes 4-6 are not reported as success: the manual does not promise feasibility
        for code in (4, 5, 6)
            @test !(NLPInterface.SNOPT_RETURN_STATUS[code] in success)
        end
        for code in (11, 12, 13, 14, 15, 16)
            @test NLPInterface.SNOPT_RETURN_STATUS[code] === :Infeasible_Problem_Detected
        end
    end

    if !has_snopt()
        @info "Skipping SNOPT tests; set SNOPT_LIB or SNOPTDIR to the libsnopt7_cpp library"
        @test_skip "SNOPT library not found"
    end
end
