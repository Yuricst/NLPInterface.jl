# Tests of the SNOPT ccall plumbing against the mock library in this directory.
# test/test_mock_snopt.jl builds the mock and runs this file in a separate Julia
# process, with SNOPT_LIB pointing at the mock wrapper and the variant
# ("sninitx" or "fallback") as the only argument. The mock does not implement
# SNOPT (see mock_wrapper.c), so the problems here are box-constrained convex
# quadratics, which its projected gradient descent solves.

using NLPInterface
using ForwardDiff
using SparseArrays
using Test

const VARIANT = Symbol(only(ARGS))
const PKG_UUID = "e604ab98-cb68-4d52-a41a-b2fbd5dba595"

quad(x) = ((x[1] - 1)^2 + (x[2] - 2)^2, nothing, nothing)
const ARGS0 = ([0.0, 0.0], [-5.0, -5.0], [5.0, 5.0], 0, 0)

@testset "mock SNOPT ($VARIANT)" begin
    @testset "library discovery" begin
        @test has_snopt()
        @test NLPInterface.libsnopt7_has_sninitx == (VARIANT === :sninitx)
        lib = realpath(NLPInterface.libsnopt7)
        if Sys.isapple()
            # The mock core's install name points to a missing directory, so the
            # wrapper is relocated, once, into the package's scratch space.
            cache = joinpath(first(DEPOT_PATH), "scratchspaces", PKG_UUID, "snopt_relocated")
            @test startswith(lib, realpath(cache))
            again = NLPInterface.relocate_snopt_library(ENV["SNOPT_LIB"])
            @test realpath(again) == lib
            t = mtime(again)
            @test NLPInterface.relocate_snopt_library(ENV["SNOPT_LIB"]) == again
            @test mtime(again) == t
            # The mock lives in a fresh temporary directory on every run, so
            # its cache entry is never reused: remove it (macOS keeps the
            # mapped files alive).
            atexit(() -> rm(dirname(lib); recursive=true, force=true))
        else
            @test lib == realpath(ENV["SNOPT_LIB"])
        end
        if VARIANT === :fallback
            @test NLPInterface.snopt_fallback_opens_files()
        end
    end

    @testset "solve deriv=$deriv" for deriv in (:forwarddiff, :finitediff)
        atol = deriv === :forwarddiff ? 1e-8 : 1e-6
        xopt, fopt, info = solve_snopt(quad, ARGS0...; deriv=deriv)
        @test info === :Solve_Succeeded
        @test xopt ≈ [1.0, 2.0] atol = atol
        @test fopt == quad(xopt)[1]

        xopt, _, info = solve_snopt(quad, [0.0, 0.0], [-5.0, -5.0], [0.5, 5.0], 0, 0; deriv=deriv)
        @test info === :Solve_Succeeded
        @test xopt ≈ [0.5, 2.0] atol = atol

        # Constraint rows are passed through (the mock ignores them).
        fc = x -> (quad(x)[1], [x[1] - 1], [x[2] - 3])
        _, _, info = solve_snopt(fc, [0.0, 0.0], [-5.0, -5.0], [5.0, 5.0], 1, 1; deriv=deriv)
        @test info === :Solve_Succeeded
        @test_throws DimensionMismatch solve_snopt(fc, [0.0, 0.0], [-5.0, -5.0], [5.0, 5.0], 0, 1; deriv=deriv)
    end

    @testset "sparse Jacobian deriv=$deriv" for deriv in (:forwarddiff, :finitediff)
        atol = deriv === :forwarddiff ? 1e-8 : 1e-6
        # The mock takes the objective gradient from the G entries of row 1, so
        # x[3] (not in the objective row of the pattern) never moves.
        f3 = x -> ((x[1] - 1)^2 + (x[2] - 2)^2, [x[1] - 1], [x[3] - 3])
        S = sparse([1, 1, 2, 3], [1, 2, 1, 3], true, 3, 3)
        args3 = ([0.0, 0.0, 0.5], fill(-5.0, 3), fill(5.0, 3), 1, 1)
        xopt, fopt, info = solve_snopt(f3, args3...; deriv=deriv, jac_sparsity=S)
        @test info === :Solve_Succeeded
        @test xopt ≈ [1.0, 2.0, 0.5] atol = atol
        @test fopt == f3(xopt)[1]
        @test_throws ArgumentError solve_snopt(f3, args3...; jac_sparsity=trues(2, 3))
        # Empty pattern: no G entries at all.
        xopt, fopt, info = solve_snopt(x -> (1.0, nothing, nothing), ARGS0...; deriv=deriv,
            jac_sparsity=spzeros(Bool, 1, 2))
        @test xopt == ARGS0[1]
        @test fopt == 1.0
    end

    @testset "NLPResult" begin
        # x[1] stops at its upper bound 0.5: ∇f = [-1, 0], so z_ub = [1, 0]
        # (the mock reports the gradient as SNOPT's reduced costs in xmul).
        ncalls = Ref(0)
        r = solve_snopt(x -> (ncalls[] += 1; quad(x)), [0.0, 0.0], [-5.0, -5.0], [0.5, 5.0], 0, 0)
        @test r isa NLPResult
        @test r.solver === :snopt
        @test r.status === :Solve_Succeeded && r.raw_status == 1
        @test r.xopt ≈ [0.5, 2.0] atol = 1e-8
        @test r.z_ub ≈ [1.0, 0.0] atol = 1e-6
        @test r.z_lb ≈ [0.0, 0.0] atol = 1e-6
        @test isempty(r.ceq) && isempty(r.cineq) && isempty(r.lambda_eq) && isempty(r.lambda_ineq)
        @test ismissing(r.iterations)
        @test r.n_fitness == ncalls[]
        @test r.fopt == quad(r.xopt)[1]
        xopt, fopt, info = r
        @test (xopt, fopt, info) == (r.xopt, r.fopt, r.status)
        # Constraint rows (ignored by the mock): values at xopt, one multiplier each.
        fc = x -> (quad(x)[1], [x[1] - 1], [x[2] - 3, x[1] + x[2] - 10])
        r = solve_snopt(fc, [0.0, 0.0], [-5.0, -5.0], [5.0, 5.0], 1, 2)
        _, ceq, cineq = fc(r.xopt)
        @test r.ceq == ceq && r.cineq == cineq
        @test length(r.lambda_eq) == 1 && length(r.lambda_ineq) == 2
        # Stopped early: complete fields, status from the INFO code.
        r = solve_snopt(quad, ARGS0...; Major_iterations_limit=1)
        @test r.status === :Maximum_Iterations_Exceeded && r.raw_status == 32
        @test length(r.z_lb) == 2
    end

    @testset "tasks spawned from fitness" begin
        # Regression (dev/gc_corruption/NOTES.md): solve_snopt used to wrap
        # fitness in a ScopedValue scope. A task spawned from fitness inherited
        # it, and on Julia 1.12 entering any `with` in that task with a GC
        # inside corrupted the heap ("GC error (probable corruption)"), which
        # aborts this whole process.
        for _ in 1:30
            go = Channel{Nothing}(1)
            worker = Ref{Task}()
            f = function (x)
                isassigned(worker) || (worker[] = Threads.@spawn (take!(go);
                    Base.CoreLogging.with_logger(() -> GC.gc(true), Base.CoreLogging.NullLogger())))
                return quad(x)
            end
            _, _, info = solve_snopt(f, ARGS0...)
            @test info === :Solve_Succeeded
            put!(go, nothing)
            wait(worker[])
            GC.gc(true)
        end
        # A solve from a task that fitness waits for throws instead of deadlocking.
        f = x -> (fetch(Threads.@spawn solve_snopt(quad, ARGS0...)); quad(x))
        @test_throws TaskFailedException solve_snopt(f, ARGS0...)
        # The lock is free again.
        @test solve_snopt(quad, ARGS0...)[3] === :Solve_Succeeded
    end

    @testset "options" begin
        _, _, info = solve_snopt(quad, ARGS0...; Major_iterations_limit=1)
        @test info === :Maximum_Iterations_Exceeded
        for val in (1, 1.0, "yes", :yes)
            @test_throws ArgumentError solve_snopt(quad, ARGS0...; Bogus_option=val)
        end
        @test_throws ArgumentError solve_snopt(quad, ARGS0...; Major_print_level=true)
        _, _, info = solve_snopt(
            quad, ARGS0...;
            Major_feasibility_tolerance=1e-6, Verify_level=-1, Hessian=:full_memory,
            Scale_option="0",
        )
        @test info === :Solve_Succeeded

        # Input checks run before SNOPT is called.
        @test_throws ArgumentError solve_snopt(quad, [0.0, 0.0], [NaN, -5.0], [5.0, 5.0], 0, 0)
        @test_throws ArgumentError solve_snopt(quad, [0.0, 0.0], [1.0, -5.0], [0.0, 5.0], 0, 0)
        @test_throws ArgumentError solve_snopt(quad, [Inf, 0.0], [-5.0, -5.0], [5.0, 5.0], 0, 0)
        @test_throws ArgumentError solve_snopt(quad, Float64[], Float64[], Float64[], 0, 0)
    end

    @testset "f_snmema failure" begin
        ncalls = Ref(0)
        g = x -> (ncalls[] += 1; quad(x))
        xopt, fopt, info = withenv(() -> solve_snopt(g, ARGS0...), "MOCK_SNOPT_MEMINFO" => "82")
        @test info === :Insufficient_Memory
        @test isnan(fopt)
        @test xopt == ARGS0[1]
        @test ncalls[] == 0
        _, _, info = solve_snopt(g, ARGS0...)
        @test info === :Solve_Succeeded
    end

    @testset "fopt is the objective at xopt" begin
        # SNOPT's final F can come from a perturbed point (derivative check).
        for deriv in (:forwarddiff, :finitediff)
            xopt, fopt, info = withenv("MOCK_SNOPT_PERTURB" => "1") do
                solve_snopt(quad, ARGS0...; Major_iterations_limit=3, deriv=deriv)
            end
            @test info === :Maximum_Iterations_Exceeded
            @test fopt == quad(xopt)[1]
        end

        # One extra plain call at xopt with ForwardDiff; none with finite
        # differences, whose last base point is xopt.
        for deriv in (:forwarddiff, :finitediff)
            plain = Vector{Vector{Float64}}()
            g = function (x)
                eltype(x) === Float64 && push!(plain, copy(x))
                return quad(x)
            end
            xopt, fopt, info = solve_snopt(g, ARGS0...; deriv=deriv)
            @test info === :Solve_Succeeded
            @test fopt == quad(xopt)[1]
            @test count(==(xopt), plain) == 1
            if deriv === :forwarddiff
                @test plain == [xopt]
            end
        end

        # A DomainError at xopt in that last call gives NaN, not an error.
        f_dual_only = function (x)
            eltype(x) === Float64 && throw(DomainError(x, "plain call"))
            return quad(x)
        end
        _, fopt, info = solve_snopt(f_dual_only, ARGS0...)
        @test info === :Solve_Succeeded
        @test isnan(fopt)
    end

    @testset "exceptions in fitness" begin
        xopt, fopt, info = @test_logs (:warn, r"undefined") match_mode = :any solve_snopt(
            x -> throw(DomainError(x)), ARGS0...,
        )
        @test info === :User_Supplied_Function_Undefined
        @test isnan(fopt)
        @test xopt == ARGS0[1]

        # A transient DomainError, also from a task spawned in fitness, makes
        # SNOPT shorten the step.
        throwers = (
            x -> throw(DomainError(x)),
            x -> fetch(Threads.@spawn throw(DomainError(1))),
            x -> (@sync Threads.@spawn throw(DomainError(1))),
        )
        for thrower in throwers, deriv in (:forwarddiff, :finitediff)
            ncalls = Ref(0)
            f = function (x)
                ncalls[] += 1
                ncalls[] == 5 && thrower(x)
                return quad(x)
            end
            xopt, _, info = solve_snopt(f, ARGS0...; deriv=deriv)
            @test info === :Solve_Succeeded
            @test xopt ≈ [1.0, 2.0] atol = 1e-6
        end

        # Other errors stop SNOPT at once and are rethrown.
        @test_throws TaskFailedException solve_snopt(
            x -> fetch(Threads.@spawn error("boom")), ARGS0...,
        )
        for err in (ErrorException("boom"), InterruptException(), ArgumentError("bad"))
            ncalls = Ref(0)
            nafter = Ref(0)
            f = function (x)
                ncalls[] += 1
                ncalls[] > 5 && (nafter[] += 1)
                ncalls[] == 5 && throw(err)
                return quad(x)
            end
            @test_throws typeof(err) solve_snopt(f, ARGS0...; deriv=:finitediff)
            @test nafter[] == 0
        end

        outer = x -> (solve_snopt(quad, ARGS0...); quad(x))
        err = try
            solve_snopt(outer, ARGS0...)
            nothing
        catch e
            e
        end
        @test err isa ErrorException && occursin("nested", err.msg)

        _, _, info = solve_snopt(quad, ARGS0...)
        @test info === :Solve_Succeeded
    end

    @testset "output files" begin
        mktempdir() do dir
            cd(dir) do
                for _ in 1:2  # the files (and Fortran units) are released after a solve
                    _, _, info = solve_snopt(quad, ARGS0...; Print_file="p.out", Summary_file="s.out")
                    @test info === :Solve_Succeeded
                    @test occursin("MOCK SNOPT print file", read("p.out", String))
                    @test occursin("MOCK SNOPT finished: INFO 1", read("p.out", String))
                    @test occursin("MOCK SNOPT summary", read("s.out", String))
                end

                # Only the exact name "screen" is the console; others are files.
                # (`readdir`, not `isfile`: on a case-insensitive volume
                # `isfile("screen")` is true once "Screen" exists.)
                for name in ("stdout", "-", "Screen")
                    _, _, info = solve_snopt(quad, ARGS0...; Summary_file=name)
                    @test info === :Solve_Succeeded
                    @test name in readdir()
                end
                _, _, info = solve_snopt(quad, ARGS0...; Summary_file="screen")
                @test info === :Solve_Succeeded
                @test !("screen" in readdir())

                @test_throws ArgumentError solve_snopt(
                    quad, ARGS0...; Print_file="p.out", Summary_file="./p.out",
                )
                # A failing check on one file creates neither.
                @test_throws ArgumentError solve_snopt(
                    quad, ARGS0...; Print_file="new.out", Summary_file="nodir/s.out",
                )
                @test !ispath("new.out")

                if !Sys.iswindows() && ccall(:geteuid, Cuint, ()) != 0
                    mkdir("ro")
                    write("readonly.out", "keep")
                    chmod("ro", 0o500)
                    chmod("readonly.out", 0o400)
                    try
                        @test_throws ArgumentError solve_snopt(quad, ARGS0...; Print_file="ro/p.out")
                        @test_throws ArgumentError solve_snopt(quad, ARGS0...; Summary_file="readonly.out")
                        @test read("readonly.out", String) == "keep"
                    finally
                        chmod("ro", 0o700)
                        chmod("readonly.out", 0o600)
                    end
                end

                # Names that differ only in case: one file only on a
                # case-insensitive volume.
                insensitive = NLPInterface._snopt_dir_is_case_insensitive(pwd())
                touch("Case.out")
                @test isfile("case.out") == insensitive
                rm("Case.out")
                if insensitive
                    @test_throws ArgumentError solve_snopt(
                        quad, ARGS0...; Print_file="c.out", Summary_file="C.out",
                    )
                else
                    _, _, info = solve_snopt(quad, ARGS0...; Print_file="c.out", Summary_file="C.out")
                    @test info === :Solve_Succeeded
                    @test isfile("c.out") && isfile("C.out")
                end
                @test !any(startswith("NLPInterfaceCaseProbe"), readdir())
            end
        end
    end

    @testset "waiting for another solve warns" begin
        old = NLPInterface.SNOPT_LOCK_WAIT_WARN_SECONDS[]
        NLPInterface.SNOPT_LOCK_WAIT_WARN_SECONDS[] = 0.2
        logger = Test.TestLogger()
        try
            lock(NLPInterface.SNOPT_LOCK)
            task = try
                Base.CoreLogging.with_logger(logger) do
                    Threads.@spawn solve_snopt(quad, ARGS0...)
                end
            finally
                sleep(1.5)
                unlock(NLPInterface.SNOPT_LOCK)
            end
            _, _, info = fetch(task)
            @test info === :Solve_Succeeded
            @test any(l -> l.level == Base.CoreLogging.Warn && occursin("waited", string(l.message)), logger.logs)

            # No warning when the lock is free.
            quiet = Test.TestLogger()
            Base.CoreLogging.with_logger(quiet) do
                solve_snopt(quad, ARGS0...)
            end
            sleep(0.5)
            @test !any(l -> occursin("waited", string(l.message)), quiet.logs)
        finally
            NLPInterface.SNOPT_LOCK_WAIT_WARN_SECONDS[] = old
        end
    end
end
