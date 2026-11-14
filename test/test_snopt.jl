# solve_snopt and its helpers (src/snopt.jl). The helper tests are pure Julia
# and run without a SNOPT library; the rest need one.

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
            noexec = joinpath(dir, "noexec")
            mkpath(joinpath(noexec, "x"))
            chmod(noexec, 0o600)  # no search permission: stat fails with EACCES
            try
                @test_throws ArgumentError check(joinpath(noexec, "p.out"), "")
                @test_throws ArgumentError check(joinpath(noexec, "x", "p.out"), "")
            finally
                chmod(noexec, 0o700)
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

if has_snopt()
    @testset "SNOPT default output after an aborted solve" begin
        f = x -> (sum(abs2, x), [x[1] + x[2] - 1.0], [x[1] - 2.0, x[2] - 2.0])
        lb, ub = [-5.0, -5.0], [5.0, 5.0]
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

    @testset "fitness evaluation counts: SNOPT" begin
        p = problems["rosenbrock"]
        # SNOPT with ForwardDiff: no plain call at points that get a Jacobian,
        # except the last one, which evaluates `fopt` at `xopt`.
        g, plain, dual = counted(p.fitness)
        xopt, fopt, info = solve_snopt(g, p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq)
        @test info === :Solve_Succeeded
        @test plain[end] == xopt
        @test fopt == p.fitness(xopt)[1]
        @test isempty(intersect(plain[1:(end - 1)], dual))
    end

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

        # From a task that fitness waits for: used to deadlock on the lock. The
        # waiting solve sees that the lock holder waits for it, and throws.
        outer_spawn = function (x)
            fetch(Threads.@spawn solve_snopt(inner, inner_args...))
            return inner(x)
        end
        err = try
            solve_snopt(outer_spawn, inner_args...)
            nothing
        catch e
            e
        end
        @test err isa TaskFailedException
        @test occursin("nested", sprint(showerror, err))
        # ... also through a chain of waits, and from @sync.
        outer_chain = function (x)
            fetch(Threads.@spawn fetch(Threads.@spawn solve_snopt(inner, inner_args...)))
            return inner(x)
        end
        @test_throws TaskFailedException solve_snopt(outer_chain, inner_args...)
        outer_sync = function (x)
            @sync Threads.@spawn solve_snopt(inner, inner_args...)
            return inner(x)
        end
        @test_throws CompositeException solve_snopt(outer_sync, inner_args...)

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

        # Neither leaves global state behind.
        xopt, fopt, info = solve_snopt(inner, inner_args...)
        @test info === :Solve_Succeeded
        @test xopt ≈ [2.0, 3.0] atol = 1e-6
        @test fopt ≈ 0.0 atol = 1e-10
    end
end
