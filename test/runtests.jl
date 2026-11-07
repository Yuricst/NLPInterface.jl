using NLPInterface
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
