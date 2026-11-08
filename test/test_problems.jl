# End-to-end solves of the problems in problems.jl with both solvers.
# Relative tolerance on fopt: with constraint violation ~1e-8 and large multipliers,
# fopt on problems with |f| in the hundreds or thousands moves by ~1e-5.

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
        @test fopt ≈ p.fopt atol = 1e-5 rtol = 1e-7
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
            @test fopt ≈ p.fopt atol = 1e-5 rtol = 1e-7
            _, ceq, cineq = p.fitness(xopt)
            @test all(abs.(ceq) .<= 1e-6)
            @test all(cineq .<= 1e-6)
        end
    end
end
