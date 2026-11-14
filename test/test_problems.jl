# End-to-end solves of the problems in problems.jl with both solvers.
# Relative tolerance on fopt: with constraint violation ~1e-8 and large multipliers,
# fopt on problems with |f| in the hundreds or thousands moves by ~1e-5.

# Forward differences (src/residuals.jl) give gradient errors ~sqrt(eps)*|f|. On the
# harder problems Ipopt cannot reach its default tol=1e-8, and whether it stops as
# "acceptable" depends on rounding (BLAS kernel, Ipopt/MUMPS build, OS): e.g. g06
# hits Maximum_Iterations_Exceeded with the minimum dependency versions on the Zen
# OpenBLAS kernel. Only these problems are solved end to end with :finitediff; all
# of them are solved with :forwarddiff.
const FINITEDIFF_PROBLEMS = Set(["rosenbrock", "mishra_bird", "townsend", "crawling", "hs078"])

# L-BFGS memory for Ipopt, as in benchmark/benchmark_problems.jl. Ipopt's default
# of 6 pairs is too little for rosenbrock_10d: its optimum lies exactly on the
# constraint boundary (sum(x.^2) = 10) with a zero multiplier, and the Ipopt
# 3.14.4 build of the minimum dependency versions then stops with
# Error_In_Step_Computation (x within 5e-5 of the optimum). With 20 pairs both
# old and current Ipopt end with Solve_Succeeded.
ipopt_lbfgs_history(n) = min(max(20, n), 75)

const ACCEPTED_STATUSES = (:Solve_Succeeded, :Solved_To_Acceptable_Level)

# The job logs of GitHub Actions need authentication; annotations do not. When a
# solve fails any of the checks below, print a workflow command so that the
# problem, solver, status and errors show up as an annotation of the CI job.
function annotate_problem_failure(solver, name, deriv, p, xopt, fopt, info)
    _, ceq, cineq = p.fitness(xopt)
    cviol = maximum(vcat(abs.(ceq), cineq, -Inf))
    ok = info in ACCEPTED_STATUSES &&
         isapprox(xopt, p.xopt; atol=1e-4) &&
         isapprox(fopt, p.fopt; atol=1e-5, rtol=1e-7) &&
         all(abs.(ceq) .<= 1e-6) && all(cineq .<= 1e-6)
    ok && return
    println(
        "::warning title=$name $solver $deriv::status=$info " *
        "xerr=$(maximum(abs, xopt - p.xopt)) ferr=$(abs(fopt - p.fopt)) cviol=$cviol",
    )
    return
end

@testset "$name" for name in sort!(collect(keys(problems)))
    p = problems[name]
    derivs = name in FINITEDIFF_PROBLEMS ? (:forwarddiff, :finitediff) : (:forwarddiff,)

    @testset "solve_ipopt deriv=$deriv" for deriv in derivs
        ipopt_tols = if deriv === :forwarddiff
            (tol=1e-6, constr_viol_tol=1e-8)
        else
            NamedTuple()
        end
        ipopt_tols = (; ipopt_tols..., limited_memory_max_history=ipopt_lbfgs_history(length(p.x0)))
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
        annotate_problem_failure("ipopt", name, deriv, p, xopt, fopt, info)
        @test p.x0 == x0_before
        @test info in ACCEPTED_STATUSES
        @test xopt ≈ p.xopt atol = 1e-4
        @test fopt ≈ p.fopt atol = 1e-5 rtol = 1e-7
        _, ceq, cineq = p.fitness(xopt)
        @test all(abs.(ceq) .<= 1e-6)
        @test all(cineq .<= 1e-6)
    end

    if has_snopt()
        @testset "solve_snopt deriv=$deriv" for deriv in derivs
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
            annotate_problem_failure("snopt", name, deriv, p, xopt, fopt, info)
            @test p.x0 == x0_before
            @test info in ACCEPTED_STATUSES
            @test xopt ≈ p.xopt atol = 1e-4
            @test fopt ≈ p.fopt atol = 1e-5 rtol = 1e-7
            _, ceq, cineq = p.fitness(xopt)
            @test all(abs.(ceq) .<= 1e-6)
            @test all(cineq .<= 1e-6)
        end
    end
end
