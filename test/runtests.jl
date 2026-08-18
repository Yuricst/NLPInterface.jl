using NLPInterface
using Test

include("problems.jl")

@testset "NLPInterface" begin
    p = problems["rosenbrock"]

    @testset "solve_ipopt deriv=$deriv" for deriv in (:forwarddiff, :finitediff)
        ipopt_tols = if deriv === :forwarddiff
            (tol=1e-6, constr_viol_tol=1e-8)
        else
            NamedTuple()
        end
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
        @test info == :Solve_Succeeded
        @test xopt ≈ [1.0, 1.0] atol = 1e-4
        @test fopt ≈ 0.0 atol = 1e-6
        _, _, cineq = p.fitness(xopt)
        @test all(cineq .<= 1e-6)
    end

    if has_snopt()
        @testset "solve_snopt deriv=$deriv" for deriv in (:forwarddiff, :finitediff)
            snopt_tols = if deriv === :forwarddiff
                (Major_feasibility_tolerance=1e-6, Major_optimality_tolerance=1e-6)
            else
                NamedTuple()
            end
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
            @test info == :Solve_Succeeded
            @test xopt ≈ [1.0, 1.0] atol = 1e-4
            @test fopt ≈ 0.0 atol = 1e-6
            _, _, cineq = p.fitness(xopt)
            @test all(cineq .<= 1e-6)
        end
    else
        @info "Skipping SNOPT tests; set SNOPT_LIB or SNOPTDIR to the libsnopt7_cpp library"
        @test_skip "SNOPT library not found"
    end
end
