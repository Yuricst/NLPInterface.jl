# Regression tests for review items #1, #2, #4, #7, #8.

# Smooth problem whose fitness is undefined for x > 1; optimum on the upper bound.
const EDGE = (fitness=x -> (-x[1] + sqrt(1 - x[1])^3, nothing, nothing),
              args=([0.5], [0.0], [1.0], 0, 0))

@testset "fitness is only evaluated inside the bounds (#1, #2)" begin
    for deriv in (:forwarddiff, :finitediff)
        seen = Float64[]
        f = x -> (push!(seen, ForwardDiff.value(x[1])); EDGE.fitness(x))
        xopt, fopt, info = solve_ipopt(f, EDGE.args...; deriv=deriv, print_level=0)
        @test info === :Solve_Succeeded
        @test xopt[1] ≈ 1.0 atol = 1e-6
        @test maximum(seen) <= 1.0
    end
    # an explicit bound_relax_factor is still honoured
    @test_throws DomainError solve_ipopt(EDGE.fitness, EDGE.args...; print_level=0, bound_relax_factor=1e-8)

    # one-sided steps: backward at the upper bound, never outside the box
    @test NLPInterface.fd_step(1.0, 0.0, 1.0) < 0
    @test NLPInterface.fd_step(0.0, 0.0, 1.0) > 0
    @test NLPInterface.fd_step(0.5, -Inf, Inf) > 0
    seen = Vector{Float64}[]
    ev = NLPInterface.ResidualEvaluator(x -> (push!(seen, copy(x)); (sum(abs2, x), nothing, nothing)),
                                        2, 0, 0, :finitediff; x_lb=[0.0, 0.0], x_ub=[1.0, 1.0])
    J = zeros(1, 2)
    NLPInterface.residual_value_and_jacobian!(ev, zeros(1), J, [1.0, 0.0], false)
    @test all(x -> all(0 .<= x .<= 1), seen)
    @test J ≈ [2.0 0.0] atol = 1e-6
end

struct ReviewQuad end
(::ReviewQuad)(x) = ((x[1] - 1)^2 + (x[2] - 2)^2, nothing, nothing)

@testset "callable objects as fitness (#7)" begin
    xopt, _, info = solve_ipopt(ReviewQuad(), [0.0, 0.0], [-5.0, -5.0], [5.0, 5.0], 0, 0; print_level=0)
    @test info === :Solve_Succeeded
    @test xopt ≈ [1.0, 2.0] atol = 1e-6
end

@testset "fitness that modifies x cannot corrupt the solver (#8)" begin
    g = function (x)
        v = (x[1] - 1)^2 + (x[2] - 2)^2
        x .= 0   # misbehaving user code
        return (v, nothing, nothing)
    end
    for deriv in (:forwarddiff, :finitediff)
        xopt, _, info = solve_ipopt(g, [0.0, 0.0], [-5.0, -5.0], [5.0, 5.0], 0, 0; deriv=deriv, print_level=0)
        @test info === :Solve_Succeeded
        @test xopt ≈ [1.0, 2.0] atol = 1e-5
    end
end

@testset "SNOPT keywords ignore case (#4)" begin
    @test NLPInterface.snopt_canonical_keyword("Major Print  Level") == "major print level"
end

if has_snopt()
    @testset "SNOPT: bounds, functors, mutation, keyword case" begin
        for deriv in (:forwarddiff, :finitediff)
            xopt, _, info = solve_snopt(EDGE.fitness, EDGE.args...; deriv=deriv)
            @test info === :Solve_Succeeded
            @test xopt[1] ≈ 1.0 atol = 1e-7   # finitediff used to stop at 0.999998
        end
        xopt, _, info = solve_snopt(ReviewQuad(), [0.0, 0.0], [-5.0, -5.0], [5.0, 5.0], 0, 0)
        @test info === :Solve_Succeeded
        mktempdir() do dir
            cd(dir) do
                _, _, info = solve_snopt(ReviewQuad(), [0.0, 0.0], [-5.0, -5.0], [5.0, 5.0], 0, 0;
                                         print_file="p.out", SUMMARY_FILE="s.out")
                @test info === :Solve_Succeeded
                @test filesize("p.out") > 0 && filesize("s.out") > 0
            end
        end
    end
end
