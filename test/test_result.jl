# NLPResult (src/result.jl): fields, multiplier convention, compatibility.

# Max-norm of the KKT stationarity residual ∇f + J_eqᵀλ_eq + J_ineqᵀλ_ineq - z_lb + z_ub.
function kkt_stationarity(fitness, r, n_ceq, n_cineq)
    J = ForwardDiff.jacobian(x -> NLPInterface.packed_residual(fitness, x, n_ceq, n_cineq), r.xopt)
    J_eq = J[2:(1 + n_ceq), :]
    J_in = J[(2 + n_ceq):end, :]
    return maximum(abs, J[1, :] + J_eq' * r.lambda_eq + J_in' * r.lambda_ineq - r.z_lb + r.z_ub)
end

# x1 at its upper bound (z_ub = 2), x2 at its lower bound (z_lb = 1), and the
# inequality x3 <= 0.2 active (λ = 0.6).
const KKT_QUAD = (
    fitness=x -> ((x[1] - 3)^2 + (x[2] + 1)^2 + (x[3] - 0.5)^2, nothing, [x[3] - 0.2]),
    x0=[0.0, 0.0, 0.0], x_lb=[-2.0, -0.5, -2.0], x_ub=[2.0, 2.0, 2.0], n_ceq=0, n_cineq=1,
)

result_solvers() = has_snopt() ? (:ipopt, :snopt) : (:ipopt,)
function solve_tight(solver, f, p; kw...)
    args = (f, p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq)
    if solver === :ipopt
        return solve_ipopt(args...; print_level=0, tol=1e-10, kw...)
    else
        return solve_snopt(args...; Major_optimality_tolerance=1e-10, kw...)
    end
end

@testset "NLPResult: $solver" for solver in result_solvers()
    statusmap = solver === :ipopt ? NLPInterface.IPOPT_RETURN_STATUS : NLPInterface.SNOPT_RETURN_STATUS

    # HS071: an equality, an active inequality and an active lower bound, with
    # known multipliers (Wächter & Biegler's Ipopt example, in our formulation
    # ceq = sum(x.^2) - 40, cineq = 25 - prod(x)).
    p = problems["hs071"]
    r = solve_tight(solver, p.fitness, p)
    @test r isa NLPResult
    @test r.solver === solver
    @test r.status === :Solve_Succeeded
    @test statusmap[r.raw_status] === r.status
    @test r.lambda_eq ≈ [0.161469] atol = 1e-5
    @test r.lambda_ineq ≈ [0.552294] atol = 1e-5
    @test r.z_lb ≈ [1.087871, 0, 0, 0] atol = 1e-5
    @test r.z_ub ≈ zeros(4) atol = 1e-5
    @test kkt_stationarity(p.fitness, r, p.n_ceq, p.n_cineq) <= 1e-6
    # Constraint values and objective at xopt.
    fx, ceq, cineq = p.fitness(r.xopt)
    @test r.ceq ≈ ceq rtol = 1e-12 atol = 1e-12
    @test r.cineq ≈ cineq rtol = 1e-12 atol = 1e-12
    @test r.fopt ≈ fx rtol = 1e-12
    solver === :snopt && @test r.fopt == fx   # a plain evaluation at xopt
    if solver === :ipopt
        @test r.iterations isa Int && r.iterations > 0
    else
        @test ismissing(r.iterations)
    end

    # Active upper bound, active lower bound, active inequality.
    q = KKT_QUAD
    r = solve_tight(solver, q.fitness, q)
    @test r.xopt ≈ [2.0, -0.5, 0.2] atol = 1e-6
    @test r.lambda_ineq ≈ [0.6] atol = 1e-6
    @test r.z_ub ≈ [2.0, 0, 0] atol = 1e-6
    @test r.z_lb ≈ [0, 1.0, 0] atol = 1e-6
    @test all(>=(-1e-8), r.lambda_ineq) && all(>=(-1e-8), r.z_lb) && all(>=(-1e-8), r.z_ub)
    @test kkt_stationarity(q.fitness, r, q.n_ceq, q.n_cineq) <= 1e-6
    @test isempty(r.ceq) && isempty(r.lambda_eq)

    # n_fitness counts every call of fitness.
    for deriv in (:forwarddiff, :finitediff)
        g, plain, dual = counted(p.fitness)
        r = solve_tight(solver, g, p; deriv=deriv)
        @test r.n_fitness == length(plain) + length(dual)
    end
    ncalls = Ref(0)
    r = solve_tight(solver, x -> (ncalls[] += 1; p.fitness(x)), p; jac_sparsity=:auto)
    @test r.n_fitness == ncalls[]   # includes the sparsity-detection call

    # A solve stopped early still returns complete fields.
    stopped = solver === :ipopt ? solve_ipopt(p.fitness, p.x0, p.x_lb, p.x_ub, 1, 1; print_level=0, max_iter=1) :
              solve_snopt(p.fitness, p.x0, p.x_lb, p.x_ub, 1, 1; Major_iterations_limit=1)
    @test stopped.status === :Maximum_Iterations_Exceeded
    @test statusmap[stopped.raw_status] === stopped.status
    @test length(stopped.ceq) == length(stopped.lambda_eq) == 1
    @test length(stopped.z_lb) == length(stopped.z_ub) == 4

    # Compatibility with the former (xopt, fopt, info) tuple.
    xopt, fopt, info = r
    @test xopt === r.xopt && fopt === r.fopt && info === r.status
    @test r[1] === r.xopt && r[2] === r.fopt && r[3] === r.status && r[end] === r.status
    @test_throws BoundsError r[4]
    @test length(r) == 3
    @test collect(r) == [r.xopt, r.fopt, r.status]
    @test occursin("NLPResult", sprint(show, MIME"text/plain"(), r))
end
