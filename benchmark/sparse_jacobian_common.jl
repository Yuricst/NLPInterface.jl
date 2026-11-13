# Settings shared by sparse_jacobian.jl and sparse_jacobian_rss.jl.

const NS_DENSE = [10, 20, 40, 80, 160]              # nodes; n = 7N variables
const NS_SPARSE = [10, 20, 40, 80, 160, 320, 640]
const DERIVS = [:forwarddiff, :finitediff]
const MODES = ["dense", "pattern", "auto"]

# Same tolerances as benchmark_problems.jl; every other option is the default.
const TOL = 1e-6

function run_solver(solver, deriv, mode, P, fitness)
    sp = mode == "dense" ? nothing : mode == "pattern" ? P.pattern : :auto
    args = (fitness, P.x0, P.x_lb, P.x_ub, P.n_ceq, P.n_cineq)
    if solver == "snopt"
        return solve_snopt(args...; deriv=deriv, jac_sparsity=sp,
            Major_feasibility_tolerance=TOL, Major_optimality_tolerance=TOL)
    elseif solver == "ipopt"
        return solve_ipopt(args...; deriv=deriv, jac_sparsity=sp, print_level=0, sb="yes",
            tol=TOL, constr_viol_tol=TOL, dual_inf_tol=TOL, compl_inf_tol=TOL)
    end
    error("unknown solver $solver")
end
