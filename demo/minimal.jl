"""Minimal example of using NLPInterface"""

using NLPInterface

x0 = [1.5, 1.5]
x_lb = [-2.0, -2.0]
x_ub = [2.0, 2.0]
n_ceq = 1
n_cineq = 1
function fitness(x)
    return (
        x[1] + x[2],
        [x[2] - x[1]^4 - 2x[1]^3 + 1.2x[1]^2 + 2x[1]],
        [-x[2] - (4/3)*x[1] - 2/3],
    )
end

# Solve with IPOPT
xopt, fopt, info = NLPInterface.solve_ipopt(
    fitness,
    x0,
    x_lb,
    x_ub,
    n_ceq,
    n_cineq;
    print_level=3,
    tol=1e-6,
    constr_viol_tol=1e-8
)

# Solve with SNOPT
xopt, fopt, info = NLPInterface.solve_snopt(
    fitness,
    x0,
    x_lb,
    x_ub,
    n_ceq,
    n_cineq;
    Major_print_level=1,
    Major_feasibility_tolerance=1e-8,
    Major_optimality_tolerance=1e-6,
)