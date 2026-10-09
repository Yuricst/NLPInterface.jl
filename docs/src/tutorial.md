```@meta
CurrentModule = NLPInterface
```

# Tutorial

This tutorial solves a small nonlinear program with one equality constraint and one inequality constraint, then shows how to switch solvers, change derivative mode, and pass solver options.

```math
\begin{aligned}
\min_{x_1, x_2} \quad & x_1 + x_2 \\
\mathrm{s.t.} \quad &
x_2 - x_1^4 - 2 x_1^3 + 1.2 x_1^2 + 2 x_1 = 0 \\
& -x_2 - \tfrac{4}{3} x_1 - \tfrac{2}{3} \le 0 \\
& -2 \le x_1 \le 2, \quad -2 \le x_2 \le 2
\end{aligned}
```

The minimizer is near ``x^\star \approx (0.529,\ -1.019)`` with ``f(x^\star) \approx -0.490``.

```julia
using NLPInterface
```

## Fitness function

Both solvers call one function. It receives the decision vector and returns a 3-tuple:

- the objective value (a scalar),
- equality constraints, stacked so that a feasible point has `ceq == 0`,
- inequality constraints, stacked so that a feasible point has `cineq <= 0`.

```julia
function fitness(x)
    obj = x[1] + x[2]
    ceq = [x[2] - x[1]^4 - 2x[1]^3 + 1.2x[1]^2 + 2x[1]]
    cineq = [-x[2] - (4/3)*x[1] - 2/3]
    return obj, ceq, cineq
end
```

The lengths of `ceq` and `cineq` must match the counts passed to the solver on every evaluation, including evaluations with `ForwardDiff.Dual` elements. If a constraint block is absent, return an empty vector and pass `0` for that count.

The body has to be generic in the element type of `x`. ForwardDiff will call `fitness` with dual numbers. Code that only makes sense for plain floats, such as logging a path for a plot, should be guarded:

```julia
function fitness_logged(x)
    if eltype(x) <: Float64
        println("x = ", x)
    end
    return fitness(x)
end
```

## Bounds and constraint counts

```julia
x0 = [1.5, 1.5]
x_lb = [-2.0, -2.0]
x_ub = [2.0, 2.0]
n_ceq = 1
n_cineq = 1
```

`x0`, `x_lb`, and `x_ub` must have the same length. Non-finite entries in the bounds are treated as unbounded: Ipopt accepts `±Inf` directly, and SNOPT maps them to its own infinite bound (`±1e20`).

## Solve with Ipopt

```julia
xopt, fopt, info = solve_ipopt(
    fitness,
    x0,
    x_lb,
    x_ub,
    n_ceq,
    n_cineq;
    print_level = 3,
    tol = 1e-6,
    constr_viol_tol = 1e-8,
)
```

The return values are

- `xopt`, a copy of the solution vector,
- `fopt`, the objective at that point,
- `info`, a symbol such as `:Solve_Succeeded` or `:Solved_To_Acceptable_Level`.

Keyword arguments other than `deriv` are Ipopt options. The option name is the keyword converted to a string, so `print_level` becomes Ipopt's `print_level`. Integer, floating-point, string, symbol, and boolean values are accepted. The Hessian is always the limited-memory (L-BFGS) approximation; a user Hessian is not passed to Ipopt.

## Solve with SNOPT

SNOPT uses the same fitness function, bounds, and counts. The call requires a loadable SNOPT library; see [SNOPT setup](@ref) for `SNOPT_LIB` and `SNOPT_LICENSE`.

```julia
if has_snopt()
    xopt, fopt, info = solve_snopt(
        fitness,
        x0,
        x_lb,
        x_ub,
        n_ceq,
        n_cineq;
        Major_print_level = 1,
        Major_feasibility_tolerance = 1e-8,
        Major_optimality_tolerance = 1e-6,
    )
end
```

SNOPT option names use spaces. Underscores in the keyword are replaced with spaces, so `Major_print_level` is sent as `Major print level`. A nonzero `Major_print_level` with no `Summary_file` prints the major-iteration summary to the console. `Print_file` and `Summary_file` choose the print and summary output paths instead of being forwarded as ordinary options.

## Derivatives

`deriv` selects how the Jacobian of `[obj; ceq; cineq]` is built.

- `:forwarddiff` (default) uses ForwardDiff. This is the right choice when `fitness` is written generically.
- `:finitediff` uses finite differences and only evaluates `fitness` on `Float64` vectors.

```julia
xopt, fopt, info = solve_ipopt(
    fitness, x0, x_lb, x_ub, n_ceq, n_cineq;
    deriv = :finitediff,
    print_level = 0,
)
```

## Checking the result

```julia
obj, ceq, cineq = fitness(xopt)
```

A successful solve has `info` equal to `:Solve_Succeeded` or `:Solved_To_Acceptable_Level`, `ceq` near zero, and `cineq` entrywise at most zero. Other symbols report failure modes such as `:Infeasible_Problem_Detected` or `:Maximum_Iterations_Exceeded`. The full code maps are `NLPInterface.IPOPT_RETURN_STATUS` and `NLPInterface.SNOPT_RETURN_STATUS`.
