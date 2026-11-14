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

The lengths of `ceq` and `cineq` must match the counts passed to the solver on every evaluation, including evaluations with `ForwardDiff.Dual` elements. A mismatch throws a `DimensionMismatch` naming the block and both lengths. Earlier versions silently dropped extra constraints or copied a single value into several rows. If a constraint block is absent, return an empty vector (or `nothing`) and pass `0` for that count. A block with a count of `1` may also be returned as a scalar. Blocks may be vectors, tuples, or other arrays (read in column-major order); a block that is itself a string, character, or `missing` is rejected with an `ArgumentError`.

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

- `xopt`, the solution vector (a new array; `x0` itself is never modified),
- `fopt`, the objective at that point,
- `info`, a symbol such as `:Solve_Succeeded` or `:Solved_To_Acceptable_Level`.

Keyword arguments other than `deriv` are Ipopt options. The option name is the keyword converted to a string, so `print_level` becomes Ipopt's `print_level`. Integer, floating-point, string, symbol, and boolean values are accepted; an integer for a real-valued option (`tol = 1`) is converted to `Float64`. `bound_relax_factor` defaults to `0.0` (Ipopt's own default is `1e-8`), so `fitness` is only evaluated inside `[x_lb, x_ub]`; pass `bound_relax_factor` to change it. The Hessian is always the limited-memory (L-BFGS) approximation, since no second derivatives are passed to Ipopt; `hessian_approximation` set to anything other than `"limited-memory"` throws an `ArgumentError`.

If `fitness` throws inside `solve_ipopt`, the exception is caught before it reaches Ipopt's C++ code, Ipopt is stopped, and the exception is rethrown once Ipopt has returned. Ctrl-C during the solve throws `InterruptException`.

Both solvers check their inputs first: `x0` must be non-empty and finite, the bounds must not be `NaN`, and `x_lb[i] <= x_ub[i]` must hold (equal bounds fix a variable). Otherwise an `ArgumentError` is thrown. `fitness` must return a real objective, and each constraint block must be a vector, tuple, real number, or `nothing` holding real numbers; strings, characters, `missing`, and similar values throw an `ArgumentError`.

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

SNOPT option names use spaces. Underscores in the keyword are replaced with spaces, so `Major_print_level` is sent as `Major print level`. A nonzero `Major_print_level` with no `Summary_file` prints the major-iteration summary to the console. `Print_file` and `Summary_file` choose the print and summary output paths instead of being forwarded as ordinary options. They must be different files. Two names for the same file (for example `p.out` and `./p.out`, or a path through a symlink) throw an `ArgumentError`, because SNOPT cannot open one file on two output units. Names that differ only in case count as the same file when the directory is on a case-insensitive volume (the usual default on macOS and Windows), which is checked with a probe file. A path that cannot be opened for writing (missing directory, a directory, no write permission) also throws `ArgumentError` before SNOPT starts. Paths are used literally: `~` is not expanded, trailing blanks are dropped, and leading blanks are kept. Leave either option unset to write no file for it; `"screen"` (exactly) sends that output to the console, and any other name, including `"stdout"`, is a file. With a library that lacks `f_sninitx`, the files are opened with SNOPT's Fortran helpers (`snFileOpenWrite`); if those are missing too, file names are ignored with a warning.

If `fitness` throws a `DomainError` (for example `sqrt` of a negative number at a line-search trial point), SNOPT treats the functions as undefined there and shortens the step. The error is not rethrown. If SNOPT cannot recover, it ends with a failure status, usually `:User_Supplied_Function_Undefined`, and the last `DomainError` is shown in a warning. A `DomainError` thrown in a task spawned by `fitness` (arriving as a `TaskFailedException` or `CompositeException`) is treated the same way. Any other exception, including `InterruptException` (Ctrl-C), stops SNOPT immediately and is rethrown by `solve_snopt`.

`fopt` comes from a plain `Float64` call of `fitness` at `xopt`, so it equals `fitness(xopt)[1]` even when SNOPT's last evaluation was elsewhere (its derivative check perturbs `x`) or used ForwardDiff Dual numbers. This costs at most one extra `fitness` call per solve. If no evaluation of `fitness` requested by SNOPT completed, `fopt` is `NaN`. With ForwardDiff, the values SNOPT sees during the solve come from the Dual evaluation; for a `fitness` with adaptive inner solvers (step-size control that also acts on the Dual partials) they can differ from a plain call at the inner solver's tolerance.

Only one SNOPT solve runs at a time. Calling `solve_snopt` from inside a `fitness` passed to `solve_snopt` throws an error, because SNOPT's C interface keeps global state; on Julia 1.11 and later this also covers tasks spawned from `fitness` while the solve runs. Handing the call to a task that already existed before the solve deadlocks, and cannot be detected; a solve that has waited a minute for another one logs a warning that points this out. Use `solve_ipopt` for an inner problem instead.

## Derivatives

`deriv` selects how the Jacobian of `[obj; ceq; cineq]` is built.

- `:forwarddiff` (default) uses ForwardDiff. This is the right choice when `fitness` is written generically.
- `:finitediff` uses finite differences and only evaluates `fitness` on `Float64` vectors.

With `:forwarddiff`, the value and the Jacobian at a point come from the same Dual evaluation. With `:finitediff` (one-sided differences), a Jacobian costs `n` extra calls, reusing the value at the base point. Each variable is stepped forward unless that would cross its upper bound, in which case it is stepped backward, so the perturbed points stay inside `[x_lb, x_ub]`. Ipopt computes a Jacobian only where it asks for one, so line-search trial points cost a single `fitness` call.

Both solvers reuse the array passed to `fitness` between calls. Do not keep a reference to `x` inside `fitness`; copy it if you need it later.

```julia
xopt, fopt, info = solve_ipopt(
    fitness, x0, x_lb, x_ub, n_ceq, n_cineq;
    deriv = :finitediff,
    print_level = 0,
)
```

## Checking the result

Both solvers return an [`NLPResult`](@ref). Destructuring it, as above, gives the solution, the objective and the status. The whole result also holds the constraint values and Lagrange multipliers at the solution, and the cost of the solve:

```julia
res = solve_ipopt(fitness, x0, x_lb, x_ub, n_ceq, n_cineq; print_level = 0)
res.status        # :Solve_Succeeded
res.ceq, res.cineq                 # constraint values at res.xopt
res.lambda_eq, res.lambda_ineq     # their Lagrange multipliers
res.z_lb, res.z_ub                 # multipliers of the bounds on x
res.iterations, res.n_fitness      # Ipopt iterations, calls of fitness
```

Both solvers report the multipliers in one convention: at a KKT point ``\nabla f + J_{\mathrm{eq}}^\top \lambda_{\mathrm{eq}} + J_{\mathrm{ineq}}^\top \lambda_{\mathrm{ineq}} - z_{\mathrm{lb}} + z_{\mathrm{ub}} = 0``, with ``\lambda_{\mathrm{ineq}}``, ``z_{\mathrm{lb}}`` and ``z_{\mathrm{ub}}`` nonnegative and zero where their constraint or bound is inactive. A multiplier is the sensitivity of the optimal objective to its constraint: here `res.lambda_ineq[1] ≈ 0` because the inequality is inactive at the solution.

A successful solve has `info` (`res.status`) equal to `:Solve_Succeeded` or `:Solved_To_Acceptable_Level`, `ceq` near zero, and `cineq` entrywise at most zero. Other symbols report failure modes such as `:Infeasible_Problem_Detected` or `:Maximum_Iterations_Exceeded`. The full code maps are `NLPInterface.IPOPT_RETURN_STATUS` and `NLPInterface.SNOPT_RETURN_STATUS`.

For SNOPT, `:Solved_To_Acceptable_Level` is INFO 3: the point is feasible, but optimality is only within a factor of 100 of `Major_optimality_tolerance`. `:Feasible_Point_Found` (INFO 2) appears only with the `Feasible point` option. SNOPT INFO 5 and 6 are reported as `:Elastic_Objective_Minimized` and `:Elastic_Infeasibilities_Minimized`. The SNOPT 7.7 manual lists them as normal endings of elastic mode, in which elastic constraints are allowed to violate their bounds, and does not promise the constraints hold. Evaluate `fitness(xopt)` and check `ceq` and `cineq` before using the point. Earlier versions reported these codes (and INFO 4, which `snOptA` is not expected to return) as `:Solved_To_Acceptable_Level`.

Checking the constraints yourself is good practice for any status. Neither solver's status replaces looking at `res.ceq` and `res.cineq` (or `fitness(xopt)`).
