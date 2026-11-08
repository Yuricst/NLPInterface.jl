"""Argument checks shared by `solve_ipopt` and `solve_snopt`."""

"""
    check_solver_inputs(x0, x_lb, x_ub, n_ceq, n_cineq, deriv)

Throw an `ArgumentError` unless the problem data are usable by both solvers:

- `x0` is non-empty, and `x0`, `x_lb`, `x_ub` have the same length;
- every entry of `x0` is a real number that is finite as a `Float64`;
- every bound is a real number;
- no bound is `NaN`, no lower bound is `+Inf`, no upper bound is `-Inf`, and
  `x_lb[i] <= x_ub[i]` for every `i` (equal bounds fix a variable);
- `n_ceq` and `n_cineq` are non-negative;
- `deriv` is `:forwarddiff` or `:finitediff`.

`x0` may lie outside the bounds; both solvers move it inside.
"""
function check_solver_inputs(x0, x_lb, x_ub, n_ceq, n_cineq, deriv)
    n = length(x0)
    n == 0 && throw(ArgumentError("x0 must be non-empty"))
    if !(n == length(x_lb) == length(x_ub))
        throw(ArgumentError("x0, x_lb, and x_ub must have the same length"))
    end
    if n_ceq < 0 || n_cineq < 0
        throw(ArgumentError("n_ceq and n_cineq must be non-negative"))
    end
    if deriv !== :forwarddiff && deriv !== :finitediff
        throw(ArgumentError("Unknown deriv=$deriv; use :forwarddiff or :finitediff"))
    end
    for i in eachindex(x0)
        v = x0[i]
        v isa Real || throw(ArgumentError("x0[$i] = $(repr(v)) is not a real number"))
        # Checked after conversion: a BigFloat beyond Float64's range becomes Inf.
        isfinite(Float64(v)) || throw(ArgumentError("x0[$i] = $v is not finite as a Float64"))
    end
    for (i, lb, ub) in zip(eachindex(x_lb), x_lb, x_ub)
        lb isa Real || throw(ArgumentError("x_lb[$i] = $(repr(lb)) is not a real number"))
        ub isa Real || throw(ArgumentError("x_ub[$i] = $(repr(ub)) is not a real number"))
        isnan(lb) && throw(ArgumentError("x_lb[$i] is NaN"))
        isnan(ub) && throw(ArgumentError("x_ub[$i] is NaN"))
        lb == Inf && throw(ArgumentError("x_lb[$i] = Inf; a lower bound must be below +Inf"))
        ub == -Inf && throw(ArgumentError("x_ub[$i] = -Inf; an upper bound must be above -Inf"))
        lb <= ub || throw(ArgumentError("x_lb[$i] = $lb is greater than x_ub[$i] = $ub"))
    end
    return nothing
end
