```@meta
CurrentModule = NLPInterface
```

# `NLPInterface.jl`

Nonlinear programming interface for the gradient-based solvers [Ipopt](https://github.com/coin-or/Ipopt) and [SNOPT](https://ccom.ucsd.edu/~optimizers/docs/snopt/introduction.html).

`NLPInterface` is not a modeling language. You supply one differentiable fitness function that returns the objective and the constraints, together with bounds on the decision variables. The same function is passed to either solver:

```math
\begin{aligned}
\min_{x} \quad & f(x) \\
\mathrm{s.t.} \quad & c_{\mathrm{eq}}(x) = 0 \\
& c_{\mathrm{ineq}}(x) \le 0 \\
& x_{\mathrm{lb}} \le x \le x_{\mathrm{ub}}
\end{aligned}
```

Derivatives of the stacked residual ``[f; c_{\mathrm{eq}}; c_{\mathrm{ineq}}]`` are taken with ForwardDiff by default, or with finite differences.

## Installation

```julia
pkg> add https://github.com/Yuricst/NLPInterface.jl.git
```

Ipopt is installed automatically through [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl). SNOPT is proprietary: a license and a prebuilt shared library are required before [`solve_snopt`](@ref) can run. See [SNOPT setup](@ref) below.

## Illustrative example

```julia
using NLPInterface

x0 = [1.5, 1.5]
x_lb = [-2.0, -2.0]
x_ub = [2.0, 2.0]

function fitness(x)
    return (
        x[1] + x[2],
        [x[2] - x[1]^4 - 2x[1]^3 + 1.2x[1]^2 + 2x[1]],
        [-x[2] - (4/3)*x[1] - 2/3],
    )
end

xopt, fopt, info = solve_ipopt(fitness, x0, x_lb, x_ub, 1, 1; print_level=0)
```

`fitness(x)` returns `(obj, ceq, cineq)`. The two integers are the number of equality and inequality constraints. The same call with [`solve_snopt`](@ref) uses SNOPT when a library is available. The [Tutorial](@ref) walks through this problem, solver options, and derivatives.

## SNOPT setup

Point `SNOPT_LICENSE` at the license file. SNOPT reads that variable itself.

```bash
export SNOPT_LICENSE="$HOME/licenses/snopt7.lic"
```

The package looks for a library that exports the C wrappers (`f_snopta`, `f_snset`, ...), in this order:

1. `SNOPT_LIB`, the full path to the shared library
2. `SNOPTDIR`, a directory containing the library
3. `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS)
4. the system loader

Preferred names are `libsnopt7_cpp`, then `libsnopt7_c`, then `libsnopt7`.

```bash
export SNOPT_LIB="$HOME/opt/bin/libsnopt7_c.dylib"
```

Restart Julia after exporting these variables, then check

```julia
using NLPInterface
has_snopt()
```

[`has_snopt`](@ref) is `false` when no usable library was loaded, and [`solve_snopt`](@ref) throws in that case. The Julia architecture must match the library (`Sys.ARCH`). Keep the C wrapper and its companion `libsnopt7` in the same directory.
