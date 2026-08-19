# NLPInterface.jl

Nonlinear programming (NLP) interface for gradient-based solvers [SNOPT](https://ccom.ucsd.edu/~optimizers/docs/snopt/introduction.html) and [Ipopt](https://github.com/coin-or/ipopt) in Julia.

Distinction from other similar libraries:
- Unlike `JuMP`, `NLPInterface` is *not* a modeling language; instead, it expects a user-defined differentiable function that evaluates the objective and constraint(s).
- It is similar to [`SNOW.jl`](https://github.com/byuflowlab/SNOW.jl)/[`Snopt.jl`](https://github.com/byuflowlab/Snopt.jl); a distinction is that `Snopt.jl` `ccall`s SNOPT's Fortran entry points (`snopta_`, `sninit_`, ...) and builds SNOPT from Fortran sources, whereas `NLPInterface` `ccall`s the C wrappers (`f_snopta`, `f_snset`, ...) in a prebuilt `libsnopt7_cpp` / `libsnopt7_c` library (which still contains the Fortran solver). This choice lets users load the official prebuilt SNOPT libraries instead of compiling Fortran themselves, and Julia's C ABI is more portable than Fortran name mangling and string arguments.
- It provides a common, easy to use interface wrapping [`Ipopt.jl`](https://juliapackages.com/p/ipopt)/the SNOPT C library.


## Overview

Given a fitness function of the form

```julia
"""Evaluate objective and constraints

Args:
- `x::Array`: decision variables

Returns:
- `objval::Real`: objective function value
- `ceq::Array`: array of equality constraints evaluated at `x`, if any
- `cineq::Array`: array of inequality constraints evaluated at `x`, if any
"""
function fitness(x)
    return obj, ceq, cineq
end
```

The interface implements the interface

```julia
"""Solve minimization problem with SNOPT

Args:
- `fitness::Function`: fitness function
- `x0::Array`: initial guess
- `x_lb::Array`: lower bound on decision variables
- `x_ub::Array`: upper bound on decision variables
- `n_ceq::Int`: number of equality constraints
- `n_cineq::Int`: number of inequality constraints
"""
function solve_snopt(
    fitness::Function,
    x0::Array,
    x_lb::Array,
    x_ub::Array,
    n_ceq::Int,
    n_cineq::Int;
    kwargs...
)
```

and

```julia
"""Solve minimization problem with IPOPT

Args:
- `fitness::Function`: fitness function
- `x0::Array`: initial guess
- `x_lb::Array`: lower bound on decision variables
- `x_ub::Array`: upper bound on decision variables
- `n_ceq::Int`: number of equality constraints
- `n_cineq::Int`: number of inequality constraints
"""
function solve_ipopt(
    fitness::Function,
    x0::Array,
    x_lb::Array,
    x_ub::Array,
    n_ceq::Int,
    n_cineq::Int;
    kwargs...
)
```

## Installation

To install, 

```julia
pkg> add /path/to/NLPInterface.jl
```

or 

```
pkg> add https://github.com/Yuricst/NLPInterface.jl.git
```

### SNOPT setup

[SNOPT](https://ccom.ucsd.edu/~optimizers/docs/snopt/introduction.html) is proprietary, so you need a license as well as SNOPT7 build whose architecture matches Julia (`Sys.ARCH`).

#### License

Point `SNOPT_LICENSE` at your license file (SNOPT reads this itself):

```bash
export SNOPT_LICENSE="$HOME/licenses/snopt7.lic"
```

On Windows: `set SNOPT_LICENSE=C:\Users\me\snopt7.lic`. See [License Setup](https://ccom.ucsd.edu/~optimizers/docs/snopt/linking.html).

#### Shared library

The package looks for a library that exports the C `f_*` wrappers (`f_snopta`, `f_snset`, ...), in this order:

1. `SNOPT_LIB` — full path to the dylib/so/dll
2. `SNOPTDIR` — directory containing the library
3. `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS)
4. the system loader

Preferred names: `libsnopt7_cpp`, then `libsnopt7_c`, then `libsnopt7` (with the usual `.dylib` / `.so` / `.dll` suffix).

```bash
# recommended: exact file
export SNOPT_LIB="$HOME/opt/bin/libsnopt7_c.dylib"

# or a directory
export SNOPTDIR="$HOME/opt/bin"
```

Put these `export`s in `~/.zshrc` or `~/.bashrc`, then restart Julia. Check with:

```julia
using NLPInterface
has_snopt()          # true if a usable library was loaded
NLPInterface.libsnopt7
```

If `has_snopt()` is `false`, `solve_snopt` throws. Common failures: Julia is ARM64 but the library is x86_64 (or the reverse), or a C wrapper cannot find a sibling `libsnopt7` in the same directory. Keep the Fortran/C libraries together; the package will try to fix macOS install-name mismatches automatically.

[Linking to the SNOPT library](https://ccom.ucsd.edu/~optimizers/docs/snopt/linking.html) has more detail on `libsnopt7` vs `libsnopt7_cpp`.


## Quick start

Here's a minimal example:

```julia
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
```
