# NLPInterface.jl

Nonlinear programming (NLP) interface for gradient-based solvers SNOPT and IPOPT.

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


## SNOPT setup

SNOPT is proprietary. You need a licensed SNOPT 7 build whose architecture matches Julia (`Sys.ARCH`). Ipopt does not need this.

### License

Point `SNOPT_LICENSE` at your license file (SNOPT reads this itself):

```bash
export SNOPT_LICENSE="$HOME/licenses/snopt7.lic"
```

On Windows: `set SNOPT_LICENSE=C:\Users\me\snopt7.lic`. See [License Setup](https://ccom.ucsd.edu/~optimizers/docs/snopt/linking.html).

### Shared library

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
