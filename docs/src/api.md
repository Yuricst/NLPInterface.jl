```@meta
CurrentModule = NLPInterface
```

# API

Both solvers minimize `fitness(x) -> (obj, ceq, cineq)` subject to `ceq == 0`, `cineq <= 0`, and box bounds. They return `(xopt, fopt, info)`.

`deriv` is `:forwarddiff` or `:finitediff`. Every other keyword is a solver option. Ipopt keywords are stringified as written (`tol` → `tol`). SNOPT keywords turn underscores into spaces (`Major_print_level` → `Major print level`).

## Ipopt

```@docs
solve_ipopt
```

## SNOPT

```@docs
solve_snopt
has_snopt
find_snopt_lib
```

## Residuals

[`solve_ipopt`](@ref) and [`solve_snopt`](@ref) stack the fitness outputs into one residual before differentiating. The helper below is the same stacking used inside the solvers.

```@docs
packed_residual
```
