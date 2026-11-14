```@meta
CurrentModule = NLPInterface
```

# API

Both solvers minimize `fitness(x) -> (obj, ceq, cineq)` subject to `ceq == 0`, `cineq <= 0`, and box bounds. They return an [`NLPResult`](@ref), which destructures like the tuple `(xopt, fopt, info)` of earlier versions. `xopt` is a new array; the caller's `x0` is not modified.

`ceq` and `cineq` must have exactly `n_ceq` and `n_cineq` entries at every evaluation. A mismatch throws `DimensionMismatch`.

`x0` must be non-empty and finite, and the bounds must satisfy `x_lb .<= x_ub` without `NaN` (a lower bound of `+Inf` or an upper bound of `-Inf` is rejected too); otherwise an `ArgumentError` is thrown before the solver starts.

```@docs
check_solver_inputs
```

`deriv` is `:forwarddiff` or `:finitediff`, and `jac_sparsity` selects a sparse Jacobian ([Sparse Jacobian](@ref)). Every other keyword is a solver option. Ipopt keywords are stringified as written (`tol` → `tol`). SNOPT keywords turn underscores into spaces (`Major_print_level` → `Major print level`). Like SNOPT itself, `solve_snopt` matches option names ignoring case (`print_file` is `Print_file`).

## Result

```@docs
NLPResult
```

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

## Sparsity detection

Both solvers take `jac_sparsity` to use a sparse Jacobian; see
[Sparse Jacobian](@ref). The pattern can be detected with:

```@docs
detect_jac_sparsity
```

## Residuals

[`solve_ipopt`](@ref) and [`solve_snopt`](@ref) stack the fitness outputs into one residual before differentiating. The helper below is the same stacking used inside the solvers.

```@docs
packed_residual
```

Internally, a [`ResidualEvaluator`](@ref) computes the residual and its Jacobian with as few `fitness` calls as possible.

```@docs
ResidualEvaluator
residual_value!
residual_value_and_jacobian!
```

With `jac_sparsity`, the sparsity pattern and its column coloring are held
in a [`JacobianPattern`](@ref), and a [`SparseResidualEvaluator`](@ref)
computes only the entries of the pattern.

```@docs
JacobianPattern
SparseResidualEvaluator
```
