# NLPInterface.jl
Interface for SNOPT/Ipopt


Given a fitness function of the form

```julia
"""
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