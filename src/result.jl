"""Result of `solve_ipopt` and `solve_snopt`."""

"""
    NLPResult

Result of [`solve_ipopt`](@ref) and [`solve_snopt`](@ref).

# Fields
- `xopt::Vector{Float64}`: the solution (a new array; `x0` is not modified).
- `fopt::Float64`: the objective at `xopt`.
- `status::Symbol`: the outcome, from `NLPInterface.IPOPT_RETURN_STATUS` or
  `NLPInterface.SNOPT_RETURN_STATUS` (for example `:Solve_Succeeded`).
- `raw_status::Int`: the solver's own code (Ipopt return status, SNOPT INFO).
- `ceq::Vector{Float64}`, `cineq::Vector{Float64}`: the constraint values at
  `xopt`.
- `lambda_eq::Vector{Float64}`, `lambda_ineq::Vector{Float64}`: Lagrange
  multipliers of `ceq` and `cineq`.
- `z_lb::Vector{Float64}`, `z_ub::Vector{Float64}`: multipliers of the lower
  and upper bounds on `x`.
- `iterations::Union{Int,Missing}`: Ipopt iterations; `missing` for SNOPT,
  whose C interface does not report its iteration count.
- `n_fitness::Int`: number of calls of `fitness` during the solve (plain,
  ForwardDiff Dual, and sparsity-detection calls).
- `solver::Symbol`: `:ipopt` or `:snopt`.

# Multipliers

Both solvers report multipliers in one convention. At a KKT point,

```math
\\nabla f + J_{\\mathrm{eq}}^\\top \\lambda_{\\mathrm{eq}}
          + J_{\\mathrm{ineq}}^\\top \\lambda_{\\mathrm{ineq}}
          - z_{\\mathrm{lb}} + z_{\\mathrm{ub}} = 0,
```

with ``\\lambda_{\\mathrm{ineq}} \\ge 0``, ``z_{\\mathrm{lb}} \\ge 0``,
``z_{\\mathrm{ub}} \\ge 0``, and each of them zero where its constraint or
bound is inactive. ``\\lambda_{\\mathrm{eq}}`` has no sign. That is, they are
the multipliers of the Lagrangian
``f + \\lambda_{\\mathrm{eq}}^\\top c_{\\mathrm{eq}} + \\lambda_{\\mathrm{ineq}}^\\top c_{\\mathrm{ineq}} - z_{\\mathrm{lb}}^\\top (x - x_{\\mathrm{lb}}) + z_{\\mathrm{ub}}^\\top (x - x_{\\mathrm{ub}})``.
A multiplier is the sensitivity of the optimal objective to its constraint:
relaxing `cineq[i] <= 0` to `cineq[i] <= ε` changes `fopt` by about
`-lambda_ineq[i] * ε`.

# Compatibility

An `NLPResult` destructures and indexes like the tuple `(xopt, fopt, status)`
that earlier versions returned:

```julia
xopt, fopt, info = solve_ipopt(fitness, x0, x_lb, x_ub, n_ceq, n_cineq)
res = solve_ipopt(fitness, x0, x_lb, x_ub, n_ceq, n_cineq)
res.lambda_ineq, res[1] == res.xopt
```
"""
struct NLPResult
    xopt::Vector{Float64}
    fopt::Float64
    status::Symbol
    raw_status::Int
    ceq::Vector{Float64}
    cineq::Vector{Float64}
    lambda_eq::Vector{Float64}
    lambda_ineq::Vector{Float64}
    z_lb::Vector{Float64}
    z_ub::Vector{Float64}
    iterations::Union{Int,Missing}
    n_fitness::Int
    solver::Symbol
end

# Destructuring and indexing as the former `(xopt, fopt, status)` tuple.
Base.length(::NLPResult) = 3
Base.firstindex(::NLPResult) = 1
Base.lastindex(::NLPResult) = 3
function Base.getindex(r::NLPResult, i::Int)
    i == 1 && return r.xopt
    i == 2 && return r.fopt
    i == 3 && return r.status
    throw(BoundsError(r, i))
end
Base.iterate(r::NLPResult, i::Int=1) = i > 3 ? nothing : (r[i], i + 1)

function Base.show(io::IO, ::MIME"text/plain", r::NLPResult)
    println(io, "NLPResult (", r.solver, "): ", r.status, " (raw status ", r.raw_status, ")")
    println(io, "  fopt        = ", r.fopt)
    println(io, "  xopt        = ", r.xopt)
    isempty(r.ceq) || println(io, "  ceq         = ", r.ceq)
    isempty(r.cineq) || println(io, "  cineq       = ", r.cineq)
    isempty(r.lambda_eq) || println(io, "  lambda_eq   = ", r.lambda_eq)
    isempty(r.lambda_ineq) || println(io, "  lambda_ineq = ", r.lambda_ineq)
    println(io, "  z_lb        = ", r.z_lb)
    println(io, "  z_ub        = ", r.z_ub)
    print(io, "  iterations  = ", r.iterations, ", fitness calls = ", r.n_fitness)
end

# Counts the calls of `fitness` during one solve.
mutable struct CountedFitness{F}
    f::F
    n::Int
end
CountedFitness(f) = CountedFitness(f, 0)
function (c::CountedFitness)(x)
    c.n += 1
    return c.f(x)
end

# Split a packed residual `[obj; ceq; cineq]` into the constraint blocks.
function constraint_blocks(r::AbstractVector, n_ceq::Integer, n_cineq::Integer)
    return Vector{Float64}(r[2:(1 + n_ceq)]), Vector{Float64}(r[(2 + n_ceq):(1 + n_ceq + n_cineq)])
end
