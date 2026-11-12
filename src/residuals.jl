"""
Functions to turn the `(obj, ceq, cineq) -> r` into a single stacked vector `r`.
"""

block_length(::Nothing) = 0
block_length(block::Real) = 1
block_length(block::Union{AbstractArray,Tuple}) = length(block)

# Accept other iterables (generators, ...) and reject everything that is not a
# collection of real numbers: strings, characters, symbols, `missing`, `Some`,
# complex numbers, and sets or dictionaries (no defined order, so rows would be
# assigned arbitrarily). This happens before `convert` or `length` would fail
# with a less helpful error. An iterable without a known size (a generator over
# a filter, whose `length` throws) is collected first, so it is iterated once.
normalize_block(block::Union{Nothing,Real,AbstractArray,Tuple}) = block
function normalize_block(block)
    if !(block isa Union{AbstractString,AbstractChar,Number,Missing,Symbol,Some,AbstractSet,AbstractDict}) &&
       applicable(iterate, block)
        size = Base.IteratorSize(block)
        if size isa Base.IsInfinite
            throw(ArgumentError("constraint block is an infinite iterator ($(typeof(block)))"))
        end
        # Julia's default `IteratorSize` is `HasLength`, even for iterators
        # without a `length` method; collect those too.
        known = size isa Union{Base.HasLength,Base.HasShape} && applicable(length, block)
        known && return block
        # Not `collect`, which itself calls `length` for `HasLength` iterators.
        values = Any[]
        for v in block
            push!(values, v)
        end
        return values
    end
    throw(
        ArgumentError(
            "constraint block must be a vector, tuple, real number, or nothing; " *
            "got $(typeof(block))",
        ),
    )
end
block_length(block) = length(block)

function bad_constraint_value(v, name)
    return ArgumentError(
        "fitness returned a $(typeof(v)) entry in $name; constraint values must be real numbers",
    )
end

# Copy one constraint block into `r[offset:(offset + n - 1)]`, after checking that
# the block has exactly `n` entries. A length mismatch would otherwise either drop
# constraints silently or broadcast one value into several rows. Each entry must
# be a real number: characters (`['a']`) would otherwise be stored as their
# character codes, and `missing` would fail inside `convert`.
function store_constraint_block!(r, offset, block, n, name)
    block = normalize_block(block)
    len = block_length(block)
    if len != n
        got = block === nothing ? "nothing" : "$len value(s)"
        throw(
            DimensionMismatch(
                "fitness returned $got for $name, but n_$name = $n was passed to the solver",
            ),
        )
    end
    n == 0 && return r
    if block isa Real
        r[offset] = block
    elseif block isa AbstractArray{<:Real}
        copyto!(r, offset, block, firstindex(block), n)
    else
        k = offset
        for v in block
            v isa Real || throw(bad_constraint_value(v, name))
            r[k] = v
            k += 1
        end
    end
    return r
end

"""
    packed_residual(fitness, x, n_ceq, n_cineq)

Evaluate `fitness(x) -> (obj, ceq, cineq)` as a single vector
`[obj; ceq; cineq]` of length `1 + n_ceq + n_cineq`.

Throws a `DimensionMismatch` if `ceq` does not have exactly `n_ceq` entries or
`cineq` does not have exactly `n_cineq` entries. A block with zero entries may
be returned as an empty collection or as `nothing`; a block with one entry may
be a real scalar. The objective and every constraint entry must be real
numbers; anything else (strings, characters, `missing`, symbols, ...) throws an
`ArgumentError`.
"""
function packed_residual(fitness, x, n_ceq, n_cineq)
    obj, ceq, cineq = fitness(x)
    obj isa Real || throw(
        ArgumentError("fitness must return a real scalar objective; got $(typeof(obj))"),
    )
    T = eltype(x)
    r = Vector{T}(undef, 1 + n_ceq + n_cineq)
    r[1] = obj
    store_constraint_block!(r, 2, ceq, n_ceq, "ceq")
    store_constraint_block!(r, 2 + n_ceq, cineq, n_cineq, "cineq")
    return r
end

# Dense Jacobian of the packed residual at `x` (used by the tests as a reference).
function jacobian_residual(fitness, x, n_ceq, n_cineq, deriv::Symbol)
    if deriv === :forwarddiff
        residual = ξ -> packed_residual(fitness, ξ, n_ceq, n_cineq)
        return ForwardDiff.jacobian(residual, x)
    elseif deriv === :finitediff
        n = length(x)
        ev = ResidualEvaluator(fitness, n, n_ceq, n_cineq, deriv)
        r = zeros(Float64, 1 + n_ceq + n_cineq)
        J = zeros(Float64, 1 + n_ceq + n_cineq, n)
        return residual_value_and_jacobian!(ev, r, J, x, false)
    else
        throw(ArgumentError("Unknown deriv=$deriv; use :forwarddiff or :finitediff"))
    end
end

"""
    ResidualEvaluator(fitness, n, n_ceq, n_cineq, deriv; x_lb, x_ub)

Evaluate the packed residual `[obj; ceq; cineq]` and its Jacobian with as few
`fitness` calls as possible. Built once per solve, so the ForwardDiff
configuration and result buffers are reused at every point.

- `:forwarddiff`: the value and the Jacobian come from the same Dual
  evaluation(s), so no separate plain call is needed.
- `:finitediff` (one-sided differences): the value at `x` is reused as the base
  point, so a Jacobian costs `n` calls when the value is already known and
  `n + 1` otherwise. Each step stays inside `[x_lb, x_ub]` where possible: it
  goes forward unless that would cross `x_ub[j]`, then backward.

`fitness` never receives the solver's own `x` array: plain calls get a copy in
a buffer owned by the evaluator, so a `fitness` that modifies its argument
cannot corrupt the solver's iterate.
"""
struct ResidualEvaluator{F,C,R}
    residual::F
    deriv::Symbol
    cfg::C      # ForwardDiff.JacobianConfig, or `nothing` for :finitediff
    result::R   # DiffResults.DiffResult, or `nothing` for :finitediff
    x_lb::Vector{Float64}
    x_ub::Vector{Float64}
    xbuf::Vector{Float64}   # the `x` handed to plain `fitness` calls
end

function ResidualEvaluator(
    fitness, n::Integer, n_ceq::Integer, n_cineq::Integer, deriv::Symbol;
    x_lb=fill(-Inf, n), x_ub=fill(Inf, n),
)
    residual = ξ -> packed_residual(fitness, ξ, n_ceq, n_cineq)
    lb = convert(Vector{Float64}, x_lb)
    ub = convert(Vector{Float64}, x_ub)
    xbuf = zeros(Float64, n)
    if deriv === :forwarddiff
        xs = zeros(Float64, n)
        cfg = ForwardDiff.JacobianConfig(residual, xs)
        result = DiffResults.JacobianResult(zeros(Float64, 1 + n_ceq + n_cineq), xs)
        return ResidualEvaluator(residual, deriv, cfg, result, lb, ub, xbuf)
    elseif deriv === :finitediff
        return ResidualEvaluator(residual, deriv, nothing, nothing, lb, ub, xbuf)
    else
        throw(ArgumentError("Unknown deriv=$deriv; use :forwarddiff or :finitediff"))
    end
end

"""
    residual_value!(ev, r, x)

Store the packed residual at `x` in `r` (one plain `fitness` call, on a copy of `x`).
"""
function residual_value!(ev::ResidualEvaluator, r, x)
    copyto!(ev.xbuf, x)
    r .= ev.residual(ev.xbuf)
    return r
end

# Step for column `j` of a one-sided difference at `xj`: `+h` unless that crosses
# the upper bound, then `-h` if that stays above the lower bound; if neither
# fits (bounds closer than `h`), step towards the wider side. Returns the step
# actually taken after rounding, so that the quotient uses the exact difference.
function fd_step(xj, lb, ub)
    h = sqrt(eps(Float64)) * max(1.0, abs(xj))
    if xj + h <= ub
        s = h
    elseif xj - h >= lb
        s = -h
    else
        s = (ub - xj >= xj - lb) ? h : -h
    end
    return (xj + s) - xj
end

"""
    residual_value_and_jacobian!(ev, r, J, x, r_valid)

Store the Jacobian at `x` in `J`. If `r_valid` is `true`, `r` must already hold
the residual at `x`; it is used as is (and left unchanged). Otherwise the residual
at `x` is computed as part of the Jacobian evaluation and stored in `r`.
"""
function residual_value_and_jacobian!(ev::ResidualEvaluator, r, J, x, r_valid::Bool)
    if ev.deriv === :forwarddiff
        # ForwardDiff copies `x` into its own Dual buffer; `fitness` never sees `x`.
        result = ForwardDiff.jacobian!(ev.result, ev.residual, x, ev.cfg)
        r_valid || (r .= DiffResults.value(result))
        J .= DiffResults.jacobian(result)
    else
        r_valid || residual_value!(ev, r, x)
        xt = ev.xbuf
        for j in eachindex(x)
            copyto!(xt, x)   # undo anything `fitness` did to `xt` last time
            s = fd_step(x[j], ev.x_lb[j], ev.x_ub[j])
            xt[j] = x[j] + s
            rj = ev.residual(xt)
            for i in eachindex(r)
                J[i, j] = (rj[i] - r[i]) / s
            end
        end
    end
    return J
end
