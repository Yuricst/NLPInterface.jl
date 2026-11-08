"""
Functions to turn the `(obj, ceq, cineq) -> r` into a single stacked vector `r`.
"""

block_length(::Nothing) = 0
block_length(block::Union{Number,AbstractArray,Tuple}) = length(block)
function block_length(block::Union{AbstractString,AbstractChar,Missing})
    throw(
        ArgumentError(
            "constraint block must be a vector, tuple, number, or nothing; got $(typeof(block))",
        ),
    )
end
block_length(block) = length(block)  # other iterables with a known length

# Copy one constraint block into `r[offset:(offset + n - 1)]`, after checking that
# the block has exactly `n` entries. A length mismatch would otherwise either drop
# constraints silently or broadcast one value into several rows.
function store_constraint_block!(r, offset, block, n, name)
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
    if block isa Number
        r[offset] = block
    elseif block isa AbstractArray
        copyto!(r, offset, block, firstindex(block), n)
    else
        k = offset
        for v in block
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
be a scalar.
"""
function packed_residual(fitness, x, n_ceq, n_cineq)
    obj, ceq, cineq = fitness(x)
    T = eltype(x)
    r = Vector{T}(undef, 1 + n_ceq + n_cineq)
    r[1] = obj
    store_constraint_block!(r, 2, ceq, n_ceq, "ceq")
    store_constraint_block!(r, 2 + n_ceq, cineq, n_cineq, "cineq")
    return r
end

function jacobian_residual(fitness, x, n_ceq, n_cineq, deriv::Symbol)
    residual = ξ -> packed_residual(fitness, ξ, n_ceq, n_cineq)
    if deriv === :forwarddiff
        return ForwardDiff.jacobian(residual, x)
    elseif deriv === :finitediff
        return FiniteDiff.finite_difference_jacobian(residual, x)
    else
        throw(ArgumentError("Unknown deriv=$deriv; use :forwarddiff or :finitediff"))
    end
end

"""
    ResidualEvaluator(fitness, n, n_ceq, n_cineq, deriv)

Evaluate the packed residual `[obj; ceq; cineq]` and its Jacobian with as few
`fitness` calls as possible. Built once per solve, so the ForwardDiff
configuration and result buffers are reused at every point.

- `:forwarddiff`: the value and the Jacobian come from the same Dual
  evaluation(s), so no separate plain call is needed.
- `:finitediff` (forward differences): the value at `x` is reused as the base
  point, so a Jacobian costs `n` calls when the value is already known and
  `n + 1` otherwise.
"""
struct ResidualEvaluator{F,C,R}
    residual::F
    deriv::Symbol
    cfg::C      # ForwardDiff.JacobianConfig, or `nothing` for :finitediff
    result::R   # DiffResults.DiffResult, or `nothing` for :finitediff
end

function ResidualEvaluator(fitness, n::Integer, n_ceq::Integer, n_cineq::Integer, deriv::Symbol)
    residual = ξ -> packed_residual(fitness, ξ, n_ceq, n_cineq)
    if deriv === :forwarddiff
        xs = zeros(Float64, n)
        cfg = ForwardDiff.JacobianConfig(residual, xs)
        result = DiffResults.JacobianResult(zeros(Float64, 1 + n_ceq + n_cineq), xs)
        return ResidualEvaluator(residual, deriv, cfg, result)
    elseif deriv === :finitediff
        return ResidualEvaluator(residual, deriv, nothing, nothing)
    else
        throw(ArgumentError("Unknown deriv=$deriv; use :forwarddiff or :finitediff"))
    end
end

"""
    residual_value!(ev, r, x)

Store the packed residual at `x` in `r` (one plain `fitness` call).
"""
function residual_value!(ev::ResidualEvaluator, r, x)
    r .= ev.residual(x)
    return r
end

"""
    residual_value_and_jacobian!(ev, r, J, x, r_valid)

Store the Jacobian at `x` in `J`. If `r_valid` is `true`, `r` must already hold
the residual at `x`; it is used as is (and left unchanged). Otherwise the residual
at `x` is computed as part of the Jacobian evaluation and stored in `r`.
"""
function residual_value_and_jacobian!(ev::ResidualEvaluator, r, J, x, r_valid::Bool)
    if ev.deriv === :forwarddiff
        result = ForwardDiff.jacobian!(ev.result, ev.residual, x, ev.cfg)
        r_valid || (r .= DiffResults.value(result))
        J .= DiffResults.jacobian(result)
    else
        r_valid || residual_value!(ev, r, x)
        # Same call as `jacobian_residual`, but with the known base value.
        J .= FiniteDiff.finite_difference_jacobian(
            ev.residual, x, Val(:forward), eltype(x), copy(r),
        )
    end
    return J
end
