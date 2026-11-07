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
