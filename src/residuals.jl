"""
Functions to turn the `(obj, ceq, cineq) -> r` into a single stacked vector `r`.
"""

"""
    packed_residual(fitness, x, n_ceq, n_cineq)

Evaluate `fitness(x) -> (obj, ceq, cineq)` as a single vector
`[obj; ceq; cineq]` of length `1 + n_ceq + n_cineq`.
"""
function packed_residual(fitness, x, n_ceq, n_cineq)
    obj, ceq, cineq = fitness(x)
    T = eltype(x)
    r = Vector{T}(undef, 1 + n_ceq + n_cineq)
    r[1] = obj
    if n_ceq > 0
        r[2:(1 + n_ceq)] .= ceq
    end
    if n_cineq > 0
        r[(2 + n_ceq):end] .= cineq
    end
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
