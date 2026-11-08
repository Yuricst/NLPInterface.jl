# Helpers shared by the test files.

# An iterator without a `length` method (its `IteratorSize` defaults to HasLength).
struct NoLengthIter end
Base.iterate(::NoLengthIter, s=0) = s < 2 ? (Float64(s), s + 1) : nothing

# Wrap `fitness` to record every point it is called at, split into plain
# (Float64) and Dual (ForwardDiff) calls. `events` keeps the call order as
# `(is_dual, x)` pairs.
function counted(f)
    plain = Vector{Vector{Float64}}()
    dual = Vector{Vector{Float64}}()
    events = Tuple{Bool,Vector{Float64}}[]
    g = function (x)
        if eltype(x) <: ForwardDiff.Dual
            push!(dual, ForwardDiff.value.(x))
            push!(events, (true, dual[end]))
        else
            push!(plain, collect(Float64, x))
            push!(events, (false, plain[end]))
        end
        return f(x)
    end
    return g, plain, dual, events
end
