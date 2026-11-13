"""
Sparse Jacobians of the packed residual `[obj; ceq; cineq]`: the sparsity
pattern, its column coloring, and an evaluator that computes only the
structurally nonzero entries.
"""

"""
    JacobianPattern

Sparsity pattern of the packed residual Jacobian (`1 + n_ceq + n_cineq` rows,
`n` columns; row 1 is the objective), with a column coloring.

Entries are numbered `k = 1:nnz` in column-major order; `rows[k]`, `cols[k]`
give the position of entry `k`. Columns of the same color share no row, so
they can be differentiated together. Columns without entries get color `0`
and are never perturbed.
"""
struct JacobianPattern
    nrows::Int
    ncols::Int
    rows::Vector{Int}
    cols::Vector{Int}
    colors::Vector{Int}                 # color of each column, 0 if empty
    ncolors::Int
    color_cols::Vector{Vector{Int}}     # columns of each color
    color_entries::Vector{Vector{Int}}  # entries `k` in columns of each color
    obj_entries::Vector{Int}            # entries in row 1 (objective)
    con_entries::Vector{Int}            # entries in rows 2:end (constraints)
end

SparseArrays.nnz(p::JacobianPattern) = length(p.rows)

# Stored entries of a sparse matrix (stored zeros included: they are structural),
# nonzero entries of any other matrix. Returned in column-major order.
function pattern_entries(S::SparseArrays.AbstractSparseMatrixCSC)
    rows = Int[]
    cols = Int[]
    rv = SparseArrays.rowvals(S)
    for j in axes(S, 2), p in SparseArrays.nzrange(S, j)
        push!(rows, rv[p])
        push!(cols, j)
    end
    return rows, cols
end
# Sparse matrices, also wrapped (transpose, adjoint, view), as a
# SparseMatrixCSC with the same stored entries; any other matrix as is.
stored_pattern(S::SparseArrays.AbstractSparseMatrixCSC) = S
stored_pattern(S::AbstractMatrix) = SparseArrays.issparse(S) ? SparseArrays.sparse(S) : S

function pattern_entries(S::AbstractMatrix)
    rows = Int[]
    cols = Int[]
    for (jj, j) in enumerate(axes(S, 2)), (ii, i) in enumerate(axes(S, 1))
        v = S[i, j]
        v isa Real || throw(
            ArgumentError("jac_sparsity entries must be Bool or real numbers; got $(typeof(v))"),
        )
        if !iszero(v)
            push!(rows, ii)
            push!(cols, jj)
        end
    end
    return rows, cols
end

"""
    JacobianPattern(S, nrows, ncols)

Build the pattern from a matrix `S` of size `(nrows, ncols)`. For a sparse
matrix, every stored entry is in the pattern (also a stored zero); for any
other matrix, every nonzero entry (`true`) is.
"""
function JacobianPattern(S::AbstractMatrix, nrows::Integer, ncols::Integer)
    if size(S) != (nrows, ncols)
        throw(
            ArgumentError(
                "jac_sparsity has size $(size(S)), but the Jacobian of [obj; ceq; cineq] " *
                "has size ($nrows, $ncols) (1 + n_ceq + n_cineq rows, length(x0) columns)",
            ),
        )
    end
    rows, cols = pattern_entries(stored_pattern(S))
    return JacobianPattern(rows, cols, Int(nrows), Int(ncols))
end

function JacobianPattern(rows::Vector{Int}, cols::Vector{Int}, nrows::Int, ncols::Int)
    nz = length(rows)
    colors = zeros(Int, ncols)
    if nz > 0
        S = SparseArrays.sparse(rows, cols, trues(nz), nrows, ncols)
        problem = SparseMatrixColorings.ColoringProblem(; structure=:nonsymmetric, partition=:column)
        algo = SparseMatrixColorings.GreedyColoringAlgorithm(SparseMatrixColorings.LargestFirst())
        raw = SparseMatrixColorings.column_colors(SparseMatrixColorings.coloring(S, problem, algo))
        # Empty columns get color 0; renumber the remaining colors 1:ncolors in
        # order of first appearance, so no color group is empty.
        nonempty = falses(ncols)
        nonempty[cols] .= true
        relabel = Dict{Int,Int}()
        for j in 1:ncols
            nonempty[j] || continue
            colors[j] = get!(relabel, raw[j], length(relabel) + 1)
        end
    end
    ncolors = maximum(colors; init=0)
    color_cols = [Int[] for _ in 1:ncolors]
    for j in 1:ncols
        colors[j] > 0 && push!(color_cols[colors[j]], j)
    end
    color_entries = [Int[] for _ in 1:ncolors]
    for k in 1:nz
        push!(color_entries[colors[cols[k]]], k)
    end
    obj_entries = findall(==(1), rows)
    con_entries = findall(>(1), rows)
    return JacobianPattern(
        nrows, ncols, rows, cols, colors, ncolors, color_cols, color_entries,
        obj_entries, con_entries,
    )
end

# Largest number of colors seeded in one Dual evaluation (ForwardDiff's own
# default chunk threshold).
const SPARSE_CHUNK_MAX = 12

"""
    SparseResidualEvaluator(fitness, pattern, n_ceq, n_cineq, deriv; x_lb, x_ub)

Like [`ResidualEvaluator`](@ref), but the Jacobian is a vector of the `nnz`
entries of `pattern` (in the order of `pattern.rows`, `pattern.cols`), and
columns of the same color are differentiated together:

- `:forwarddiff`: each Dual evaluation seeds up to 12 colors, so a Jacobian
  costs `cld(ncolors, 12)` `fitness` calls (one plain call if the pattern is
  empty and the value is needed).
- `:finitediff`: all columns of a color are stepped at once, each with its own
  bound-aware step, so a Jacobian costs `ncolors` calls, plus one if the value
  at `x` is not already known.

Entries outside the pattern are assumed to be zero and are never computed.
"""
struct SparseResidualEvaluator{F,D}
    residual::F
    deriv::Symbol
    pattern::JacobianPattern
    duals::D                # Vector of Duals for :forwarddiff, `nothing` otherwise
    x_lb::Vector{Float64}
    x_ub::Vector{Float64}
    xbuf::Vector{Float64}   # the `x` handed to plain `fitness` calls
    steps::Vector{Float64}  # finite-difference step of each column
end

function SparseResidualEvaluator(
    fitness, pattern::JacobianPattern, n_ceq::Integer, n_cineq::Integer, deriv::Symbol;
    x_lb=fill(-Inf, pattern.ncols), x_ub=fill(Inf, pattern.ncols),
)
    n = pattern.ncols
    residual = ξ -> packed_residual(fitness, ξ, n_ceq, n_cineq)
    lb = convert(Vector{Float64}, x_lb)
    ub = convert(Vector{Float64}, x_ub)
    xbuf = zeros(Float64, n)
    steps = zeros(Float64, n)
    if deriv === :forwarddiff
        N = clamp(pattern.ncolors, 1, SPARSE_CHUNK_MAX)
        T = typeof(ForwardDiff.Tag(residual, Float64))
        duals = Vector{ForwardDiff.Dual{T,Float64,N}}(undef, n)
        return SparseResidualEvaluator(residual, deriv, pattern, duals, lb, ub, xbuf, steps)
    elseif deriv === :finitediff
        return SparseResidualEvaluator(residual, deriv, pattern, nothing, lb, ub, xbuf, steps)
    else
        throw(ArgumentError("Unknown deriv=$deriv; use :forwarddiff or :finitediff"))
    end
end

function residual_value!(ev::SparseResidualEvaluator, r, x)
    copyto!(ev.xbuf, x)
    r .= ev.residual(ev.xbuf)
    return r
end

"""
    residual_value_and_jacobian!(ev::SparseResidualEvaluator, r, Jnz, x, r_valid)

Store the pattern entries of the Jacobian at `x` in the vector `Jnz`. `r` and
`r_valid` work as for the dense evaluator.
"""
function residual_value_and_jacobian!(
    ev::SparseResidualEvaluator, r, Jnz::AbstractVector, x, r_valid::Bool,
)
    p = ev.pattern
    if p.ncolors == 0
        # Empty pattern: nothing to differentiate.
        r_valid || residual_value!(ev, r, x)
    elseif ev.deriv === :forwarddiff
        sparse_dual_jacobian!(ev, ev.duals, r, Jnz, x, r_valid)
    else
        r_valid || residual_value!(ev, r, x)
        xt = ev.xbuf
        for c in 1:p.ncolors
            copyto!(xt, x)   # undo anything `fitness` did to `xt` last time
            for j in p.color_cols[c]
                s = fd_step(x[j], ev.x_lb[j], ev.x_ub[j])
                ev.steps[j] = s
                xt[j] = x[j] + s
            end
            rc = ev.residual(xt)
            for k in p.color_entries[c]
                i = p.rows[k]
                Jnz[k] = (rc[i] - r[i]) / ev.steps[p.cols[k]]
            end
        end
    end
    return Jnz
end

# Function barrier: `N` (the chunk size) is a type parameter here.
function sparse_dual_jacobian!(
    ev::SparseResidualEvaluator, xd::Vector{ForwardDiff.Dual{T,Float64,N}}, r, Jnz, x, r_valid,
) where {T,N}
    p = ev.pattern
    zero_seed = ForwardDiff.Partials{N,Float64}(ntuple(_ -> 0.0, Val(N)))
    seeds = ntuple(k -> ForwardDiff.Partials{N,Float64}(ntuple(i -> Float64(i == k), Val(N))), Val(N))
    for c0 in 0:N:(p.ncolors - 1)   # colors c0+1 : c0+N in this pass
        for j in eachindex(xd)
            k = p.colors[j] - c0
            seed = 1 <= k <= N ? seeds[k] : zero_seed
            xd[j] = ForwardDiff.Dual{T,Float64,N}(Float64(x[j]), seed)
        end
        rd = ev.residual(xd)
        if c0 == 0 && !r_valid
            for i in eachindex(r)
                r[i] = ForwardDiff.value(rd[i])
            end
        end
        for c in (c0 + 1):min(c0 + N, p.ncolors), k in p.color_entries[c]
            Jnz[k] = ForwardDiff.partials(rd[p.rows[k]], c - c0)
        end
    end
    return Jnz
end

const SCT_EXT = :NLPInterfaceSparseConnectivityTracerExt

# The detector used by `jac_sparsity=:auto` and by `detect_jac_sparsity` without
# a `detector`: SparseConnectivityTracer's global `TracerSparsityDetector`,
# provided by the package extension.
function default_sparsity_detector()
    ext = Base.get_extension(@__MODULE__, SCT_EXT)
    if ext === nothing
        throw(
            ArgumentError(
                "automatic sparsity detection needs SparseConnectivityTracer.jl; " *
                "run `using SparseConnectivityTracer` first, or pass a sparsity " *
                "detector or the pattern itself",
            ),
        )
    end
    return ext.default_detector()
end

"""
    detect_jac_sparsity(fitness, x0, n_ceq, n_cineq; detector=nothing)

Detect the sparsity pattern of the Jacobian of `[obj; ceq; cineq]` with respect
to `x`, as a `(1 + n_ceq + n_cineq) × length(x0)` sparse Boolean matrix. Row 1
is the objective, then the rows of `ceq`, then those of `cineq`.

`detector` is any `ADTypes.AbstractSparsityDetector`. By default (`nothing`)
it is SparseConnectivityTracer's `TracerSparsityDetector()`, which needs
`using SparseConnectivityTracer` and gives a pattern that holds at every `x`.
`TracerLocalSparsityDetector()` gives the pattern at `x0` only, which may miss
entries that are zero at `x0` but not elsewhere.

`fitness` is called with tracer numbers instead of `Float64`, so it must be
generic in the element type of `x`, as for `deriv=:forwarddiff`. Branches on
the value of `x` (`if x[1] > 0`) fail with the global detector.

The result can be passed to `solve_ipopt` and `solve_snopt` as `jac_sparsity`.
`jac_sparsity=:auto` calls this function with the default detector at `x0`.
"""
function detect_jac_sparsity(fitness, x0::AbstractVector, n_ceq::Integer, n_cineq::Integer; detector=nothing)
    sd = detector === nothing ? default_sparsity_detector() : detector
    sd isa ADTypes.AbstractSparsityDetector || throw(
        ArgumentError("detector must be an ADTypes.AbstractSparsityDetector; got $(typeof(sd))"),
    )
    x = convert(Vector{Float64}, x0)
    residual = ξ -> packed_residual(fitness, ξ, n_ceq, n_cineq)
    S = try
        ADTypes.jacobian_sparsity(residual, x, sd)
    catch err
        # Code that is not generic in the element type of `x` fails with a
        # TypeError (a branch on a tracer, a type assertion) or a MethodError (a
        # conversion to Float64); explain those. Anything else (our checks of the
        # fitness output, an error thrown by `fitness` itself) passes through.
        err isa Union{TypeError,MethodError} || rethrow()
        throw(
            ArgumentError(
                "sparsity detection with $(nameof(typeof(sd))) failed. `fitness` must be " *
                "generic in the element type of x; a branch on the value of x needs " *
                "TracerLocalSparsityDetector (pattern at x0 only) or a hand-made " *
                "pattern. Original error: " * sprint(showerror, err),
            ),
        )
    end
    nr, nc = 1 + Int(n_ceq) + Int(n_cineq), length(x)
    size(S) == (nr, nc) || throw(
        DimensionMismatch("the detector returned a $(size(S)) pattern; expected ($nr, $nc)"),
    )
    return bool_pattern(S)
end

# Boolean sparse pattern of a detector's result: the stored entries of a
# sparse matrix (stored zeros included), the nonzero entries of a dense one.
function bool_pattern(S::AbstractMatrix)
    if SparseArrays.issparse(S)
        Sc = SparseArrays.sparse(S)
        return SparseArrays.SparseMatrixCSC{Bool,Int}(
            size(Sc)..., Vector{Int}(SparseArrays.getcolptr(Sc)),
            Vector{Int}(SparseArrays.rowvals(Sc)), fill(true, SparseArrays.nnz(Sc)),
        )
    end
    return SparseArrays.sparse(S .!= 0)
end

# Turn the solver keyword `jac_sparsity` into a `JacobianPattern`, or `nothing`
# for a dense Jacobian. `x0` is the (already validated) starting point.
function resolve_jac_sparsity(jac_sparsity, fitness, x0, n_ceq, n_cineq)
    jac_sparsity === nothing && return nothing
    nrows, ncols = 1 + Int(n_ceq) + Int(n_cineq), length(x0)
    if jac_sparsity === :auto
        S = detect_jac_sparsity(fitness, x0, n_ceq, n_cineq)
    elseif jac_sparsity isa ADTypes.AbstractSparsityDetector
        S = detect_jac_sparsity(fitness, x0, n_ceq, n_cineq; detector=jac_sparsity)
    elseif jac_sparsity isa AbstractMatrix
        S = jac_sparsity
    else
        throw(
            ArgumentError(
                "jac_sparsity must be nothing, :auto, a sparsity detector, or a " *
                "($nrows × $ncols) matrix; got " *
                (jac_sparsity isa Symbol ? repr(jac_sparsity) : string(typeof(jac_sparsity))),
            ),
        )
    end
    return JacobianPattern(S, nrows, ncols)
end

# Pick the evaluator and the Jacobian buffer for a pattern (or dense, `nothing`).
function make_evaluator(fitness, pattern::Nothing, n, n_ceq, n_cineq, deriv; x_lb, x_ub)
    ev = ResidualEvaluator(fitness, n, n_ceq, n_cineq, deriv; x_lb=x_lb, x_ub=x_ub)
    return ev, zeros(Float64, 1 + n_ceq + n_cineq, n)
end
function make_evaluator(fitness, pattern::JacobianPattern, n, n_ceq, n_cineq, deriv; x_lb, x_ub)
    ev = SparseResidualEvaluator(fitness, pattern, n_ceq, n_cineq, deriv; x_lb=x_lb, x_ub=x_ub)
    return ev, zeros(Float64, SparseArrays.nnz(pattern))
end
