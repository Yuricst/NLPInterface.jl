# Step 4: the trigger without NLPInterface. A ScopedValue holding a Ref, set
# with the same code as NLPInterface._with_snopt_in_solve; a task spawned
# inside that scope runs later, re-enters the scope, and calls C code that
# calls back into Julia (qsort with a Julia comparator).
#
#   julia --heap-size-hint=300M repro/scoped.jl VARIANT [N]
# Variants:
#   scoped_qsort      spawned inside the scope; later withflag + qsort with callback
#   plain_qsort       spawned outside any scope; later withflag + qsort with callback
#   scoped_alloc      spawned inside the scope; later withflag + allocation only
#   scoped_qsort_noref  like scoped_qsort, but the scoped value is a Bool, not a Ref
#   scoped_qsort_gc   like scoped_qsort, with a full GC inside the comparator now and then
#   plain_qsort_gc    like plain_qsort (no scope at spawn), with GC inside the comparator
#   scoped_noreenter_qsort_gc  spawned inside the scope; later qsort (GC in callback) without withflag
#   reenter_only_qsort_gc      spawned outside any scope; later withflag + qsort (GC in callback)
#   scoped_alloc_gc   spawned inside the scope; later withflag + GC.gc(true) in plain Julia
# With NLPInterface (run in dev/gc_corruption's environment):
#   nlpi_noreenter    spawned in NLPInterface's solve scope; later solve_ipopt (no re-entry)
#   nlpi_reenter      same, later solve_ipopt inside _with_snopt_in_solve (= scope_worker_ipopt)
#   nlpi_reenter_fd   same as nlpi_reenter with deriv = :finitediff
#   own_scope_ipopt   spawned in a Main-defined scope (Ref); later withflag + solve_ipopt
#   own_scope_rawipopt  spawned in withflag scope; later withflag + a raw Ipopt.jl solve
using Base.ScopedValues

variant = ARGS[1]
N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 300
println("scoped.jl: ", variant, " N=", N, " julia ", VERSION, " threads ",
    Threads.nthreads(), "+", Threads.nthreads(:interactive))
flush(stdout)

const SV = ScopedValue{Union{Nothing,Base.RefValue{Bool}}}(nothing)
const SVB = ScopedValue{Bool}(false)

# Same shape as NLPInterface._with_snopt_in_solve.
function withflag(f)
    active = Ref(true)
    try
        return with(f, SV => active)
    finally
        active[] = false
    end
end
withbool(f) = with(f, SVB => true)

# A comparator that allocates, so the GC can run inside the callback.
const NCMP = Ref(0)
const GC_IN_CMP = Ref(false)
function cmp_alloc(a::Ptr{Float64}, b::Ptr{Float64})::Cint
    tmp = [unsafe_load(a), unsafe_load(b)]
    if GC_IN_CMP[] && (NCMP[] += 1) % 5000 == 0
        GC.gc(true)
    end
    return Cint(sign(tmp[1] - tmp[2]))
end
function qsort_with_callback()
    v = rand(2000)
    c = @cfunction(cmp_alloc, Cint, (Ptr{Float64}, Ptr{Float64}))
    ccall(:qsort, Cvoid, (Ptr{Float64}, Csize_t, Csize_t, Ptr{Cvoid}), v, length(v), sizeof(Float64), c)
    issorted(v) || error("qsort failed")
end
alloc_only() = sum(sum, [rand(100) for _ in 1:20_000])

function spawn_then_run(make_scope, work)
    go = Channel{Nothing}(1)
    t = make_scope() do
        Threads.@spawn (take!(go); work())
    end
    put!(go, nothing)
    fetch(t)
end

if startswith(variant, "nlpi") || occursin("ipopt", variant)
    @eval import NLPInterface as NLPI
    @eval const Ipopt = NLPI.Ipopt
end
const ARGS0 = ([0.0, 0.0], [-10.0, -10.0], [10.0, 10.0], 0, 0)
quad(x) = ((x[1] - 2)^2 + (x[2] - 3)^2, nothing, nothing)

# min (x1-2)^2 + (x2-3)^2 with Ipopt.jl's C API directly (no NLPInterface).
function raw_ipopt_solve()
    eval_f(x) = (x[1] - 2)^2 + (x[2] - 3)^2
    eval_g(x, g) = nothing
    function eval_grad_f(x, g)
        g[1] = 2(x[1] - 2); g[2] = 2(x[2] - 3); return
    end
    eval_jac_g(x, rows, cols, values) = nothing
    prob = Base.invokelatest(Ipopt.CreateIpoptProblem, 2, [-10.0, -10.0], [10.0, 10.0], 0, Float64[], Float64[],
        0, 0, eval_f, eval_g, eval_grad_f, eval_jac_g, nothing)
    Base.invokelatest(Ipopt.AddIpoptIntOption, prob, "print_level", 0)
    Base.invokelatest(Ipopt.AddIpoptStrOption, prob, "hessian_approximation", "limited-memory")
    prob.x = [0.0, 0.0]
    Base.invokelatest(Ipopt.IpoptSolve, prob)
end

for i in 1:N
    if variant == "scoped_qsort"
        spawn_then_run(withflag, () -> withflag(qsort_with_callback))
    elseif variant == "plain_qsort"
        spawn_then_run(f -> f(), () -> withflag(qsort_with_callback))
    elseif variant == "scoped_alloc"
        spawn_then_run(withflag, () -> withflag(alloc_only))
    elseif variant == "scoped_qsort_noref"
        spawn_then_run(withbool, () -> withbool(qsort_with_callback))
    elseif variant == "scoped_qsort_gc"
        GC_IN_CMP[] = true
        spawn_then_run(withflag, () -> withflag(qsort_with_callback))
    elseif variant == "plain_qsort_gc"
        GC_IN_CMP[] = true
        spawn_then_run(f -> f(), qsort_with_callback)
    elseif variant == "scoped_noreenter_qsort_gc"
        GC_IN_CMP[] = true
        spawn_then_run(withflag, qsort_with_callback)
    elseif variant == "reenter_only_qsort_gc"
        GC_IN_CMP[] = true
        spawn_then_run(f -> f(), () -> withflag(qsort_with_callback))
    elseif variant == "scoped_alloc_gc"
        spawn_then_run(withflag, () -> withflag(() -> (alloc_only(); GC.gc(true))))
    elseif variant == "nlpi_noreenter"
        spawn_then_run(NLPI._with_snopt_in_solve, () -> NLPI.solve_ipopt(quad, ARGS0...; print_level=0))
    elseif variant == "nlpi_reenter"
        spawn_then_run(NLPI._with_snopt_in_solve,
            () -> NLPI._with_snopt_in_solve(() -> NLPI.solve_ipopt(quad, ARGS0...; print_level=0)))
    elseif variant == "nlpi_reenter_fd"
        spawn_then_run(NLPI._with_snopt_in_solve,
            () -> NLPI._with_snopt_in_solve(() -> NLPI.solve_ipopt(quad, ARGS0...; print_level=0, deriv=:finitediff)))
    elseif variant == "own_scope_ipopt"
        spawn_then_run(withflag, () -> withflag(() -> NLPI.solve_ipopt(quad, ARGS0...; print_level=0)))
    elseif variant == "own_scope_rawipopt"
        spawn_then_run(withflag, () -> withflag(raw_ipopt_solve))
    else
        error("unknown variant $variant")
    end
    GC.gc(true)
end
println("scoped.jl: completed ", N, " iterations")
