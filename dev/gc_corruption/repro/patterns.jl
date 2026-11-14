# Step 3: loop one spawn pattern from test_snopt.jl, forcing a full GC after
# every iteration, so damage is detected close to where it happens.
#
#   julia --project=dev/gc_corruption --heap-size-hint=300M repro/patterns.jl VARIANT [N]
#
# Variants:
#   plain                       plain SNOPT solves (control)
#   spawn_outside               wait(Threads.@spawn throw(e)), no solve (test_snopt.jl:8)
#   spawn_in_callback_trivial   fetch(Threads.@spawn 1 + 1) inside fitness of a SNOPT solve
#   spawn_in_callback_nested    fetch(Threads.@spawn solve_snopt(...)) inside fitness (test_snopt.jl:311)
#   worker_solve                a whole SNOPT solve in a task on another thread (test_snopt.jl:322)
#   spawn_in_callback_ipopt     fetch(Threads.@spawn 1 + 1) inside fitness of an Ipopt solve
#   nested_same_task            fitness calls solve_snopt itself (it throws; test_snopt.jl:295)
#   worker_created_in_callback  fitness spawns a task that waits on a channel and solves
#                               after the outer solve ends (test_snopt.jl:318-333)
#   nested_testset              the whole "nested solve_snopt" testset body
#   wcic_noop                   worker created in the SNOPT callback, later only allocates
#   wcic_ipopt                  worker created in the SNOPT callback, later solves with Ipopt
#   wcic_from_ipopt             worker created in an Ipopt callback, later solves with SNOPT
#   scope_worker                worker created outside any callback but inside the
#                               solve's ScopedValue scope, later solves with SNOPT
#   scope_worker_with           same, but later only enters _with_snopt_in_solve and allocates
#   scope_worker_ipopt          same, but later solves with Ipopt inside _with_snopt_in_solve
using NLPInterface

variant = ARGS[1]
N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 300
println("patterns.jl: ", variant, " N=", N, " julia ", VERSION, " threads ",
    Threads.nthreads(), "+", Threads.nthreads(:interactive), " has_snopt=", has_snopt())
flush(stdout)

const ARGS0 = ([0.0, 0.0], [-10.0, -10.0], [10.0, 10.0], 0, 0)
quad(x) = ((x[1] - 2)^2 + (x[2] - 3)^2, nothing, nothing)

function one_iteration(variant)
    if variant == "plain"
        solve_snopt(quad, ARGS0...)
    elseif variant == "spawn_outside"
        try
            wait(Threads.@spawn throw(DomainError(1.0)))
        catch
        end
    elseif variant == "spawn_in_callback_trivial"
        f = x -> (fetch(Threads.@spawn 1 + 1); quad(x))
        solve_snopt(f, ARGS0...)
    elseif variant == "spawn_in_callback_nested"
        f = x -> (fetch(Threads.@spawn solve_snopt(quad, ARGS0...)); quad(x))
        try
            solve_snopt(f, ARGS0...)
        catch
        end
    elseif variant == "worker_solve"
        fetch(Threads.@spawn solve_snopt(quad, ARGS0...))
    elseif variant == "spawn_in_callback_ipopt"
        f = x -> (fetch(Threads.@spawn 1 + 1); quad(x))
        solve_ipopt(f, ARGS0...; print_level=0)
    elseif variant == "nested_same_task"
        f = x -> (solve_snopt(quad, ARGS0...); quad(x))
        try
            solve_snopt(f, ARGS0...)
        catch
        end
    elseif variant == "worker_created_in_callback"
        worker_created_in_callback()
    elseif variant == "nested_testset"
        one_iteration("nested_same_task")
        one_iteration("spawn_in_callback_nested")
        worker_created_in_callback()
        solve_snopt(quad, ARGS0...)
    elseif variant == "wcic_noop"
        worker_created_in_callback(solve_snopt, () -> sum(sum, [rand(100) for _ in 1:1000]))
    elseif variant == "wcic_ipopt"
        worker_created_in_callback(solve_snopt, () -> solve_ipopt(quad, ARGS0...; print_level=0))
    elseif variant == "wcic_from_ipopt"
        worker_created_in_callback((f, a...) -> solve_ipopt(f, a...; print_level=0),
            () -> solve_snopt(quad, ARGS0...))
    elseif variant == "scope_worker"
        scope_worker(() -> solve_snopt(quad, ARGS0...))
    elseif variant == "scope_worker_with"
        scope_worker(() -> NLPInterface._with_snopt_in_solve(() -> sum(sum, [rand(100) for _ in 1:1000])))
    elseif variant == "scope_worker_ipopt"
        scope_worker(() -> NLPInterface._with_snopt_in_solve(() -> solve_ipopt(quad, ARGS0...; print_level=0)))
    else
        error("unknown variant $variant")
    end
end

# A worker created inside a solve's ScopedValue scope (as a task spawned from
# fitness would be), which runs `work` after that scope has ended.
function scope_worker(work)
    go = Channel{Nothing}(1)
    t = NLPInterface._with_snopt_in_solve() do
        Threads.@spawn (take!(go); work())
    end
    put!(go, nothing)
    fetch(t)
end

# `outer` runs the solve whose fitness creates the worker; the worker waits
# until that solve has ended, then runs `work`.
function worker_created_in_callback(outer=solve_snopt, work=() -> solve_snopt(quad, ARGS0...))
    go = Channel{Nothing}(1)
    worker = Ref{Task}()
    f = function (x)
        if !isassigned(worker)
            worker[] = Threads.@spawn (take!(go); work())
        end
        return quad(x)
    end
    outer(f, ARGS0...)
    put!(go, nothing)
    fetch(worker[])
end

for i in 1:N
    one_iteration(variant)
    GC.gc(true)
end
println("patterns.jl: completed ", N, " iterations")
