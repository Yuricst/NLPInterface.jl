#=
Peak resident memory of a dense vs. a sparse Jacobian (Linux only).

    julia benchmark/sparse_jacobian_rss.jl

For each (solver, deriv, mode) this starts a fresh worker process (the same
script with arguments), which, after a warm-up solve at the smallest N
(compilation), solves each N and reports the peak resident memory during the
solve minus the resident memory just before it. Unlike `@allocated`, this
includes the memory Ipopt and SNOPT allocate themselves.

The peak is the kernel's high-water mark `VmHWM` (/proc/self/status), reset
before each solve by writing 5 to /proc/self/clear_refs. `Sys.maxrss()` cannot
be used: on Linux the getrusage maximum survives fork and exec, so a worker
would start with its parent's peak.

Output: benchmark/sparse_rss.csv
=#

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "demo"); io=devnull)

using NLPInterface
using SparseArrays
using SparseConnectivityTracer

include(joinpath(@__DIR__, "..", "test", "transcription.jl"))
include(joinpath(@__DIR__, "sparse_jacobian_common.jl"))

# Resident memory now and its high-water mark, in MiB.
function proc_status_mib(key)
    for line in eachline("/proc/self/status")
        startswith(line, key * ":") && return parse(Int, split(line)[2]) / 1024   # kB
    end
    error("$key not found in /proc/self/status")
end

function worker(solver, deriv, mode, Ns)
    P = transcription_problem(minimum(Ns))
    run_solver(solver, deriv, mode, P, P.fitness)   # warm-up (compilation)
    for N in Ns
        P = transcription_problem(N)
        GC.gc()
        write("/proc/self/clear_refs", "5")            # reset VmHWM to VmRSS
        before = proc_status_mib("VmRSS")
        run_solver(solver, deriv, mode, P, P.fitness)
        println("RSS ", N, " ", proc_status_mib("VmHWM") - before)
        flush(stdout)
    end
end

function driver()
    Sys.islinux() || error("sparse_jacobian_rss.jl reads /proc and runs on Linux only")
    solvers = has_snopt() ? ["snopt", "ipopt"] : ["ipopt"]
    outfile = joinpath(@__DIR__, "sparse_rss.csv")
    open(outfile, "w") do io
        println(io, "solver,deriv,mode,N,n,rss_peak_increase_mib")
        for solver in solvers, deriv in DERIVS, mode in MODES
            Ns = mode == "dense" ? NS_DENSE : NS_SPARSE
            cmd = `$(Base.julia_cmd()) --startup-file=no $(@__FILE__) $solver $deriv $mode $(join(Ns, ","))`
            for line in eachline(cmd)
                m = match(r"^RSS (\d+) (-?[0-9.]+)$", line)
                m === nothing && continue
                N = parse(Int, m[1])
                println(io, join((solver, deriv, mode, N, 7N, m[2]), ","))
                println(join((solver, deriv, mode, N, m[2]), "  "))
            end
        end
    end
    println("saved ", outfile)
end

if isempty(ARGS)
    driver()
else
    worker(ARGS[1], Symbol(ARGS[2]), ARGS[3], sort(parse.(Int, split(ARGS[4], ","))))
end
