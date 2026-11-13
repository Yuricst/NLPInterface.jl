#=
Benchmark a dense Jacobian against a sparse one (`jac_sparsity`).

Problem: `transcription_problem(N)` from test/transcription.jl, a trapezoidal
collocation of a planar minimum-energy transfer around an obstacle, with
n = 7N variables and 6N + 5 rows in the Jacobian of [obj; ceq; cineq].

For each solver (SNOPT, Ipopt), each `deriv` (:forwarddiff, :finitediff), each
Jacobian mode, and each N, this records:
  - CPU time over M solves (median and range), after one untimed warm-up
    solve; the modes are interleaved (see `benchmark`);
  - bytes allocated by Julia per solve (`@timed`), which includes SNOPT's
    workspace (allocated in Julia) but not Ipopt's internal C++ memory;
  - the number of `fitness` calls, the return status, and fopt.
The peak resident memory, which also counts the solvers' own memory, is
measured separately by sparse_jacobian_rss.jl.

Jacobian modes:
  dense    `jac_sparsity = nothing` (the default)
  pattern  `jac_sparsity = P.pattern`, the hand-made pattern
  auto     `jac_sparsity = :auto` (SparseConnectivityTracer; detection is part
           of every timed solve)

Run from anywhere (uses the demo environment):
    julia benchmark/sparse_jacobian.jl        # timing, writes sparse_results.csv
    julia benchmark/sparse_jacobian_rss.jl    # peak memory, writes sparse_rss.csv
    julia benchmark/plot_sparse.jl            # figures from both CSV files
Outputs: benchmark/sparse_results.csv, and (through plot_sparse.jl)
docs/src/assets/sparse_cpu_time.png and docs/src/assets/sparse_memory.png
=#

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "demo"); io=devnull)

using NLPInterface
using Printf
using SparseArrays
using SparseConnectivityTracer
using Statistics

include(joinpath(@__DIR__, "..", "test", "transcription.jl"))
include(joinpath(@__DIR__, "sparse_jacobian_common.jl"))

const M = 7   # timed solves per configuration

solvers = has_snopt() ? ["snopt", "ipopt"] : ["ipopt"]
has_snopt() || @warn "SNOPT library not found; benchmarking Ipopt only"

# Process CPU time in seconds (POSIX `clock`, CLOCKS_PER_SEC = 10^6).
cpu_time() = ccall(:clock, Clong, ()) / 1_000_000

# All modes of one (solver, deriv, N) are timed together and interleaved: each
# of the M rounds runs every mode once, in turn, so a change of machine load
# affects all modes alike rather than one block of them.
function benchmark(solver, deriv, N, modes)
    P = transcription_problem(N)
    for mode in modes
        run_solver(solver, deriv, mode, P, P.fitness)   # warm-up (compilation)
    end
    times = Dict(mode => zeros(M) for mode in modes)
    bytes = Dict(mode => zeros(Int, M) for mode in modes)
    for k in 1:M, mode in modes
        GC.gc()
        t0 = cpu_time()
        stats = @timed run_solver(solver, deriv, mode, P, P.fitness)
        times[mode][k] = cpu_time() - t0
        bytes[mode][k] = stats.bytes
    end
    out = []
    for mode in modes
        # Untimed run that counts the `fitness` calls (`:auto` adds one tracer call).
        ncalls = Ref(0)
        counted_fitness = x -> (ncalls[] += 1; P.fitness(x))
        xopt, fopt, info = run_solver(solver, deriv, mode, P, counted_fitness)
        push!(out, (; solver, deriv, mode, N, n=P.n, nrows=size(P.pattern, 1),
            nnz=nnz(P.pattern), info, fopt, n_evals=ncalls[], times=times[mode],
            bytes=minimum(bytes[mode])))
    end
    return out
end

results = []
for solver in solvers, deriv in DERIVS, N in NS_SPARSE
    modes = N in NS_DENSE ? MODES : filter(!=("dense"), MODES)
    for r in benchmark(solver, deriv, N, modes)
        push!(results, r)
        @printf("%-6s %-12s %-8s N=%4d n=%5d  %-27s evals=%7d  cpu=%.3e s  alloc=%8.1f MiB\n",
            solver, deriv, r.mode, N, r.n, r.info, r.n_evals, median(r.times), r.bytes / 2^20)
        flush(stdout)
    end
end

csvfile = joinpath(@__DIR__, "sparse_results.csv")
open(csvfile, "w") do io
    println(io, "solver,deriv,mode,N,n,jac_rows,jac_nnz,info,fopt,n_evals,cpu_median_s,cpu_mean_s,cpu_min_s,cpu_max_s,alloc_bytes,M")
    for r in results
        println(io, join((r.solver, r.deriv, r.mode, r.N, r.n, r.nrows, r.nnz, r.info, r.fopt,
            r.n_evals, median(r.times), mean(r.times), minimum(r.times), maximum(r.times), r.bytes, M), ","))
    end
end
println("saved ", csvfile)

include(joinpath(@__DIR__, "plot_sparse.jl"))
plot_sparse()
