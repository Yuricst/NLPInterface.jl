#=
Benchmark SNOPT and Ipopt on every problem in test/problems.jl.

For each (problem, solver) pair this records
  - CPU time over M solves (mean and range), after one untimed warm-up solve,
  - iteration count (SNOPT major and minor iterations / Ipopt iterations),
  - number of `fitness` evaluation points (consecutive calls at the same point
    count once),
  - return status, xopt and fopt,
  - the error of xopt against the reference and the largest constraint violation,
and, for 2-D problems only, saves a contour plot of the iterates, one axis per
solver side by side.

Run from anywhere (uses the demo environment):
    julia benchmark/benchmark_problems.jl
Outputs: benchmark/results.csv, benchmark/times.csv (every timed solve),
benchmark/plots/<problem>_path.png (2-D only), benchmark/plots/cpu_time_boxplot.png
and benchmark/plots/evals_barplot.png

With the argument `auto`, every solve passes `jac_sparsity=:auto` (sparsity
detection with SparseConnectivityTracer is part of every timed solve), and the
outputs get the suffix `_auto` (results_auto.csv, times_auto.csv,
plots/cpu_time_boxplot_auto.png, plots/evals_barplot_auto.png; no path plots):
    julia benchmark/benchmark_problems.jl auto
=#

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "demo"); io=devnull)

using CairoMakie
using ForwardDiff
using NLPInterface
using Printf
using SparseConnectivityTracer
using Statistics

# `jac_sparsity` passed to every solve, and the suffix of the output files.
const JAC_SPARSITY = "auto" in ARGS ? :auto : nothing
const SUFFIX = JAC_SPARSITY === :auto ? "_auto" : ""

include(joinpath(@__DIR__, "..", "test", "problems.jl"))

const M = 10    # timed solves per (problem, solver)

solvers = has_snopt() ? ["snopt", "ipopt"] : ["ipopt"]
has_snopt() || @warn "SNOPT library not found; benchmarking Ipopt only"

# Process CPU time in seconds (POSIX `clock`, CLOCKS_PER_SEC = 10^6).
cpu_time() = ccall(:clock, Clong, ()) / 1_000_000

# Feasibility and optimality tolerance, the same threshold for both solvers. The
# two solvers measure these differently (SNOPT: relative to the size of x and of
# the multipliers; Ipopt: `tol` on its scaled overall error, the others absolute),
# so the thresholds match but the tests are not identical.
const TOL = 1e-6

# Benchmark-only Ipopt setting. Ipopt's default L-BFGS memory is 6 pairs, while
# SNOPT keeps a full-memory BFGS Hessian when it has at most 75 nonlinear
# variables. Keep at least 20 pairs (at least n), capped at SNOPT's 75 cutoff.
ipopt_lbfgs_history(n) = min(max(20, n), 75)

# Quiet solve. Extra keyword arguments are passed to the solver.
function run_solver(solver, fitness, p; kwargs...)
    if solver == "snopt"
        return solve_snopt(
            fitness, p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq;
            Major_feasibility_tolerance=TOL, Major_optimality_tolerance=TOL,
            jac_sparsity=JAC_SPARSITY, kwargs...,
        )
    elseif solver == "ipopt"
        return solve_ipopt(
            fitness, p.x0, p.x_lb, p.x_ub, p.n_ceq, p.n_cineq;
            print_level=0, sb="yes",
            tol=TOL, constr_viol_tol=TOL, dual_inf_tol=TOL, compl_inf_tol=TOL,
            limited_memory_max_history=ipopt_lbfgs_history(length(p.x0)),
            jac_sparsity=JAC_SPARSITY, kwargs...,
        )
    end
    error("unknown solver $solver")
end

# One untimed solve that logs the iterates and writes the solver's print file,
# from which the iteration counts are read. `n_iter` is SNOPT's major iterations
# or Ipopt's iterations; `n_minor` is SNOPT's total minor (QP) iterations and
# `missing` for Ipopt.
function record_run(solver, p)
    path = Vector{Vector{Float64}}()
    function fitness_logged(x)
        # Sparsity detection (`auto`) calls fitness with tracer numbers; not a point.
        eltype(x) <: Union{Float64,ForwardDiff.Dual} || return p.fitness(x)
        # With deriv=:forwarddiff most calls carry Dual numbers; log their primal value.
        # Skip repeats: one Jacobian may call fitness several times at the same point.
        xv = ForwardDiff.value.(x)
        if isempty(path) || xv != path[end]
            push!(path, collect(Float64, xv))
        end
        return p.fitness(x)
    end

    return mktempdir() do dir
        out = joinpath(dir, "$(solver).out")
        if solver == "snopt"
            xopt, fopt, info = run_solver(solver, fitness_logged, p; Print_file=out)
            txt = read(out, String)
            m = match(r"No\. of major iterations\s+(\d+)", txt)
            # "No. of iterations" is the total number of minor iterations.
            m_minor = match(r"No\. of iterations\s+(\d+)", txt)
        else
            xopt, fopt, info = run_solver(
                solver, fitness_logged, p; output_file=out, file_print_level=3,
            )
            m = match(r"Number of Iterations\.*:\s*(\d+)", read(out, String))
            m_minor = nothing
        end
        n_iter = m === nothing ? missing : parse(Int, m[1])
        n_minor = m_minor === nothing ? missing : parse(Int, m_minor[1])
        (; xopt, fopt, info, n_iter, n_minor, path)
    end
end

function benchmark(solver, p)
    run_solver(solver, p.fitness, p)    # warm-up (compilation)
    times = zeros(M)
    for k in 1:M
        GC.gc()
        t0 = cpu_time()
        run_solver(solver, p.fitness, p)
        times[k] = cpu_time() - t0
    end
    rec = record_run(solver, p)
    return (; rec..., times)
end

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

display_names = Dict(
    "rosenbrock" => "Rosenbrock",
    "mishra_bird" => "Mishra's Bird",
    "townsend" => "Townsend",
    "crawling" => "Crawling",
    "rosenbrock_cubic_line" => "Rosenbrock (cubic + line)",
    "gomez_levy" => "Gomez & Levy",
    "simionescu" => "Simionescu",
    "hs015" => "HS15",
    "g06" => "G06",
    "g08" => "G08",
    "g24" => "G24",
    "hs071" => "HS71",
    "hs078" => "HS78",
    "hartmann6" => "Hartmann 6-D",
    "hs100" => "HS100",
    "hs104" => "HS104",
    "rosenbrock_10d" => "Rosenbrock 10-D",
)
# Problems whose objective is shown on a log scale, log₁₀(f + ε).
log_scale = Set(["rosenbrock", "rosenbrock_cubic_line", "hs015"])

function plot_paths(name, p, results)
    n_grid = 250
    xs = range(p.x_lb[1], p.x_ub[1]; length=n_grid)
    ys = range(p.x_lb[2], p.x_ub[2]; length=n_grid)
    vals = [p.fitness([x, y]) for x in xs, y in ys]
    F = [v[1] for v in vals]
    if name in log_scale
        Z = log10.(F .+ 1e-3)
        zlabel = "log₁₀(f + ε)"
    else
        Z = F
        zlabel = "f"
    end
    ceqs = [[v[2][i] for v in vals] for i in 1:p.n_ceq]
    cineqs = [[v[3][i] for v in vals] for i in 1:p.n_cineq]

    # One eval-index color scale shared by both axes so the paths are comparable.
    n_max = maximum(length(results[s].path) for s in solvers)
    colorrange = (1, max(n_max, 2))

    # Fixed axis size matching the domain's aspect ratio, so the layout has no gaps.
    # Clamped so that very tall or wide boxes (e.g. G06) still give a readable axis.
    ax_height = 600
    ax_width = clamp(
        round(Int, ax_height * (p.x_ub[1] - p.x_lb[1]) / (p.x_ub[2] - p.x_lb[2])),
        450, 1000,
    )

    fig = Figure()
    hm = nothing
    sc = nothing
    for (j, solver) in enumerate(solvers)
        r = results[solver]
        path = r.path
        ax = Axis(
            fig[1, j];
            xlabel="x₁",
            ylabel="x₂",
            title="$(solver) path on $(display_names[name]) ($(zlabel))",
            width=ax_width,
            height=ax_height,
        )
        hm = contourf!(ax, xs, ys, Z; levels=25, colormap=:viridis)

        for C in cineqs
            contour!(ax, xs, ys, C; levels=[0.0], color=:white, linestyle=:dash, linewidth=1.0)
        end
        for C in ceqs
            contour!(ax, xs, ys, C; levels=[0.0], color=:white, linewidth=1.5)
        end
        # Legend entries for the constraint curves.
        p.n_cineq > 0 && lines!(
            ax, [NaN], [NaN];
            color=:white, linestyle=:dash, linewidth=1.0, label="cineq = 0",
        )
        p.n_ceq > 0 && lines!(
            ax, [NaN], [NaN]; color=:white, linewidth=1.5, label="ceq = 0",
        )

        if !isempty(path)
            px = getindex.(path, 1)
            py = getindex.(path, 2)
            t = eachindex(path)
            lines!(
                ax,
                px,
                py;
                color=t,
                colormap=:cool,
                colorrange=colorrange,
                linewidth=1.5,
                label="fitness evals",
            )
            sc = scatter!(
                ax,
                px,
                py;
                color=t,
                colormap=:cool,
                colorrange=colorrange,
                markersize=10,
            )
        end

        scatter!(
            ax,
            [p.x0[1]],
            [p.x0[2]];
            color=:white,
            strokecolor=:black,
            strokewidth=1.5,
            markersize=16,
            marker=:circle,
            label="x0",
        )
        scatter!(
            ax,
            [r.xopt[1]],
            [r.xopt[2]];
            color=:white,
            strokecolor=:black,
            strokewidth=1.5,
            markersize=18,
            marker=:star5,
            label="xopt",
        )
        # Gray background so the white constraint lines show in the legend.
        axislegend(ax; position=:lt, backgroundcolor=(:gray60, 0.9))
        xlims!(ax, p.x_lb[1], p.x_ub[1])
        ylims!(ax, p.x_lb[2], p.x_ub[2])
    end
    nc = length(solvers)
    Colorbar(fig[1, nc + 1], hm; label=zlabel)
    sc === nothing || Colorbar(fig[1, nc + 2], sc; label="eval index")
    resize_to_layout!(fig)

    outfile = joinpath(@__DIR__, "plots", "$(name)_path.png")
    save(outfile, fig)
    println("saved ", outfile)
    return fig
end

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

mkpath(joinpath(@__DIR__, "plots"))

# Largest constraint violation at x: |ceq| and positive part of cineq.
function max_violation(p, x)
    _, ceq, cineq = p.fitness(x)
    return max(maximum(abs, ceq; init=0.0), maximum(cineq; init=0.0))
end

rows = []
# 2-D problems first (these get contour plots), then by dimension and name.
for name in sort(collect(keys(problems)); by=k -> (length(problems[k].x0), k))
    p = problems[name]
    n = length(p.x0)
    println("== ", name, " (n = $n)")
    results = Dict{String,Any}()
    for solver in solvers
        r = benchmark(solver, p)
        results[solver] = r
        push!(rows, (;
            problem=name, solver, n, r...,
            xerr=maximum(abs, r.xopt .- p.xopt), viol=max_violation(p, r.xopt),
        ))
    end
    # The `auto` run takes the same paths; its path plots would only repeat them.
    n == 2 && JAC_SPARSITY === nothing && plot_paths(name, p, results)
end

println()
@printf(
    "%-22s %3s %-6s %-26s %6s %6s %6s %16s %16s %9s %9s %11s %25s\n",
    "problem", "n", "solver", "info", "iter", "minor", "evals", "fopt", "fopt (ref)",
    "|x-xref|∞", "max viol", "cpu mean[s]", "cpu range [min, max] [s]",
)
println("-"^177)
for r in rows
    @printf(
        "%-22s %3d %-6s %-26s %6s %6s %6d %16.8e %16.8e %9.1e %9.1e %11.3e   [%.3e, %.3e]\n",
        r.problem, r.n, r.solver, r.info, string(r.n_iter), string(coalesce(r.n_minor, "-")),
        length(r.path), r.fopt,
        problems[r.problem].fopt, r.xerr, r.viol,
        mean(r.times), minimum(r.times), maximum(r.times),
    )
end

csvfile = joinpath(@__DIR__, "results$(SUFFIX).csv")
open(csvfile, "w") do io
    println(
        io,
        "problem,label,n,solver,info,n_iter,n_minor,n_evals,fopt,fopt_ref,xerr_inf,max_viol,xopt," *
        "cpu_mean_s,cpu_min_s,cpu_max_s,M",
    )
    for r in rows
        println(
            io,
            join(
                (
                    r.problem, display_names[r.problem], r.n, r.solver, r.info,
                    coalesce(r.n_iter, ""), coalesce(r.n_minor, ""), length(r.path), r.fopt, problems[r.problem].fopt, r.xerr, r.viol,
                    # Semicolon-separated so the field needs no CSV quoting.
                    join(r.xopt, ";"),
                    mean(r.times), minimum(r.times), maximum(r.times), M,
                ),
                ",",
            ),
        )
    end
end
println("saved ", csvfile)

# Every timed solve, one row each, for the CPU-time box plot.
timesfile = joinpath(@__DIR__, "times$(SUFFIX).csv")
open(timesfile, "w") do io
    println(io, "problem,label,n,solver,run,cpu_s")
    for r in rows, (k, t) in enumerate(r.times)
        println(io, join((r.problem, display_names[r.problem], r.n, r.solver, k, t), ","))
    end
end
println("saved ", timesfile)

include(joinpath(@__DIR__, "plot_cpu_times.jl"))
plot_cpu_times(timesfile; outfile=joinpath(@__DIR__, "plots", "cpu_time_boxplot$(SUFFIX).png"))

include(joinpath(@__DIR__, "plot_evals.jl"))
plot_evals(csvfile; outfile=joinpath(@__DIR__, "plots", "evals_barplot$(SUFFIX).png"))
