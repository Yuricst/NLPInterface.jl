#=
Box plot of the CPU time per solve, SNOPT vs. Ipopt side by side, with one
category per problem.

Reads bechmark/times.csv (one row per timed solve, written by
benchmark_problems.jl, which also calls this script at the end). To redraw the
plot without re-running the benchmark:
    julia bechmark/plot_cpu_times.jl
Output: bechmark/plots/cpu_time_boxplot.png
=#

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "demo"); io=devnull)

using CairoMakie

function plot_cpu_times(
    timesfile=joinpath(@__DIR__, "times.csv");
    outfile=joinpath(@__DIR__, "plots", "cpu_time_boxplot.png"),
)
    # Columns: problem,label,n,solver,run,cpu_s (no field contains a comma).
    lines_ = filter(!isempty, readlines(timesfile))
    header = split(lines_[1], ",")
    col = Dict(h => i for (i, h) in enumerate(header))
    recs = [split(l, ",") for l in lines_[2:end]]

    # Categories in file order (the benchmark writes 2-D problems first, then by n).
    problems = unique(r[col["problem"]] for r in recs)
    solvers = unique(r[col["solver"]] for r in recs)
    labels = map(problems) do p
        r = recs[findfirst(r -> r[col["problem"]] == p, recs)]
        "$(r[col["label"]]) (n = $(r[col["n"]]))"
    end

    category = [findfirst(==(r[col["problem"]]), problems) for r in recs]
    dodge = [findfirst(==(r[col["solver"]]), solvers) for r in recs]
    cpu_ms = [1e3 * parse(Float64, r[col["cpu_s"]]) for r in recs]
    n_runs = maximum(parse(Int, r[col["run"]]) for r in recs)

    colors = Makie.wong_colors()[1:length(solvers)]
    fig = Figure(size=(1900, 750))
    ax = Axis(
        fig[1, 1];
        yscale=log10,
        ylabel="CPU time per solve [ms]",
        title="CPU time per solve, $(n_runs) solves per (problem, solver)",
        xticks=(eachindex(problems), labels),
        xticklabelrotation=π / 6,
        xgridvisible=false,
        yminorticksvisible=true,
        yminorgridvisible=true,
        yminorticks=IntervalsBetween(9),
    )
    # Light separators between problems.
    vlines!(ax, 1.5:1:(length(problems) - 0.5); color=(:gray, 0.3), linewidth=1)
    # One call per solver so the box outline, whiskers, median and outliers all take
    # the solver's color: run-to-run spread is small, so most boxes are thin and
    # the colored strokes are what make them visible on the log axis.
    for (j, solver) in enumerate(solvers)
        idx = dodge .== j
        c = colors[j]
        boxplot!(
            ax, category[idx], cpu_ms[idx];
            dodge=dodge[idx], n_dodge=length(solvers), width=0.8,
            color=(c, 0.35), strokecolor=c, strokewidth=1.5,
            whiskercolor=c, whiskerwidth=0.6, whiskerlinewidth=1.5,
            mediancolor=c, medianlinewidth=2,
            show_outliers=true, outliercolor=c, markersize=6,
            label=solver == "snopt" ? "SNOPT" : "Ipopt",
        )
    end
    xlims!(ax, 0.4, length(problems) + 0.6)
    Legend(fig[1, 2], ax, "solver")

    mkpath(dirname(outfile))
    save(outfile, fig)
    println("saved ", outfile)
    return fig
end

# Run when executed as a script (not when included by benchmark_problems.jl).
if abspath(PROGRAM_FILE) == @__FILE__
    plot_cpu_times()
end
