#=
Grouped bar plot of the number of `fitness` evaluation points (calls at a point
different from the previous call's point), SNOPT vs. Ipopt side by side, with one
category per problem. Unlike iteration counts, which mean different units of work
for the two solvers, this measure means the same for both. It is not the number
of `fitness` calls (see README.md, "Solver settings and comparability").

Reads bechmark/results.csv, written by benchmark_problems.jl, which also calls
this script at the end. To redraw the plot without re-running the benchmark:
    julia bechmark/plot_evals.jl
Output: bechmark/plots/evals_barplot.png
=#

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "demo"); io=devnull)

using CairoMakie

# Read a CSV file with a header row and no quoted fields into a vector of
# Dict(column => value) rows.
function _read_csv_rows(file)
    lines_ = filter(!isempty, readlines(file))
    header = split(lines_[1], ",")
    return [Dict(zip(header, split(l, ","))) for l in lines_[2:end]]
end

function plot_evals(
    resultsfile=joinpath(@__DIR__, "results.csv");
    outfile=joinpath(@__DIR__, "plots", "evals_barplot.png"),
)
    # results.csv columns used: problem, label, n, solver, n_evals.
    recs = _read_csv_rows(resultsfile)

    # Categories in file order (the benchmark writes 2-D problems first, then by n).
    problems = unique(r["problem"] for r in recs)
    solvers = unique(r["solver"] for r in recs)
    labels = map(problems) do p
        r = recs[findfirst(r -> r["problem"] == p, recs)]
        "$(r["label"]) (n = $(r["n"]))"
    end

    category = [findfirst(==(r["problem"]), problems) for r in recs]
    dodge = [findfirst(==(r["solver"]), solvers) for r in recs]
    n_evals = [parse(Int, r["n_evals"]) for r in recs]

    colors = Makie.wong_colors()[1:length(solvers)]
    fig = Figure(size=(1900, 750))
    ax = Axis(
        fig[1, 1];
        yscale=log10,
        ylabel="fitness evaluation points",
        title="fitness evaluation points per solve (consecutive repeats counted once)",
        xticks=(eachindex(problems), labels),
        xticklabelrotation=π / 6,
        xgridvisible=false,
        yminorticksvisible=true,
        yminorgridvisible=true,
        yminorticks=IntervalsBetween(9),
    )
    # Light separators between problems.
    vlines!(ax, 1.5:1:(length(problems) - 0.5); color=(:gray, 0.3), linewidth=1)
    # The counts span about an order of magnitude or more, so the y axis is
    # logarithmic like the CPU-time plot. Bars start at 1 (log10(1) = 0) and
    # carry their value as a label.
    barplot!(
        ax, category, n_evals;
        dodge=dodge, n_dodge=length(solvers), width=0.8, gap=0.2, dodge_gap=0.05,
        fillto=1,
        color=[(colors[d], 0.35) for d in dodge],
        strokecolor=colors[dodge], strokewidth=1.5,
        bar_labels=:y, label_formatter=y -> string(round(Int, y)),
        label_size=13, label_offset=3,
        label_color=colors[dodge],
    )
    xlims!(ax, 0.4, length(problems) + 0.6)
    # Headroom for the bar labels above the tallest bar.
    ylims!(ax, 1, 2 * maximum(n_evals))
    Legend(
        fig[1, 2],
        [PolyElement(; color=(c, 0.35), strokecolor=c, strokewidth=1.5) for c in colors],
        [s == "snopt" ? "SNOPT" : "Ipopt" for s in solvers],
        "solver",
    )

    mkpath(dirname(outfile))
    save(outfile, fig)
    println("saved ", outfile)
    return fig
end

# Run when executed as a script (not when included by benchmark_problems.jl).
if abspath(PROGRAM_FILE) == @__FILE__
    plot_evals()
end
