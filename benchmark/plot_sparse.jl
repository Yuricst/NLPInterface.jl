#=
Figures for the dense vs. sparse Jacobian benchmark.

Reads benchmark/sparse_results.csv (sparse_jacobian.jl, which also calls this
script at the end) and, if present, benchmark/sparse_rss.csv
(sparse_jacobian_rss.jl). To redraw the figures without re-running the benchmark:
    julia benchmark/plot_sparse.jl
Outputs: docs/src/assets/sparse_cpu_time.png (CPU time per solve, rows: deriv,
columns: solver) and docs/src/assets/sparse_memory.png (rows: memory allocated
by Julia per solve, and the peak resident memory of the solve; columns: solver;
deriv = :forwarddiff).
=#

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "demo"); io=devnull)

using CairoMakie

# Read a CSV file with a header line into a vector of Dicts (values as strings).
function read_rows(file)
    lines = readlines(file)
    header = split(lines[1], ",")
    return [Dict(zip(header, split(l, ","))) for l in lines[2:end] if !isempty(l)]
end

# Categorical slots 1-3 of the reference palette (validated all-pairs), with a
# distinct marker per mode as secondary encoding.
const MODE_STYLE = [
    "dense" => (color="#2a78d6", marker=:circle, label="dense"),
    "pattern" => (color="#eb6834", marker=:rect, label="pattern"),
    "auto" => (color="#1baf7a", marker=:diamond, label=":auto"),
]
const SOLVER_NAME = Dict("snopt" => "SNOPT", "ipopt" => "Ipopt")

# One panel per (row, column); `select(row, col)` filters the CSV rows of a
# panel, `value(r)` gives y. x is the number of variables n.
function panel_figure(file, rows_csv, panel_rows, panel_cols; title, ylabel, select, value, panel_title)
    fig = Figure(; size=(900, 320 * length(panel_rows) + 60), fontsize=14)
    Label(fig[0, 1:length(panel_cols)], title; fontsize=16, font=:bold)
    ns = sort(unique(parse(Int, r["n"]) for r in rows_csv))
    for (i, pr) in enumerate(panel_rows), (j, pc) in enumerate(panel_cols)
        ax = Axis(fig[i, j];
            title=panel_title(pr, pc),
            xlabel=i == length(panel_rows) ? "number of variables n" : "",
            ylabel=j == 1 ? ylabel(pr) : "",
            xscale=log10, yscale=log10,
            xticks=(ns, string.(ns)),
            xgridcolor=(:black, 0.08), ygridcolor=(:black, 0.08))
        for (mode, st) in MODE_STYLE
            rs = filter(r -> select(r, pr, pc) && r["mode"] == mode, rows_csv)
            isempty(rs) && continue
            sort!(rs; by=r -> parse(Int, r["n"]))
            xs = [parse(Int, r["n"]) for r in rs]
            ys = [value(r, pr) for r in rs]
            keep = isfinite.(ys) .& (ys .> 0)   # 0 (not measurable) has no place on a log scale
            scatterlines!(ax, xs[keep], ys[keep]; color=st.color, marker=st.marker,
                markersize=9, linewidth=2, label=st.label)
        end
    end
    Legend(fig[length(panel_rows) + 1, 1:length(panel_cols)],
        [MarkerElement(color=st.color, marker=st.marker, markersize=10) for (_, st) in MODE_STYLE],
        [st.label for (_, st) in MODE_STYLE]; orientation=:horizontal, framevisible=false)
    save(file, fig; px_per_unit=2)
    println("saved ", file)
    return fig
end

function plot_sparse(;
    resultsfile=joinpath(@__DIR__, "sparse_results.csv"),
    rssfile=joinpath(@__DIR__, "sparse_rss.csv"),
    assets=joinpath(@__DIR__, "..", "docs", "src", "assets"),
)
    res = read_rows(resultsfile)
    solvers = filter(s -> any(r -> r["solver"] == s, res), ["snopt", "ipopt"])

    panel_figure(joinpath(assets, "sparse_cpu_time.png"), res, ["forwarddiff", "finitediff"], solvers;
        title="CPU time per solve (median of $(res[1]["M"]) solves)",
        ylabel=_ -> "CPU time [s]",
        select=(r, deriv, solver) -> r["deriv"] == deriv && r["solver"] == solver,
        value=(r, _) -> parse(Float64, r["cpu_median_s"]),
        panel_title=(deriv, solver) -> "$(SOLVER_NAME[solver]), deriv = :$deriv")

    # Memory: Julia allocations from the timing run, peak RSS from the RSS run.
    mem = [merge(r, Dict("metric" => "alloc", "mib" => string(parse(Float64, r["alloc_bytes"]) / 2^20)))
           for r in res if r["deriv"] == "forwarddiff"]
    metrics = ["alloc"]
    if isfile(rssfile)
        append!(mem, [merge(r, Dict("metric" => "rss", "mib" => r["rss_peak_increase_mib"]))
                      for r in read_rows(rssfile) if r["deriv"] == "forwarddiff"])
        push!(metrics, "rss")
    end
    metric_label = Dict("alloc" => "allocated by Julia [MiB]", "rss" => "peak RSS increase [MiB]")
    metric_title = Dict("alloc" => "allocated by Julia per solve", "rss" => "peak resident memory of the solve")
    panel_figure(joinpath(assets, "sparse_memory.png"), mem, metrics, solvers;
        title="Memory per solve (deriv = :forwarddiff)",
        ylabel=m -> metric_label[m],
        select=(r, metric, solver) -> r["metric"] == metric && r["solver"] == solver,
        value=(r, _) -> parse(Float64, r["mib"]),
        panel_title=(metric, solver) -> "$(SOLVER_NAME[solver]): $(metric_title[metric])")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_sparse()
end
