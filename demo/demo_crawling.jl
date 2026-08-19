using GLMakie
using NLPInterface

include(joinpath(@__DIR__, "..", "test", "problems.jl"))

solver = "ipopt"    # "snopt" or "ipopt"

p = problems["crawling"]

path = Vector{Vector{Float64}}()
function fitness_logged(x)
    if eltype(x) <: Float64
        push!(path, collect(Float64, x))
    end
    return p.fitness(x)
end

if solver == "ipopt"
    xopt, fopt, info = solve_ipopt(
        fitness_logged,
        p.x0,
        p.x_lb,
        p.x_ub,
        p.n_ceq,
        p.n_cineq;
        print_level=5,
    )
elseif solver == "snopt"
    xopt, fopt, info = solve_snopt(
        fitness_logged,
        p.x0,
        p.x_lb,
        p.x_ub,
        p.n_ceq,
        p.n_cineq;
        Major_print_level=1,
    )
end

println("info = ", info)
println("xopt = ", xopt)
println("fopt = ", fopt)
println("n fitness evals = ", length(path))

obj(x1, x2) = x1 + x2
n_grid = 250
xs = range(p.x_lb[1], p.x_ub[1]; length=n_grid)
ys = range(p.x_lb[2], p.x_ub[2]; length=n_grid)
Z = [obj(x, y) for x in xs, y in ys]

fig = Figure(size=(950, 700))
ax = Axis(
    fig[1, 1];
    xlabel="x₁",
    ylabel="x₂",
    title="$(solver) path on Crawling (f = x₁ + x₂)",
    aspect=DataAspect(),
)
hm = contourf!(ax, xs, ys, Z; levels=25, colormap=:viridis)
Colorbar(fig[1, 2], hm; label="f")

x_eq = range(p.x_lb[1], p.x_ub[1]; length=400)
y_eq = @. x_eq^4 + 2 * x_eq^3 - 1.2 * x_eq^2 - 2 * x_eq
lines!(
    ax,
    x_eq,
    y_eq;
    color=:white,
    linestyle=:dash,
    linewidth=1.5,
    label="ceq = 0",
)

x_ineq = range(p.x_lb[1], p.x_ub[1]; length=64)
y_ineq = @. -(4 / 3) * x_ineq - 2 / 3
lines!(
    ax,
    x_ineq,
    y_ineq;
    color=:white,
    linestyle=:dot,
    linewidth=1.5,
    label="cineq = 0",
)

if !isempty(path)
    px = getindex.(path, 1)
    py = getindex.(path, 2)
    t = eachindex(path)
    colorrange = (first(t), last(t))
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
    Colorbar(fig[1, 3], sc; label="eval index")
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
    [xopt[1]],
    [xopt[2]];
    color=:white,
    strokecolor=:black,
    strokewidth=1.5,
    markersize=18,
    marker=:star5,
    label="xopt",
)
axislegend(ax; position=:lt)
xlims!(ax, p.x_lb[1], p.x_ub[1])
ylims!(ax, p.x_lb[2], p.x_ub[2])

outfile = joinpath(@__DIR__, "crawling_path_$(solver).png")
save(outfile, fig)
display(fig)
println("saved ", outfile)
