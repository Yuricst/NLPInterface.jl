"""Demo for trajectory design with forward-backward shooting"""

using GLMakie
using OrdinaryDiffEq
using LinearAlgebra

using NLPInterface

using Random
Random.seed!(42)

MU = 1.0
nseg = 20
nseg_fwd = fld(nseg, 2)
nseg_bck = nseg - nseg_fwd

t0_bounds = [0.0, 0.0]
tof_bounds = [0.5π, 4π]
umax = 0.2

x_lb = [[t0_bounds[1], tof_bounds[1]]; -ones(3*nseg);  zeros(nseg)]
x_ub = [[t0_bounds[2], tof_bounds[2]];  ones(3*nseg);  umax * ones(nseg)]

x0_t0 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
xf_t0 = [1.5, 0.0, 0.0, 0.0, sqrt(MU/1.5), 0.0]

# dynamics & base ODE problem (to be reused within fitness function)
function dynamics!(deriv, state, p, t)
    deriv[1:3] = state[4:6]
    deriv[4:6] = -MU * state[1:3]/norm(state[1:3])^3 + p[1:3]
    return
end

prob_base = ODEProblem(dynamics!, x0_t0, (0.0, 1.0), zeros(3))

function construct_trajectory(x)
    # unpack decision variables
    t0, tof = x[1:2]
    us = reshape(x[3:end], nseg, 4)

    # propagate forward in time
    sols_fwd = Vector{ODESolution}(undef, nseg_fwd)
    ti = t0
    x_fwd_t0 = x0_t0
    if t0 > 1e-15
        prob_t0 = remake(prob_base, u0=x0_t0, tspan=(0.0, ti), p=zeros(3))
        x_fwd_t0 = solve(prob_t0, Tsit5(); reltol=1e-11, abstol=1e-12).u[end]
    else
        x_fwd_t0 = x0_t0
    end
    for i in 1:nseg_fwd
        prob = remake(prob_base, u0=x_fwd_t0, tspan=(ti, ti + tof/nseg), p=us[i,1:3] * us[i,4])
        sols_fwd[i] = solve(prob, Tsit5(); reltol=1e-11, abstol=1e-12)
        x_fwd_t0 = sols_fwd[i].u[end]
        ti += tof/nseg
    end

    # propagate backward in time
    ti = tof + t0
    sols_bck = Vector{ODESolution}(undef, nseg_bck)
    prob_tf = remake(prob_base, u0=xf_t0, tspan=(0.0, t0 + tof), p=zeros(3))
    x_bck_tf = solve(prob_tf, Tsit5(); reltol=1e-11, abstol=1e-12).u[end]
    for i in 1:nseg_bck
        prob = remake(prob_base, u0=x_bck_tf, tspan=(ti, ti - tof/nseg), p=us[end+1-i,1:3] * us[end+1-i,4])
        sols_bck[i] = solve(prob, Tsit5(); reltol=1e-11, abstol=1e-12)
        x_bck_tf = sols_bck[i].u[end]
        ti -= tof/nseg
    end

    # return trajectory
    return sols_fwd, sols_bck, sols_bck[end].u[end] - sols_fwd[end].u[end]
end

# fitness function
n_ceq = 6
n_cineq = nseg
function fitness(x)
    us = reshape(x[3:end], nseg, 4)
    obj = sum(us[:,4])
    _, _, ceq = construct_trajectory(x)
    cineq = [
        us[i,1:3]'us[i,1:3] - 1.0 for i in 1:nseg
    ]
    return obj, ceq, cineq
end

# solve NLP
x0 = [[0.0, π]; rand(4*nseg)] #ones(3*nseg); umax * ones(nseg)]
xopt, fopt, info = solve_snopt(
    fitness,
    x0,
    x_lb,
    x_ub,
    n_ceq,
    n_cineq;
    Major_print_level=1,
    Major_feasibility_tolerance=1e-8,
    Major_optimality_tolerance=1e-6,
    Major_iterations=200
)
# xopt, fopt, info = solve_ipopt(
#     fitness,
#     x0,
#     x_lb,
#     x_ub,
#     n_ceq,
#     n_cineq;
#     print_level=5,
#     tol=1e-6,
#     constr_viol_tol=1e-8,
# )
uopt = reshape(xopt[3:end], nseg, 4)

# reconstruct solved trajectory
sols_fwd, sols_bck, ceq = construct_trajectory(xopt)

# initial and final orbit for plotting
initial_orbit = solve(
    remake(prob_base, u0=x0_t0, tspan=(0.0, 4π), p=zeros(3)),
    Tsit5(); reltol=1e-11, abstol=1e-12
)
final_orbit = solve(
    remake(prob_base, u0=xf_t0, tspan=(0.0, 4π), p=zeros(3)),
    Tsit5(); reltol=1e-11, abstol=1e-12
)

# plot trajectory
quiver_scale = 1.0
n_quiver = 5
fig = Figure()
ax = Axis3(fig[1, 1]; aspect=:data)
lines!(ax, Array(initial_orbit)[1,:], Array(initial_orbit)[2,:], Array(initial_orbit)[3,:], color=:black)
lines!(ax, Array(final_orbit)[1,:], Array(final_orbit)[2,:], Array(final_orbit)[3,:], color=:black)

for (id_sol, _sol) in enumerate(sols_fwd)
    if id_sol == 1
        scatter!(ax, _sol.u[1][1], _sol.u[1][2], _sol.u[1][3], color=:blue)
    end
    lines!(ax, Array(_sol)[1,:], Array(_sol)[2,:], Array(_sol)[3,:])
    
    t_quiver = LinRange(_sol.t[1], _sol.t[end], n_quiver)
    u_dir = uopt[id_sol,1:3] * uopt[id_sol,4] * quiver_scale
    arrows3d!(ax, Array(_sol(t_quiver))[1,:], Array(_sol(t_quiver))[2,:], Array(_sol(t_quiver))[3,:],
        u_dir[1]*ones(n_quiver), u_dir[2]*ones(n_quiver), u_dir[3]*ones(n_quiver), color=:red)
end
for (id_sol, _sol) in enumerate(sols_bck)
    if id_sol == 1
        scatter!(ax, _sol.u[1][1], _sol.u[1][2], _sol.u[1][3], color=:red)
    end
    lines!(ax, Array(_sol)[1,:], Array(_sol)[2,:], Array(_sol)[3,:])

    t_quiver = LinRange(_sol.t[1], _sol.t[end], n_quiver)
    u_dir = uopt[id_sol,1:3] * uopt[id_sol,4] * quiver_scale
    arrows3d!(ax, Array(_sol(t_quiver))[1,:], Array(_sol(t_quiver))[2,:], Array(_sol(t_quiver))[3,:],
        u_dir[1]*ones(n_quiver), u_dir[2]*ones(n_quiver), u_dir[3]*ones(n_quiver), color=:red)
end
display(fig)