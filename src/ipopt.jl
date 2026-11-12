"""Interface for IPOPT"""

# Ipopt return codes: https://coin-or.github.io/Ipopt/OUTPUT.html
const IPOPT_RETURN_STATUS = Dict(
    0 => :Solve_Succeeded,
    1 => :Solved_To_Acceptable_Level,
    2 => :Infeasible_Problem_Detected,
    3 => :Search_Direction_Becomes_Too_Small,
    4 => :Diverging_Iterates,
    5 => :User_Requested_Stop,
    6 => :Feasible_Point_Found,
    -1 => :Maximum_Iterations_Exceeded,
    -2 => :Restoration_Failed,
    -3 => :Error_In_Step_Computation,
    -4 => :Maximum_CpuTime_Exceeded,
    -5 => :Maximum_WallTime_Exceeded,
    -10 => :Not_Enough_Degrees_Of_Freedom,
    -11 => :Invalid_Problem_Definition,
    -12 => :Invalid_Option,
    -13 => :Invalid_Number_Detected,
    -100 => :Unrecoverable_Exception,
    -101 => :NonIpopt_Exception_Thrown,
    -102 => :Insufficient_Memory,
    -199 => :Internal_Error,
)

# Ipopt options of type "real" (Ipopt 3.14, from `print_options_documentation`).
# An integer value for one of these (e.g. `tol=1`) is sent as a Float64:
# `AddIpoptIntOption` rejects it, and Ipopt's C interface cannot report an
# option's type. Integer values for any other name are sent as integers.
const IPOPT_REAL_OPTIONS = Set([
    "acceptable_compl_inf_tol", "acceptable_constr_viol_tol", "acceptable_dual_inf_tol",
    "acceptable_obj_change_tol", "acceptable_tol", "adaptive_mu_kkterror_red_fact",
    "adaptive_mu_monotone_init_factor", "alpha_for_y_tol", "alpha_min_frac",
    "alpha_red_factor", "barrier_tol_factor", "bound_frac", "bound_mult_init_val",
    "bound_mult_reset_threshold", "bound_push", "bound_relax_factor", "compl_inf_tol",
    "constr_mult_init_max", "constr_mult_reset_threshold", "constr_viol_tol",
    "corrector_compl_avrg_red_fact", "delta", "derivative_test_perturbation",
    "derivative_test_tol", "diverging_iterates_tol", "dual_inf_tol", "eta_phi",
    "expect_infeasible_problem_ctol", "expect_infeasible_problem_ytol",
    "filter_margin_fact", "filter_max_margin", "findiff_perturbation",
    "first_hessian_perturbation", "gamma_phi", "gamma_theta",
    "jacobian_regularization_exponent", "jacobian_regularization_value", "kappa_d",
    "kappa_sigma", "kappa_soc", "limited_memory_init_val", "limited_memory_init_val_max",
    "limited_memory_init_val_min", "ma27_la_init_factor", "ma27_liw_init_factor",
    "ma27_meminc_factor", "ma27_pivtol", "ma27_pivtolmax", "ma57_pivtol", "ma57_pivtolmax",
    "ma57_pre_alloc", "ma77_small", "ma77_static", "ma77_u", "ma77_umax", "ma86_small",
    "ma86_static", "ma86_u", "ma86_umax", "ma97_small", "ma97_u", "ma97_umax",
    "max_cpu_time", "max_hessian_perturbation", "max_wall_time", "min_hessian_perturbation",
    "mu_init", "mu_linear_decrease_factor", "mu_max", "mu_max_fact", "mu_min",
    "mu_superlinear_decrease_power", "mu_target", "mumps_dep_tol", "mumps_pivtol",
    "mumps_pivtolmax", "neg_curv_test_tol", "nlp_lower_bound_inf",
    "nlp_scaling_constr_target_gradient", "nlp_scaling_max_gradient",
    "nlp_scaling_min_value", "nlp_scaling_obj_target_gradient", "nlp_upper_bound_inf",
    "nu_inc", "nu_init", "obj_max_inc", "obj_scaling_factor",
    "pardiso_iter_dropping_factor", "pardiso_iter_dropping_schur",
    "pardiso_iter_inverse_norm_factor", "pardiso_iter_relative_tol", "perturb_dec_fact",
    "perturb_inc_fact", "perturb_inc_fact_first", "point_perturbation_radius",
    "print_frequency_time", "quality_function_section_qf_tol",
    "quality_function_section_sigma_tol", "recalc_y_feas_tol",
    "required_infeasibility_reduction", "residual_improvement_factor", "residual_ratio_max",
    "residual_ratio_singular", "resto_failure_feasibility_threshold",
    "resto_penalty_parameter", "resto_proximity_weight", "rho", "s_max", "s_phi", "s_theta",
    "sigma_max", "sigma_min", "slack_bound_frac", "slack_bound_push", "slack_move",
    "soft_resto_pderror_reduction_factor", "spral_gpu_perf_coeff",
    "spral_max_load_inbalance", "spral_min_gpu_work", "spral_small",
    "spral_small_subtree_threshold", "spral_u", "spral_umax", "tau_min", "theta_max_fact",
    "theta_min_fact", "tiny_step_tol", "tiny_step_y_tol", "tol", "warm_start_bound_frac",
    "warm_start_bound_push", "warm_start_mult_bound_push", "warm_start_mult_init_max",
    "warm_start_slack_bound_frac", "warm_start_slack_bound_push", "warm_start_target_mu",
])

function add_ipopt_option!(prob, key::AbstractString, val::AbstractString)
    return Ipopt.AddIpoptStrOption(prob, key, val)
end
function add_ipopt_option!(prob, key::AbstractString, val::Symbol)
    return Ipopt.AddIpoptStrOption(prob, key, String(val))
end
function add_ipopt_option!(prob, key::AbstractString, val::Bool)
    return Ipopt.AddIpoptStrOption(prob, key, val ? "yes" : "no")
end
function add_ipopt_option!(prob, key::AbstractString, val::Integer)
    if key in IPOPT_REAL_OPTIONS
        return Ipopt.AddIpoptNumOption(prob, key, Float64(val))
    end
    return Ipopt.AddIpoptIntOption(prob, key, Int(val))
end
function add_ipopt_option!(prob, key::AbstractString, val::AbstractFloat)
    return Ipopt.AddIpoptNumOption(prob, key, Float64(val))
end

# No Hessian callback is supplied, so Ipopt must use its L-BFGS approximation.
function check_ipopt_options(kwargs)
    for (key, val) in kwargs
        String(key) == "hessian_approximation" || continue
        # Ipopt compares string option values case-insensitively.
        if !(val isa Union{AbstractString,Symbol} && lowercase(String(val)) == "limited-memory")
            throw(
                ArgumentError(
                    "hessian_approximation=$(repr(val)) is not supported: solve_ipopt " *
                    "does not provide second derivatives, so it always uses " *
                    "hessian_approximation=\"limited-memory\"",
                ),
            )
        end
    end
    return nothing
end

"""
    solve_ipopt(fitness, x0, x_lb, x_ub, n_ceq, n_cineq; kwargs...)

Solve a constrained minimization problem with Ipopt.

`fitness(x)` must return `(obj, ceq, cineq)` with `ceq == 0` and `cineq <= 0`.

`fitness` may be any callable object, not only a `Function`.

Keyword `deriv` selects derivatives (`:forwarddiff` default, or `:finitediff`:
one-sided differences that step backward where a forward step would cross
`x_ub`). All other keyword arguments are passed through as Ipopt options. An integer
value for a real-valued option (e.g. `tol=1`) is converted to `Float64`. The
Hessian is always approximated with L-BFGS (`hessian_approximation =
"limited-memory"`); any other `hessian_approximation` throws an `ArgumentError`.

Bounds must satisfy `x_lb <= x_ub` and not be `NaN`, and `x0` must be
non-empty and finite; otherwise an `ArgumentError` is thrown.

If `fitness` throws, Ipopt is stopped and `solve_ipopt` rethrows the exception
after Ipopt returns, so no Julia exception unwinds through Ipopt's C++ code.
The rethrown exception's stacktrace starts in `solve_ipopt`; the backtrace from
inside `fitness` is logged at debug level (`ENV["JULIA_DEBUG"] = "NLPInterface"`).
An interrupt (Ctrl-C) during the solve throws `InterruptException`.

The lengths of `ceq` and `cineq` must equal `n_ceq` and `n_cineq` at every
evaluation; otherwise a `DimensionMismatch` is thrown. `x0` is not modified.

Unlike Ipopt's own default, `bound_relax_factor` is `0.0`, so `fitness` is only
evaluated inside `[x_lb, x_ub]` (Ipopt would otherwise step up to ~1e-8 outside
them, where `fitness` may be undefined). Pass `bound_relax_factor` to override.

`fitness` receives a copy of the solver's iterate, so modifying `x` does not
affect the solve. It must not keep a reference to `x`: the array is reused
between calls (copy it to keep it).

Returns `(xopt, fopt, info)`. `info` is a symbol from
`NLPInterface.IPOPT_RETURN_STATUS`.
"""
function solve_ipopt(
    fitness,
    x0::AbstractVector,
    x_lb::AbstractVector,
    x_ub::AbstractVector,
    n_ceq::Integer,
    n_cineq::Integer;
    deriv::Symbol=:forwarddiff,
    kwargs...,
)
    check_solver_inputs(x0, x_lb, x_ub, n_ceq, n_cineq, deriv)
    check_ipopt_options(kwargs)
    n = length(x0)

    x0 = convert(Vector{Float64}, x0)
    x_L = convert(Vector{Float64}, x_lb)
    x_U = convert(Vector{Float64}, x_ub)
    m = Int(n_ceq) + Int(n_cineq)
    g_L = vcat(zeros(Float64, n_ceq), fill(-Inf, n_cineq))
    g_U = zeros(Float64, m)

    # Ipopt asks for f, g, grad f and the Jacobian of g in separate callbacks.
    # Cache the residual and its Jacobian separately, each keyed on the `x` it was
    # computed at, so that trial points that only need f or g (line search) do
    # not pay for a Jacobian, and no point is evaluated twice.
    evaluator = ResidualEvaluator(fitness, n, n_ceq, n_cineq, deriv; x_lb=x_L, x_ub=x_U)
    x_r = zeros(Float64, n)
    x_J = zeros(Float64, n)
    r_valid = Ref(false)
    J_valid = Ref(false)
    r_cache = zeros(Float64, 1 + m)
    J_cache = zeros(Float64, 1 + m, n)

    function update_value!(x)
        r_valid[] && isequal(x, x_r) && return
        r_valid[] = false  # stays invalid if `fitness` throws
        residual_value!(evaluator, r_cache, x)
        x_r .= x
        r_valid[] = true
        return
    end

    function update_jacobian!(x)
        J_valid[] && isequal(x, x_J) && return
        have_r = r_valid[] && isequal(x, x_r)
        J_valid[] = false
        have_r || (r_valid[] = false)
        residual_value_and_jacobian!(evaluator, r_cache, J_cache, x, have_r)
        x_r .= x
        x_J .= x
        r_valid[] = true
        J_valid[] = true
        return
    end

    # Exceptions must not unwind through Ipopt's C++ frames. Each callback
    # catches them, keeps the first one, and hands Ipopt NaN; the intermediate
    # callback then stops Ipopt, and the exception is rethrown after
    # `IpoptSolve` returns. After the first error `fitness` is not called again.
    callback_error = Ref{Any}(nothing)
    callback_backtrace = Ref{Any}(nothing)
    function record_error!(err)  # call only inside `catch`
        if callback_error[] === nothing
            callback_error[] = err
            callback_backtrace[] = catch_backtrace()
        end
        return nothing
    end

    function eval_f(x)
        callback_error[] === nothing || return NaN
        try
            update_value!(x)
            return r_cache[1]
        catch err
            record_error!(err)
            return NaN
        end
    end

    function eval_g(x, g)
        m > 0 || return
        if callback_error[] === nothing
            try
                update_value!(x)
                g .= view(r_cache, 2:(1 + m))
                return
            catch err
                record_error!(err)
            end
        end
        g .= NaN
        return
    end

    function eval_grad_f(x, grad_f)
        if callback_error[] === nothing
            try
                update_jacobian!(x)
                grad_f .= view(J_cache, 1, :)
                return
            catch err
                record_error!(err)
            end
        end
        grad_f .= NaN
        return
    end

    function eval_jac_g(x, rows, cols, values)
        if values === nothing
            # Do not access `x` here; Ipopt.jl may pass an undefined object.
            k = 1
            for i in 1:m
                for j in 1:n
                    rows[k] = i
                    cols[k] = j
                    k += 1
                end
            end
            return
        end
        if callback_error[] === nothing
            try
                update_jacobian!(x)
                k = 1
                for i in 1:m
                    for j in 1:n
                        values[k] = J_cache[i + 1, j]
                        k += 1
                    end
                end
                return
            catch err
                record_error!(err)
            end
        end
        values .= NaN
        return
    end

    # Returning `false` makes Ipopt stop with `User_Requested_Stop`.
    intermediate(args...) = callback_error[] === nothing

    nele_jac = m * n
    nele_hess = 0
    prob = Ipopt.CreateIpoptProblem(
        n,
        x_L,
        x_U,
        m,
        g_L,
        g_U,
        nele_jac,
        nele_hess,
        eval_f,
        eval_g,
        eval_grad_f,
        eval_jac_g,
        nothing,
    )

    for (key, val) in kwargs
        add_ipopt_option!(prob, String(key), val)
    end
    # Ipopt relaxes the bounds by `bound_relax_factor` (default 1e-8) and would
    # then call `fitness` slightly outside [x_lb, x_ub]. Keep it inside, unless
    # the caller chose a value.
    haskey(kwargs, :bound_relax_factor) ||
        Ipopt.AddIpoptNumOption(prob, "bound_relax_factor", 0.0)
    # Always use an L-BFGS Hessian; we do not supply eval_h.
    Ipopt.AddIpoptStrOption(prob, "hessian_approximation", "limited-memory")
    Ipopt.SetIntermediateCallback(prob, intermediate)

    prob.x = copy(x0)
    status = Ipopt.IpoptSolve(prob)
    err = callback_error[]
    if err !== nothing
        # `throw` cannot reuse the original backtrace; it is kept for debugging.
        @debug "fitness threw inside solve_ipopt; backtrace at the throw site:" exception = (err, callback_backtrace[])
        throw(err)
    end
    if status == 5
        # Our intermediate callback never asks to stop without an error, so this
        # stop comes from Ctrl-C: Ipopt.jl catches the `InterruptException` in
        # its intermediate-callback wrapper and returns "stop" to Ipopt.
        throw(InterruptException())
    end
    info = get(IPOPT_RETURN_STATUS, status, :Unknown)
    return copy(prob.x), prob.obj_val, info
end
