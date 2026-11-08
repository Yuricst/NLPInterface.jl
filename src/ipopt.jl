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
    return Ipopt.AddIpoptIntOption(prob, key, Int(val))
end
function add_ipopt_option!(prob, key::AbstractString, val::AbstractFloat)
    return Ipopt.AddIpoptNumOption(prob, key, Float64(val))
end

"""
    solve_ipopt(fitness, x0, x_lb, x_ub, n_ceq, n_cineq; kwargs...)

Solve a constrained minimization problem with Ipopt.

`fitness(x)` must return `(obj, ceq, cineq)` with `ceq == 0` and `cineq <= 0`.

Keyword `deriv` selects derivatives (`:forwarddiff` default, or `:finitediff`).
All other keyword arguments are passed through as Ipopt options.

The lengths of `ceq` and `cineq` must equal `n_ceq` and `n_cineq` at every
evaluation; otherwise a `DimensionMismatch` is thrown. `x0` is not modified.

`fitness` must not keep a reference to `x`: the array is reused between calls
(copy it to keep it).

Returns `(xopt, fopt, info)`. `info` is a symbol from
`NLPInterface.IPOPT_RETURN_STATUS`.
"""
function solve_ipopt(
    fitness::Function,
    x0::AbstractVector,
    x_lb::AbstractVector,
    x_ub::AbstractVector,
    n_ceq::Integer,
    n_cineq::Integer;
    deriv::Symbol=:forwarddiff,
    kwargs...,
)
    n = length(x0)
    if !(n == length(x_lb) == length(x_ub))
        throw(ArgumentError("x0, x_lb, and x_ub must have the same length"))
    end
    if n_ceq < 0 || n_cineq < 0
        throw(ArgumentError("n_ceq and n_cineq must be non-negative"))
    end
    if deriv !== :forwarddiff && deriv !== :finitediff
        throw(ArgumentError("Unknown deriv=$deriv; use :forwarddiff or :finitediff"))
    end

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
    evaluator = ResidualEvaluator(fitness, n, n_ceq, n_cineq, deriv)
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

    function eval_f(x)
        update_value!(x)
        return r_cache[1]
    end

    function eval_g(x, g)
        update_value!(x)
        if m > 0
            g .= view(r_cache, 2:(1 + m))
        end
        return
    end

    function eval_grad_f(x, grad_f)
        update_jacobian!(x)
        grad_f .= view(J_cache, 1, :)
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
        else
            update_jacobian!(x)
            k = 1
            for i in 1:m
                for j in 1:n
                    values[k] = J_cache[i + 1, j]
                    k += 1
                end
            end
        end
        return
    end

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
    # Always use an L-BFGS Hessian; we do not supply eval_h.
    Ipopt.AddIpoptStrOption(prob, "hessian_approximation", "limited-memory")

    prob.x = copy(x0)
    status = Ipopt.IpoptSolve(prob)
    info = get(IPOPT_RETURN_STATUS, status, :Unknown)
    return copy(prob.x), prob.obj_val, info
end
