using SparseArrays

"""
A scalable trajectory problem with a sparse Jacobian, used by the sparse
Jacobian tests and by benchmark/sparse_jacobian.jl.

Minimum-energy transfer of a planar double integrator around a circular
obstacle, by trapezoidal collocation on `N` nodes with fixed final time.
Node `k` holds the state `(px, py, vx, vy, e)` and the control `(ux, uy)`,
where `e' = (ux^2 + uy^2)/2` accumulates the energy. The objective is the
final energy `e_N` (Mayer form), so the objective row has one entry and the
column coloring does not grow with `N`.

    x = [z_1; z_2; ...; z_N],   z_k = (px, py, vx, vy, e, ux, uy)

- `ceq`: 5 (N - 1) collocation defects, 5 initial conditions, 4 final conditions;
- `cineq`: one obstacle constraint per node, `R^2 - |p_k - c|^2 <= 0`.
"""
function transcription_problem(N::Integer; tf=2.0, center=(0.5, 0.15), radius=0.2)
    nz = 7
    n = nz * N
    h = tf / (N - 1)
    n_ceq = 5 * (N - 1) + 5 + 4
    n_cineq = N
    p0 = (0.0, 0.0, 0.0, 0.0, 0.0)   # px, py, vx, vy, e at t = 0
    pf = (1.0, 0.0, 0.0, 0.0)        # px, py, vx, vy at t = tf
    rhs(z) = (z[3], z[4], z[6], z[7], (z[6]^2 + z[7]^2) / 2)
    function fitness(x)
        T = eltype(x)
        z(k) = view(x, (nz * (k - 1) + 1):(nz * k))
        ceq = Vector{T}(undef, n_ceq)
        i = 0
        for k in 1:(N - 1)
            zk, zk1 = z(k), z(k + 1)
            fk, fk1 = rhs(zk), rhs(zk1)
            for s in 1:5
                ceq[i += 1] = zk1[s] - zk[s] - h / 2 * (fk[s] + fk1[s])
            end
        end
        z1, zN = z(1), z(N)
        for s in 1:5
            ceq[i += 1] = z1[s] - p0[s]
        end
        for s in 1:4
            ceq[i += 1] = zN[s] - pf[s]
        end
        cineq = [radius^2 - (z(k)[1] - center[1])^2 - (z(k)[2] - center[2])^2 for k in 1:N]
        return zN[5], ceq, cineq
    end
    # Straight-line initial guess at constant speed, zero control.
    x0 = zeros(n)
    for k in 1:N
        t = (k - 1) / (N - 1)
        x0[nz * (k - 1) + 1] = t
        x0[nz * (k - 1) + 3] = 1 / tf
    end
    x_lb = repeat([-2.0, -2.0, -5.0, -5.0, 0.0, -10.0, -10.0], N)
    x_ub = repeat([2.0, 2.0, 5.0, 5.0, 100.0, 10.0, 10.0], N)
    # Hand-made pattern of the packed Jacobian [obj; ceq; cineq].
    I = Int[]
    J = Int[]
    col(k, s) = nz * (k - 1) + s
    push!(I, 1); push!(J, col(N, 5))                       # objective: e_N
    row = 1
    deps = ((1, 3), (2, 4), (3, 6), (4, 7), (5, 6, 7))     # state s depends on
    for k in 1:(N - 1), s in 1:5
        row += 1
        for kk in (k, k + 1)
            push!(I, row); push!(J, col(kk, s))
            for d in deps[s]
                d == s && continue
                push!(I, row); push!(J, col(kk, d))
            end
        end
    end
    for s in 1:5
        row += 1; push!(I, row); push!(J, col(1, s))
    end
    for s in 1:4
        row += 1; push!(I, row); push!(J, col(N, s))
    end
    for k in 1:N, s in 1:2
        push!(I, row + k); push!(J, col(k, s))
    end
    pattern = sparse(I, J, true, 1 + n_ceq + n_cineq, n)
    return (; fitness, x0, x_lb, x_ub, n_ceq, n_cineq, pattern, N, n)
end
