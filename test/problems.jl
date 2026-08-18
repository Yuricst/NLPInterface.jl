"""Database of sample problems"""

problems = {
    "rosenbrock" => (
        x0 = [-0.5, 1.0],
        x_lb = [-1.5, 1.5],
        x_ub = [-1.5, 1.5],
        n_ceq = 0,
        n_cineq = 1,
        fitness = (x) -> (
            (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2,
            [],
            [x[1]^2 + x[2]^2 - 2],
        )
    ),
}