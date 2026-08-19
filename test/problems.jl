"""Database of sample problems"""

problems = Dict(
    "rosenbrock" => (
        x0=[-0.5, 1.0],
        x_lb=[-1.5, -1.5],
        x_ub=[1.5, 1.5],
        n_ceq=0,
        n_cineq=1,
        fitness=(x) -> (
            (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2,
            [],
            [x[1]^2 + x[2]^2 - 2],
        ),
        xopt=[1.0,1.0],
        fopt=0.0,
    ),
    # Mishra's Bird, constrained: (x+5)^2 + (y+5)^2 < 25
    "mishra_bird" => (
        x0=[-3.0, -1.5],
        x_lb=[-10.0, -6.5],
        x_ub=[0.0, 0.0],
        n_ceq=0,
        n_cineq=1,
        fitness=(x) -> (
            sin(x[2]) * exp((1 - cos(x[1]))^2) +
            cos(x[1]) * exp((1 - sin(x[2]))^2) +
            (x[1] - x[2])^2,
            [],
            [(x[1] + 5)^2 + (x[2] + 5)^2 - 25],
        ),
        xopt=[-3.1302468, -1.5821422],
        fopt=-106.7645367,
    ),
    # Townsend (modified): polar heart constraint, t = atan2(x, y)
    "townsend" => (
        x0=[1.5, 1.0],
        x_lb=[-2.25, -2.5],
        x_ub=[2.25, 1.75],
        n_ceq=0,
        n_cineq=1,
        fitness=function (x)
            t = atan(x[1], x[2])
            r2max =
                (2 * cos(t) - 0.5 * cos(2t) - 0.25 * cos(3t) - 0.125 * cos(4t))^2 +
                (2 * sin(t))^2
            return (
                -cos((x[1] - 0.1) * x[2])^2 - x[1] * sin(3 * x[1] + x[2]),
                [],
                [x[1]^2 + x[2]^2 - r2max],
            )
        end,
        xopt=[2.0052938, 1.1944509],
        fopt=-2.0239884,
    ),
    # Test function from scp
    "crawling" => (
        x0=[1.5, 1.5],
        x_lb=[-2.0, -2.0],
        x_ub=[2.0, 2.0],
        n_ceq=1,
        n_cineq=1,
        fitness=function (x)
            return (
                x[1] + x[2],
                [x[2] - x[1]^4 - 2x[1]^3 + 1.2x[1]^2 + 2x[1]],
                [-x[2] - (4/3)*x[1] - 2/3],
            )
        end,
        xopt=[0.5287823, -1.0192089],
        fopt=-0.4904266,
    ),
)
