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
    # ------------------------------------------------------------------
    # Additional 2-D problems
    # ------------------------------------------------------------------
    # Rosenbrock constrained by a cubic and a line. Many starts (e.g. x0 = (-1, 2))
    # end at a local minimum f ≈ 0.99889 near the origin on the cubic.
    "rosenbrock_cubic_line" => (
        x0=[0.5, 1.0],
        x_lb=[-1.5, -0.5],
        x_ub=[1.5, 2.5],
        n_ceq=0,
        n_cineq=2,
        fitness=(x) -> (
            (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2,
            [],
            [(x[1] - 1)^3 - x[2] + 1, x[1] + x[2] - 2],
        ),
        xopt=[1.0, 1.0],
        fopt=0.0,
    ),
    # Gomez & Levy (modified): six-hump camel with a periodic constraint whose
    # feasible set is many disconnected islands.
    "gomez_levy" => (
        x0=[0.5, -0.3],
        x_lb=[-1.0, -1.0],
        x_ub=[0.75, 1.0],
        n_ceq=0,
        n_cineq=1,
        fitness=(x) -> (
            4x[1]^2 - 2.1x[1]^4 + x[1]^6 / 3 + x[1] * x[2] - 4x[2]^2 + 4x[2]^4,
            [],
            [-sin(4π * x[1]) + 2 * sin(2π * x[2])^2 - 1.5],
        ),
        xopt=[0.0898420131, -0.7126564030],
        fopt=-1.0316284535,
    ),
    # Simionescu: flower-shaped feasible set r <= 1 + 0.2 cos(8θ). The minimum
    # at (-0.8485, 0.8485) is equivalent; x0 selects the 4th-quadrant one.
    "simionescu" => (
        x0=[0.5, -0.3],
        x_lb=[-1.25, -1.25],
        x_ub=[1.25, 1.25],
        n_ceq=0,
        n_cineq=1,
        fitness=(x) -> (
            0.1 * x[1] * x[2],
            [],
            [x[1]^2 + x[2]^2 - (1 + 0.2 * cos(8 * atan(x[1], x[2])))^2],
        ),
        xopt=[0.8485281374, -0.8485281374],
        fopt=-0.072,
    ),
    # Hock-Schittkowski 15: Rosenbrock with a hyperbola, x1 <= 0.5 (from HS);
    # the other bounds are added to make a finite box.
    "hs015" => (
        x0=[-2.0, 1.0],
        x_lb=[-2.5, -3.0],
        x_ub=[0.5, 3.0],
        n_ceq=0,
        n_cineq=2,
        fitness=(x) -> (
            100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2,
            [],
            [1 - x[1] * x[2], -x[1] - x[2]^2],
        ),
        xopt=[0.5, 2.0],
        fopt=306.5,
    ),
    # CEC 2006 g06: cubic objective on a thin crescent between two circles.
    # Box narrowed from [13, 100] x [0, 100]; it still contains the feasible set.
    "g06" => (
        x0=[15.0, 3.0],
        x_lb=[13.0, 0.0],
        x_ub=[16.0, 11.0],
        n_ceq=0,
        n_cineq=2,
        fitness=(x) -> (
            (x[1] - 10)^3 + (x[2] - 20)^3,
            [],
            [
                -(x[1] - 5)^2 - (x[2] - 5)^2 + 100,
                (x[1] - 6)^2 + (x[2] - 5)^2 - 82.81,
            ],
        ),
        xopt=[14.095, 0.8429607892],
        fopt=-6961.8138755802,
    ),
    # CEC 2006 g08: highly multimodal objective on a small lens-shaped set.
    # Box narrowed from [0, 10]^2 (avoids the 1/x1^3 pole at x1 = 0).
    "g08" => (
        x0=[1.3, 4.1],
        x_lb=[0.5, 2.0],
        x_ub=[2.5, 6.0],
        n_ceq=0,
        n_cineq=2,
        fitness=(x) -> (
            -sin(2π * x[1])^3 * sin(2π * x[2]) / (x[1]^3 * (x[1] + x[2])),
            [],
            [x[1]^2 - x[2] + 1, 1 - x[1] + (x[2] - 4)^2],
        ),
        xopt=[1.2279713526, 4.2453733661],
        fopt=-0.0958250414,
    ),
    # CEC 2006 g24: linear objective, two quartic constraints, disconnected
    # feasible set.
    "g24" => (
        x0=[2.0, 2.0],
        x_lb=[0.0, 0.0],
        x_ub=[3.0, 4.0],
        n_ceq=0,
        n_cineq=2,
        fitness=(x) -> (
            -x[1] - x[2],
            [],
            [
                -2x[1]^4 + 8x[1]^3 - 8x[1]^2 + x[2] - 2,
                -4x[1]^4 + 32x[1]^3 - 88x[1]^2 + 96x[1] + x[2] - 36,
            ],
        ),
        xopt=[2.3295201975, 3.1784930741],
        fopt=-5.5080132716,
    ),
    # ------------------------------------------------------------------
    # Higher-dimensional problems
    # ------------------------------------------------------------------
    # Hock-Schittkowski 71 (4-D), the Ipopt tutorial problem.
    "hs071" => (
        x0=[1.0, 5.0, 5.0, 1.0],
        x_lb=fill(1.0, 4),
        x_ub=fill(5.0, 4),
        n_ceq=1,
        n_cineq=1,
        fitness=(x) -> (
            x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3],
            [x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 - 40],
            [25 - x[1] * x[2] * x[3] * x[4]],
        ),
        xopt=[1.0, 4.7429996373, 3.8211499842, 1.3794082932],
        fopt=17.0140172892,
    ),
    # Hock-Schittkowski 78 (5-D): product objective, three nonlinear equalities.
    # HS has no bounds; the box is added and inactive.
    "hs078" => (
        x0=[-2.0, 1.5, 2.0, -1.0, -1.0],
        x_lb=fill(-3.0, 5),
        x_ub=fill(3.0, 5),
        n_ceq=3,
        n_cineq=0,
        fitness=(x) -> (
            x[1] * x[2] * x[3] * x[4] * x[5],
            [
                x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 + x[5]^2 - 10,
                x[2] * x[3] - 5 * x[4] * x[5],
                x[1]^3 + x[2]^3 + 1,
            ],
            [],
        ),
        xopt=[-1.7171435704, 1.5957096902, 1.8272457529, -0.7636430782, -0.7636430782],
        fopt=-2.9197004090,
    ),
    # Hartmann 6-D: multimodal, bound constraints only.
    "hartmann6" => (
        x0=fill(0.25, 6),
        x_lb=zeros(6),
        x_ub=ones(6),
        n_ceq=0,
        n_cineq=0,
        fitness=function (x)
            α = (1.0, 1.2, 3.0, 3.2)
            A = (
                (10.0, 3.0, 17.0, 3.5, 1.7, 8.0),
                (0.05, 10.0, 17.0, 0.1, 8.0, 14.0),
                (3.0, 3.5, 1.7, 10.0, 17.0, 8.0),
                (17.0, 8.0, 0.05, 10.0, 0.1, 14.0),
            )
            P = (
                (0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886),
                (0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991),
                (0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650),
                (0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381),
            )
            f = -sum(
                α[i] * exp(-sum(A[i][j] * (x[j] - P[i][j])^2 for j in 1:6)) for i in 1:4
            )
            return (f, [], [])
        end,
        xopt=[0.2016895110, 0.1500106918, 0.4768739742, 0.2753324305, 0.3116516166, 0.6573005341],
        fopt=-3.3223680114,
    ),
    # Hock-Schittkowski 100 (7-D), also CEC 2006 g09. HS has no bounds; the box
    # is added and inactive.
    "hs100" => (
        x0=[1.0, 2.0, 0.0, 4.0, 0.0, 1.0, 1.0],
        x_lb=fill(-10.0, 7),
        x_ub=fill(10.0, 7),
        n_ceq=0,
        n_cineq=4,
        fitness=(x) -> (
            (x[1] - 10)^2 + 5 * (x[2] - 12)^2 + x[3]^4 + 3 * (x[4] - 11)^2 +
            10 * x[5]^6 + 7 * x[6]^2 + x[7]^4 - 4 * x[6] * x[7] - 10 * x[6] - 8 * x[7],
            [],
            [
                2x[1]^2 + 3x[2]^4 + x[3] + 4x[4]^2 + 5x[5] - 127,
                7x[1] + 3x[2] + 10x[3]^2 + x[4] - x[5] - 282,
                23x[1] + x[2]^2 + 6x[6]^2 - 8x[7] - 196,
                4x[1]^2 + x[2]^2 - 3x[1] * x[2] + 2x[3]^2 + 5x[6] - 11x[7],
            ],
        ),
        xopt=[
            2.3304993729, 1.9513723729, -0.4775413924, 4.3657262337,
            -0.6244869705, 1.0381310186, 1.5942267116,
        ],
        fopt=680.6300573744,
    ),
    # Hock-Schittkowski 104 (8-D): signomial reactor design, objective bounded
    # by two of the constraints.
    "hs104" => (
        x0=[6.0, 3.0, 0.4, 0.2, 6.0, 6.0, 1.0, 0.5],
        x_lb=fill(0.1, 8),
        x_ub=fill(10.0, 8),
        n_ceq=0,
        n_cineq=6,
        fitness=function (x)
            f = 0.4 * x[1]^0.67 * x[7]^-0.67 + 0.4 * x[2]^0.67 * x[8]^-0.67 +
                10 - x[1] - x[2]
            return (
                f,
                [],
                [
                    0.0588 * x[5] * x[7] + 0.1 * x[1] - 1,
                    0.0588 * x[6] * x[8] + 0.1 * x[1] + 0.1 * x[2] - 1,
                    (4 * x[3] + 2 * x[3]^-0.71) / x[5] + 0.0588 * x[3]^-1.3 * x[7] - 1,
                    (4 * x[4] + 2 * x[4]^-0.71) / x[6] + 0.0588 * x[4]^-1.3 * x[8] - 1,
                    1 - f,
                    f - 4.2,
                ],
            )
        end,
        xopt=[
            6.4651140284, 2.2327086480, 0.6673974913, 0.5957564229,
            5.9326756779, 5.5272345650, 1.0133220089, 0.4006682291,
        ],
        fopt=3.9511634401,
    ),
    # Chained Rosenbrock (10-D) in a ball of radius sqrt(10); the optimum
    # lies on the ball's surface.
    "rosenbrock_10d" => (
        x0=repeat([-1.2, 1.0], 5),
        x_lb=fill(-2.0, 10),
        x_ub=fill(2.0, 10),
        n_ceq=0,
        n_cineq=1,
        fitness=(x) -> (
            sum(100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2 for i in 1:9),
            [],
            [sum(abs2, x) - 10],
        ),
        xopt=ones(10),
        fopt=0.0,
    ),
)
