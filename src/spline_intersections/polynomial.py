import numpy as np


def poly_derivative(coeffs: np.ndarray) -> np.ndarray:
    degree = coeffs.shape[-1] - 1

    if degree <= 0:
        return np.zeros_like(coeffs)

    powers = np.arange(degree, 0, -1)
    return coeffs[..., :-1] * powers


def poly_eval(coeffs: np.ndarray, t: np.ndarray | float) -> np.ndarray | float:
    is_float = isinstance(t, float)
    t = np.asarray(t)

    output = coeffs[..., 0, None]
    for i in range(1, coeffs.shape[-1]):
        output = output * t + coeffs[..., i, None]

    if is_float:
        output = float(output.squeeze())
    return output


type PolyCoeffs = np.ndarray[tuple[int], np.dtype[np.float64]]


def poly_cauchy_bound(coeffs: PolyCoeffs) -> float:
    degree = coeffs.shape[0] - 1
    if degree <= 1:
        return 0.0

    if np.isclose(coeffs[0], 0.0):
        return 0.0

    alpha = 1.0 + np.max(np.abs(coeffs[1:] / coeffs[0]))
    return alpha


def poly_newton_bisect(
    coeffs: PolyCoeffs,
    d_coeffs: PolyCoeffs,
    t0: float,
    t1: float,
    max_iterations: int = 10000,
) -> float:
    f0 = poly_eval(coeffs, t0)
    # f1 = poly_eval(coeffs, t1)

    t = 0.5 * (t0 + t1)
    for _ in range(max_iterations):
        f = poly_eval(coeffs, t)
        if np.isclose(f, 0.0):
            break

        df = poly_eval(d_coeffs, t)
        use_newton = not np.isclose(df, 0.0)
        if use_newton:
            t_next = t - f / df

            if t_next <= t0 or t_next >= t1:
                use_newton = False

        if not use_newton:
            t_next = 0.5 * (t0 + t1)

        f_next = poly_eval(coeffs, t_next)

        if np.sign(f0) != np.sign(f_next):
            t1 = t_next
            # f1 = f_next
        else:
            t0 = t_next
            f0 = f_next

        if np.isclose(t0 - t1, 0.0):
            t = 0.5 * (t0 + t1)
            break

        t = t_next
    return t


def poly_real_roots(coeffs: PolyCoeffs) -> list[float]:
    while np.isclose(coeffs[0], 0.0):
        coeffs = coeffs[1:]

    degree = coeffs.shape[0] - 1

    if degree <= 0:
        return []

    elif degree == 1:
        a, b = coeffs

        return [-b / a]

    elif degree == 2:
        a, b, c = coeffs

        if np.isclose(b, 0.0) and np.isclose(c, 0.0):
            return [0.0]

        elif np.isclose(b, 0.0) and not np.isclose(c, 0.0):
            x2 = -c / a

            if x2 < 0:
                return []

            r = np.sqrt(x2)
            return [-r, r]

        elif not np.isclose(b, 0.0) and np.isclose(c, 0.0):
            r = -b / a
            return sorted([0.0, r])

        D = b * b - 4.0 * a * c
        if D < 0.0:
            return []

        elif np.isclose(D, 0.0):
            return [-b / (2.0 * a)]

        else:
            q = -0.5 * (b + np.copysign(np.sqrt(D), b))

            r1 = q / a
            if np.isclose(q, 0.0):
                return [r1]

            r2 = c / q
            return sorted([r2, r1])

    else:
        roots = []

        d_coeffs = poly_derivative(coeffs)
        d_roots = poly_real_roots(d_coeffs)

        for r in d_roots:
            f = poly_eval(coeffs, r)
            if np.isclose(f, 0.0):
                roots.append(r)

        alpha = poly_cauchy_bound(coeffs)

        search_points = sorted([-alpha] + d_roots + [alpha])
        for i in range(len(search_points) - 1):
            r0 = search_points[i + 0]
            r1 = search_points[i + 1]

            f0 = poly_eval(coeffs, r0)
            f1 = poly_eval(coeffs, r1)

            if (f0 > 0 and f1 < 0) or (f0 < 0 and f1 > 0):
                r = poly_newton_bisect(coeffs, d_coeffs, r0, r1)
                roots.append(r)

        roots.sort()

        unique = []
        for r in roots:
            if len(unique) == 0 or not np.isclose(r, unique[-1]):
                unique.append(r)

        return unique


class Polynomial:
    def __init__(self, coeffs: np.ndarray):
        self.coeffs = coeffs

        while self.coeffs.shape[0] > 0 and np.isclose(self.coeffs[0], 0.0):
            self.coeffs = self.coeffs[1:]

        self.degree = np.max(self.coeffs.shape[0] - 1, 0)

    def eval(self, t: np.ndarray | float) -> np.ndarray | float:
        return poly_eval(self.coeffs, t)

    def derivative(self) -> Polynomial:
        return Polynomial(poly_derivative(self.coeffs))

    def get_real_roots(self) -> list[float]:
        return poly_real_roots(self.coeffs)

    def __call__(self, t: np.ndarray | float) -> np.ndarray | float:
        return self.eval(t)

    def __add__(self, other: Polynomial) -> Polynomial:
        coeffs_self = self.coeffs
        coeffs_other = other.coeffs

        if self.degree > other.degree:
            coeffs_other = np.concatenate([np.zeros((self.degree - other.degree,)), coeffs_other], axis=0)
        elif self.degree < other.degree:
            coeffs_self = np.concatenate([np.zeros((other.degree - self.degree,)), coeffs_self], axis=0)

        coeffs_added = coeffs_self + coeffs_other
        return Polynomial(coeffs_added)

    def __neg__(self) -> Polynomial:
        return Polynomial(-self.coeffs)

    def __sub__(self, other: Polynomial) -> Polynomial:
        return self + (-other)

    def __mul__(self, other: Polynomial) -> Polynomial:
        # degree_multiplied = self.degree + other.degree
        # coeffs_multiplied = np.zeros((degree_multiplied,))
        # for i in range(self.degree):
        #     for j in range(other.degree):
        #         pass

        coeffs_multiplied = np.convolve(self.coeffs, other.coeffs) # TODO ?
        return Polynomial(coeffs_multiplied)
