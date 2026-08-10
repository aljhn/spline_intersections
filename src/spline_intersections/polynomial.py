import numpy as np


def poly_derivative(coeffs: np.ndarray) -> np.ndarray:
    degree = coeffs.shape[-1] - 1

    if degree <= 0:
        return np.zeros_like(coeffs)

    powers = np.arange(degree, 0, -1)
    return coeffs[..., :-1] * powers


def poly_eval(coeffs: np.ndarray, t: float | np.ndarray) -> float | np.ndarray:
    t = np.asarray(t)

    if t.ndim == 0:
        output = coeffs[..., 0]
        for i in range(1, coeffs.shape[-1]):
            output = output * t + coeffs[..., i]

        if coeffs.ndim == 1:
            return float(output)

        return output

    output = coeffs[..., 0, None]
    for i in range(1, coeffs.shape[-1]):
        output = output * t + coeffs[..., i, None]

    return output


type PolyCoeffs = np.ndarray[tuple[int], np.dtype[np.float64]]


def poly_cauchy_bound(coeffs: PolyCoeffs) -> float:
    degree = coeffs.shape[0] - 1
    if degree < 1:
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


def poly_real_roots(coeffs: PolyCoeffs, t_min: float = -np.inf, t_max: float = np.inf) -> list[float]:
    while coeffs.shape[0] > 0 and np.isclose(coeffs[0], 0.0):
        coeffs = coeffs[1:]

    degree = coeffs.shape[0] - 1

    roots = []

    if degree <= 0:
        return roots

    a = poly_cauchy_bound(coeffs)
    if t_min == -np.inf:
        t_min = -a
    if t_max == np.inf:
        t_max = a

    if degree == 1:
        a, b = coeffs

        r = -b / a
        if r >= t_min and r <= t_max:
            roots.append(r)

        return roots

    elif degree == 2:
        a, b, c = coeffs

        if np.isclose(b, 0.0) and np.isclose(c, 0.0):
            r = 0.0
            if r >= t_min and r <= t_max:
                roots.append(r)
            return roots

        elif np.isclose(b, 0.0) and not np.isclose(c, 0.0):
            x2 = -c / a

            if x2 < 0:
                return roots

            r1 = np.sqrt(x2)
            r2 = -r1
            if r2 >= t_min and r2 <= t_max:
                roots.append(r2)
            if r1 >= t_min and r1 <= t_max:
                roots.append(r1)
            return roots

        elif not np.isclose(b, 0.0) and np.isclose(c, 0.0):
            r1 = -b / a
            r2 = 0.0
            if r2 >= t_min and r2 <= t_max:
                roots.append(r2)
            if r1 >= t_min and r1 <= t_max:
                roots.append(r1)
            return roots

        D = b * b - 4.0 * a * c
        if D < 0.0:
            return roots

        elif np.isclose(D, 0.0):
            r = -b / (2.0 * a)
            if r >= t_min and r <= t_max:
                roots.append(r)
            return roots

        else:
            q = -0.5 * (b + np.copysign(np.sqrt(D), b))

            r1 = q / a

            if r1 >= t_min and r1 <= t_max:
                roots.append(r1)

            if not np.isclose(q, 0.0):
                r2 = c / q
                if r2 >= t_min and r2 <= t_max:
                    roots.append(r2)

            return sorted(roots)

    else:
        roots = []

        f_min = poly_eval(coeffs, t_min)
        f_max = poly_eval(coeffs, t_max)

        if np.isclose(f_min, 0.0):
            roots.append(t_min)

        if np.isclose(f_max, 0.0):
            roots.append(t_max)

        d_coeffs = poly_derivative(coeffs)
        d_roots = poly_real_roots(d_coeffs, t_min, t_max)

        for r in d_roots:
            f = poly_eval(coeffs, r)
            if np.isclose(f, 0.0):
                roots.append(r)

        search_points = sorted([t_min] + d_roots + [t_max])
        for i in range(len(search_points) - 1):
            r0 = search_points[i + 0]
            r1 = search_points[i + 1]

            f0 = poly_eval(coeffs, r0)
            f1 = poly_eval(coeffs, r1)

            if (f0 > 0 and f1 < 0) or (f0 < 0 and f1 > 0):
                r = poly_newton_bisect(coeffs, d_coeffs, r0, r1)
                roots.append(r)

        roots.sort()

        roots_unique = []
        for r in roots:
            if len(roots_unique) == 0 or not np.isclose(r, roots_unique[-1]):
                roots_unique.append(r)

        return roots_unique


class Polynomial:
    def __init__(self, coeffs: np.ndarray):
        self.coeffs = coeffs

        while self.coeffs.shape[0] >= 2 and np.isclose(self.coeffs[0], 0.0):
            self.coeffs = self.coeffs[1:]

        self.degree = np.max(self.coeffs.shape[0] - 1, 0)

    def eval(self, t: np.ndarray | float) -> np.ndarray | float:
        return poly_eval(self.coeffs, t)

    def derivative(self) -> Polynomial:
        return Polynomial(poly_derivative(self.coeffs))

    def get_real_roots(self, t_min: float = -np.inf, t_max: float = np.inf) -> list[float]:
        alpha = poly_cauchy_bound(self.coeffs)
        if t_min == -np.inf:
            t_min = -alpha
        if t_max == np.inf:
            t_max = alpha
        return poly_real_roots(self.coeffs, t_min, t_max)

    def __call__(self, t: np.ndarray | float) -> np.ndarray | float:
        return self.eval(t)

    def __add__(self, other: Polynomial) -> Polynomial:
        coeffs_self = self.coeffs
        coeffs_other = other.coeffs

        if self.degree > other.degree:
            coeffs_other = np.pad(coeffs_other, (self.degree - other.degree, 0))
        elif self.degree < other.degree:
            coeffs_self = np.pad(coeffs_self, (other.degree - self.degree, 0))

        coeffs_added = coeffs_self + coeffs_other
        return Polynomial(coeffs_added)

    def __neg__(self) -> Polynomial:
        return Polynomial(-self.coeffs)

    def __sub__(self, other: Polynomial) -> Polynomial:
        return self + (-other)

    def __mul__(self, other: Polynomial) -> Polynomial:
        coeffs_multiplied = np.convolve(self.coeffs, other.coeffs)
        return Polynomial(coeffs_multiplied)
