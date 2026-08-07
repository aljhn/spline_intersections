import numpy as np


# # First dimension = which polynomial it is
# # Second dimension = which coefficient of that it is
# type PolyCoeffs = np.ndarray[tuple[int, int], np.dtype[np.float64]]
#
#
# def poly_derivative(coeffs: PolyCoeffs) -> PolyCoeffs:
#     degree = coeffs.shape[1] - 1
#
#     if degree <= 0:
#         return np.zeros_like(coeffs)
#
#     powers = np.arange(degree, 0, -1)
#     return coeffs[:, :-1] * powers
#
#
# def poly_eval(
#     coeffs: PolyCoeffs, t: npt.NDArray[np.float64]
# ) -> npt.NDArray[np.float64]:
#     t = np.asarray(t).reshape(-1)
#     output = coeffs[:, 0, None]
#     for i in range(1, coeffs.shape[1]):
#         output = output * t + coeffs[:, i, None]
#     return output


type PolyCoeffs = np.ndarray[tuple[int], np.dtype[np.float64]]


def poly_derivative(coeffs: PolyCoeffs) -> PolyCoeffs:
    degree = coeffs.shape[0] - 1

    if degree <= 0:
        return np.zeros_like(coeffs)

    powers = np.arange(degree, 0, -1)
    return coeffs[:-1] * powers


def poly_eval(coeffs: PolyCoeffs, t: float) -> float:
    output = coeffs[0]
    for i in range(1, coeffs.shape[0]):
        output = output * t + coeffs[i]
    return output


def poly_cauchy_bound(coeffs: PolyCoeffs) -> float:
    degree = coeffs.shape[0] - 1
    if degree <= 1:
        return 0.0

    if np.abs(coeffs[0]) == 0.0:
        return 0.0

    alpha = 1.0 + np.max(np.abs(coeffs[1:] / coeffs[0]))
    return alpha


def poly_newton(
    coeffs: PolyCoeffs,
    d_coeffs: PolyCoeffs,
    t0: float,
    max_iterations: int = 10000,
) -> float:
    t = t0
    for _ in range(max_iterations):
        f = poly_eval(coeffs, t)
        if np.isclose(f, 0.0):
            break

        df = poly_eval(d_coeffs, t)
        if np.isclose(df, 0.0):
            break

        t_next = t - f / df
        if np.isclose(t_next, t):
            t = t_next
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
            return [0.0, r]

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
            return [r1, r2]

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
                r = poly_newton(coeffs, d_coeffs, 0.5 * (r0 + r1))
                roots.append(r)

        roots.sort()

        unique = []
        for r in roots:
            if len(unique) == 0 or not np.isclose(r, unique[-1]):
                unique.append(r)

        return unique
