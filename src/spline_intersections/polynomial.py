import numpy as np
import numpy.typing as npt


def poly_derivative(coeffs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    degree = coeffs.shape[1] - 1

    if degree <= 0:
        return np.zeros_like(coeffs)

    powers = np.arange(degree, 0, -1)
    return coeffs[:, :-1] * powers


def poly_eval(
    coeffs: npt.NDArray[np.float64], t: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    t = np.asarray(t).reshape(-1)
    output = coeffs[:, 0, None]
    for i in range(1, coeffs.shape[1]):
        output = output * t + coeffs[:, i, None]
    return output


def poly_real_roots(coeffs: npt.NDArray[np.float64]) -> list[float]:
    eps = 1e-8

    while np.abs(coeffs[0]) < eps:
        coeffs = coeffs[1:]

    degree = coeffs.shape[1] - 1

    if degree <= 0:
        return []

    elif degree == 1:
        a, b = coeffs

        return [-b / a]

    elif degree == 2:
        a, b, c = coeffs

        if np.abs(b) < eps and np.abs(c) < eps:
            return [0.0]

        elif np.abs(b) < eps and np.abs(c) > eps:
            x2 = -c / a

            if x2 < 0:
                return []

            r = np.sqrt(x2)
            return [-r, r]

        elif np.abs(b) > eps and np.abs(c) < eps:
            r = -b / a
            return [0.0, r]

        D = b * b - 4.0 * a * c
        if D < -eps:
            return []

        elif np.abs(D) < eps:
            return [-b / (2.0 * a)]

        else:
            q = -0.5 * (b + np.copysign(np.sqrt(D), b))

            r1 = q / a
            r2 = c / q
            return [r1, r2]

    else:
        return []

