import numpy as np
import numpy.typing as npt


def derivative(coeffs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    degree = coeffs.shape[1] - 1

    if degree == 0:
        return np.zeros_like(coeffs)

    powers = np.arange(degree, 0, -1)
    return coeffs[:, :-1] * powers


def get_spline(
    t0: float,
    p0: npt.NDArray[np.float64],
    v0: npt.NDArray[np.float64],
    t1: float,
    p1: npt.NDArray[np.float64],
    v1: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    t0_2 = t0 * t0
    t0_3 = t0_2 * t0

    t1_2 = t1 * t1
    t1_3 = t1_2 * t1

    M = np.array(
        [
            [t0_3, t0_2, t0, 1.0],
            [t1_3, t1_2, t1, 1.0],
            [3.0 * t0_2, 2.0 * t0, 1.0, 0.0],
            [3.0 * t1_2, 2.0 * t1, 1.0, 0.0],
        ]
    )

    B = np.stack([p0, p1, v0, v1], axis=0)
    coeffs = np.linalg.solve(M, B).T

    return coeffs


def spline_eval(
    coeffs: npt.NDArray[np.float64], t: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    t = np.asarray(t).reshape(-1)
    output = coeffs[:, 0, None]
    for i in range(1, coeffs.shape[1]):
        output = output * t + coeffs[:, i, None]
    return output
