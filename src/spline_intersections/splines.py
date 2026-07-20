import numpy as np
import numpy.typing as npt


def derivative(coeffs: npt.NDArray) -> npt.NDArray:
    n = coeffs.shape[0] - 1
    if n == 0:
        return np.array([0.0])
    return coeffs[:-1] * np.arange(n, 0, -1)


def get_spline(
    t0: float,
    p0: npt.NDArray,
    v0: npt.NDArray,
    t1: float,
    p1: npt.NDArray,
    v1: npt.NDArray,
) -> npt.NDArray:
    n = p0.shape[0]
    coeffs = np.zeros((n, 4))

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

    for i in range(n):
        b = np.array([p0[i], p1[i], v0[i], v1[i]])
        c, _, _, _ = np.linalg.lstsq(M, b)
        coeffs[i, :] = c

    return coeffs


def spline_eval(coeffs: npt.NDArray, t: npt.NDArray) -> npt.NDArray:
    n = coeffs.shape[0]
    x = np.zeros((n, t.shape[0]))
    for i in range(n):
        x[i, :] = np.polyval(coeffs[i, :], t)
    return x
