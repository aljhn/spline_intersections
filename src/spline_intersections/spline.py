import numpy as np


type SplineCoeffs = np.ndarray[tuple[int, int], np.dtype[np.float64]]


def spline_get(
    t0: float,
    p0: np.ndarray[tuple[int], np.dtype[np.float64]],
    v0: np.ndarray[tuple[int], np.dtype[np.float64]],
    t1: float,
    p1: np.ndarray[tuple[int], np.dtype[np.float64]],
    v1: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> SplineCoeffs:
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
