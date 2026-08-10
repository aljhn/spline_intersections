from bisect import bisect_right

import numpy as np
from dataclasses import dataclass

from .spline import spline_get_cubic, spline_get_linear
from .polynomial import poly_eval, Polynomial


@dataclass
class State:
    t: float
    p: np.ndarray
    v: np.ndarray


@dataclass
class Segment:
    t: float
    coeffs: np.ndarray


class Trajectory:
    def __init__(self):
        self.states: list[State] = []
        self.segments: list[Segment] = []

    def add_state(self, state: State) -> None:
        self.states.append(state)

        if len(self.states) == 1:
            t = self.states[-1].t
            p = self.states[-1].p
            v = self.states[-1].v

            coeffs = spline_get_linear(t, p, v)
            segment = Segment(t, coeffs)
            self.segments.append(segment)

        elif len(self.states) == 2 and len(self.segments) == 1:
            self.segments.clear()

        if len(self.states) >= 2:
            t0 = self.states[-2].t
            p0 = self.states[-2].p
            v0 = self.states[-2].v

            t1 = self.states[-1].t
            p1 = self.states[-1].p
            v1 = self.states[-1].v

            coeffs = spline_get_cubic(t0, p0, v0, t1, p1, v1)
            segment = Segment(t0, coeffs)
            self.segments.append(segment)

    def eval(self, t: float) -> float | np.ndarray:
        if len(self.segments) == 0:
            return 0.0

        elif len(self.segments) == 1:
            return poly_eval(self.segments[0].coeffs, t)

        else:
            if t < self.segments[1].t:
                return poly_eval(self.segments[0].coeffs, t)

            i = 1
            while i < len(self.segments) - 1 and t < self.segments[i + 1].t:
                i += 1

            return poly_eval(self.segments[i].coeffs, t)

    def __call__(self, t: float) -> float | np.ndarray:
        return self.eval(t)

    def find_segment_index(self, t: float) -> int:
        low = 0
        high = len(self.segments)

        while low < high:
            mid = (low + high) // 2
            if self.segments[mid].t <= t:
                low = mid + 1
            else:
                high = mid

        return max(low - 1, 0)

    def intersect(self, other: Trajectory, threshold: float) -> list[float]:
        if len(self.segments) == 0 or len(other.segments) == 0:
            return []

        t_all_self = []
        for state in self.states:
            t_all_self.append(state.t)

        t_all_other = []
        for state in other.states:
            t_all_other.append(state.t)

        t_union = t_all_self + t_all_other
        t_union.sort()

        t_all = []
        t_all.append(-np.inf)
        for t in t_union:
            if len(t_all) == 0 or not np.isclose(t, t_all[-1]):
                t_all.append(t)
        t_all.append(np.inf)

        t_intersections = []

        for i in range(len(t_all) - 1):
            t0 = t_all[i + 0]
            t1 = t_all[i + 1]

            coeffs0_index = self.find_segment_index(t0)
            coeffs0 = self.segments[coeffs0_index].coeffs
 
            coeffs1_index = other.find_segment_index(t1)
            coeffs1 = other.segments[coeffs1_index].coeffs

            coeffs_diff = coeffs0 - coeffs1
            poly_x = Polynomial(coeffs_diff[0, :])
            poly_y = Polynomial(coeffs_diff[1, :])
            poly_z = Polynomial(coeffs_diff[2, :])
            poly_norm_squared = (
                (poly_x * poly_x) + (poly_y * poly_y) + (poly_z * poly_z)
            )

            poly_norm_squared_derivative = poly_norm_squared.derivative()
            roots = poly_norm_squared_derivative.get_real_roots(t_min=t0, t_max=t1)
            roots = sorted([t0] + roots + [t1])
            for r in roots:
                f = poly_norm_squared.eval(r)

                if f < threshold:
                    t_intersections.append(r) 

        t_intersections.sort()

        t_unique = []
        for t in t_intersections:
            if len(t_unique) == 0 or not np.isclose(t, t_unique[-1]):
                t_unique.append(t)

        return t_unique

