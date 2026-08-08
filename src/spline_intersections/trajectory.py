import numpy as np
from dataclasses import dataclass

from .spline import spline_get_cubic, spline_get_linear
from .polynomial import poly_eval


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

    def eval(self, t: float) -> float:
        if len(self.segments) == 0:
            return 0.0

        elif len(self.segments) == 1:
            return float(poly_eval(self.segments[0].coeffs, t))

        else:
            if t < self.segments[1].t:
                return float(poly_eval(self.segments[0].coeffs, t))

            i = 1
            while i < len(self.segments) - 1 and t < self.segments[i + 1].t:
                i += 1

            return float(poly_eval(self.segments[i].coeffs, t))

    def intersect(self, other: Trajectory) -> list[float]:
        t_all_self = []
        for state in self.states:
            t_all_self.append(state.t)

        t_all_other = []
        for state in other.states:
            t_all_other.append(state.t)

        t_union = t_all_self + t_all_other
        t_union.sort()

        t_all = []
        for t in t_union:
            if len(t_all) == 0 or not np.isclose(t, t_all[-1]):
                t_all.append(t)

        t_intersections = []

        for i in range(len(t_all) - 1):
            t0 = t_all[i + 0]
            t1 = t_all[i + 1]

            #find spline segments from each trajectoru in the interval

            # get the spline segment / polynomial difference

            # find the minimum value of the norm of the difference

            # if the minimum value is close to zero, append the time to the list

        return t_intersections
