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


@dataclass
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

    def eval(self, t: float) -> np.ndarray:
        if len(self.segments) == 0:
            return np.zeros(())

        elif len(self.segments) == 1:
            return poly_eval(self.segments[0].coeffs, t)

        else:
            if t < self.segments[1].t:
                return poly_eval(self.segments[0].coeffs, t)

            i = 1
            while i < len(self.segments) - 1 and t < self.segments[i + 1].t:
                i += 1

            return poly_eval(self.segments[i].coeffs, t)
