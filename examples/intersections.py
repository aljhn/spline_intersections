import numpy as np

from spline_intersections import State, Trajectory, poly_eval


def trajectory1_symbolic(t: float) -> tuple[np.ndarray, np.ndarray]:
    px = t - 3.0
    py = t * t - 6.0 * t + 9.0
    pz = t * t * t - 9.0 * t * t - 9.0 * t + 27.0
    p = np.array([px, py, pz])

    vx = 1.0
    vy = 2.0 * t - 6.0
    vz = 3.0 * t * t - 18.0 * t - 9.0
    v = np.array([vx, vy, vz])

    return p, v


def trajectory2_symbolic(t: float) -> tuple[np.ndarray, np.ndarray]:
    px = t - 3.0
    py = -t * t + 6.0 * t - 1.0
    pz = t * t * t - 9.0 * t * t - 9.0 * t + 27.0
    p = np.array([px, py, pz])

    vx = 1.0
    vy = -2.0 * t + 6.0
    vz = 3.0 * t * t - 18.0 * t - 9.0
    v = np.array([vx, vy, vz])

    return p, v


def main():
    t0 = 0.0
    p0, v0 = trajectory1_symbolic(t0)
    s0 = State(t0, p0, v0)

    t1 = 2.0
    p1, v1 = trajectory1_symbolic(t1)
    s1 = State(t1, p1, v1)

    t2 = 7.5
    p2, v2 = trajectory1_symbolic(t2)
    s2 = State(t2, p2, v2)

    trajectory1 = Trajectory()
    trajectory1.add_state(s0)
    trajectory1.add_state(s1)
    trajectory1.add_state(s2)

    t3 = 0.5
    p3, v3 = trajectory2_symbolic(t3)
    s3 = State(t3, p3, v3)

    t4 = 3.5
    p4, v4 = trajectory2_symbolic(t4)
    s4 = State(t4, p4, v4)

    t5 = 6.0
    p5, v5 = trajectory2_symbolic(t5)
    s5 = State(t5, p5, v5)

    t6 = 9.0
    p6, v6 = trajectory2_symbolic(t6)
    s6 = State(t6, p6, v6)

    trajectory2 = Trajectory()
    trajectory2.add_state(s3)
    trajectory2.add_state(s4)
    trajectory2.add_state(s5)
    trajectory2.add_state(s6)

    t_intersections = trajectory1.intersect(trajectory2)
    print(t_intersections)


if __name__ == "__main__":
    main()
