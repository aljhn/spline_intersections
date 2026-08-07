import numpy as np

from spline_intersections import poly_real_roots


def main():
    # f(x) = (x+2)(x-1)(x+1)(x-2)(x-3)
    # = x^5 - 3x^4 - 5x^3 + 15x^2 + 4x - 12
    coeffs = np.array([1.0, -3.0, -5.0, 15.0, 4.0, -12.0])
    roots = poly_real_roots(coeffs)
    print(roots)


if __name__ == "__main__":
    main()
