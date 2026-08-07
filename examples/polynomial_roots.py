import numpy as np

from spline_intersections import poly_real_roots


def main():
    coeffs = np.array([1.0, 1.0])
    roots = poly_real_roots(coeffs)
    print("x + 1")
    print(np.round(roots))
    print()

    coeffs = np.array([1.0, -3.0, 2.0])
    roots = poly_real_roots(coeffs)
    print("x^2 - 3x + 2")
    print("= (x-1)(x-2)")
    print(np.round(roots))
    print()

    coeffs = np.array([1.0, 1.0, -2.0])
    roots = poly_real_roots(coeffs)
    print("x^2 + x - 2")
    print("= (x+2)(x-1)")
    print(np.round(roots))
    print()

    coeffs = np.array([1.0, -5.0, 6.0])
    roots = poly_real_roots(coeffs)
    print("x^2 - 5x + 6")
    print("= (x-2)(x-3)")
    print(np.round(roots))
    print()

    coeffs = np.array([1.0, -3.0, -5.0, 15.0, 4.0, -12.0])
    roots = poly_real_roots(coeffs)
    print("x^5 - 3x^4 - 5x^3 + 15x^2 + 4x - 12")
    print("= (x+2)(x+1)(x-1)(x-2)(x-3)")
    print(np.round(roots))
    print()


if __name__ == "__main__":
    main()
