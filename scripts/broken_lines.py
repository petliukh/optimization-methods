import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def dsk(f, x, h):
    k = 0

    if f(x + h) > f(x):
        h = -h

    print("k\th_k\tx_k\tf_k\tx_kp1\tf_kp1")
    print(f"{k}\t{h}\t{x}\t{f(x)}\t{x + h}\t{f(x + h)}")

    while f(x + h) < f(x):
        x += h
        h *= 2
        k += 1
        print(f"{k}\t{h}\t{x}\t{f(x)}\t{x + h}\t{f(x + h)}")

    x_mm2 = x - h
    x_mm1 = x
    x_m = x + h
    h /= 2
    x_mp1 = x_m - h
    x_mm2, x_mm1, x_mp1, x_m = sorted((x_mm2, x_mm1, x_mp1, x_m))
    print("x_mm2\tx_mm1\tx_mp1\tx_m\th")
    print(f"{x_mm2:.2f}\t{x_mm1:.2f}\t{x_mp1:.2f}\t{x_m:.2f}\t{h}")

    if f(x_mp1) < f(x_mm1):
        return x_mm1, x_mp1, x_m
    return x_mm2, x_mm1, x_mp1


def find_lipschitz_const(f, a, b, npoints, eps):
    lnspace = np.linspace(a, b, npoints)
    return max(abs(f(x + eps) - f(x)) / eps for x in lnspace)


def fnmin(f, L, xks, a, b, num):
    lnspace = np.linspace(a, b, num)
    rhos = [max(f(xk) - L * abs(x - xk) for xk in xks) for x in lnspace]
    min_pos = np.argmin(rhos)
    return lnspace[min_pos], rhos[min_pos]


def plot_rho(f, L, xks, a, b, num, min=None):
    lnspace = np.linspace(a, b, num)
    f_ys = np.array([f(x) for x in lnspace])
    rhos = [max(f(xk) - L * abs(x - xk) for xk in xks) for x in lnspace]
    plt.plot(lnspace, f_ys)
    plt.plot(lnspace, rhos)

    if min:
        x, y = min
        plt.scatter(x, y, s=50, c="red")
    plt.show()


def broken_lines_method(f, a, b, L, npoints, eps):
    xks = [a]

    while True:
        xkp1, rho_min = fnmin(f, L, xks, a, b, npoints)

        if abs(rho_min - f(xkp1)) < eps:
            print(f"xks: {xks}")
            plot_rho(f, L, xks, a, b, npoints, (xkp1, rho_min))
            return xkp1
        xks.append(xkp1)


def test_broken_lines_method():
    f = lambda x: x * x - 5 * x + 4
    # f = lambda x: x * x + 2 * x + 1
    eps = 1e-2
    npoints = 500
    a, x_m, b = dsk(f, 3, 2)
    print(f"[a, b] = [{a:.2f},{b:.2f}], x* = {x_m:.2f}")

    L = find_lipschitz_const(f, a, b, npoints, eps)
    x_min = broken_lines_method(f, a, b, L, npoints, eps)

    print(f"L = {L}")
    print(f"Mimimum of f = ({x_min}, {f(x_min)}) for eps {eps}")


def main():
    test_broken_lines_method()


if __name__ == "__main__":
    main()
