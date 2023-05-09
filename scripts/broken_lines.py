import numpy as np
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

    if f(x_mp1) < f(x_mm1):
        return x_mm1, x_mp1, x_m
    return x_mm2, x_mm1, x_mp1


def find_lipschitz_const(f, a, b):
    # | f(x) - f(y) | <= L * | x - y |
    # L >= | f(x) - f(y) | / | x - y |
    return abs(f(a) - f(b)) / abs(a - b) + 0.1


def fnmin(f, L, xks, a, b, num):
    lnspace = np.linspace(a, b, num)
    rhos = [max(f(xk) - L * abs(x - xk) for xk in xks) for x in lnspace]
    min_pos = np.argmin(rhos)
    return lnspace[min_pos], rhos[min_pos]


def plot_rho(f, L, xks, a, b, num):
    lnspace = np.linspace(a, b, num)
    f_ys = np.array([f(x) for x in lnspace])
    rhos = [max(f(xk) - L * abs(x - xk) for xk in xks) for x in lnspace]

    plt.plot(lnspace, f_ys)
    plt.plot(lnspace, rhos)
    plt.show()


def broken_lines_method(f, a, b, L, eps):
    xks = [a]

    while True:
        xkp1, rho_min = fnmin(f, L, xks, a, b, 1000)

        if abs(rho_min - f(xkp1)) < eps:
            plot_rho(f, L, xks, a, b, 1000)
            return xkp1
        xks.append(xkp1)


def test_gradient_descent_method():
    f = lambda x: x * x + 2 * x + 1
    eps = 1e-2
    a, x_m, b = dsk(f, 3, 2)
    print(f"[a, b] = [{a:.2f},{b:.2f}], x* = {x_m:.2f}")

    # L = find_lipschitz_const(f, a, b)
    L = 8
    x_min = broken_lines_method(f, a, b, L, eps)

    print(f"L = {L}")
    print(f"Mimimum of f = ({x_min}, {f(x_min)}) for eps {eps}")


def main():
    test_gradient_descent_method()


if __name__ == "__main__":
    main()
