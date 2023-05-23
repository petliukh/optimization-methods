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
    print("x_mm2\tx_mm1\tx_mp1\tx_m\th")
    print(f"{x_mm2:.2f}\t{x_mm1:.2f}\t{x_mp1:.2f}\t{x_m:.2f}\t{h}")

    if f(x_mp1) < f(x_mm1):
        return x_mm1, x_mp1, x_m
    return x_mm2, x_mm1, x_mp1


def first_derivative(f, x, h=1e-6):
    return (f(x + h) - f(x)) / h


def second_derivative(f, x, h=1e-6):
    fppx = (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)
    return fppx


def plot_slope(x, f, a, b):
    def line(x_i):
        return (x_i - x) * first_derivative(f, x) + f(x)
    xs = np.linspace(a, b, 1000)
    ys = [line(x_i) for x_i in xs]
    plt.plot(xs, ys)


def newtons_minimize(f, x, eps, a, b):
    k = 0
    x_prev = x
    plt.scatter(x, f(x))
    plot_slope(x, f, a, b)
    x = x_prev - first_derivative(f, x_prev) / second_derivative(f, x_prev)
    plt.scatter(x, f(x))
    plot_slope(x, f, a, b)

    print(f"Iteration {k}")
    print(f"x{k}={x_prev}, x{k+1}={x}")
    print(f"f(x{k})={f(x_prev)}, f(x{k+1})={f(x)}")

    while abs(f(x) - f(x_prev)) > eps:
        k += 1
        x_prev = x
        x = x_prev - first_derivative(f, x_prev) / second_derivative(f, x_prev)
        plt.scatter(x, f(x))
        plot_slope(x, f, a, b)
        print(f"Iteration {k}")
        print(f"x{k}={x_prev}, x{k+1}={x}")
        print(f"f(x{k})={f(x_prev)}, f(x{k+1})={f(x)}")

    return x


def plot(f, a, b, x, y):
    lnspace = np.linspace(a, b, 100)
    plt.plot(lnspace, [f(x) for x in lnspace])
    plt.scatter(x, y)
    plt.show()


def main():
    f = lambda x: x * x - 5 * x + 4
    eps = 1e-2
    x0 = 3
    h = 2
    a, x_m, b = dsk(f, x0, h)
    min = newtons_minimize(f, x_m, eps, a, b)

    plot(f, a, b, min, f(min))
    print(f"x_min = {min:.2f}, f(x_min) = {f(min):.2f}")


if __name__ == "__main__":
    main()
