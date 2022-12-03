import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit


@jit(nopython=True)
def func1(x):
    # Sphere函数
    f = 0
    for i in x:
        f += i ** 2
    return f

@jit(nopython=True)
def func2(x):
    # Schwefel 2.22函数
    f = 0
    sum = 0
    prod = 1
    for i in x:
        sum += abs(i)
        prod *= abs(i)
    f = sum + prod
    return f

@jit(nopython=True)
def func3(x):
    # Schwefel 1.2函数
    f = 0
    for i in range(len(x)):
        sum = 0
        for j in range(i):
            sum += x[j]
        f += sum ** 2
    return f

@jit(nopython=True)
def func4(x):
    # Schwefel 2.21函数
    f = 0
    for i in x:
        f = np.maximum(f, abs(i))
    return f

@jit(nopython=True)
def func5(x):
    # Rosenbrock函数
    f = 0
    for i in range(len(x) - 1):
        f += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return f

@jit(nopython=True)
def func6(x):
    # Rastrigin函数
    f = 0
    for i in x:
        f += i ** 2 - 10 * np.cos(2 * np.pi * i) + 10
    return f

@jit(nopython=True)
def func7(x):
    # Ackley函数
    d = len(x)
    f = 0
    sum1 = 0
    sum2 = 0
    for i in x:
        sum1 += i ** 2
        sum2 += np.cos(2 * np.pi * i)
    f = -20 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.e
    return f

@jit(nopython=True)
def func8(x):
    # Griewank函数
    f = 0
    prod = 1
    for i in range(len(x)):
        f += x[i] ** 2
        prod *= np.cos(x[i] / np.sqrt(i + 1))
    f = f / 4000 - prod + 1
    return f

@jit(nopython=True)
def func9(x):
    # Penalized 1函数
    def func_y(xi):
        yi = 1 + (xi + 1) / 4
        return yi

    y = func_y(x)

    sum1 = 0
    sum2 = 0
    for i in range(len(y) - 1):
        sum1 += (y[i] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[i + 1]) ** 2)
        sum2 += func_u(y[i], 10, 100, 4)
    sum2 += func_u(y[-1], 10, 100, 4)
    f = np.pi / len(y) * (10 * np.sin(np.pi * y[0]) ** 2 + sum1 + (y[-1] - 1) ** 2) + sum2
    return f

@jit(nopython=True)
def func_u(xi, a, k, m):
    gta_mask = np.where(xi > a)
    lta_mask = np.where(xi < -a)
    zero_mask = np.where(np.abs(xi) <= a)
    if type(xi) is np.ndarray:
        if len(gta_mask[0]) > 0:
            xi[gta_mask] = k * (xi[gta_mask] - a) ** m + a ** m
        if len(lta_mask[0]) > 0:
            xi[lta_mask] = k * (-xi[lta_mask] - a) ** m + a ** m
        if len(zero_mask[0]) > 0:
            xi[zero_mask] = 0
    else:
        if xi > a:
            xi = k * (xi - a) ** m + a ** m
        elif xi < -a:
            xi = k * (-xi - a) ** m + a ** m
        else:
            xi = 0
    return xi


if __name__ == '__main__':
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)


    Z = func9(np.asarray([X, Y]))


    fig = plt.figure()
    ax = Axes3D(fig)
    ax.contour(X, Y, Z, 10,offset=0, cmap='rainbow', linestyles="solid", alpha=0.5)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

    plt.show()