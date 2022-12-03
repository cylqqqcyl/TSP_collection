import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit


def func1(x):
    # Sphere函数
    f = 0
    for i in x:
        f += i ** 2
    return f


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


def func3(x):
    # Schwefel 1.2函数
    f = 0
    for i in range(len(x)):
        sum = 0
        for j in range(i):
            sum += x[j]
        f += sum ** 2
    return f


def func4(x):
    # Schwefel 2.21函数
    f = 0
    for i in x:
        f = np.maximum(f, abs(i))
    return f


def func5(x):
    # Rosenbrock函数
    f = 0
    for i in range(len(x) - 1):
        f += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return f


def func6(x):
    # Rastrigin函数
    f = 0
    for i in x:
        f += i ** 2 - 10 * np.cos(2 * np.pi * i) + 10
    return f


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


def func8(x):
    # Griewank函数
    f = 0
    prod = 1
    for i in range(len(x)):
        f += x[i] ** 2
        prod *= np.cos(x[i] / np.sqrt(i + 1))
    f = f / 4000 - prod + 1
    return f


def func9(x):
    # Alpine函数
    f = 0
    for i in x:
        f += abs(i * np.sin(i) + 0.1 * i)
    return f




if __name__ == '__main__':
    funcs = [func1, func2, func3, func4, func5, func6, func7, func8, func9]
    funcs = [func9]
    for func in funcs:
        test_range = 100
        if func.__name__[-1] == '2':
            test_range = 10
        elif func.__name__[-1] == '5':
            test_range = 30
        elif func.__name__[-1] == '6':
            test_range = 5.12
        elif func.__name__[-1] == '7':
            test_range = 32
        elif func.__name__[-1] == '8':
            test_range = 600
        elif func.__name__[-1] == '9':
            test_range = 10
        x = np.linspace(-test_range, test_range, 100)
        y = np.linspace(-test_range, test_range, 100)
        X, Y = np.meshgrid(x, y)

        Z = func(np.asarray([X, Y]))

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.contour(X, Y, Z, 10, offset=0, cmap='rainbow', linestyles="solid", alpha=0.5)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
        plt.savefig(u'figure/{}_3d.png'.format(func.__name__))
        plt.show()
