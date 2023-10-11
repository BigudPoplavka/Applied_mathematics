# © Павленко Б.В. ИВТ-2 МАГ2 
# Л.Р.1

import math

eps = 0.000000000001

def f(x):
    return x**2 - 21*x + 90


def df(x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def solve_bisection_method(a, b):
    print("Метод бисекции \n")
    step = 1

    while b - a >= eps:
        x0 = (a + b) / 2
        fx0 = f(x0)
        fa = f(a)
        
        print("Шаг {0}: a = {1}, b = {2}, x0 = {3}, fx0 = {4}".format(step, a, b, x0, fx0))

        if (fa > 0 and fx0 < 0) or (fa < 0 and fx0 > 0):
            b = x0
        else:
            a = x0
        step += 1
    return (a + b) / 2


def solve_iteration_method(a, b):
    print("Метод простой итерации \n")
    step = 1

    if f(a) > f(a + 1):
        xk = a
    elif f(a) < f(a + 1):
        xk = b

    x_next = xk + 1
    delta_x = 1 # x[i+1] - x[i]
    bk = delta_x / f(xk)

    while not math.isclose(f(xk), 0, rel_tol=1e-12):
        fxk = f(xk)
        x_next = xk + bk * fxk
        f_x_next = f(x_next)

        print("Шаг {0}: xk = {1}, x_next = {2}, bk = {3}, f(xk) = {4}".format(step, xk, x_next, bk, fxk))

        if (f_x_next < fxk) and (f_x_next > 0 and fxk > 0):
            bk += 0.2
        elif (f_x_next < fxk) and (f_x_next < 0 and fxk > 0):
            bk /= 2 

        xk = x_next
        step += 1
    return xk


def solve_relaxation_method(a, b):
    print("Метод релаксации \n")
    step = 1

    if f(a) > f(a + 1):
        xk = a
    elif f(a) < f(a + 1):
        xk = b

    x_next = xk + 1
    delta_x = 1 # x[i+1] - x[i]
    bk = delta_x / f(xk)

    while not math.isclose(f(xk), 0, rel_tol=1e-12):
        fxk = f(xk)
        x_next = xk + bk * fxk

        print("Шаг {0}: xk = {1}, x_next = {2}, f(xk) = {3}".format(step, xk, x_next, fxk))

        xk = x_next
        step += 1
    return xk


def solve_newton_method(x0, h):
    print("Метод Ньютона \n")
    step = 1
    xk = x0

    while not math.isclose(f(xk), 0, rel_tol=1e-12):
        fxk = f(xk)
        xk = xk - fxk / df(xk, h)
        print("Шаг {0}: xk = {1}, f(xk) = {2}".format(step, xk, fxk))
        step += 1
    return xk


def solve_secant_method(x0, x1):
    print("Метод Секущих \n")
    step = 1

    while abs(f(x1)) > 1e-12:
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0 = x1
        x1 = x2
        step += 1
    return x1


def main():
    print("\nx ~ {0}\n{1}\n".format(solve_bisection_method(3, 8), "\n" + "-"*20))
    print("\nОтвет: x ~ {0}\n{1}\n".format(solve_iteration_method(3, 8), "\n" + "-"*20))
    print("\nОтвет: x ~ {0}\n{1}\n".format(solve_relaxation_method(3, 8), "\n" + "-"*20))
    print("\nОтвет: x ~ {0}\n{1}\n".format(solve_newton_method(3, 0.01), "\n" + "-"*20))
    print("\nОтвет: x ~ {0}\n{1}\n".format(solve_secant_method(3, 0.01), "\n" + "-"*20))


main()