import numpy as np
import matplotlib.pyplot as plt


eps = 1e-13
u1_values, u2_values = [], []

def J(u):
    return (u[0]**2 - 1)**2 + 5*(u[1] - u[0]**2)**2


def grad_J(u1, u2):
    return np.array([4*u1*(u1**2 - 1) + 10*u1*(u2 - u1**2), 10*u2 - 10*u1**2])


def hess_J(u1):
    return np.array([[12*u1**2 - 4, 10*u1], [10*u1, 10]])


def find_optimum_by_newton_method(u0):
    u1 = u0[0]
    u2 = u0[1]
    step = 1
    grad_J_uk = grad_J(u1, u2)

    while np.linalg.norm(grad_J_uk) > eps:
        grad_J_uk = grad_J(u1, u2)
        hess = hess_J(u1)
        p = np.linalg.solve(hess, -grad_J_uk)
        u1 += p[0]
        u2 += p[1]

        print("Шаг {}: точка  [{:.6f}, {:.6f}], градиент = {}".format(step, u1, u2, grad_J_uk))
        u1_values.append(u1)
        u2_values.append(u2)

        return np.array([u1, u2])


def show_method_result(u0, u1_values, u2_values):
    u1 = np.linspace(-10, 10, 100)
    u2 = np.linspace(-10, 30, 100)
    U1, U2 = np.meshgrid(u1, u2)
    J_u = J(np.array([U1, U2]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(U1, U2, J_u, cmap='viridis')
    ax.set_xlabel('u1')
    ax.set_ylabel('u2')
    ax.set_zlabel('J(u)')

    plt.plot(u1_values, u2_values, "vr")
    plt.plot(u0[0], u0[1], "Dk")
    u1_values.append(u0[0])
    u2_values.append(u0[1])
    plt.plot(u1_values, u2_values, "k-")
    plt.show()


def main():
    u0 = np.array([3, 3])
    a, b = [-7, 0], [9, 0]
    u1_min, u2_min = find_optimum_by_newton_method(u0)
    print("Минимум на отрезке [{}, {}] в точке [{:.6f}, {:.6f}]".format(a, b, u1_min, u2_min))
    show_method_result(u0, u1_values, u2_values)


main()
