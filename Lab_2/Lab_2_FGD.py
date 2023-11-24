import numpy as np
import matplotlib.pyplot as plt

delta_u = 0.1
eps = 1e-13
u1_values, u2_values = [], []
golden_num = 0.6

def J(u):
    u1, u2 = u
    return (u1**2 - 1)**2 + 5*(u2 - u1**2)**2


def grad_J(u):
    u1, u2 = u
    return [2*(u1**2 - 1)*2*u1 - 20*u1*(u2 - u1**2), 10*(u2 - u1**2)]


def optimize_golden_section(f, ua, ub):
    u1 = ub - golden_num * (ub - ua)
    u2 = ua + golden_num * (ub - ua)

    while abs(ub - ua) > eps:
        if f(u1) < f(u2):
            ub = u2
        else:
            ua = u1
        u1 = ub - golden_num * (ub - ua)
        u2 = ua + golden_num * (ub - ua)
        
    return (ub + ua) / 2


def gradient_descent(u0, max_iter=1000):
    uk = u0
    u_next = [0, 0]
    step = 1

    u1_values.append(u0[0])
    u2_values.append(u0[1])

    for _ in range(max_iter):
        grad_J_uk = np.array(grad_J(uk))

        def f_lr(b): return J(uk - b * grad_J_uk)
        
        b = optimize_golden_section(f_lr, 0, delta_u)  
        u_next = uk - b * grad_J_uk

        u1_values.append(u_next[0])
        u2_values.append(u_next[1])

        print("Шаг {}: точка  [{:.13f}, {:.13f}], градиент = {}".format(step, u_next[0], u_next[1], grad_J_uk))
        step += 1

        if np.linalg.norm(u_next - uk) < eps:
            break
        uk = u_next

    return uk


def show_method_result(u0, u1_values, u2_values):
    u1 = np.linspace(0, 4, 10)
    u2 = np.linspace(0, 4, 10)
    U1, U2 = np.meshgrid(u1, u2)
    J_u = J(np.array([U1, U2]))

    fig = plt.figure(figsize=(16, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(U1, U2, J_u, cmap='viridis', alpha=0.7)
    ax.set_xlabel('u1')
    ax.set_ylabel('u2')
    ax.set_zlabel('J(u)')
    
    plt.plot(u0[0], u0[1], "Dk")
    plt.plot(u1_values, u2_values, "vr")
    
    
    plt.plot(u1_values, u2_values, "k-")
    plt.show()


u0 = np.array([3, 3])

u = gradient_descent(u0)
print(u)
show_method_result(u0, u1_values, u2_values)
