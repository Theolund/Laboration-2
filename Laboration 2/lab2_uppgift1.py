import numpy as np
import matplotlib.pyplot as plt
def y1(t,y):
    return 1+t-y

def y1_exact(t, y0):
    return t + y0 * np.e**(-t)

def direction_field(f, tmin, tmax, ymin, ymax, density, scale, y0):
    xs = np.linspace(tmin, tmax, density)
    ys = np.linspace(ymin, ymax, density)
    X, Y = np.meshgrid(xs, ys)

    S = f(X, Y)
    U = np.ones_like(S)
    V = S
    L = np.hypot(U, V)
    U /= L
    V /= L

    plt.quiver(X, Y, U, V, scale = scale)

    t_vec = np.linspace(tmin,tmax,200)
    y_vec = y1_exact(t_vec, y0)
    plt.plot(t_vec, y_vec, color='red', label=f'y(0) = {y0}', linewidth=2)

    plt.xlim(tmin, tmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("Riktningsfält: dy/dt = 1 + t - y")
    plt.legend()
    plt.grid(True, alpha=0.3)  
    plt.tight_layout()
    plt.show()

def FE(f, t_start, t_end, y0, h):

    n_steps = int((t_end - t_start) / h)

    t_values = np.zeros(n_steps + 1)
    y_values = np.zeros(n_steps + 1)

    t_values[0] = t_start
    y_values[0] = y0

    for k in range(n_steps):
        y_values[k + 1] = y_values[k] + h * f(t_values[k], y_values[k])
        t_values[k + 1] = t_values[k] + h

    return t_values, y_values

def plot_solutions(t_vals, y_vals, y0):
    plt.plot(t_vals, y_vals, 'blue', label='Eulers metod', markersize=4)
    plt.plot(t_vals, y1_exact(t_vals, y0), 'red', label='Analytisk lösning', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.title(f"Framåt Euler: dy/dt = 1 + t - y, y(0) = {y0}")
    plt.grid(True, alpha=0.3)
    plt.show()  

def main():
    direction_field(y1, 0, 1, 0, 2.0, density=25, scale=25, y0=1.0)
    t_vals, y_vals = FE(y1,0,t_end=1.2,y0=1,h=0.1)
    plot_solutions(t_vals,y_vals,y0=1)
    e_k = np.abs(y_vals[-1] - y1_exact(t_vals[-1], y0=1.0))
    print(f"Felet: {e_k:.4f} som är ≈ 0.0188")
main()

    