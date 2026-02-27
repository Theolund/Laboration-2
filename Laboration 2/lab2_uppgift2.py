import numpy as np
import matplotlib.pyplot as plt


def y1(t, y):
    return 1 + t - y


def y1_exact(t, y0):
    return t + y0 * np.exp(-t)

def FE(f, t_start, t_end, y0, h):

    n_steps = round((t_end - t_start) / h)

    t_values = np.zeros(n_steps + 1)
    y_values = np.zeros(n_steps + 1)

    t_values[0] = t_start
    y_values[0] = y0

    for k in range(n_steps):
        y_values[k + 1] = y_values[k] + h * f(t_values[k], y_values[k])
        t_values[k + 1] = t_values[k] + h

    return t_values, y_values

def main():
    y0 = 1.0
    t_start = 0.0
    T = 1.2
    h_start = 0.2
    num_halvings = 4
    y_exact = y1_exact(T, y0)

    h_values = []
    y_numerical = []
    errors = []
    n_steps_list = []

    h = h_start
    for i in range(num_halvings + 1):
        t_vals, y_vals = FE(y1, t_start, T, y0, h)
        y_at_T = y_vals[-1]
        n_steps = round((T - t_start) / h)
        error = abs(y_at_T - y_exact)

        h_values.append(h)
        y_numerical.append(y_at_T)
        errors.append(error)
        n_steps_list.append(n_steps)

        h = h / 2

    for i in range(len(h_values)):
        if i < len(h_values) - 1:
            p = np.log(errors[i] / errors[i+1]) * 1/np.log(2)
            print(f"{h_values[i]:12.4f} {y_numerical[i]:12.6f} {y_exact:12.6f} {errors[i]:12.2e} {p:12.3f} {n_steps_list[i]:12d}")
        else:
            print(f"{h_values[i]:12.4f} {y_numerical[i]:12.6f} {y_exact:12.6f} {errors[i]:12.2e} {'-':>12s} {n_steps_list[i]:12d}")

main()
