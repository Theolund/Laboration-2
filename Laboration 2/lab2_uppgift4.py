import numpy as np
import matplotlib.pyplot as plt

def q(x):
    return 50 * x**3 * np.log(x + 1)

def discretize_temperature(N, q, k, TL, TR, L=1.0):
    h = L / N
    x = np.linspace(0, L, N + 1)

    n = N - 1

    diag = -2*k/h**2 * np.ones(n)
    off_diag = k/h**2 * np.ones(n - 1)
    A1 = np.diag(diag)
    A2 = np.diag(off_diag, 1)
    A3 = np.diag(off_diag, -1)
    A = A1 + A2 + A3

    HL = np.array([q(x[i]) for i in range(1, N)])
    HL[0] -= k/h**2 * TL
    HL[-1] -= k/h**2 * TR

    return A, HL

def plot_graph(x, y):
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("T(x)")
    plt.title("Temperatur som funktion av x")
    plt.grid(True)
    plt.show()

#del c
N = 4
k = 2
TL = 2
TR = 2
L = 1

A, HL = discretize_temperature(N, q, k, TL, TR, L)
print(A)
print(HL)
print("="*100)

#del d
N = 100
A, HL = discretize_temperature(N, q, k, TL, TR, L)
temperature = np.linalg.solve(A, HL)
x = np.linspace(0, L, N + 1)
T = np.concatenate([[TL], temperature, [TR]])
plot_graph(x, T)
target1 = 0.2
index = np.argmin(np.abs(x - target1))
print(f"Approximativ temperatur vid x = 0.2: {T[index]:.4f} grader")
print("="*100)


#del e
target2  = 0.7
T_exact  = 1.6379544
N_vector = 50 * np.array([2**j for j in range(10)], dtype=int)
T_numerical = []

for N in N_vector:
    A, HL = discretize_temperature(N, q, k, TL, TR, L)
    T_inner = np.linalg.solve(A, HL)
    T_full = np.concatenate([[TL], T_inner, [TR]])
    index = round(target2 * N)
    T_numerical.append(T_full[index])

errors = [abs(T - T_exact) for T in T_numerical]

print("\nKonvergensstudie vid x = 0.7  (T_exact = 1.6379544):")
print(f"{'N':>8} {'h':>12} {'T(0.7)':>14} {'fel':>12} {'Konv.ordning p':>16}")
print("="*100)
for i, N in enumerate(N_vector):
    h_i = L / N
    if i == 0:
        p_str = "-"
    else:
        p = np.log(errors[i-1] / errors[i]) / np.log(2)
        
        p_str = f"{p:.3f}"
    print(f"{N:8d} {h_i:12.6f} {T_numerical[i]:14.7f} {errors[i]:12.2e} {p_str:>16}")

#del f
N = 100
k = 2
TL = 9
TR = 10
L = 1

A, HL = discretize_temperature(N, q, k, TL, TR, L)
temperature = np.linalg.solve(A, HL)
x = np.linspace(0, L, N + 1)
T = np.concatenate([[TL], temperature, [TR]])
print("="*100)
plot_graph(x,T)