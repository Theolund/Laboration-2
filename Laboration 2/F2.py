
#Återigen, mesta taget från Luciano's python.

import numpy as np
import matplotlib.pyplot as plt

# Definera funktionen f(t,y)
def f(t,y):
    return 1 + t - y

def fexakt(t):
    return np.exp(-t) + t

def framatEuler(f, tspan, y0, h):
    """
    Framåt Euler för ODE-skalär
    f: står för funktionen f(t,y)
    tspan: tidsintervall
    y0: beggynnelsevärdet y(t0) = y0
    h: steglängd (tidsteglängd)
    """
    a, b = tspan[0], tspan[1]
    # Diskretiseringsteg: Generate n+1 griddpunkter
    n = round(np.abs(b-a)/h)
    t = np.linspace(a, b, n+1)
    
    #Skapa arrayen y[k] med värden noll
    y = np.zeros(n+1)

    #Begynnelsevillkor
    y[0] = y0

    # Iterate med Framåt Euler. Formler EF.1
    for k in np.arange(n):
        y[k+1] = y[k] + h*f(t[k], y[k])
        
    return t, y
    
def main():
    tspan = np.array([0,1.2])
    y0 = 1
    h_lista = [0.2, 0.1, 0.05, 0.025, 0.0125]

    y_exakt = fexakt(t=1.2)
    
    felen = []
    
    for h in h_lista:
        tk, yk = framatEuler(f,tspan,y0,h)

        y_T = yk[-1]
        #y_T är värdet vid T = 1.2

        e_k = np.abs(y_T - y_exakt)
        felen.append(e_k)
        
        print(f"h = {h:<10} y_T = {y_T:<15.6f} e_k = {e_k:<15.2e}")
    
    for i in range(len(felen) - 1):
        e_kh = felen[i]
        e_khalv = felen[i + 1]

        kvot = e_kh / e_khalv

        p = np.log(kvot) / np.log(2)

        print(f"p = {p:.3f}")

if __name__ == "__main__":
  main()