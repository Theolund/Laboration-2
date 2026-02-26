
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
    
def plot_analys(tk,yk,h):
    
    fexakt = lambda t: np.exp(-t) + t
    tv = np.arange(0, 1.3, h)
    
    fig, ax = plt.subplots()
    #Plot approximativa lösningarna
    ax.plot(tk,yk, color='red', label="Approximativa lösningar")
    ax.plot(tk,yk,'r-.')
    #Plot analystiska lösningarna
    ax.plot(tv,fexakt(tv), color='blue', label="Analytiska lösningar")
    ax.plot(tv,fexakt(tv),'b-')
    
    ax.set_xlabel('t',fontsize=14)
    ax.set_ylabel('$y(t)$',fontsize=14)
    ax.tick_params(labelsize=14)
    plt.grid(True)
    plt.legend(loc=4)
    plt.show()

def main():
    tspan = np.array([0,1.2])
    y0 = 1
    h = 0.1
    tk, yk = framatEuler(f,tspan,y0,h)
    plot_analys(tk,yk,h)

    y_exakt = fexakt(t=1.2)
    y_approx = yk[-1]
    e_k = np.abs(y_approx - y_exakt)
    print(round(e_k, ndigits=4))

if __name__ == "__main__":
  main()