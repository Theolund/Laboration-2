#Tagen från Lucianos python-filer.

"""
Created on Wed Jul 16 11:31:20 2025
Exampel: Rita riktningsfält
@author: Luciano Triguero
Rev. jan 2026
"""

import numpy as np
import matplotlib.pyplot as plt

# Definera funktionen f(t,y)
def f(t,y):
    return 1 + t - y

# Exakta lösningen
def fexakt(t):
    return np.exp(-t) + t

#Rita riktningsfält
def direction_field(f, tmin, tmax, ymin, ymax, density, scale):
    
    xs = np.linspace(tmin, tmax, density)
    ys = np.linspace(ymin, ymax, density)
    X, Y = np.meshgrid(xs, ys)

    # Vektorer (1, f(x,y)) normaliserade till enhetlängd
    S = f(X, Y)
    U = np.ones_like(S)
    V = S
    L = np.hypot(U, V)
    U /= L
    V /= L

    fig, ax = plt.subplots(figsize=(5,5))
    plt.quiver(X, Y, U, V, scale=scale)
 
    #Plotta analystiska lösningar i samma figur som riktningsfältet
    t_vec = np.linspace(tmin,tmax,200)
    y_vec = fexakt(t_vec)
    plt.plot(t_vec, y_vec, color='red', label='f(x)')

    
    plt.xlim(tmin, tmax)
    plt.ylim(ymin, ymax)
    #ax.set_aspect('equal')
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("Riktningsfält: dy/dt = 1 + t - y")
    plt.tight_layout()
    plt.show()

def main():
    direction_field(f,0, 1.2, 1,2.0,density=25, scale=25)
    
    
if __name__ == "__main__":
    main()
