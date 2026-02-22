import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def F(t,y,R,L,C):
    q=y[0]
    i=y[1]
    dq_dt=i
    di_dt= -(1/(L*C))*q-(R/L)*i
    return np.array([dq_dt,di_dt])

def solveivp():
    tspan=(0,20)
    Q0=[1,0]
    L=2
    C=0.5
    R=1
    sol=solve_ivp(F,tspan,Q0,method='RK45',args=(R,L,C), dense_output=True)
    return sol

sol= solveivp()


def q(t):
    return sol.sol(t)[0]

def framatEulerSystem(F, tspan, U0, h):
    
    """ 
    Implementerar Framåt Euler för ODE-System
    
    Parameters:
        F       : Vektorvärd function F(t, U)
        tspan   : Tidsintervall
        U0      : initial state U0
        h       : steglängd, eller tidssteg
    
    Returns:
        tk     : numpy array of time points
        Uk     : numpy array of state values
    """
    
    n_steps = round(np.abs(tspan[1]-tspan[0])/h)
    tk = np.zeros(n_steps+1)
    Uk = np.zeros((n_steps+1, len(U0)))

    tk[0] = tspan[0]
    Uk[0] = U0

    for k in np.arange(n_steps):
        Uk[k+1] = Uk[k] + h * F(tk[k], Uk[k])
        tk[k+1] = tk[k] + h

    return tk, Uk 

def main():
    tspan=np.array([0,20])
    U0=[1,0]
    R=1
    L=2
    C=0.5
    N_values=[80,160,320,640]
    for N in N_values:
        h=(tspan[1]-tspan[0])/N
        tk,Uk=framatEulerSystem(lambda t, U:F(t,U,R,L,C),tspan,U0,h)
        plt.plot(tk,Uk[:,0],label=f'FE N={N}')
    
    t_ref=np.linspace(0,20,1000)
    plt.plot(t_ref,q(t_ref),label='RK45 (referens)')
    plt.legend()
    plt.grid(True)
    plt.ylim(-6,6)
    plt.show()
main()