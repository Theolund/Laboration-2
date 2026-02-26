import numpy as np
from scipy.integrate import solve_ivp


def F(t,y,R,L,C):
    q=y[0]
    i=y[1]
    dq_dt=i
    di_dt= -(1/(L*C))*q-(R/L)*i
    return [dq_dt, di_dt]

def i():
    tspan=(0,20)
    Q0=[1,0]
    L=2
    C=0.5
    R=1
    svar=solve_ivp(F,tspan,Q0,method='RK45',args=(R,L,C))
    print(svar.t)
    print(svar.y)

def ii():
    tspan=(0,20)
    Q0=[1,0]
    L=2
    C=0.5
    R=0
    svar=solve_ivp(F,tspan,Q0,method='RK45',args=(R,L,C))
    print(svar.t)
    print(svar.y)

ii()
