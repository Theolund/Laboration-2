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

answer= solveivp()


def q(t):
    return answer.sol(t)[0]

def framatEulerSystem(F, tspan, U0, h):
    
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
    N_values=[20,40,80,160]
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

def konvergensstudie():
    tspan=np.array([0,20])
    U0=[1,0]
    R=1
    L=2
    C=0.5
    N_values=[160,320,640,1280]

    fel=[]
    for N in N_values:
        h=(tspan[1]-tspan[0])/N
        tk,Uk = framatEulerSystem(lambda t, U:F(t,U,R,L,C),tspan, U0, h)

        q_ref=answer.sol(20)[0]
        i_ref=answer.sol(20)[1]

        fel_q=np.abs(Uk[-1,0]-q_ref)
        fel_i=np.abs(Uk[-1,1]-i_ref)
        fel.append((fel_q,fel_i))
        print(f'N={N}: fel_q={fel_q:.6f}, fel_i={fel_i:.6f}')
    print('\nNoggrannhetsordning:')
    for k in range(1,len(N_values)):
        p_q=np.log(fel[k-1][0]/fel[k][0])/np.log(2)
        p_i=np.log(fel[k-1][1]/fel[k][1])/np.log(2)
        print(f'N={N_values[k-1]}->{N_values[k]}: p_q={p_q:.4f},p_i={p_i:.4f}')

konvergensstudie()
