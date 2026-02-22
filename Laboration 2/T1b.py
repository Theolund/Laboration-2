import numpy as np

def F(t,y,R,L,C):
    q=y[0]
    i=y[1]
    dq_dt=i
    di_dt= -(1/(L*C))*q-(R/L)*i
    return [dq_dt,di_dt]

