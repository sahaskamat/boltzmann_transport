from numbalsoda import lsoda_sig, lsoda
from numba import njit, cfunc
from scipy.integrate import solve_ivp 
import numpy as np
from matplotlib import pyplot as plt



@cfunc(lsoda_sig)
def f(t, u, du, p):
    du[0] = u[0]-u[0]*u[1]
    du[1] = u[0]*u[1]-u[1]

@njit
def f_scipy(t, u):
    return np.array([u[0]-u[0]*u[1],u[0]*u[1]-u[1]])

funcptr = f.address
u0 = np.array([5.,0.8])
data = np.array([1.0])
t_eval = np.linspace(0.0,50.0,1000)

usol, success = lsoda(funcptr, u0, t_eval, data,rtol=1e-7,atol=1e-8)

plt.rcParams.update({'font.size': 15})
fig,ax = plt.subplots(1,1,figsize=[7,5])

ax.plot(t_eval,usol[:,0],label='u1')
ax.plot(t_eval,usol[:,1],label='u2')
ax.legend()
ax.set_xlabel('t')

plt.show()



