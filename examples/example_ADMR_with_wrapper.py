from transport.wrappers import create_dispersion,create_FS,create_SM,rhozz_vs_theta
import matplotlib.pyplot as plt
import numpy as np

dispersionInstance = create_dispersion(T=190e-3,T1multvalue=-0.134,T11multvalue=0.066,Tzmultvalue=0.033,mumultvalue=0.81)
FSinstance = create_FS(dispersionInstance,res_z=20,res_xy=100)
FSinstance.calculateDoping()

scattering_params_list = [
    ["pipidelta_exp",True,{"strength":13.4,"spread_xy":0.136,"n":5.167}],
    ["pipi",False,{"strength":0,"spread_xy":0.136,"n":5.167}]
]
SMinstance = create_SM(dispersionInstance,FSinstance,invtau_iso=10,scattering_params_list=scattering_params_list)

theta_array = np.linspace(-14,100,80)
rho_array_0 = rhozz_vs_theta(B=41.5,phi=0,theta_array=theta_array,scatteringMatrixInstance=SMinstance)
rho_array_45 = rhozz_vs_theta(B=41.5,phi=45,theta_array=theta_array,scatteringMatrixInstance=SMinstance)

plt.plot(theta_array,rho_array_0)
plt.plot(theta_array,rho_array_45)
plt.show()