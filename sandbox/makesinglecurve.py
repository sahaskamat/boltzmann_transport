from transport.wrappers import create_dispersion,create_FS,create_SM,rhozz_vs_theta,load_data_nov2026
import matplotlib.pyplot as plt
import numpy as np
import time

start = time.perf_counter()

dispersionInstance = create_dispersion(T=190e-3,T1multvalue=-0.132,T11multvalue=0.066,Tzmultvalue=0.0376,mumultvalue=0.86 )
FSinstance = create_FS(dispersionInstance,res_z=20,res_xy=100)
FSinstance.calculateDoping()

scattering_params_list = [
    ["pipidelta_exp",True,{"strength":2.5,"spread_xy":0.145,"n":4}],
    ["pipi",False,{"strength":0,"spread_xy":0.136,"n":5.167}]
]
SMinstance = create_SM(dispersionInstance,FSinstance,invtau_iso=9.5,scattering_params_list=scattering_params_list)

theta_array = np.sort(np.concatenate((np.linspace(-14,100,20),[0])))
rho_array_0 = rhozz_vs_theta(B=41.5,phi=0,theta_array=theta_array,scatteringMatrixInstance=SMinstance)
rho_array_45 = rhozz_vs_theta(B=41.5,phi=45,theta_array=theta_array,scatteringMatrixInstance=SMinstance)

end = time.perf_counter()
elapsed = end - start
print(f"Execution time: {elapsed:.6f} seconds")

fig,(axes,axes1) = plt.subplots(1,2,figsize=(8,4))

axes.plot(theta_array,rho_array_0/10000,"-o",ms=3)
axes.plot(theta_array,rho_array_45/10000,"-o",ms=3)

axes1.plot(theta_array,rho_array_0/rho_array_0[np.argmin(theta_array**2)],"-o",ms=3)
axes1.plot(theta_array,rho_array_45/rho_array_45[np.argmin(theta_array**2)],"-o",ms=3)

for phi in [0,15,30,45]:
    axes.plot(*load_data_nov2026(sample="LSCO-2609-27E",field=41.5,phi=phi,temp=35))
    theta,rho = load_data_nov2026(sample="LSCO-2609-27E",field=41.5,phi=phi,temp=35)
    axes1.plot(theta,rho/rho[np.argmin(theta**2)])

plt.show()
