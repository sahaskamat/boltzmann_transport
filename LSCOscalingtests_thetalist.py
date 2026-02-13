import numpy as np
import matplotlib.pyplot as plt
import dispersion
import orbitcreation
import conductivity
from makesigmalist import makelist_parallel
from time import time

starttime_global = time()
thetalist = np.linspace(0,180,80)
phi = 0
phi_rad = np.deg2rad(phi)

res_z = 20
res_xy = 100

dispersionInstance = dispersion.LSCOdispersion(T= 190e-3,T1multvalue=-0.134,T11multvalue=0.067,Tzmultvalue=0.03,mumultvalue=0.805)
FSorbitsInstance = orbitcreation.fermiSurfaceOrbits(res_z,res_xy,dispersionInstance,True)
starttime = time()
FSorbitsInstance.createFS(tilingformat="variable",alpha=0.1,parallelised=False)
endtime = time()
print(f"Time taken to create Fermi Surface = {endtime - starttime}")

conductivityInstance = conductivity.Conductivity(dispersionInstance,FSorbitsInstance,invtau_iso=12.595,invtau_aniso=63.823)
starttime = time()
conductivityInstance.createAmatrix_Bindependent()
endtime = time()
print(f"Time taken to create B independent Amatrix =  {endtime - starttime}")

def getsigma(theta):
    B = [45*np.sin(np.deg2rad(theta))*np.cos(phi_rad),45*np.sin(np.deg2rad(theta))*np.sin(phi_rad),45*np.cos(np.deg2rad(theta))]
    conductivityInstance.createAmatrix_Bdependent(B)
    conductivityInstance.createAlpha()
    conductivityInstance.createSigma()

    print(f"Theta={theta}. Calculated total area: {conductivityInstance.areasum}. Number of orbits used {len(conductivityInstance.FSorbitsInstance.FSorbits)}. Size of Amatrix: {conductivityInstance.n}")
    return conductivityInstance.sigma,conductivityInstance.areasum

sigmalist,rholist,arealist = makelist_parallel(getsigma,thetalist)
rhoxylist= [rho[2,2]*10e-5 for rho in rholist]

endtime_global = time()
print(f"execution time: {endtime_global-starttime_global}")

np.savetxt("rhoxyvstPhi"+str(phi)+".dat",np.transpose([thetalist,rhoxylist]))

FSorbitsInstance.plotpoints()

fig,axes = plt.subplots()

axes.plot(thetalist,rhoxylist,ls="-",marker="o",ms=2)
axes.set_ylabel(r"$\rho_{zz}$ ($m\Omega$ cm )")
axes.set_xlabel(r'$\theta$')
axes.text(0.1,0.1,f"Res={res_z}x{res_xy}\nPhi={phi} degrees",fontsize=10, transform=axes.transAxes)

plt.show()