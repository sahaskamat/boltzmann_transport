import numpy as np
import matplotlib.pyplot as plt
import dispersion
import orbitcreation
import conductivity
from makesigmalist import makelist_parallel
from time import time

starttime_global = time()
thetalist = np.linspace(-14,99,80)

res_z = 20
res_xy = 100

params = 0.07992756258272564,12.656955338072684,252.81365144783192 #Tzmultvalue,invtau_iso,invtau_aniso

dispersionInstance = dispersion.LSCOdispersion(T= 190e-3,T1multvalue=-0.134,T11multvalue=0.067,Tzmultvalue=params[0],mumultvalue=0.805)
FSorbitsInstance = orbitcreation.fermiSurfaceOrbits(res_z,res_xy,dispersionInstance,True)
starttime = time()
FSorbitsInstance.createFS(tilingformat="variable",alpha=0.1,parallelised=False)
endtime = time()
print(f"Time taken to create Fermi Surface = {endtime - starttime}")

def create_rhozz(phi,Bmag):
    phi_rad = np.deg2rad(phi)
    conductivityInstance = conductivity.Conductivity(dispersionInstance,FSorbitsInstance,invtau_iso=params[1],invtau_aniso=params[2])
    starttime = time()
    conductivityInstance.createAmatrix_Bindependent()
    endtime = time()
    print(f"Time taken to create B independent Amatrix =  {endtime - starttime}")

    def getsigma(theta):
        B = [Bmag*np.sin(np.deg2rad(theta))*np.cos(phi_rad),Bmag*np.sin(np.deg2rad(theta))*np.sin(phi_rad),Bmag*np.cos(np.deg2rad(theta))]
        conductivityInstance.createAmatrix_Bdependent(B)
        conductivityInstance.createAlpha()
        conductivityInstance.createSigma()

        print(f"Theta={theta}. Calculated total area: {conductivityInstance.areasum}. Number of orbits used {len(conductivityInstance.FSorbitsInstance.FSorbits)}. Size of Amatrix: {conductivityInstance.n}")
        return conductivityInstance.sigma,conductivityInstance.areasum

    sigmalist,rholist,arealist = makelist_parallel(getsigma,thetalist)
    rhozzlist= [rho[2,2]*10e-5 for rho in rholist]

    endtime_global = time()
    print(f"execution time: {endtime_global-starttime_global}")

    return rhozzlist

#p.savetxt("rhoxyvstPhi"+str(phi)+".dat",np.transpose([thetalist,rhoxylist]))

FSorbitsInstance.plotpoints()
fig,axes = plt.subplots()

#load data
data_theta0,data_rhozz0 = np.loadtxt("data/2601C/2601C_phi0_T30K_B45.0T.txt",unpack=True,skiprows=1,delimiter=",",usecols=(0,1))
data_theta45,data_rhozz45 = np.loadtxt("data/2601C/2601C_phi45_T30K_B45.0T.txt",unpack=True,skiprows=1,delimiter=",",usecols=(0,1))

#generate data
rhozzlist0 = create_rhozz(phi=0,Bmag=45)
rhozzlist45= create_rhozz(phi=45,Bmag=45)

axes.plot(thetalist,rhozzlist0,ls="-",marker="o",ms=2,label=f"Model, $\phi=0$")
axes.plot(thetalist,rhozzlist45,ls="-",marker="o",ms=2,label=f"Model, $\phi=45$")
axes.plot(data_theta0,data_rhozz0,ls="-",marker="o",ms=2,label="Data, $\phi=0$")
axes.plot(data_theta45,data_rhozz45,ls="-",marker="o",ms=2,label="Data, $\phi=0$")
axes.set_ylabel(r"$\rho_{zz}$ ($m\Omega$ cm )")
axes.set_xlabel(r'$\theta$')
axes.text(0.1,0.1,f"LSCO x=0.22\nT=30 K\nRes={res_z}x{res_xy}",fontsize=10, transform=axes.transAxes)

plt.show()