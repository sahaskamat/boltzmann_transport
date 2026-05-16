import numpy as np
import matplotlib.pyplot as plt
import transport.dispersion as dispersion
import transport.orbitcreation as orbitcreation
import transport.conductivity as conductivity
from transport.makesigmalist import makelist_parallel,makelist_serial
from time import time

starttime_global = time()
thetalist = np.linspace(-14,99,20)

res_z = 20
res_xy = 100

#0.11254623711633413,13.64440988094479,83.15247343896175,0.10296026520227448,1.5230931197615067

Tzmultvalue = 0.07
invtau_iso =13.64440988094479
strength = 83.15247343896175
spread_xy=0.10296026520227448
spread_z = 0.1
n=1.5230931197615067
mumultvalue=0.81

plotScattering=False

dispersionInstance = dispersion.LSCOdispersion(T= 190e-3,T1multvalue=-0.132,T11multvalue=0.066,Tzmultvalue=Tzmultvalue,mumultvalue=mumultvalue)
FSorbitsInstance = orbitcreation.fermiSurfaceOrbits(res_z,res_xy,dispersionInstance,True)
starttime = time()
FSorbitsInstance.createFS(tilingformat="variable",alpha=0.1,parallelised=False)
endtime = time()
print(f"Time taken to create Fermi Surface = {endtime - starttime}")
print(f"Doping={FSorbitsInstance.calculateDoping()}")

def create_rhozz(phi,Bmag,scattering_in=True):
    phi_rad = np.deg2rad(phi)
    if scattering_in: conductivityInstance = conductivity.Conductivity(dispersionInstance,FSorbitsInstance,invtau_iso=invtau_iso,plotScattering=False)
    else: conductivityInstance = conductivity.Conductivity(dispersionInstance,FSorbitsInstance,invtau_iso=invtau_iso,plotScattering=plotScattering)
    starttime = time()
    conductivityInstance.createAmatrix_Bindependent_isotropic()
    conductivityInstance.create_Hfunc(scatteringmodel="pipizero",strength=strength,spread_xy=spread_xy,spread_z=spread_z,n=n)
    conductivityInstance.createAmatrix_Bindependent_fwdscatter_out()
    if scattering_in: conductivityInstance.createAmatrix_Bindependent_fwdscatter_in()
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

#load data
data_theta0,data_rhozz0 = np.loadtxt("data/admr_data/2511A/2511A_phi0_T30K_B45.0T.txt",unpack=True,skiprows=1,delimiter=",",usecols=(0,1))
data_theta45,data_rhozz45 = np.loadtxt("data/admr_data/2511A/2511A_phi45_T30K_B45.0T.txt",unpack=True,skiprows=1,delimiter=",",usecols=(0,1))

#generate data
rhozzlist0 = create_rhozz(phi=0,Bmag=45,scattering_in=True)
#rhozzlist0_withoutSCin = create_rhozz(phi=0,Bmag=45,scattering_in=False)
#rhozzlist45 = create_rhozz(phi=45,Bmag=45,scattering_in=True)

#FSorbitsInstance.plotpoints()
fig,axes = plt.subplots(nrows=1,ncols=2, figsize=(10, 5))

axes[0].plot(thetalist,rhozzlist0,ls="-",marker="o",ms=2,label=f"Model, $\phi=0$")
#axes[0].plot(thetalist,rhozzlist45,ls="-",marker="o",ms=2,label=f"Model, $\phi=45$")
#axes[0].plot(thetalist,rhozzlist0_withoutSCin,ls="-",marker="o",ms=2,label=f"Model, $\phi=0$, no scattering in")
axes[0].plot(data_theta0,data_rhozz0,ls="-",marker="o",ms=2,label="Data, $\phi=0$")
axes[0].plot(data_theta45,data_rhozz45,ls="-",marker="o",ms=2,label="Data, $\phi=45$")
axes[0].set_ylabel(r"$\rho_{zz}$ ($m\Omega$ cm )") 
axes[0].set_xlabel(r'$\theta$')
axes[0].text(0.1,0.1,f"LSCO x=0.24\nT=30 K\nRes={res_z}x{res_xy}\n$\pi-\pi$ scatterers + isotropic\n$t_z = 0.06 - 0.08$",fontsize=10, transform=axes[0].transAxes)
axes[0].legend()

axes[1].plot(thetalist,rhozzlist0/rhozzlist0[0],ls="-",marker="o",ms=2,label=f"Model, $\phi=0$")
#axes[1].plot(thetalist,rhozzlist0_withoutSCin/rhozzlist0_withoutSCin[0],ls="-",marker="o",ms=2,label=f"Model, $\phi=0$, no scattering in")
axes[1].plot(data_theta0,data_rhozz0/data_rhozz0[-1],ls="-",marker="o",ms=2,label="Data, $\phi=0$")
axes[1].set_ylabel(r"$\rho_{zz}/\rho_{zz0}$") #($m\Omega$ cm )
axes[1].set_xlabel(r'$\theta$')

plt.tight_layout()
plt.show()