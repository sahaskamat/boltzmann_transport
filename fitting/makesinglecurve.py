import numpy as np
import matplotlib.pyplot as plt
import transport.dispersion as dispersion
import transport.orbitcreation as orbitcreation
import transport.conductivity as conductivity
from transport.makesigmalist import makelist_parallel,makelist_serial
from time import time

starttime_global = time()
thetalist = np.sort(np.concatenate([np.linspace(-14,99,20),[0]])) #list of 20 numbers from -14 to 99 but 0 is forced in there

res_z = 20
res_xy = 100

Tzmultvalue,invtau_iso,strength,spread_xy,n = 0.03,8.0,16.0,0.136,5
mumultvalue=0.805

plotScattering=False

dispersionInstance = dispersion.LSCOdispersion(T= 190e-3,T1multvalue=-0.134,T11multvalue=0.067,Tzmultvalue=Tzmultvalue,mumultvalue=mumultvalue)
FSorbitsInstance = orbitcreation.fermiSurfaceOrbits(res_z,res_xy,dispersionInstance,True)
starttime = time()
FSorbitsInstance.createFS(tilingformat="variable",alpha=0.1,parallelised=False)
endtime = time() 
print(f"Time taken to create Fermi Surface = {endtime - starttime}")
print(f"Doping={FSorbitsInstance.calculateDoping()}")

def create_rhozz(phi,Bmag,scattering_in=True,plotScattering=plotScattering):
    phi_rad = np.deg2rad(phi)
    B_for_LU = [Bmag*np.sin(np.deg2rad(70))*np.cos(phi_rad),Bmag*np.sin(np.deg2rad(70))*np.sin(phi_rad),Bmag*np.cos(np.deg2rad(70))]
    conductivityInstance = conductivity.Conductivity(dispersionInstance,FSorbitsInstance,invtau_iso=invtau_iso,delta_in_k=True)
    starttime = time()
    conductivityInstance.createAmatrix_Bindependent_isotropic()

    conductivityInstance.create_Hfunc(scatteringmodel="pipidelta_exp",strength=strength,spread_xy=spread_xy,n=n)
    conductivityInstance.createAmatrix_Bindependent_fwdscatter_out()
    if scattering_in: conductivityInstance.createAmatrix_Bindependent_fwdscatter_in()

    #conductivityInstance.delta_in_k = False
    #conductivityInstance.create_Hfunc(scatteringmodel="pancake",strength=strength_fwd,spread_xy=spread_fwd,spread_z=spread_fwd)
    #conductivityInstance.createAmatrix_Bindependent_fwdscatter_out()
    #if scattering_in: conductivityInstance.createAmatrix_Bindependent_fwdscatter_in()

    if plotScattering: conductivityInstance.plotScatteringOut()
    #conductivityInstance.LUdecomp(B_for_LU)
    endtime = time()
    print(f"Time taken to create B independent Amatrix =  {endtime - starttime}")

    def getsigma(theta,Bmag=Bmag):
        B = [Bmag*np.sin(np.deg2rad(theta))*np.cos(phi_rad),Bmag*np.sin(np.deg2rad(theta))*np.sin(phi_rad),Bmag*np.cos(np.deg2rad(theta))]
        conductivityInstance.createAmatrix_Bdependent(B)
        conductivityInstance.createAlpha()
        conductivityInstance.createSigma()

        print(f"Theta={theta}. Calculated total area: {conductivityInstance.areasum}. Number of orbits used {len(conductivityInstance.FSorbitsInstance.FSorbits)}. Size of Amatrix: {conductivityInstance.n}")
        return conductivityInstance.sigma,conductivityInstance.areasum


    sigmalist,rholist,arealist = makelist_parallel(getsigma,thetalist,workers=5)
    sigma_zero,area_zero = getsigma(theta=0,Bmag=0)
    sigma_9T,area_9T = getsigma(theta=0,Bmag=9)

    rho_zero = np.linalg.inv(sigma_zero)
    rho_9T = np.linalg.inv(sigma_9T)

    rhozzlist= [rho[2,2]*10e-5 for rho in rholist]
    print(f"rho_xx = {rho_zero[0,0]*10e-2}, rho_xy (9 T) = {rho_9T[0,1]*10e-2}")
    

    endtime_global = time()
    print(f"execution time: {endtime_global-starttime_global}")

    return rhozzlist

#p.savetxt("rhoxyvstPhi"+str(phi)+".dat",np.transpose([thetalist,rhoxylist]))

#generate data
rhozzlist0 = create_rhozz(phi=0,Bmag=45,scattering_in=True)
#rhozzlist0_withoutSCin = create_rhozz(phi=0,Bmag=45,scattering_in=False)
rhozzlist45 = create_rhozz(phi=45,Bmag=45,scattering_in=True,plotScattering=False)

#FSorbitsInstance.plotpoints()
fig,axes = plt.subplots(nrows=1,ncols=2, figsize=(10, 5))

axes[0].plot(thetalist,rhozzlist0,ls="-",marker="o",ms=2,label=f"Model, $\phi=0$")
axes[0].plot(thetalist,rhozzlist45,ls="-",marker="o",ms=2,label=f"Model, $\phi=45$")
#axes[0].plot(thetalist,rhozzlist0_withoutSCin,ls="-",marker="o",ms=2,label=f"Model, $\phi=0$, no scattering in")
axes[1].plot(thetalist,rhozzlist0/rhozzlist0[np.argmin(thetalist**2)],ls="-",marker="o",ms=2,label=f"Model, $\phi=0$")
axes[1].plot(thetalist,rhozzlist45/rhozzlist45[np.argmin(thetalist**2)],ls="-",marker="o",ms=2,label=f"Model, $\phi=45$")
#axes[1].plot(thetalist,rhozzlist0_withoutSCin/rhozzlist0_withoutSCin[0],ls="-",marker="o",ms=2,label=f"Model, $\phi=0$, no scattering in")

#load data
data_theta0,data_rhozz0 = np.loadtxt("data/admr_data/2511A/2511A_phi0_T30K_B45.0T.txt",unpack=True,skiprows=1,delimiter=",",usecols=(0,1))
data_theta45,data_rhozz45 = np.loadtxt("data/admr_data/2511A/2511A_phi45_T30K_B45.0T.txt",unpack=True,skiprows=1,delimiter=",",usecols=(0,1))
axes[0].plot(data_theta0,data_rhozz0,ls="-",marker="o",ms=2,label="Data, $\phi=0$")
axes[0].plot(data_theta45,data_rhozz45,ls="-",marker="o",ms=2,label="Data, $\phi=45$")
axes[1].plot(data_theta0,data_rhozz0/data_rhozz0[np.argmin(data_theta0**2)],ls="-",marker="o",ms=2,label="Data, $\phi=0$")
axes[1].plot(data_theta45,data_rhozz45/data_rhozz45[np.argmin(data_theta45**2)],ls="-",marker="o",ms=2,label="Data, $\phi=45$")

data_theta0,data_rhozz0 = np.loadtxt("data/admr_data/2601C/2601C_phi0_T30K_B45.0T.txt",unpack=True,skiprows=1,delimiter=",",usecols=(0,1))
data_theta45,data_rhozz45 = np.loadtxt("data/admr_data/2601C/2601C_phi45_T30K_B45.0T.txt",unpack=True,skiprows=1,delimiter=",",usecols=(0,1))
axes[0].plot(data_theta0,data_rhozz0,ls="-",marker="o",ms=2,label="Data, $\phi=0$")
axes[0].plot(data_theta45,data_rhozz45,ls="-",marker="o",ms=2,label="Data, $\phi=45$")
axes[1].plot(data_theta0,data_rhozz0/data_rhozz0[np.argmin(data_theta0**2)],ls="-",marker="o",ms=2,label="Data, $\phi=0$")
axes[1].plot(data_theta45,data_rhozz45/data_rhozz45[np.argmin(data_theta45**2)],ls="-",marker="o",ms=2,label="Data, $\phi=45$")


axes[0].set_ylabel(r"$\rho_{zz}$ ($m\Omega$ cm )") 
axes[0].set_xlabel(r'$\theta$')
axes[0].text(0.1,0.1,f"LSCO x=0.24\nT=35 K\nRes={res_z}x{res_xy}\n$\pi-\pi-\delta(q_z)$ scatterers + isotropic\n$t_z = 0.03$",fontsize=10, transform=axes[0].transAxes)
axes[0].legend()
axes[1].set_ylabel(r"$\rho_{zz}/\rho_{zz0}$") #($m\Omega$ cm )
axes[1].set_xlabel(r'$\theta$')

plt.tight_layout()
plt.show()