import numpy as np
import matplotlib.pyplot as plt
import dispersion
import orbitcreation
import conductivity
from makesigmalist import makelist_parallel
from time import time
from scipy.optimize import basinhopping

#global params for the fit
starttime_global = time()
thetalist = np.linspace(-14,99,80)
phi = 0
phi_rad = np.deg2rad(phi)

res_z = 20
res_xy = 100

#load data
data_theta,data_rhozz = np.loadtxt("data/2601C/2601C_phi0_T30K_B45.0T.txt",unpack=True,skiprows=1,delimiter=",",usecols=(0,1))
data_theta,data_rhozz = zip(*sorted(zip(data_theta,data_rhozz)))
data_rhozz_interp = np.interp(thetalist,data_theta,data_rhozz)

#start a log file to save fit logs
with open("fit_logs.txt","w") as f:
    f.write(f"New fit started at {time()}. Res={res_z}x{res_xy}, Phi={phi} degrees\ncost,Tz,invtau_iso,invtau_aniso\n")

def costfunction(params,fitting=False):
    Tzmultvalue,invtau_iso,invtau_aniso = params
    dispersionInstance = dispersion.LSCOdispersion(T= 190e-3,T1multvalue=-0.134,T11multvalue=0.067,Tzmultvalue=Tzmultvalue,mumultvalue=0.805)
    FSorbitsInstance = orbitcreation.fermiSurfaceOrbits(res_z,res_xy,dispersionInstance,True)
    starttime = time()
    FSorbitsInstance.createFS(tilingformat="variable",alpha=0.1,parallelised=False)
    endtime = time()
    print(f"Time taken to create Fermi Surface = {endtime - starttime}")

    conductivityInstance = conductivity.Conductivity(dispersionInstance,FSorbitsInstance,invtau_iso=invtau_iso,invtau_aniso=invtau_aniso)
    starttime = time()
    conductivityInstance.createAmatrix_Bindependent()
    endtime = time()
    print(f"Time taken to create B independent Amatrix =  {endtime - starttime}")

    def getsigma(theta):
        B = [45*np.sin(np.deg2rad(theta))*np.cos(phi_rad),45*np.sin(np.deg2rad(theta))*np.sin(phi_rad),45*np.cos(np.deg2rad(theta))]
        conductivityInstance.createAmatrix_Bdependent(B)
        conductivityInstance.createAlpha()
        conductivityInstance.createSigma()

        #print(f"Theta={theta}. Calculated total area: {conductivityInstance.areasum}. Number of orbits used {len(conductivityInstance.FSorbitsInstance.FSorbits)}. Size of Amatrix: {conductivityInstance.n}")
        return conductivityInstance.sigma,conductivityInstance.areasum

    sigmalist,rholist,arealist = makelist_parallel(getsigma,thetalist)
    rhozzlist= [rho[2,2]*10e-5 for rho in rholist]
    cost = np.sum((rhozzlist - data_rhozz_interp)**2)

    endtime_global = time()
    print(f"Fit complete. Execution time: {endtime_global-starttime_global}. Cost: {cost}")

    if not fitting:
        return rhozzlist,cost
    else:
        with open("fit_logs.txt","a") as f:
                f.write(f"{cost},{Tzmultvalue},{invtau_iso},{invtau_aniso}\n")
        return cost
            
#FSorbitsInstance.plotpoints()

res = basinhopping(costfunction,x0=(0.051,11,63.823),niter=10,minimizer_kwargs={"args":(True,)})
rhozzlist = costfunction(res.x,fitting=False)[0]

fig,axes = plt.subplots()

axes.plot(thetalist,rhozzlist,ls="-",marker="o",ms=2)
axes.plot(data_theta,data_rhozz,ls="-",marker="o",ms=2)
axes.plot(thetalist,data_rhozz_interp,ls="-",marker="o",ms=2)
axes.set_ylabel(r"$\rho_{zz}$ ($m\Omega$ cm )")
axes.set_xticks([0,45,90])
axes.set_xlabel(r'$\theta$')
axes.text(0.1,0.1,f"Res={res_z}x{res_xy}\nPhi={phi} degrees",fontsize=10, transform=axes.transAxes)

plt.show()