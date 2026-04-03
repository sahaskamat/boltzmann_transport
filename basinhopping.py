import numpy as np
import matplotlib.pyplot as plt
import dispersion
import orbitcreation
import conductivity
from makesigmalist import makelist_parallel
from time import time
from scipy.optimize import basinhopping
import os

plt.ion()

def fit_data(sample="2511A",doping="24",temp=35,theta_max=99,field=45,fixedparams=(190e-3,-0.132,0.066,0.81),fixTz=False,tzfixedvalue=0.077):
    T = temp
    #global params for the fit
    starttime_global = time()
    theta_min = -14
    thetalist = np.linspace(theta_min,theta_max,20)

    res_z = 20
    res_xy = 100

    #load data
    data_theta0,data_rhozz0 = np.loadtxt(f"data/{sample}/{sample}_phi0_T{T}K_B{field}.0T.txt",unpack=True,skiprows=1,delimiter=",",usecols=(0,1))
    data_theta0,data_rhozz0 = zip(*sorted(zip(data_theta0,data_rhozz0)))
    data_rhozz0_interp = np.interp(thetalist,data_theta0,data_rhozz0)

    data_theta45,data_rhozz45 = np.loadtxt(f"data/{sample}/{sample}_phi45_T{T}K_B{field}.0T.txt",unpack=True,skiprows=1,delimiter=",",usecols=(0,1))
    data_theta45,data_rhozz45 = zip(*sorted(zip(data_theta45,data_rhozz45)))
    data_rhozz45_interp = np.interp(thetalist,data_theta45,data_rhozz45)

    def costfunction(params):
        if fixTz:
            Tzmultvalue = np.abs(tzfixedvalue)
            invtau_iso,invtau_aniso = np.abs(params)
        else:
            Tzmultvalue,invtau_iso,invtau_aniso = np.abs(params)

        dispersionInstance = dispersion.LSCOdispersion(T=fixedparams[0],T1multvalue=fixedparams[1],T11multvalue=fixedparams[2],Tzmultvalue=Tzmultvalue,mumultvalue=fixedparams[3])

        #0.22 params
        #T= 190e-3,T1multvalue=-0.134,T11multvalue=0.067,Tzmultvalue=Tzmultvalue,mumultvalue=0.805

        FSorbitsInstance = orbitcreation.fermiSurfaceOrbits(res_z,res_xy,dispersionInstance,True)
        starttime = time()
        FSorbitsInstance.createFS(tilingformat="variable",alpha=0.1,parallelised=False)
        calculated_doping = FSorbitsInstance.calculateDoping()
        endtime = time()
        print(f"Time taken to create Fermi Surface = {endtime - starttime}. Doping = {calculated_doping}%")

        def create_rhozz(phi,Bmag):
            phi_rad = np.deg2rad(phi)

            conductivityInstance = conductivity.Conductivity(dispersionInstance,FSorbitsInstance,invtau_iso=invtau_iso,invtau_aniso=invtau_aniso)
            starttime = time()
            conductivityInstance.createAmatrix_Bindependent()
            endtime = time()
            print(f"Time taken to create B independent Amatrix =  {endtime - starttime}")

            def getsigma(theta):
                B = [Bmag*np.sin(np.deg2rad(theta))*np.cos(phi_rad),Bmag*np.sin(np.deg2rad(theta))*np.sin(phi_rad),Bmag*np.cos(np.deg2rad(theta))]
                conductivityInstance.createAmatrix_Bdependent(B)
                conductivityInstance.createAlpha()
                conductivityInstance.createSigma()

                #print(f"Theta={theta}. Calculated total area: {conductivityInstance.areasum}. Number of orbits used {len(conductivityInstance.FSorbitsInstance.FSorbits)}. Size of Amatrix: {conductivityInstance.n}")
                return conductivityInstance.sigma,conductivityInstance.areasum

            sigmalist,rholist,arealist = makelist_parallel(getsigma,thetalist)
            rhozzlist= [rho[2,2]*10e-5 for rho in rholist]

            return rhozzlist
        
        rhozz0list = create_rhozz(phi=0,Bmag=field)
        rhozz45list = create_rhozz(phi=45,Bmag=field)

        cost = np.sum((rhozz0list - data_rhozz0_interp)**2) + np.sum((rhozz45list - data_rhozz45_interp)**2)
        endtime_global = time()
        print(f"Fit complete. Execution time: {endtime_global-starttime_global}. Cost: {cost}")

        plt.clf()
        plt.plot(thetalist,rhozz0list,ls="-",marker="o",ms=2)
        plt.plot(thetalist,rhozz45list,ls="-",marker="o",ms=2)
        plt.plot(data_theta0,data_rhozz0,ls="-",marker="o",ms=2)
        plt.plot(data_theta45,data_rhozz45,ls="-",marker="o",ms=2)
        plt.ylabel(r"$\rho_{zz}$ ($m\Omega$ cm )")
        plt.xticks([0,45,90])
        plt.xlabel(r'$\theta$')
        plt.show(block=False)
        plt.pause(0.1)

        file_path = f"fit_logs_{doping}perc_{T}K.txt"
        if os.path.exists(file_path):
            with open(f"fit_logs_{doping}perc_{T}K.txt","a") as f:
                    f.write(f"{cost},{Tzmultvalue},{invtau_iso},{invtau_aniso}\n")
        else:
            with open(f"fit_logs_{doping}perc_{T}K.txt","w") as f:
                    f.write(f"LSCO {sample}, x = {doping}%\n")
                    f.write(f"T={T} K, B = {field} T, theta= {theta_min} to {theta_max}\n")
                    f.write("Fixed parameters:\n")
                    f.write(f"T={fixedparams[0]},T1multvalue={fixedparams[1]},T11multvalue={fixedparams[2]},mumultvalue={fixedparams[3]}\n")
                    f.write("cost,Tzmultvalue,invtau_iso,invtau_aniso\n")
                    f.write(f"{cost},{Tzmultvalue},{invtau_iso},{invtau_aniso}\n")

        return cost
    
    if fixTz:
        res = basinhopping(costfunction,x0=(14.04683404951196,189.85506941378708),niter=3,niter_success=100)
    else:
        res = basinhopping(costfunction,x0=(0.07942100653562356,14.04683404951196,189.85506941378708),niter=3,niter_success=100)
        return res.x[0]

if __name__ == "__main__":
    fit_data()