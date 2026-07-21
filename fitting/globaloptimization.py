import numpy as np
import matplotlib.pyplot as plt
import transport.dispersion as dispersion
import transport.orbitcreation as orbitcreation
import transport.conductivity as conductivity
from transport.makesigmalist import makelist_parallel,makelist_serial
from time import time
from scipy.optimize import differential_evolution,Bounds
from multiprocessing.pool import ThreadPool
import os

plt.ion()

def fit_data(sample="2511A",doping="24",temp=35,theta_max=99,field=45.0,fixedparams=(190e-3,-0.132,0.066,0.81),scatteringmodel="pipidelta_exp",delta_in_k=False,initialguess=(0.03411,9.9,120000,0.1235),bounds = Bounds([0.02,0.1,12,0.05],[0.045,30,30,0.5]),parallel_over_theta="True",plot=False):
    """
    Inputs: 
    scatteringmodel (string, corresponding to the name of a function scatteringmodel(deltak,g,**kwargs) in transport.scattering_kernels)
    initialguess (tuple of (tzmultvalue,**kwargs) to be passed to the global optimizer as an initial guess)
    bounds (instance of type scipy.optimize.Bounds that specifies search space for solutions)
    parallel_over_theta (bool, if True, values of resistivity are calculated paralelly for all values of theta. If False, values are calculated paralelly over all generations)
    plot (bool, if True, fit is plotted at each costfunction evaluation.)
    """

    #global params for the fit
    starttime_global = time()
    theta_min = -14
    thetalist = np.sort(np.concatenate([np.linspace(theta_min,theta_max,20),[0]])) #list of 20 numbers from theta_min to theta_max but 0 is forced in there

    res_z = 20
    res_xy = 100

    #load data
    data_theta0,data_rhozz0 = np.loadtxt(f"data/admr_data/{sample}/{sample}_phi0_T{temp}K_B{field}T.txt",unpack=True,skiprows=1,delimiter=",",usecols=(0,1))
    data_theta0,data_rhozz0 = data_theta0[np.argsort(data_theta0)],data_rhozz0[np.argsort(data_theta0)]
    data_rhozz0_interp = np.interp(thetalist,data_theta0,data_rhozz0)
    data_rhozz0_normalized = data_rhozz0_interp/data_rhozz0_interp[np.argmin(thetalist**2)]

    data_theta45,data_rhozz45 = np.loadtxt(f"data/admr_data/{sample}/{sample}_phi45_T{temp}K_B{field}T.txt",unpack=True,skiprows=1,delimiter=",",usecols=(0,1))
    data_theta45,data_rhozz45 = data_theta45[np.argsort(data_theta45)],data_rhozz45[np.argsort(data_theta45)]
    data_rhozz45_interp = np.interp(thetalist,data_theta45,data_rhozz45)
    data_rhozz45_normalized = data_rhozz45_interp/data_rhozz45_interp[np.argmin(thetalist**2)]

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    def costfunction(params):
        Tzmultvalue,invtau_iso,strength,spread_xy = np.abs(params)
        T,T1multvalue,T11multvalue,mumultvalue,n = fixedparams
        
        dispersionInstance = dispersion.LSCOdispersion(T=T,T1multvalue=T1multvalue,T11multvalue=T11multvalue,Tzmultvalue=Tzmultvalue,mumultvalue=mumultvalue)

        FSorbitsInstance = orbitcreation.fermiSurfaceOrbits(res_z,res_xy,dispersionInstance,True)
        FSorbitsInstance.createFS(tilingformat="variable",alpha=0.1,parallelised=False)
        calculated_doping = FSorbitsInstance.calculateDoping()

        def create_rhozz(phi,Bmag):
            phi_rad = np.deg2rad(phi)

            conductivityInstance = conductivity.Conductivity(dispersionInstance,FSorbitsInstance,invtau_iso=invtau_iso,delta_in_k=delta_in_k)
            conductivityInstance.createAmatrix_Bindependent_isotropic()
            conductivityInstance.create_Hfunc(scatteringmodel=scatteringmodel,strength=strength,spread_xy=spread_xy,n=n)
            conductivityInstance.createAmatrix_Bindependent_fwdscatter_out()
            conductivityInstance.createAmatrix_Bindependent_fwdscatter_in()

            def getsigma(theta):
                B = [Bmag*np.sin(np.deg2rad(theta))*np.cos(phi_rad),Bmag*np.sin(np.deg2rad(theta))*np.sin(phi_rad),Bmag*np.cos(np.deg2rad(theta))]
                conductivityInstance.createAmatrix_Bdependent(B)
                conductivityInstance.createAlpha()
                conductivityInstance.createSigma()

                #print(f"Theta={theta}. Calculated total area: {conductivityInstance.areasum}. Number of orbits used {len(conductivityInstance.FSorbitsInstance.FSorbits)}. Size of Amatrix: {conductivityInstance.n}")
                return conductivityInstance.sigma,conductivityInstance.areasum

            if parallel_over_theta: sigmalist,rholist,arealist = makelist_serial(getsigma,thetalist)
            else: sigmalist,rholist,arealist = makelist_serial(getsigma,thetalist)

            rhozzlist= [rho[2,2]*10e-5 for rho in rholist]

            return rhozzlist
        
        rhozz0list = create_rhozz(phi=0,Bmag=field)
        rhozz45list = create_rhozz(phi=45,Bmag=field)

        rhozz0list_normalized = rhozz0list/rhozz0list[np.argmin(thetalist**2)]
        rhozz45list_normalized = rhozz45list/rhozz45list[np.argmin(thetalist**2)]

        cost_leastsq = np.sum((rhozz0list - data_rhozz0_interp)**2) + np.sum((rhozz45list - data_rhozz45_interp)**2)
        cost_shape = np.sum((rhozz0list_normalized-data_rhozz0_normalized)**2) + np.sum((rhozz45list_normalized-data_rhozz45_normalized)**2)
        cost = cost_leastsq + cost_shape*500
        #print(f"Cost_leastsq={cost_leastsq},cost_diff={cost_diff}")

        if plot:
            ax[0].cla()
            ax[1].cla()

            ax[0].plot(thetalist, rhozz0list, ls="-", marker="o", ms=2)
            ax[0].plot(thetalist, rhozz45list, ls="-", marker="o", ms=2)
            ax[0].plot(data_theta0, data_rhozz0, ls="-", marker="o", ms=2)
            ax[0].plot(data_theta45, data_rhozz45, ls="-", marker="o", ms=2)
            ax[0].set_ylabel(r"$\rho_{zz}$ ($m\Omega$ cm)")
            ax[0].set_xlabel(r"$\theta$")
            ax[0].set_xticks([0, 45, 90])

            ax[1].plot(thetalist, rhozz0list_normalized, ls="-", marker="o", ms=2)
            ax[1].plot(thetalist, rhozz45list_normalized, ls="-", marker="o", ms=2)
            ax[1].plot(data_theta0, data_rhozz0/data_rhozz0[np.argmin(data_theta0**2)], ls="-", marker="o", ms=2)
            ax[1].plot(data_theta45, data_rhozz45/data_rhozz45[np.argmin(data_theta45**2)], ls="-", marker="o", ms=2)
            ax[1].set_ylabel(r"$\rho_{zz}/\rho_{zz0}$ ($m\Omega$ cm)")
            ax[1].set_xlabel(r"$\theta$")
            ax[1].set_xticks([0, 45, 90])

            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)

        file_path = f"fitting/fit_logs/fit_logs_nonRTA/fit_logs_{doping}perc_{temp}K_{scatteringmodel}.txt"
        if os.path.exists(file_path):
            with open(file_path,"a") as f:
                    f.write(f"{cost},{Tzmultvalue},{invtau_iso},{strength},{spread_xy}\n")
                    print(f"Cost={cost},Tzmultvalue={Tzmultvalue},invtau_iso={invtau_iso},strength={strength},spread_xy={spread_xy}")
        else:
            with open(file_path,"w") as f:
                    f.write(f"LSCO {sample}, x = {doping}%\n")
                    f.write(f"T={temp} K, B = {field} T, theta= {theta_min} to {theta_max}, phi = 0 and 45\n")
                    f.write("Fixed parameters:\n")
                    f.write(f"T,T1multvalue,T11multvalue,mumultvalue,n")
                    f.write(f"{T},{T1multvalue},{T11multvalue},{mumultvalue},{n}")
                    f.write("cost_leastsq+cost_shape,Tzmultvalue,invtau_iso,strength,spread_xy\n")
                    f.write(f"{cost},{Tzmultvalue},{invtau_iso},{strength},{spread_xy}\n")
                    print(f"Cost={cost},Tzmultvalue={Tzmultvalue},invtau_iso={invtau_iso},strength={strength},spread_xy={spread_xy}")

        return cost
    
    """Differential evolution"""
    if parallel_over_theta: res = differential_evolution(costfunction,bounds,x0=initialguess,popsize=5,maxiter=10000)
    else: 
         pool = ThreadPool(4)
         res = differential_evolution(
              costfunction,
              bounds,
              x0=initialguess,
              strategy="rand2bin",
              popsize=40,
              mutation=(1.0, 1.9),
              recombination=0.95,
              init="sobol",
              workers=pool.map,
              updating="deferred")

    return res.x

if __name__ == "__main__":
    fit_data()