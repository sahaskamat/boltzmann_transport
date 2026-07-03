import numpy as np
import matplotlib.pyplot as plt
import transport.dispersion as dispersion
import transport.orbitcreation as orbitcreation
import transport.conductivity as conductivity
from transport.makesigmalist import makelist_parallel,makelist_serial
from time import time
from scipy.optimize import differential_evolution,Bounds,minimize
import os

plt.ion()

def fit_data(sample="2511A",doping="24",temp=35,theta_max=99,field=45.0,fixedparams=(190e-3,-0.132,0.066,0.81),scatteringmodel="pipidelta",initialguess=(0.03411,9.9,120000,0.1235,3.4),plot=True):
    """
    Inputs: 
    scatteringmodel (string, corresponding to the name of a function scatteringmodel(deltak,g,**kwargs) in transport.scattering_kernels)
    initialguess (tuple of (tzmultvalue,**kwargs) to be passed to the global optimizer as an initial guess)
    bounds (instance of type scipy.optimize.Bounds that specifies search space for solutions)
    parallel_over_theta (bool, if True, values of resistivity are calculated paralelly for all values of theta. If False, values are calculated paralelly over all generations)
    plot (bool, if True, fit is plotted at each costfunction evaluation.)
    """

    T = temp
    #global params for the fit
    starttime_global = time()
    theta_min = -14
    thetalist = np.linspace(theta_min,theta_max,20)

    res_z = 20
    res_xy = 100

    #load data
    data_theta0,data_rhozz0 = np.loadtxt(f"data/admr_data/{sample}/{sample}_phi0_T{T}K_B{field}T.txt",unpack=True,skiprows=1,delimiter=",",usecols=(0,1))
    data_theta0,data_rhozz0 = zip(*sorted(zip(data_theta0,data_rhozz0)))
    data_rhozz0_interp = np.interp(thetalist,data_theta0,data_rhozz0)

    data_theta45,data_rhozz45 = np.loadtxt(f"data/admr_data/{sample}/{sample}_phi45_T{T}K_B{field}T.txt",unpack=True,skiprows=1,delimiter=",",usecols=(0,1))
    data_theta45,data_rhozz45 = zip(*sorted(zip(data_theta45,data_rhozz45)))
    data_rhozz45_interp = np.interp(thetalist,data_theta45,data_rhozz45)

    def costfunction(params):
        Tzmultvalue,invtau_iso,strength,spread_xy,n = np.abs(params)

        dispersionInstance = dispersion.LSCOdispersion(T=fixedparams[0],T1multvalue=fixedparams[1],T11multvalue=fixedparams[2],Tzmultvalue=Tzmultvalue,mumultvalue=fixedparams[3])

        FSorbitsInstance = orbitcreation.fermiSurfaceOrbits(res_z,res_xy,dispersionInstance,True)
        FSorbitsInstance.createFS(tilingformat="variable",alpha=0.1,parallelised=False)
        calculated_doping = FSorbitsInstance.calculateDoping()

        def create_rhozz(phi,Bmag):
            phi_rad = np.deg2rad(phi)

            conductivityInstance = conductivity.Conductivity(dispersionInstance,FSorbitsInstance,invtau_iso=invtau_iso,delta_in_k=True)
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

            sigmalist,rholist,arealist = makelist_serial(getsigma,thetalist)

            rhozzlist= [rho[2,2]*10e-5 for rho in rholist]

            return rhozzlist
        
        rhozz0list = create_rhozz(phi=0,Bmag=field)
        rhozz45list = create_rhozz(phi=45,Bmag=field)

        cost = np.sum((rhozz0list - data_rhozz0_interp)**2) + np.sum((rhozz45list - data_rhozz45_interp)**2)

        if plot:
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

        file_path = f"fitting/fit_logs/fit_logs_nonRTA/fit_logs_{doping}perc_{T}K_{scatteringmodel}.txt"
        if os.path.exists(file_path):
            with open(file_path,"a") as f:
                    #f.write(f"{cost},{Tzmultvalue},{invtau_iso},{strength},{spread_xy},{spread_z},{n}\n")
                    print(f"Cost={cost},Tzmultvalue={Tzmultvalue},invtau_iso={invtau_iso},strength={strength},spread_xy={spread_xy},n={n}")
        else:
            with open(file_path,"w") as f:
                    #f.write(f"LSCO {sample}, x = {doping}%\n")
                    #f.write(f"T={T} K, B = {field} T, theta= {theta_min} to {theta_max}, phi = 0 and 45\n")
                    #f.write("Fixed parameters:\n")
                    #f.write(f"T={fixedparams[0]},T1multvalue={fixedparams[1]},T11multvalue={fixedparams[2]}\n")
                    #f.write("cost,Tzmultvalue,invtau_iso,strength,spread_xy,spread_z,n\n")
                    #f.write(f"{cost},{Tzmultvalue},{invtau_iso},{strength},{spread_xy},{spread_z},{n}\n")
                    print(f"Cost={cost},Tzmultvalue={Tzmultvalue},invtau_iso={invtau_iso},strength={strength},spread_xy={spread_xy},n={n}")

        return cost
    
    """Differential evolution"""
    res = minimize(
        costfunction,
        x0=initialguess,
        method="Powell"
    )

    return res.x

if __name__ == "__main__":
    fit_data()