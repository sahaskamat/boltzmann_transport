import numpy as np
import matplotlib.pyplot as plt
import transport.dispersion as dispersion
import transport.orbitcreation as orbitcreation
import transport.conductivity as conductivity
from transport.makesigmalist import makelist_parallel
from time import time
from scipy.optimize import basinhopping
from scipy.optimize import direct,Bounds
from scipy.optimize import differential_evolution
import os

plt.ion()

def fit_data_shape(sample="2511A",doping="24",temp=30,theta_max=99,phi=0,field=45.0,fixedparams=(190e-3,-0.132,0.066),mumultvalue=0.81):
    #fixed params: (T,T1multvalue,T11multvalue)

    T = temp
    #global params for the fit
    starttime_global = time()
    theta_min = -14
    thetalist = np.linspace(theta_min,theta_max,20)

    res_z = 20
    res_xy = 100

    scatteringmodel="pipi"

    #load data
    data_theta,data_rhozz = np.loadtxt(f"data/admr_data/{sample}/{sample}_phi{phi}_T{T}K_B{field}T.txt",unpack=True,skiprows=1,delimiter=",",usecols=(0,1))
    data_theta,data_rhozz = zip(*sorted(zip(data_theta,data_rhozz)))
    data_rhozz_interp = np.interp(thetalist,data_theta,data_rhozz)

    #keeping spread fixed
    def costfunction(params):
        #define parameters used to construct dispersion,orbits and conductivity
        Tzmultvalue,invtau_iso,strength,spread_xy,n = np.abs(params)

        dispersionInstance = dispersion.LSCOdispersion(T=fixedparams[0],T1multvalue=fixedparams[1],T11multvalue=fixedparams[2],Tzmultvalue=Tzmultvalue,mumultvalue=mumultvalue)

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

            conductivityInstance = conductivity.Conductivity(dispersionInstance,FSorbitsInstance,invtau_iso=invtau_iso)
            starttime = time()
            conductivityInstance.createAmatrix_Bindependent_isotropic()
            conductivityInstance.create_Hfunc(scatteringmodel=scatteringmodel,strength=strength,spread_xy=spread_xy,n=n)
            conductivityInstance.createAmatrix_Bindependent_fwdscatter_out()
            conductivityInstance.createAmatrix_Bindependent_fwdscatter_in()
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
        
        rhozzlist = create_rhozz(phi,Bmag=field)

        cost = np.sum((rhozzlist - data_rhozz_interp)**2)
        endtime_global = time()
        print(f"Fit complete. Execution time: {endtime_global-starttime_global}. Cost: {cost}")

        plt.clf()
        plt.plot(thetalist,rhozzlist,ls="-",marker="o",ms=2)
        plt.plot(data_theta,data_rhozz,ls="-",marker="o",ms=2)
        plt.ylabel(r"$\rho_{zz}$ ($m\Omega$ cm )")
        plt.xticks([0,45,90])
        plt.xlabel(r'$\theta$')
        plt.show(block=False)
        plt.pause(0.1)

        file_path = f"fit_logs_nonRTA/fit_logs_{doping}perc_{T}K_{scatteringmodel}.txt"
        if os.path.exists(file_path):
            with open(file_path,"a") as f:
                    f.write(f"{cost},{Tzmultvalue},{invtau_iso},{strength},{spread_xy}\n")
                    print(f"Cost={cost},Tzmultvalue={Tzmultvalue},invtau_iso={invtau_iso},strength={strength}")
        else:
            with open(file_path,"w") as f:
                    f.write(f"LSCO {sample}, x = {doping}%\n")
                    f.write(f"T={T} K, B = {field} T, theta= {theta_min} to {theta_max}, phi = {phi}\n")
                    f.write("Fixed parameters:\n")
                    f.write(f"T={fixedparams[0]},T1multvalue={fixedparams[1]},T11multvalue={fixedparams[2]}\n")
                    f.write("cost,Tzmultvalue,invtau_iso,strength,spread_xy\n")
                    f.write(f"{cost},{Tzmultvalue},{invtau_iso},{strength},{spread_xy}\n")

        return cost
    
    """Basinhopping:
    # Define local optimizer settings (optional but recommended)
    minimizer_args = {"method": "L-BFGS-B", "options": {"maxiter": 100}}

    #define a custom step taker
    rng = np.random.default_rng()
    def step(x):
        x[0] += rng.uniform(-0.01, 0.01) #Tzmultvalue
        x[1] += rng.uniform(-5,5) #invtau_iso
        x[2] += rng.uniform(-200,200) #strength
        x[3] += rng.uniform(-0.5,0.5) #spread_xy
        x[4] += rng.uniform(-2,2) #angular var
        return x

    res = basinhopping(
        costfunction, 
        x0=initial_guess, 
        niter=10,
        niter_success=1000,
        take_step=step
    )
    return res.x
    """

    """DIRECT
    #initial guess (params): Tzmultvalue,invtau_iso,strength,spread_xy
    bounds = Bounds([0.01,0,0,0.05],[0.12,30,1000,0.4])
    res = direct(costfunction,bounds)
    return res.x
    """

    """Differential evolution"""
    bounds = Bounds([0.01,0,0,0.05,0],[0.12,30,100,0.4,12])
    res = differential_evolution(costfunction,bounds,x0=(0.074,12,30,0.1,2),popsize=10,maxiter=10000)
    

if __name__ == "__main__":
    print(fit_data_shape())