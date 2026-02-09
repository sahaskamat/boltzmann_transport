import numpy as np
import matplotlib.pyplot as plt
import dispersion
import orbitcreation
import conductivity
from makesigmalist import makelist_parallel
from time import time

starttime_global = time()
thetalist = np.linspace(0,180,50)
phi = 0

res_z = 20
res_xy = 150

phi_rad = np.deg2rad(phi)
dispersionInstance = dispersion.LSCOdispersion()
initialpointsInstance = orbitcreation.InterpolatedCurves(res_z,res_xy,dispersionInstance,True)
orbitsinstance = orbitcreation.NewOrbits(dispersionInstance,initialpointsInstance)
starttime = time()
initialpointsInstance.solveforpoints(parallelised=False)
endtime = time()
print(f"Time taken to create initialcurves = {endtime - starttime}")

starttime = time()
def getsigma(theta):
    B = [45*np.sin(np.deg2rad(theta))*np.cos(phi_rad),45*np.sin(np.deg2rad(theta))*np.sin(phi_rad),45*np.cos(np.deg2rad(theta))]
    #print(f'orbitcreation completed for {theta} degrees')
    #if theta>=60: orbitsinstance.plotOrbitsEQS() #enable plotting for diagnostic purposes
    conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance,B)
    conductivityInstance.createAMatrix()
    conductivityInstance.createAlpha()
    conductivityInstance.createSigma()

    #orbitsinstance.orbitdiagnosticplot()
    #print(f'matrixinversion performed for {theta}')
    print(f"Theta={theta}. Calculated total area: {conductivityInstance.areasum}. Number of orbits used {len(conductivityInstance.orbitsInstance.orbitsEQS)}. Size of Amatrix: {conductivityInstance.n}")
    return conductivityInstance.sigma,conductivityInstance.areasum


sigmalist,rholist,arealist = makelist_parallel(getsigma,thetalist)
endtime = time()
print("Time taken to create orbits:",endtime-starttime)

rhoxylist= [rho[2,2]*10e-5 for rho in rholist]

endtime_global = time()
print(f"execution time: {endtime_global-starttime_global}")

np.savetxt("rhoxyvstPhi"+str(phi)+".dat",np.transpose([thetalist,rhoxylist]))

fig,axes = plt.subplots()

axes.plot(thetalist,rhoxylist,ls="-",marker="o",ms=2)
axes.set_ylabel(r"$\rho_{zz}$ ($m\Omega$ cm )")
axes.set_xlabel(r'$\theta$')
axes.text(0.1,0.1,f"Res={res_z}x{res_xy}\nPhi={phi} degrees",fontsize=10, transform=axes.transAxes)

plt.show()