import numpy as np
import matplotlib.pyplot as plt
import dispersion
import orbitcreation
import conductivity
from makesigmalist import makelist_parallel
from time import time

starttime_global = time()
thetalist = np.linspace(0,80,40)
phi = 0

phi_rad = np.deg2rad(phi)
dispersionInstance = dispersion.LSCOdispersion()
initialpointsInstance = orbitcreation.InterpolatedCurves(200,dispersionInstance,True)
starttime = time()
initialpointsInstance.solveforpoints(parallelised=False)
initialpointsInstance.extendedZoneMultiply(5)
initialpointsInstance.createPlaneAnchors(60)
endtime = time()
print(f"Time taken to create initialcurves = {endtime - starttime}")

starttime = time()
def getsigma(theta):
    B = [45*np.sin(np.deg2rad(theta))*np.cos(phi_rad),45*np.sin(np.deg2rad(theta))*np.sin(phi_rad),45*np.cos(np.deg2rad(theta))]
    orbitsinstance = orbitcreation.NewOrbits(dispersionInstance,initialpointsInstance)
    orbitsinstance.createOrbits(B,termination_resolution=0.01,mult_factor=10)
    orbitsinstance.createOrbitsEQS(integration_resolution=0.01)
    endtime = time()
    #print(f'orbitcreation completed for {theta} degrees')
    #if theta>=60: orbitsinstance.plotOrbitsEQS() #enable plotting for diagnostic purposes
    conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
    conductivityInstance.createAMatrix()
    conductivityInstance.createAlpha()
    conductivityInstance.createSigma()

    #orbitsinstance.orbitdiagnosticplot()
    #print(f'matrixinversion performed for {theta}')
    print(f"Theta={theta},Calculated total area: {conductivityInstance.areasum}, number of orbits used {len(conductivityInstance.orbitsInstance.orbitsEQS)}")
    return conductivityInstance.sigma,conductivityInstance.areasum


sigmalist,rholist,arealist = makelist_parallel(getsigma,thetalist)
endtime = time()
print("Time taken to create orbits:",endtime-starttime)

rhoxylist= [rho[2,2]*10e-5 for rho in rholist]

endtime_global = time()
print(f"execution time: {endtime_global-starttime_global}")

plt.scatter(thetalist,rhoxylist)
plt.ylabel(r"$\rho_{zz}$ ($m\Omega$ cm )")
plt.xlabel(r'$\theta$')
plt.show()

plt.scatter(thetalist,arealist)
plt.ylabel(r"$Area (\AA)$ cm )")
plt.xlabel(r'$\theta$')
plt.show()