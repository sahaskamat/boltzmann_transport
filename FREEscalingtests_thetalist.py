import numpy as np
import matplotlib.pyplot as plt
import dispersion
import orbitcreation
import conductivity
from makesigmalist import makelist_parallel
from time import time

starttime_global = time()
thetalist = np.linspace(0,75,40)
phi = 0

phi_rad = np.deg2rad(phi)
dispersionInstance = dispersion.FREEdispersion()
initialpointsInstance = orbitcreation.InterpolatedCurves(200,dispersionInstance,False)
starttime = time()
initialpointsInstance.solveforpoints(parallelised=False)
initialpointsInstance.extendedZoneMultiply(5)
initialpointsInstance.createPlaneAnchors(20)
endtime = time()
print(f"Time taken to create initialcurves = {endtime - starttime}")

starttime = time()
def getsigma(theta):
    #Bz stays constant and field angle changes
    B = [(1*np.sin(np.deg2rad(theta))*np.cos(phi_rad))/(1*np.cos(np.deg2rad(theta))),(1*np.sin(np.deg2rad(theta))*np.sin(phi_rad))/(1*np.cos(np.deg2rad(theta))),1]
    orbitsinstance = orbitcreation.NewOrbits(dispersionInstance,initialpointsInstance)
    orbitsinstance.createOrbits(B,termination_resolution=0.1,mult_factor=40,sampletimes= np.linspace(0,1,100000),rtol=1e-11,atol=1e-12)
    orbitsinstance.createOrbitsEQS(integration_resolution=0.1)
    endtime = time()

    conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
    conductivityInstance.createAMatrix()
    conductivityInstance.createAlpha()
    conductivityInstance.createSigma()

    #if theta<65 and theta>55: orbitsinstance.orbitdiagnosticplot()
    #print(f'matrixinversion performed for {theta}')
    print(f"Theta={theta},Calculated total area: {conductivityInstance.areasum}, number of orbits used {len(conductivityInstance.orbitsInstance.orbitsEQS)}")
    return conductivityInstance.sigma,conductivityInstance.areasum


sigmalist,rholist,arealist = makelist_parallel(getsigma,thetalist)
endtime = time()
print("Time taken to create orbits:",endtime-starttime)

rhoxylist= [rho[0,0]*10e-5 for rho in rholist]

endtime_global = time()
print(f"execution time: {endtime_global-starttime_global}")

plt.scatter(thetalist,rhoxylist)
plt.ylabel(r"$\rho_{xy}$ ($m\Omega$ cm )")
plt.xlabel(r'$\theta$')
plt.show()

plt.scatter(thetalist,arealist)
plt.ylabel(r"$Area (\AA)$ cm )")
plt.xlabel(r'$\theta$')
plt.show()