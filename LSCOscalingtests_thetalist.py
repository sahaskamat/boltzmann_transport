import numpy as np
import matplotlib.pyplot as plt
import dispersion
import orbitcreation
import conductivity
from makesigmalist import makelist_parallel
from time import time

startime = time()

thetalist = np.linspace(0,80,20)

dispersionInstance = dispersion.LSCOdispersion()
initialpointsInstance = orbitcreation.InitialPoints(40,dispersionInstance,True)

def getsigma(theta):
    orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
    orbitsinstance.createOrbits([0,45*np.sin(np.deg2rad(theta)),45*np.cos(np.deg2rad(theta))],0.1)
    orbitsinstance.createOrbitsEQS(0.101)
    print(f'orbitcreation completed for {theta} degrees')
    #orbitsinstance.plotOrbitsEQS() #enable plotting for diagnostic purposes
    conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
    conductivityInstance.createAMatrix()
    conductivityInstance.createAlpha()
    conductivityInstance.createSigma()
    print(f'matrixinversion performed for {theta}')
    return conductivityInstance.sigma,conductivityInstance.areasum

sigmalist,rholist,arealist = makelist_parallel(getsigma,thetalist)

rhoxylist= [rho[2,2]*1E-4 for rho in rholist]

endtime = time()
print(f"execution time: {endtime-startime}")

plt.scatter(thetalist,rhoxylist)
plt.ylabel(r"$\rho$ ($10^{-9} m\Omega$ cm )")
plt.xlabel(r'$\theta$')
plt.show()

plt.scatter(thetalist,arealist)
plt.ylabel(r"$\rho$ ($10^{-9} m\Omega$ cm )")
plt.xlabel(r'$\theta$')
plt.show()