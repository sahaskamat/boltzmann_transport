import numpy as np
import matplotlib.pyplot as plt
import dispersion
import orbitcreation
import conductivity
from makesigmalist import makelist_parallel
from time import time

startime = time()

thetalist = np.linspace(0,80,20)

def getsigma(theta):
    B = [0,45*np.sin(np.deg2rad(theta)),45*np.cos(np.deg2rad(theta))]

    dispersionInstance = dispersion.LSCOdispersion()
    initialpointsInstance = orbitcreation.InitialPoints(40,dispersionInstance,True,B)

    orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
    orbitsinstance.createOrbits(B,0.1,mult_factor=0.5)
    orbitsinstance.createOrbitsEQS(0.0701)
    print(f'orbitcreation completed for {theta} degrees')
    if theta>=60: orbitsinstance.plotOrbitsEQS() #enable plotting for diagnostic purposes
    conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
    conductivityInstance.createAMatrix()
    conductivityInstance.createAlpha()
    conductivityInstance.createSigma()
    print(f'matrixinversion performed for {theta}')
    print(f"Calculated total area: {conductivityInstance.areasum}, number of orbits used {len(conductivityInstance.orbitsInstance.orbitsEQS)}")
    return conductivityInstance.sigma,conductivityInstance.areasum

sigmalist,rholist,arealist = makelist_parallel(getsigma,thetalist)

rhoxylist= [rho[2,2] for rho in rholist]

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