import numpy as np
import matplotlib.pyplot as plt
import dispersion
import orbitcreation
import conductivity
from makesigmalist import makelist_parallel

thetalist = np.linspace(0,80,20)

dispersionInstance = dispersion.LSCOdispersion()
initialpointsInstance = orbitcreation.InitialPoints(40,dispersionInstance,True)

def getsigma(theta):
    orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
    orbitsinstance.createOrbits([45*np.sin(np.deg2rad(theta)),0,45*np.cos(np.deg2rad(theta))],0.05)
    orbitsinstance.createOrbitsEQS(0.051)
    print(f'orbitcreation completed for {theta} degrees')
    #orbitsinstance.plotOrbitsEQS() #enable plotting for diagnostic purposes
    conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
    conductivityInstance.createAMatrix()
    print(f'matrixinversion performed for {theta}')
    conductivityInstance.createAlpha()
    conductivityInstance.createSigma()
    return conductivityInstance.sigma,conductivityInstance.areasum

sigmalist,rholist,arealist = makelist_parallel(getsigma,thetalist)

rhoxylist= [rho[2,2] for rho in rholist]

plt.scatter(thetalist,rhoxylist)
plt.ylabel(r"$\rho$ ($10^{-9} \Omega$ cm )")
plt.xlabel(r'$\theta$')
plt.show()

plt.scatter(thetalist,arealist)
plt.ylabel(r"$\rho$ ($10^{-9} \Omega$ cm )")
plt.xlabel(r'$\theta$')
plt.show()