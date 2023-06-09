import dispersion
import orbitcreation
import conductivity
import numpy as np
import matplotlib.pyplot as plt
from makesigmalist import makelist_parallel

thetalist = np.linspace(10,60,10)

dispersionInstance = dispersion.LSCOdispersion()
initialpointsInstance = orbitcreation.InitialPoints(10,dispersionInstance,True)

def getsigma(theta):
    orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
    orbitsinstance.createOrbits([45*np.sin(np.deg2rad(theta)),0,45*np.cos(np.deg2rad(theta))],0.05)
    orbitsinstance.createOrbitsEQS(0.051)
    orbitsinstance.plotOrbitsEQS()
    conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
    conductivityInstance.createAMatrix()
    conductivityInstance.createAlpha()
    conductivityInstance.createSigma()
    return conductivityInstance.sigma,conductivityInstance.areasum

sigmalist,rholist,arealist = makelist_parallel(getsigma,thetalist)

rhoxylist= [rho[2,2] for rho in rholist]

plt.scatter(thetalist,rhoxylist)
plt.ylabel(r"$\rho$ ($10^{-9} \Omega$ cm )")
plt.xlabel(r'$\theta$')
plt.show()