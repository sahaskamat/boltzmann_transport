import dispersion
import orbitcreation
import conductivity
import numpy as np
import matplotlib.pyplot as plt
from makesigmalist import makelist_parallel

thetalist = np.linspace(10,60,10)

dispersionInstance = dispersion.FreeElectronDispersion(1,2,7)
initialpointsInstance = orbitcreation.InitialPoints(5,dispersionInstance,False)

def getsigma(theta):
    orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
    orbitsinstance.createOrbits([1*np.sin(np.deg2rad(theta)),0,1*np.cos(np.deg2rad(theta))],0.04)
    orbitsinstance.createOrbitsEQS(0.041)

    conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
    conductivityInstance.createAMatrix()
    conductivityInstance.createAlpha()
    conductivityInstance.createSigma()
    return conductivityInstance.sigma,conductivityInstance.areasum

sigmalist,rholist,arealist = makelist_parallel(getsigma,thetalist)

rhoxylist= [rho[0,0] for rho in rholist]

plt.scatter(thetalist,rhoxylist)
plt.ylabel(r"$\rho$ ($10^{-9} \Omega$ cm )")
plt.xlabel(r'$\theta$')
plt.show()