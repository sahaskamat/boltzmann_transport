import dispersion
import orbitcreation
import conductivity
import numpy as np
import matplotlib.pyplot as plt
from makesigmalist import makelist_parallel

mulist = np.linspace(1,10,10)

def getsigma(mu):
    dispersionInstance = dispersion.FreeElectronDispersion(1,2,mu)
    initialpointsInstance = orbitcreation.InitialPoints(5,dispersionInstance,False)

    orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
    orbitsinstance.createOrbits([0,0,1],0.04)
    orbitsinstance.createOrbitsEQS(0.041)

    conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
    conductivityInstance.createAMatrix()
    conductivityInstance.createAlpha()
    conductivityInstance.createSigma()
    return conductivityInstance.sigma,conductivityInstance.areasum

sigmalist,rholist,arealist = makelist_parallel(getsigma,mulist)

rhoxylist= [rho[0,1] for rho in rholist]

plt.scatter(mulist,rhoxylist)
plt.ylabel(r"$\rho$ ($10^{-9} \Omega$ m )")
plt.xlabel(r'$\mu$')
plt.show()