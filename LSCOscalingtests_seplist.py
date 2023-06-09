import dispersion
import orbitcreation
import conductivity
import numpy as np
import matplotlib.pyplot as plt
from makesigmalist import makelist_parallel

seplist = np.linspace(0.05,0.1,10)
theta = 60

def getsigma(sep):
    dispersionInstance = dispersion.LSCOdispersion()
    initialpointsInstance = orbitcreation.InitialPoints(5,dispersionInstance,False)
    orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
    orbitsinstance.createOrbits([45*np.sin(np.deg2rad(theta)),0,45*np.cos(np.deg2rad(theta))],sep,mult_factor=1)
    orbitsinstance.createOrbitsEQS(sep + 0.001)

    conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
    conductivityInstance.createAMatrix()
    conductivityInstance.createAlpha()
    conductivityInstance.createSigma()
    return conductivityInstance.sigma,conductivityInstance.areasum

sigmalist,rholist,arealist = makelist_parallel(getsigma,seplist)

rhoxylist= [rho[2,2] for rho in rholist]

plt.scatter(seplist,rhoxylist)
plt.ylabel(r"$\rho$ ($10^{-9} \Omega$ m )")
plt.xlabel(r'integration resolution')
plt.show()