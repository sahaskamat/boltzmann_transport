import dispersion
import orbitcreation
import conductivity
import numpy as np
import matplotlib.pyplot as plt
from makesigmalist import makelist_parallel

maglist = np.linspace(0,10,10)

dispersionInstance = dispersion.FreeElectronDispersion(1,2,0.01)
initialpointsInstance = orbitcreation.InitialPoints(1,dispersionInstance,False)

orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
orbitsinstance.createOrbits([0,0,1],0.01)
orbitsinstance.createOrbitsEQS(0.68)
orbitsinstance.plotOrbitsEQS()

def getsigma(mag):
    orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
    orbitsinstance.createOrbits([0,0,mag],0.01)
    orbitsinstance.createOrbitsEQS(0.68)

    conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
    conductivityInstance.createAMatrix()
    conductivityInstance.createAlpha()
    conductivityInstance.createSigma()
    return conductivityInstance.sigma,conductivityInstance.areasum

sigmalist,rholist,arealist = makelist_parallel(getsigma,maglist)

rhoxxlist= [rho[0,0] for rho in rholist]
rhoxylist = [rho[0,1] for rho in rholist]

plt.scatter(maglist,rhoxxlist)
plt.ylabel(r"$\rho_{xx}$")
plt.xlabel(r'H')
plt.show()

plt.scatter(maglist,rhoxylist)
plt.ylabel(r"$\rho_{xy}$")
plt.xlabel(r'H')
plt.show()