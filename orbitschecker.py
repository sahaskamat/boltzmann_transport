import dispersion
import orbitcreation
import conductivity
import numpy as np
import matplotlib.pyplot as plt

dispersionInstance = dispersion.FreeElectronDispersion(1,2,7)
initialpointsInstance = orbitcreation.InitialPoints(10,dispersionInstance,False)


orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
orbitsinstance.createOrbits([0,0,1],0.02)
orbitsinstance.createOrbitsEQS(0.021)

conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
conductivityInstance.createAMatrix()
conductivityInstance.createAlpha()
conductivityInstance.createSigma()
rhoxy = np.linalg.inv(conductivityInstance.sigma)[0,1]
rhoxx = np.linalg.inv(conductivityInstance.sigma)[0,0]

print(rhoxy)
print((42.689E-3)/(rhoxy*np.pi*2))

print(rhoxx)
print(2.4271E-3/rhoxx)

orbitsinstance.plotOrbits()
orbitsinstance.plotOrbitsEQS()
