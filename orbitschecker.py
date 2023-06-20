import dispersion
import orbitcreation
import conductivity
import numpy as np
import matplotlib.pyplot as plt

dispersionInstance = dispersion.FreeElectronDispersion(1,2,7)

initialpointsInstance = orbitcreation.InitialPoints(20,dispersionInstance,False)

orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
orbitsinstance.createOrbits([0,0,1],0.01)
orbitsinstance.createOrbitsEQS(0.011)

conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
conductivityInstance.createAMatrix()
conductivityInstance.createAlpha()
conductivityInstance.createSigma()
rhoxy = np.linalg.inv(conductivityInstance.sigma)[0,1]
rhoxx = np.linalg.inv(conductivityInstance.sigma)[0,0]

print(rhoxy)
print((42.689824241E-3)/(rhoxy))

print(rhoxx)
print(2.42718549066E-3/rhoxx)

orbitsinstance.plotOrbits()
orbitsinstance.plotOrbitsEQS()
