import dispersion
import orbitcreation
import conductivity
import numpy as np
import matplotlib.pyplot as plt

dispersionInstance = dispersion.LSCOdispersion()
initialpointsInstance = orbitcreation.InitialPoints(5,dispersionInstance,False)

orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
orbitsinstance.createOrbits([1*np.sin(np.deg2rad(70)),0,1*np.sin(np.deg2rad(70))],0.1)
orbitsinstance.createOrbitsEQS(0.05)

conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
conductivityInstance.createAMatrix()
conductivityInstance.createAlpha()
conductivityInstance.createSigma()

print(initialpointsInstance.k0)
orbitsinstance.plotOrbitsEQS()
rhoxy = np.linalg.inv(conductivityInstance.sigma)[0,1]

print(rhoxy)
print((42E-3)/(rhoxy))