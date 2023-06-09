import dispersion
import orbitcreation
import conductivity
import numpy as np
import matplotlib.pyplot as plt

dispersionInstance = dispersion.LSCOdispersion()
initialpointsInstance = orbitcreation.InitialPoints(5,dispersionInstance,False)

orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
orbitsinstance.createOrbits([0,0,1],0.02)
orbitsinstance.createOrbitsEQS(0.021)

conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
conductivityInstance.createAMatrix()
conductivityInstance.createAlpha()
conductivityInstance.createSigma()
rhoxy = np.linalg.inv(conductivityInstance.sigma)[0,1]

print(rhoxy)
print((42E-3)/(rhoxy))