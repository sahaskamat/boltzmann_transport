import dispersion
import orbitcreation
import conductivity
import numpy as np
import matplotlib.pyplot as plt

B = [0,1,1]

dispersionInstance = dispersion.FreeElectronDispersion(1,2,7)

initialpointsInstance = orbitcreation.InitialPoints(5,dispersionInstance,False,B)

orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
orbitsinstance.createOrbits(B,0.100)
orbitsinstance.createOrbitsEQS(0.101)
print(len(orbitsinstance.orbitsEQS))
orbitsinstance.plotOrbitsEQS()

conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
conductivityInstance.createAMatrix()
conductivityInstance.createAlpha()
conductivityInstance.createSigma()
rhoxy = np.linalg.inv(conductivityInstance.sigma)[0,1]
rhoxx = np.linalg.inv(conductivityInstance.sigma)[0,0]

print(f"Absolute value for rhoxy: {rhoxy}")
print(f"Relative value of rhoxy: {(42.689824241E-3)/(rhoxy)}")

print(f"Absolute value for rhoxx: {rhoxx}")
print(f"Relative value of rhoxx: {2.42718549066E-3/rhoxx}")

print(f"Calculated total area: {conductivityInstance.areasum}")

