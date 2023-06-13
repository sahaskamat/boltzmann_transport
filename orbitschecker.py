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
rhoxy = np.linalg.inv(np.array([[conductivityInstance.sigma[0,0],conductivityInstance.sigma[0,1]],[conductivityInstance.sigma[1,0],conductivityInstance.sigma[1,1]]]))[0,1]

print(rhoxy)
print((42E-3)/(rhoxy))