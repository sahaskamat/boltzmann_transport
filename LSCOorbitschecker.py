import dispersion
import orbitcreation
import conductivity
import numpy as np
import matplotlib.pyplot as plt
from time import time

starttime = time()

dispersionInstance = dispersion.LSCOdispersion()
initialpointsInstance = orbitcreation.InitialPoints(20,dispersionInstance,True)

orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
orbitsinstance.createOrbits([0,1*np.sin(np.deg2rad(80)),1*np.cos(np.deg2rad(80))],0.1)
orbitsinstance.createOrbitsEQS(0.05)

conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
conductivityInstance.createAMatrix()
conductivityInstance.createAlpha()
conductivityInstance.createSigma()

endtime = time()
print(f"Execution time= {endtime-starttime}")

orbitsinstance.plotOrbitsEQS()
rhoxy = np.linalg.inv(conductivityInstance.sigma)[0,1]
