import dispersion
import orbitcreation
import conductivity
import numpy as np
import matplotlib.pyplot as plt


dispersionInstance = dispersion.FreeElectronDispersion(1,2,0.01)
initialpointsInstance = orbitcreation.InitialPoints(1,dispersionInstance,False)


orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
orbitsinstance.createOrbits([0,0,1],0.1)
orbitsinstance.createOrbitsEQS(0.68)
orbitsinstance.plotOrbitsEQS()

conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
conductivityInstance.createAMatrix()
np.savetxt("A.txt",conductivityInstance.A)
