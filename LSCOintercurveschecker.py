import dispersion
import orbitcreation
import conductivity
import numpy as np
import matplotlib.pyplot as plt
from time import time

dispersionInstance = dispersion.LSCOdispersion()
initialpointsInstance = orbitcreation.InterpolatedCurves(500,dispersionInstance,True,B_parr=[1,0])

starttime = time()
initialpointsInstance.solveforpoints("positive",parallelised=False)
initialpointsInstance.solveforpoints("negative",parallelised=False)
initialpointsInstance.extendedZoneMultiply(5)
initialpointsInstance.createPlaneAnchors(20)
endtime = time()

print(f"Time taken to create initialcurves = {endtime - starttime}")

#ax = plt.figure().add_subplot(projection='3d')

#plotting extendedcurveslist
#for curve in initialpointsInstance.extendedcurvesList:
#    ax.scatter(curve[:,0],curve[:,1], curve[:,2], label='parametric curve',s=1)

theta = np.deg2rad(60)
B = [1*np.sin(theta),0,1*np.cos(theta)]
intersections = initialpointsInstance.findintersections(B,[0,0,0])

#plottingintersections
#ax.scatter(intersections[:,0],intersections[:,1],intersections[:,2],c='#FF0000',s=10)

starttime = time()
orbitsinstance = orbitcreation.NewOrbits(dispersionInstance,initialpointsInstance)
orbitsinstance.createOrbits(B)
orbitsinstance.createOrbitsEQS()
listoforbits = orbitsinstance.orbitsEQS
endtime = time()
print(f"Time taken to create orbits = {endtime - starttime}")

starttime = time()
conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
conductivityInstance.createAMatrix()
conductivityInstance.createAlpha()
conductivityInstance.createSigma()
endtime = time()
print(f"Time taken to calculate conductivity = {endtime - starttime}")

orbitsinstance.orbitdiagnosticplot()

#for orbit in listoforbits: ax.scatter(orbit[:,0],orbit[:,1],orbit[:,2],s=1)
#plt.show()