import dispersion
import orbitcreation
import numpy as np
import matplotlib.pyplot as plt
from time import time

dispersionInstance = dispersion.LSCOdispersion()
initialpointsInstance = orbitcreation.InterpolatedCurves(500,dispersionInstance,True,B_parr=[1,0])

starttime = time()
initialpointsInstance.solveforpoints("positive",parallelised=False)
initialpointsInstance.solveforpoints("negative",parallelised=False)
initialpointsInstance.extendedZoneMultiply(5)
endtime = time()

print(f"Time taken to create initialcurves = {endtime - starttime}")

#plotting elements of initialpointsInstance
ax = plt.figure().add_subplot(projection='3d')

for curve in initialpointsInstance.extendedcurvesList:
    ax.scatter(curve[:,0],curve[:,1], curve[:,2], label='parametric curve',s=1)

theta = np.deg2rad(80)
B = [1*np.sin(theta),0,1*np.cos(theta)]
intersections = initialpointsInstance.findintersections(B,[0,0,0])
print(intersections)
ax.scatter(intersections[:,0],intersections[:,1],intersections[:,2],c='#FF0000',s=10)

plt.show()

starttime = time()
orbitsinstance = orbitcreation.NewOrbits(dispersionInstance,initialpointsInstance)
listoforbitsinplane = orbitsinstance.createOrbitsInPlane(B,[0,0,0])
endtime = time()
print(f"Time taken to create orbit = {endtime - starttime}")

ax2 = plt.figure().add_subplot(projection='3d')
for orbit in listoforbitsinplane: ax2.scatter(orbit[:,0],orbit[:,1],orbit[:,2],s=1)
plt.show()