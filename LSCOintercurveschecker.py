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

#plotting elements of initialpointsInstance.
ax = plt.figure().add_subplot(projection='3d')

for curve in initialpointsInstance.extendedcurvesList:
    ax.scatter(curve[:,0],curve[:,1], curve[:,2], label='parametric curve',s=1)

theta = np.deg2rad(0)
intersections = initialpointsInstance.findintersections([1*np.sin(theta),0,1*np.cos(theta)],[0,0,0])
print(intersections)
ax.scatter(intersections[:,0],intersections[:,1],intersections[:,2],c='#FF0000',s=10)

plt.show()