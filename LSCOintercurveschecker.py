import dispersion
import orbitcreation
import conductivity
import numpy as np
import matplotlib.pyplot as plt
from time import time

B = [0,1*np.sin(np.deg2rad(70)),1*np.cos(np.deg2rad(70))]

dispersionInstance = dispersion.LSCOdispersion()
initialpointsInstance = orbitcreation.InterpolatedCurves(1000,dispersionInstance,True,B_parr=[1,0])

starttime = time()
initialpointsInstance.solveforpoints("positive",parallelised=False)
initialpointsInstance.solveforpoints("negative",parallelised=False)
initialpointsInstance.extendedZoneMultiply(1)
endtime = time()

print(f"Time taken to create initialcurves = {endtime - starttime}")

#plotting elements of initialpointsInstance.
ax = plt.figure().add_subplot(projection='3d')

for curve in initialpointsInstance.extendedcurvesList:
    ax.scatter(curve[:,0],curve[:,1], curve[:,2], label='parametric curve')

plt.show()