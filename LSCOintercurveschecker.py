import dispersion
import orbitcreation
import numpy as np
import matplotlib.pyplot as plt
from time import time

dispersionInstance = dispersion.LSCOdispersion()
initialpointsInstance = orbitcreation.InterpolatedCurves(10,dispersionInstance,True,B_parr=[0,1])

starttime = time()
initialpointsInstance.solveforpoints("positive",parallelised=False)
initialpointsInstance.solveforpoints("negative",parallelised=False)
#initialpointsInstance.extendedZoneMultiply(1)
endtime = time()

print(f"Time taken to create initialcurves = {endtime - starttime}")

#plotting elements of initialpointsInstance.
ax = plt.figure().add_subplot(projection='3d')

for curve in initialpointsInstance.initialcurvesList:
    ax.scatter(curve[:,0],curve[:,1], curve[:,2], label='parametric curve')

plt.show()