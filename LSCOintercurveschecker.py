#computation libraries
import numpy as np
import matplotlib.pyplot as plt

#benchmarking libraries
from time import time
import cProfile, io
import pstats

#homemade libraries
import dispersion
import orbitcreation
import conductivity

def main():

    dispersionInstance = dispersion.LSCOdispersion()
    initialpointsInstance = orbitcreation.InterpolatedCurves(200,dispersionInstance,True)

    starttime = time()
    initialpointsInstance.solveforpoints(parallelised=False)
    initialpointsInstance.extendedZoneMultiply(0)
    initialpointsInstance.createPlaneAnchors(20)
    #initialpointsInstance.plotpoints()
    endtime = time()

    print(f"Time taken to create initialcurves = {endtime - starttime}")


    #ax = plt.figure().add_subplot(projection='3d')

    #plotting extendedcurveslist
    #for curve in initialpointsInstance.extendedcurvesList:
    #    ax.scatter(curve[:,0],curve[:,1], curve[:,2], label='parametric curve',s=1)

    theta = np.deg2rad(0)
    phi = np.deg2rad(0)
    B = [45*np.sin(theta)*np.cos(phi),45*np.sin(theta)*np.sin(phi),45*np.cos(theta)]

    #plottingintersections
    #ax.scatter(intersections[:,0],intersections[:,1],intersections[:,2],c='#FF0000',s=10)

    starttime = time()
    orbitsinstance = orbitcreation.NewOrbits(dispersionInstance,initialpointsInstance)
    orbitsinstance.createOrbits(B,termination_resolution=0.05,mult_factor=10)
    orbitsinstance.createOrbitsEQS(integration_resolution=0.05)

    #creates the point cloud for 3d printing
    pointcloud = np.concatenate(orbitsinstance.orbitsEQS)
    print(pointcloud)

    ax = plt.figure().add_subplot(projection='3d')

    ax.scatter(pointcloud[:,0],pointcloud[:,1],pointcloud[:,2])
    plt.show()

main()