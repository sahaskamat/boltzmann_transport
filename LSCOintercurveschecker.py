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
    initialpointsInstance.extendedZoneMultiply(1)
    initialpointsInstance.createPlaneAnchors(1)
    #initialpointsInstance.plotpoints()
    endtime = time()

    print(f"Time taken to create initialcurves = {endtime - starttime}")


    #ax = plt.figure().add_subplot(projection='3d')

    #plotting extendedcurveslist
    #for curve in initialpointsInstance.extendedcurvesList:
    #    ax.scatter(curve[:,0],curve[:,1], curve[:,2], label='parametric curve',s=1)

    theta = np.deg2rad(10)
    phi = np.deg2rad(0)
    B = [45*np.sin(theta)*np.cos(phi),45*np.sin(theta)*np.sin(phi),45*np.cos(theta)]
    intersections = initialpointsInstance.findintersections(B,[0,0,0])

    #plottingintersections
    #ax.scatter(intersections[:,0],intersections[:,1],intersections[:,2],c='#FF0000',s=10)

    starttime = time()
    orbitsinstance = orbitcreation.NewOrbits(dispersionInstance,initialpointsInstance)
    orbitsinstance.createOrbits(B,termination_resolution=0.01,mult_factor=10)
    orbitsinstance.createOrbitsEQS(integration_resolution=0.01)
    """
    listoforbits = orbitsinstance.orbitsEQS
    endtime = time()
    print(f"Time taken to create orbits = {endtime - starttime}, number of orbits created {len(orbitsinstance.orbitsEQS)}, time spent finding intitialpoints {orbitsinstance.timespentfindingpoints}")

    starttime = time()
    conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
    endtime = time()
    print(f"Time taken to create conductivityInstance =  {endtime - starttime}")

    starttime = time()
    conductivityInstance.createAMatrix()
    print(conductivityInstance.A.shape)
    endtime = time()
    print(f"Time taken to create Amatrix =  {endtime - starttime}")

    starttime = time()
    conductivityInstance.createAlpha()
    endtime = time()
    print(f"Time taken to create Alpha =  {endtime - starttime}")

    starttime = time()
    conductivityInstance.createSigma()
    endtime = time()
    print(f"Time taken to calculate conductivity = {endtime - starttime}")

    orbitsinstance.orbitdiagnosticplot()

    #for orbit in listoforbits: ax.scatter(orbit[:,0],orbit[:,1],orbit[:,2],s=1)
    #plt.show()    
    """

#cProfile.run('main()',filename='stats.prof')
main()
plt.show()
