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

    dispersionInstance = dispersion.FREEdispersion()
    initialpointsInstance = orbitcreation.InterpolatedCurves(200,dispersionInstance,False)

    starttime = time()
    initialpointsInstance.solveforpoints(parallelised=False)
    initialpointsInstance.extendedZoneMultiply(3)
    initialpointsInstance.createPlaneAnchors(10)
    #initialpointsInstance.plotpoints()
    endtime = time()

    print(f"Time taken to create initialcurves = {endtime - starttime}")


    #ax = plt.figure().add_subplot(projection='3d')

    #plotting extendedcurveslist
    #for curve in initialpointsInstance.extendedcurvesList:
    #    ax.scatter(curve[:,0],curve[:,1], curve[:,2], label='parametric curve',s=1)

    theta = np.deg2rad(80)
    phi = np.deg2rad(0)
    B = [np.sin(theta)/np.cos(theta),0,1]

    #plottingintersections
    #ax.scatter(intersections[:,0],intersections[:,1],intersections[:,2],c='#FF0000',s=10)

    starttime = time()
    orbitsinstance = orbitcreation.NewOrbits(dispersionInstance,initialpointsInstance)
    orbitsinstance.createOrbits(B,termination_resolution=0.1,mult_factor=15,sampletimes=np.linspace(0,1,100000),rtol=1e-11,atol=1e-12)
    orbitsinstance.createOrbitsEQS(integration_resolution=0.1)

    plt.show()
    orbitsinstance.orbitdiagnosticplot()
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

cProfile.run('main()',filename='stats.prof')
#main()
#plt.show()
