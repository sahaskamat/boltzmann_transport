import dispersion
import orbitcreation
import conductivity
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.optimize import fsolve

"""
Defining initialpoints using older method, for comparison:
This is to diagnose where the singularities came from
"""
class InitialPoints:
    """
    Inputs:
    n (number of initial points)
    dispersion (object of class dispersion)
    doublefermisurface (True if fermi surface extends from 2Pi/c to -2Pi/c like LSCO)

    Computes array of initial points k0
    Computes array of vectors denoting separation between initial poins dkz

    """

    def __init__(self,n,dispersion,doublefermisurface):
        starttime = time()
            
        self.k0 = [] #this is the list of initial conditions evenly spaced in the z direction
        self.dkz = [] #list of differences of starting points of orbits (vector)
        self.dispersion = dispersion

        if not isinstance(doublefermisurface,bool):
            raise Exception("Argument doublefermisurface is not a boolean")

        c = self.dispersion.c/(1+int(doublefermisurface)) #this makes c = dispersion.c/2 if doublefermisurface is True

        #solve numeric function along the line kz = kx = 0
        for iter_num, kz0 in enumerate(np.linspace(-(np.pi)/c,(np.pi)/c,n+1)):

            if iter_num==0: #dont append first solution, this is only used to keep dkz consistent
                ky0 = fsolve(lambda ky_numeric : self.dispersion.en_numeric(0,ky_numeric,kz0),0.6)[0]
                previousk0 = np.array([0,ky0,kz0])
            else:
                ky0 = fsolve(lambda ky_numeric : self.dispersion.en_numeric(0,ky_numeric,kz0),0.6)[0]
                thisk0 = np.array([0,ky0,kz0])

                self.k0.append(thisk0)
                self.dkz.append(thisk0 - previousk0)
                previousk0 = thisk0

        endtime = time()
        print(f"Time to create initialpoints: {endtime-starttime}")


#First calculation with improved initialpoints:
"""
starttime = time()

dispersionInstance = dispersion.LSCOdispersion()
initialpointsInstance = orbitcreation.InitialPoints(20,dispersionInstance,True)

orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
orbitsinstance.createOrbits([0,1*np.sin(np.deg2rad(80)),1*np.cos(np.deg2rad(80))],0.1)
orbitsinstance.createOrbitsEQS(0.05)
orbitsinstance.plotOrbitsEQS()

conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
conductivityInstance.createAMatrix()
conductivityInstance.createAlpha()
conductivityInstance.createSigma()

endtime = time()
print(f"Execution time= {endtime-starttime}").orbi
rhoxy = np.linalg.inv(conductivityInstance.sigma)[0,1]
"""

#Second calculation with old initialpoints

starttime = time()

dispersionInstance = dispersion.LSCOdispersion()
initialpointsInstance = orbitcreation.InitialPoints(20,dispersionInstance,True)

orbitsinstance = orbitcreation.Orbits(dispersionInstance,initialpointsInstance)
orbitsinstance.createOrbits([1*np.sin(np.deg2rad(80)),0,1*np.cos(np.deg2rad(80))],0.1)
orbitsinstance.createOrbitsEQS(0.05)
orbitsinstance.plotOrbitsEQS()

conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
conductivityInstance.createAMatrix()
conductivityInstance.createAlpha()
conductivityInstance.createSigma()

endtime = time()
print(f"Execution time= {endtime-starttime}")
rhoxy = np.linalg.inv(conductivityInstance.sigma)[0,1]