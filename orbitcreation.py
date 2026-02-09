import numpy as np
from scipy.optimize import root
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from time import time
import dispersion
from numba import njit,cfunc
from numbalsoda import lsoda_sig, lsoda

########################
# Module to find number of CPUSnumbalsoda
##########################
import multiprocessing
from joblib import delayed, Parallel

#find number of cpus
try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2   # arbitrary default

if cpus >60: cpus =60 #joblib breaks if you use too many CPUS (>61)
########################
# Module to find number of CPUS
##########################

class InterpolatedCurves:
    """
    Inputs:
    npoints (number of points to solve for on each side of FS)
    dispersion (object of class dispersion)
    doublefermisurface (bool, True if unit cell size is c/2)
    B_parr (list containing two floats, representing the in plane direction of B)
    B (if B_parr is not supplied, in plane direction of B will be inferred)

    This class replaces the older InitialPoints class, and is compatible with the Conductivity class out of the box
    """

    def __init__(self,n_points,n_cuts,dispersion,doublefermisurface):
        self.dispersion = dispersion
        self.n_cuts = n_cuts
        self.n_points = n_points

        if not isinstance(doublefermisurface,bool): #check if doublefermisurface is correctly specified
            raise Exception("Argument doublefermisurface is not a boolean")

        self.doublefermisurface = doublefermisurface
        self.c = self.dispersion.c/(1+int(self.doublefermisurface)) #this makes c = dispersion.c/2 if doublefermisurface is True

        self.planeZcoords = np.linspace(-(np.pi)/self.c,(np.pi)/self.c,n_points+1) #create zcoordinates, each defining a plane on which points used for interpolation will be found. Exclude endpoint so that zone can be multiplied easily
        self.dkz = np.array([0,0,self.planeZcoords[1] - self.planeZcoords[0]]) #vector connecting two planes used for area calculations in conductivity

        self.initialcurvesList = np.zeros((n_points,n_cuts,3)) #list of list of initialpoints lying on the fermi surface. each sublist should be a contiguous set of points. eg: [[point1-,point2-,point3-],[point1+,point2+,point3+]]

    def solveforpoints(self,parallelised=False):
        """
        Solves for points on four sides of the fermi surface
        Inputs:
        n_cuts (number of cuts along which to solve for points. each cut is a line along which points lying on the fermi surface are solved for)
        parallelised (bool, True if solving for points is to be parallelised across cores)
        Creates:
        initialcurvesList (a list containing two numpy arrays, with each numpy array containing contiguous points lying along the fermi surface)
        """

        #angularwidth = np.pi/10 #angular width around van hole points to solve for points
        #philist = np.concatenate([np.linspace(-angularwidth+alpha,angularwidth+alpha,6) for alpha in np.linspace(0,2*np.pi,4,endpoint=False)]) #list of phis along which to find curves lying on the fermi surface
        philist = np.arange(0,2*np.pi,2*np.pi/self.n_cuts) #list of phis along which to find curves lying on the fermi surface

        def getpoints(startingZcoords,phi):
            """
            solve for points lying on FS for a given array of z coordinates and phi
            """
            def energyAlongPhi(r0):
                """
                returns the value of self.dispersion.en_numeric() along a fixed phi, for a distance from origin r0 at z coordinates in startingZcoords
                """
                return self.dispersion.en_numeric(r0*np.cos(phi),r0*np.sin(phi),startingZcoords)

            sol = root(energyAlongPhi,0.5*np.ones(startingZcoords.size))
            r0 = sol.x #list of radius vector moduli corresponding to points lying on the FS

            return np.transpose(np.array([r0*np.cos(phi),r0*np.sin(phi),startingZcoords]))

        #create startingpoints by iterating getpoints() over self.planeZcoords
        for id,phi in enumerate(philist):
            startingpointsarray = getpoints(self.planeZcoords,phi)

            self.initialcurvesList[:,id] = np.delete(startingpointsarray,-1,axis=0)

    def plotpoints(self):
        ax = plt.figure().add_subplot(projection='3d')

        #plotting extendedcurveslist
        for curve in self.initialcurvesList:
            ax.scatter(curve[:,0],curve[:,1], curve[:,2], label='parametric curve',s=10)

        #plotting interpolatedcurves
        #interpolatedcurveslist = np.array([[[interpolatingfunction(kz)[0],interpolatingfunction(kz)[1],kz] for kz in np.linspace((-np.pi)/self.c,(np.pi)/self.c,1000)] for interpolatingfunction in self.interpolatedcurveslist])

        #for curve in interpolatedcurveslist:
        #    ax.scatter(curve[:,0],curve[:,1], curve[:,2], label='parametric curve',s=1)

        plt.show()


class NewOrbits:
    """
    Inputs:
    dispersion (object of type dispersion)
    interpolatedcurves (onject of type interpolatedcurves)
    B (magnetic field as a 3-vector)
    """
    def __init__(self,dispersion,interpolatedcurves,B):
        self.dispersion = dispersion
        self.interpolatedcurves = interpolatedcurves

        self.timespentfindingpoints = 0
        self.orbitsEQS = self.interpolatedcurves.initialcurvesList
        self.B = B #this definition is legacy since conductivity assumes NewOrbits.B exits, but needs to go and enter as an input to conductivity