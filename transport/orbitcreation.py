import numpy as np
from scipy.optimize import root
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from time import time
import transport.dispersion as dispersion
from numba import njit,cfunc
from itertools import product, combinations

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

class fermiSurfaceOrbits:
    """
    Inputs:
    res_z (number of slices of the fermi surface in the z direction, with each slice lying in the xy plane)
    res_xy (number of points to solve for in each slice)
    dispersion (object of class dispersion)
    doublefermisurface (bool, True if unit cell size is c/2)

    Initializes an object to create orbits on the fermi surface
    """

    def __init__(self,res_z,res_xy,dispersion,doublefermisurface):
        self.dispersion = dispersion
        self.n_cuts = res_xy
        self.n_points = res_z

        if not isinstance(doublefermisurface,bool): #check if doublefermisurface is correctly specified
            raise Exception("Argument doublefermisurface is not a boolean")

        self.doublefermisurface = doublefermisurface
        self.c = self.dispersion.c/(1+int(self.doublefermisurface)) #this makes c = dispersion.c/2 if doublefermisurface is True

        self.planeZcoords = np.linspace(-(np.pi)/self.c,(np.pi)/self.c,self.n_points+1) #create zcoordinates, each defining a plane on which points used for interpolation will be found. Exclude endpoint so that zone can be multiplied easily
        self.dkz = np.array([0,0,self.planeZcoords[1] - self.planeZcoords[0]]) #vector connecting two planes used for area calculations in conductivity

        #create reciprocal lattice vectors
        self.g1 = (2*np.pi)/self.dispersion.a
        self.g2 = (2*np.pi)/self.dispersion.b
        self.g3 = (2*np.pi)/self.c

        self.initialcurvesList = np.zeros((self.n_points,self.n_cuts,3)) #list of list of initialpoints lying on the fermi surface. each sublist should be a contiguous set of points. eg: [[point1-,point2-,point3-],[point1+,point2+,point3+]]

    def createFS(self,tilingformat="variable",alpha=0,parallelised=False):
        """
        Solves for points on the fermi surface
        Inputs:
        parallelised (bool, True if solving for points is to be parallelised across cores)
        tilingformat ("variable" if using an increased density of points near VHS, "regular" for uniform density of points)
        alpha (larger alphas correspond to higher density of points near VHS)
        Creates:
        FSorbits (a numpy array with FSorbits[i] representing a single in-plane orbit)
        """
        if tilingformat=="regular":
            philist = np.linspace(0, 2*np.pi, self.n_cuts, endpoint=False) #list of phis along which to find curves lying on the fermi surface
        else:
            philist = np.linspace(0, 2*np.pi, self.n_cuts, endpoint=False)
            philist = philist - alpha * np.sin(4*philist)

            isascending = np.all(np.diff(philist) > 0)
            if not isascending:
                raise Exception("alpha is too large and leads to overlapping points!")


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

        self.FSorbits = self.initialcurvesList

    def calculateDoping(self): 
        FSvolume = 0
        for orbit in self.FSorbits:
            orbit_plus1 = np.roll(orbit,-1,axis=0) #orbit_plus1[i] is the next point after orbit[i] on the orbit
            dk = orbit_plus1 - orbit #dk[i] = orbit[i+1] - orbit[i], vector connecting successive points
            k_cross_dk = np.cross(orbit,dk) #cross product of k and dk
            orbitarea = 0.5*np.sum(k_cross_dk[:,2]) #area of the orbit is given by the sum of the z components of k cross dk
            FSvolume += orbitarea*np.linalg.norm(self.dkz) #multiply by dkz to get volume of each orbit
        BZvolume = (2*np.pi/self.dispersion.a)*(2*np.pi/self.dispersion.b)*(2*np.pi/self.c) #volume of the Brillouin zone
        calculated_doping = 2*(0.5 - FSvolume/BZvolume) #return number of extra holes
        print(f"Doping = {calculated_doping}%")
        return calculated_doping 

    def plotpoints(self):
        ax = plt.figure().add_subplot(projection='3d')

        #plotting extendedcurveslist
        for curve in self.initialcurvesList:
            ax.scatter(curve[:,0],curve[:,1], curve[:,2], label='parametric curve',s=5)

        #making unit cell
        hx, hy, hz = np.pi/self.dispersion.a, np.pi/self.dispersion.b, np.pi/self.c  

        pts = list(product([-hx, hx], [-hy, hy], [-hz, hz]))

        for s, e in combinations(pts, 2):
            if sum(abs(a-b) for a,b in zip(s,e)) in (2*hx, 2*hy, 2*hz):
                ax.plot3D(*zip(s,e), color="k")

        plt.show()

