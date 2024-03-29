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

    def __init__(self,npoints,dispersion,doublefermisurface):
        self.dispersion = dispersion

        if not isinstance(doublefermisurface,bool): #check if doublefermisurface is correctly specified
            raise Exception("Argument doublefermisurface is not a boolean")

        self.doublefermisurface = doublefermisurface
        self.c = self.dispersion.c/(1+int(self.doublefermisurface)) #this makes c = dispersion.c/2 if doublefermisurface is True

        self.planeZcoords = np.linspace(-(np.pi)/self.c,(np.pi)/self.c,npoints+1) #create zcoordinates, each defining a plane on which points used for interpolation will be found. Exclude endpoint so that zone can be multiplied easily

        self.initialcurvesList = [] #list of list of initialpoints. each sublist should be a contiguous set of points. eg: [[point1-,point2-,point3-],[point1+,point2+,point3+]]
        self.interpolatedcurveslist = [] #list of interpolated functions that output [x,y] coordinates along a set of points when given a z-coordinate

    def solveforpoints(self,parallelised=False):
        """
        Solves for points on four sides of the fermi surface
        Inputs:
        parallelised (bool, True if solving for points is to be parallelised across cores)
        Creates:
        initialcurvesList (a list containing two numpy arrays, with each numpy array containing contiguous points lying along the fermi surface)
        """

        #angularwidth = np.pi/10 #angular width around van hole points to solve for points
        #philist = np.concatenate([np.linspace(-angularwidth+alpha,angularwidth+alpha,6) for alpha in np.linspace(0,2*np.pi,4,endpoint=False)]) #list of phis along which to find curves lying on the fermi surface
        philist = [0,np.pi]

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
        for phi in philist:
            startingpointsarray = getpoints(self.planeZcoords,phi)

            self.initialcurvesList.append(np.delete(startingpointsarray,-1,axis=0))

            self.interpolatedcurveslist.append(interp1d(np.transpose(startingpointsarray[:,2]),np.transpose(startingpointsarray[:,0:2])))

    def plotpoints(self):
        ax = plt.figure().add_subplot(projection='3d')

        #plotting extendedcurveslist
        for curve in self.initialcurvesList:
            ax.scatter(curve[:,0],curve[:,1], curve[:,2], label='parametric curve',s=10)

        #plotting interpolatedcurves
        interpolatedcurveslist = np.array([[[interpolatingfunction(kz)[0],interpolatingfunction(kz)[1],kz] for kz in np.linspace((-np.pi)/self.c,(np.pi)/self.c,1000)] for interpolatingfunction in self.interpolatedcurveslist])

        for curve in interpolatedcurveslist:
            ax.scatter(curve[:,0],curve[:,1], curve[:,2], label='parametric curve',s=1)

        plt.show()

    def extendedZoneMultiply(self,nzones=1):
        """
        Extends initialcurvesList to 2*nzones zones lying along the kz direction. nzones in positive kz, nzones in negative kz.
        Creates:
        extendedcurvesList (a list containing two numpy arrays, with each numpy array containing contiguous points lying along the fermi surface)
        """
        self.extendedcurvesList = []

        c = self.dispersion.c/(1+int(self.doublefermisurface)) #this makes c = dispersion.c/2 if doublefermisurface is True

        if not hasattr(self,'initialcurvesList'): raise Exception("initialcurvesList has not been created. Call solveforpoints() first.")

        for curve in self.initialcurvesList:

            extendedcurve = curve

            for zonenumber in range(1,nzones+1): #iterates from 1 to nzones
                previouszoneCurve = curve + [0,0,-(2*np.pi*zonenumber)/c] #create initial curve in the BZ below extendedcurve
                nextzoneCurve = curve + [0,0,+(2*np.pi*zonenumber)/c]   #create initial curve in the BZ above extendedcurve
                extendedcurve = np.concatenate((previouszoneCurve,extendedcurve,nextzoneCurve),axis=0) #concatenate all three curves to extend extendedcurve by one BZ on each side

            self.extendedcurvesList.append(extendedcurve)


    def findintersections(self,normalvector,pointonplane):
        """
        Find intersections between self.extendedcurveslist and the plane defined by normalvector and pointonplane
        Returns an array of intersection points
        """
        def planeequation(kvector):
            """
            planeequation(kvector) = 0 is the equation of the plane defined by normalvector and pointonplane
            """
            return np.dot(kvector,normalvector) - np.dot(pointonplane,normalvector)

        def binaryfindintersection(upperbound,lowerbound,planeequation,interpolatedcurve):
            """
            Inputs:
            upperbound (3-vector denoting one end of curve in which planeequation=0 has to be found)
            lowerbound (3-vector denoting the other end of curve in which planeequation=0 has to be found)
            planeequation (function of [x,y,z] whose roots have to be found)
            extendedinterpolatedcurve (function [x(z),y(z)] that parameterizes a curve between upperbound and lowerbound, along which a root will be found)
            """
            def extendedinterpolatedcurve(kz):
                return interpolatedcurve(((kz+np.pi/self.c)%(2*np.pi/self.c) - np.pi/self.c))

            upperz = upperbound[2]
            lowerz = lowerbound[2]

            zcoords = np.linspace(lowerz,upperz,1000) #array of z coordinates between upperz and lowerz to be used for finding a solution

            interpolatedpoints = np.transpose(np.vstack((extendedinterpolatedcurve(zcoords),zcoords)))  #array of three vectors lying on the curve between upperz and lowerz, along whcih to find solutions
            intersectionindices = [np.nonzero(np.diff(np.sign(planeequation(interpolatedpoints))))] #coordinate where planequation flips sign
            
            intersectionpoint = np.reshape(interpolatedpoints[intersectionindices],(3))

            return intersectionpoint

        intersectionindices = [np.nonzero(np.diff(np.sign(planeequation(curve)))) for curve in self.extendedcurvesList] #contains a list of indices for each set of points denoting intersections with the plane
        intersectionpoints = []

        for (curve_id,curve) in enumerate(self.extendedcurvesList):
            for id in intersectionindices[curve_id][0]:
                lowerbound = curve[id]
                upperbound = curve[id+1]
                intersectionpoints.append(binaryfindintersection(upperbound, lowerbound, planeequation, self.interpolatedcurveslist[curve_id]))

        return np.array(intersectionpoints)

    def createPlaneAnchors(self,n):
        """
        Inputs:
        n (this denotes the number of plane anchors to create, each will lie along the z-axis. corresponds to the number of planes used to create orbits later)
        """

        c = self.dispersion.c/(1+int(self.doublefermisurface)) #this makes c = dispersion.c/2 if doublefermisurface is True

        planeAnchors_zcoords = np.linspace(-(np.pi)/c,(np.pi)/c,n+1,endpoint=False) #create zcoordinates, each defining a plane on which orbits will be created
        self.planeAnchors = [np.array([0,0,z]) for z in planeAnchors_zcoords] #make points out of zcoordinates, to be passed to createorbitsinplane()

        self.dkz = self.planeAnchors[1] - self.planeAnchors[0] #this creates a vector that takes you from one plane to another


class NewOrbits:
    """
    Inputs:
    dispersion (object of type dispersion)
    interpolatedcurves (onject of type interpolatedcurves)

    """
    def __init__(self,dispersion,interpolatedcurves):
        self.dispersion = dispersion
        self.interpolatedcurves = interpolatedcurves

        self.timespentfindingpoints = 0

        try:
            self.interpolatedcurves.extendedcurvesList #since many methods for this class are defined using extendedcurvesList, this makes sure it exists
        except NameError:
            print("initialcurvesList not extended for interpolatedcurves, extending with nzones =1")
            self.interpolatedcurves.extendedZoneMultiply()

        #the force v \cross B term but now with fixed B
        RHS_numeric = self.dispersion.RHS_numeric

        @cfunc(lsoda_sig)
        def RHS_withB(t,k,dk,B ):
            [dk[0],dk[1],dk[2]] = RHS_numeric([k[0],k[1],k[2]],[B[0],B[1],B[2]])

        self.RHS_withB_address = RHS_withB.address



    def createOrbitsInPlane(self,B,pointonplane,termination_resolution,sampletimes,mult_factor,rtol,atol):
        """
        Inputs:
        B (3-vector specifying direction of magnetic field)
        pointonplane (a point lying in the plane of orbit to be created. Does not have to lie on the orbit itself.)
        termination_resolution (radius in which integration terminates, default value 0.05)
        sampletimes (times at which to sample solution)
        mult_factor (multiplication factor for B during integration: larger means faster integration, but greater than 10 and integration breaks ihavenoideawhy)

        Creates all possible orbits lying in the plane defined by B and pointonplane
        """

        #find starting points lying on plane and also time it
        starttime = time()
        initialpointslist = self.interpolatedcurves.findintersections(B,pointonplane)
        endtime = time()
        self.timespentfindingpoints += (endtime-starttime)

        #list of orbits in plane
        orbitsinplane = []

        while initialpointslist.size > 0:
            initial = np.array(initialpointslist[0])

            #starttime = time()
            solution,success = lsoda(self.RHS_withB_address, initial, sampletimes,data=B*mult_factor,rtol=rtol,atol=atol)
            #endtime = time()
            orbit = (solution)

            #print("Time taken to create orbits = ",endtime - starttime)
            
            #now check if any other elements of initialpointslist appear in orbit

            elementstobedeleted = []
            for id,initialpoint in enumerate(initialpointslist):
                normsarray = np.linalg.norm(orbit - initialpoint,axis=1)
                closenessarray = np.isclose(normsarray,0,atol=termination_resolution)
                if np.any(closenessarray):
                    elementstobedeleted.append(id) 

            initialpointslist = np.delete(initialpointslist,elementstobedeleted,axis=0)#deletes initial point elements that lie on the current orbit

            orbitsinplane.append(orbit)

        #print("Number of orbits created in plane:",len(orbitsinplane)) diagnostic to make sure all extra orbits are created
        return orbitsinplane

    def createOrbits(self,B,termination_resolution = 0.05,sampletimes = np.linspace(0,4,10000),mult_factor=1,rtol=1e-7,atol=1e-8):
        """
        Inputs:
        B (3-vector specifying direction of magnetic field)
        termination_resolution (radius in which integration terminates, default value 0.05)
        sampletimes (times at which to sample solution) (default value is value for LSCO at critial doping)
        mult_factor (multiplication factor for B during integration: larger means faster integration, but greater than 10 and integration breaks ihavenoideawhy)

        Creates all possible orbits lying on planes defined by B and self.interpolatedcurves.planeAnchors
        """
        self.orbits = []
        self.termination_resolution = termination_resolution
        self.B = B
        self.B_normalized = np.array(B)/dispersion.norm(B)

        for point in self.interpolatedcurves.planeAnchors: #creates orbits for each planeanchor and then appends it to self.orbits
            listoforbitsinplane = self.createOrbitsInPlane(self.B_normalized,point,termination_resolution = termination_resolution,sampletimes = sampletimes,mult_factor=mult_factor,rtol=rtol,atol=atol)
            for orbit in listoforbitsinplane: self.orbits.append(orbit)

    def createOrbitsEQS(self,integration_resolution=0.05):
        """
        Inputs:
        integration_resolution (spacing between points, this becomes the resolution for integration when used in conjunction with a Conductivity object)

        Creates:
        List of equally spaced orbits orbitsEQS
        """
        self.orbitsEQS = []

        def appendSingleOrbitEQS(orbit):
            """
            This function creates an equally spaced orbit out of the input orbit, and appends it to self.orbitsEQS if it has mroe than three points
            """
            #Find distances of points from initial point to tell where orbit closes
            diffvectors = orbit - np.outer(np.ones(orbit.shape[0]),orbit[0])
            diffvectorsnorm = np.linalg.norm(diffvectors,axis=1) #array of distances from initial point
            d_diffvectorsnorm = np.diff(diffvectorsnorm,append=0) #derivative of diffvectorsnorm. Value of 0 appended to keep length same as diffvectorsnorm

            #orbit is close to completion when 1.diffvectorsnorm is close to zero 2.the derivative of diffvectorssnorm is negative 
            whereclosetozero = np.isclose(diffvectorsnorm,0,atol=integration_resolution)
            wherenegativederivative = d_diffvectorsnorm < 0
            orbitcompletionindices = np.arange(diffvectorsnorm.size)[np.logical_and(whereclosetozero,wherenegativederivative)] #set of indices where orbit closes

            if orbitcompletionindices.size == 0:
                raise Exception("Some orbits did not close! Increase integration time or mult_factor, and try again")

            firstindexofcompletion = orbitcompletionindices[0] #first index where orbit closes
            firstorbit = orbit[:firstindexofcompletion,:] #array elements corresponding to the first completed orbit

            #plt.plot(firstorbit[:,0],ls ="",marker="o",ms = 1)
            shiftedfirstorbit = np.roll(firstorbit,-1,axis=0)
            cumnormslist = np.cumsum(np.linalg.norm(shiftedfirstorbit-firstorbit,axis=1))
            equallyspacesindices = np.nonzero(np.diff(np.sign(np.sin(cumnormslist*np.pi/integration_resolution))))
            singleorbitEQS = firstorbit[equallyspacesindices]

            #if orbits1EQS has less than three points, we discard the orbit as numerical path derivatives won't be well defined
            if not singleorbitEQS.shape[0] < 3:
                self.orbitsEQS.append(singleorbitEQS)

        for orbit in self.orbits: appendSingleOrbitEQS(orbit)

    def orbitdiagnosticplot(self):
        ax = plt.figure().add_subplot(projection='3d')

        for curve in self.interpolatedcurves.extendedcurvesList:
            ax.scatter(curve[:,0],curve[:,1], curve[:,2], label='parametric curve',s=1)

        for orbit in self.orbitsEQS: ax.scatter(orbit[:,0],orbit[:,1],orbit[:,2],s=1)

        plt.show()