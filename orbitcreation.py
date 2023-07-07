import numpy as np
from scipy.optimize import fsolve
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from time import time



########################
# Module to find number of CPUS
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
    """

    def __init__(self,npoints,dispersion,doublefermisurface,B_parr=None,B=None):
        if B!=None:
            self.theta = np.arctan(np.float64(B[1])/np.float64(B[0])) #theta = arctan(By/Bx) numpy.float64 is used to ensure division by zero returns Inf and not an error
        elif B_parr!=None:
            self.theta = np.arctan(np.float64(B_parr[1])/np.float64(B_parr[0])) #theta = arctan(By/Bx) numpy.float64 is used to ensure division by zero returns Inf and not an error
        else:
            raise Exception("Both B and B_parr are unspecified: cannot create interpolatedCurves")
        
        self.dispersion = dispersion

        if not isinstance(doublefermisurface,bool): #check if doublefermisurface is correctly specified
            raise Exception("Argument doublefermisurface is not a boolean")

        self.doublefermisurface = doublefermisurface
        c = self.dispersion.c/(1+int(self.doublefermisurface)) #this makes c = dispersion.c/2 if doublefermisurface is True

        self.planeZcoords = np.linspace(-(np.pi)/c,(np.pi)/c,npoints+1,endpoint=False) #create zcoordinates, each defining a plane on which points used for interpolation will be found. Exclude endpoint so that zone can be multiplied easily

        self.initialcurvesList = [] #list of list of initialpoints. each sublist should be a contiguous set of points. eg: [[point1-,point2-,point3-],[point1+,point2+,point3+]]

    def solveforpoints(self,sign,parallelised=False):
        """
        Solves for points on each side of the fermi surface
        Inputs:
        sign (decides the sign of theta along which to search for solutions. For a complete set, use both "positive" and "negative")
        parallelised (bool, True if solving for points is to be parallelised across cores)
        Creates:
        initialcurvesList (a list containing two numpy arrays, with each numpy array containing contiguous points lying along the fermi surface)
        """

        if sign == "positive":
            signfactor = 1
        elif sign == "negative":
            signfactor = -1
        else:
            raise Exception("Sign should be a string with value 'positive' or 'negative'")

        def getpoint(startingZcoord):
            """
            solve for point lying on FS for a given z coordinate (sign is inherited from the method variable signfactor) 
            """
            r0 = fsolve(lambda r0 : self.dispersion.en_numeric(r0*np.cos(self.theta),r0*np.sin(self.theta),startingZcoord),0.5*signfactor,factor=0.1,maxfev=500000)[0] 
            return np.array([r0*np.cos(self.theta),r0*np.sin(self.theta),startingZcoord])
        
        #create startingpoints by iterating getpoints() over self.planeZcoords
        if parallelised:
            startingpointslist = Parallel(n_jobs=int(cpus))(delayed(getpoint)(startingZcoord) for startingZcoord in self.planeZcoords)
        else:
            startingpointslist = [getpoint(startingZcoord) for startingZcoord in self.planeZcoords]

        self.initialcurvesList.append(np.array(startingpointslist))

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
        
        intersectionindices = [np.nonzero(np.diff(np.sign(list(map(planeequation,curve))))) for curve in self.extendedcurvesList] #contains a list of indices for each set of points denoting intersections with the plane

        intersectionpoints = [curve[id] for (curve_id,curve) in enumerate(self.extendedcurvesList) for id in intersectionindices[curve_id][0] ] #extracts intersection coordinates from intersection indices

        return np.array(intersectionpoints)

class NewOrbits:
    """
    Inputs:
    dispersion (object of type dispersion)
    interpolatedcurves (onject of type interpolatedcurves)
    
    """
    def __init__(self,dispersion,interpolatedcurves):
        self.dispersion = dispersion
        self.interpolatedcurves = interpolatedcurves

        try: 
            self.interpolatedcurves.extendedcurvesList #since many methods for this class are defined using extendedcurvesList, this makes sure it exists
        except NameError:
            print("initialcurvesList not extended for interpolatedcurves, extending with nzones =1")
            self.interpolatedcurves.extendedZoneMultiply()


    def createOrbitsInPlane(self,B,pointonplane,termination_resolution = 0.05,sampletimes = np.linspace(0,400,100000),mult_factor=1):
        """
        Inputs:
        B (3-vector specifying direction of magnetic field)
        pointonplane (a point lying in the plane of orbit to be created. Does not have to lie on the orbit itself.)
        termination_resolution (radius in which integration terminates, default value 0.05)
        sampletimes (times at which to sample solution)
        mult_factor (multiplication factor for B during integration: larger means faster integration, but greater than 10 and integration breaks ihavenoideawhy)

        Creates all possible orbits lying in the plane defined by B and pointonplane
        """
        self.B = np.array(B)
        B_normalized = np.array(B)/np.linalg.norm(B)
        self.termination_resolution = termination_resolution

        initialpointslist = self.interpolatedcurves.findintersections(B,pointonplane)

        #the force v \cross B term but now with fixed B
        RHS_withB = lambda t,k : self.dispersion.RHS_numeric(k,mult_factor*B_normalized)

        #timestamps on which to start and end integration (in principle, the endpoint will not be required)
        t_span = (sampletimes[0],sampletimes[-1])

        #list of orbits in plane
        orbitsinplane = []

        while initialpointslist.size > 0:
            initial = initialpointslist[0]

            # solve differential equation to find a single orbit
            event_fun = lambda t,k : np.linalg.norm(np.array(k)-np.array(initial)) - termination_resolution # define event function
            event_fun.terminal = True # make event function terminal -- this is the terminating event
            event_fun.direction = -1 #event function only triggered when it is decreasing

            solution = solve_ivp(RHS_withB, t_span, initial, t_eval = sampletimes, dense_output=True, events=event_fun,method='LSODA',rtol=1e-4,atol=1e-5)
            orbit = np.transpose(solution.y)

            #now check if any other elements of initialpointslist appear in orbit
            elementstobedeleted= []

            for orbitpoint in orbit:
                for id,initialpoint in enumerate(initialpointslist):
                    if np.linalg.norm(orbitpoint-initialpoint) < termination_resolution: #this means that initialpoint lies on orbit
                        elementstobedeleted.append(id) #add index of initialpoint to the array of positions to be deleted

            initialpointslist = np.delete(initialpointslist,elementstobedeleted,axis=0)#deletes initial point elements that lie on the current orbit

            orbitsinplane.append(orbit)
            print("Orbit created and appended")

        return orbitsinplane

class InitialPoints:
    """
    Inputs:
    n (number of initial points)
    dispersion (object of class dispersion)
    doublefermisurface (True if fermi surface extends from 2Pi/c to -2Pi/c like LSCO)

    Computes array of initial points k0
    Computes array of vectors denoting separation between initial poins dkz

    """

    def __init__(self,n,dispersion,doublefermisurface,B):
        """
        n: number of initial points to create
        dispersion: object of type dispersion
        doublefermisurface: true if unit cell height is c/2 and not c
        B: magnetic field vector. Note that code is currently only compatible with Bx = 0

        """
        self.dispersion = dispersion

        if not isinstance(doublefermisurface,bool):
            raise Exception("Argument doublefermisurface is not a boolean")

        c = self.dispersion.c/(1+int(doublefermisurface)) #this makes c = dispersion.c/2 if doublefermisurface is True
        #starttime = time()

        self.k0 = np.array([[0,0,kz0] for kz0 in np.linspace(-(np.pi)/c,(np.pi)/c,n+1)])
        #this is the list of initial conditions evenly spaced in the z direction, lying along x=y=0.
        #these will be used to create initial conditions lying on the fermi surface

        self.dkz = np.diff(self.k0,axis=0)[0] #dkz is a vector that takes you from one initial plane to the other
        #print(self.dkz)

        self.klist1 = []
        self.klist2 = []
        self.k0 = np.delete(self.k0,0,axis=0) #this deletes the first element of k0 to avoid over-tiling the fermi surface

        #solve numeric function along the kx = 0 lying in the plane of kz0 and perpendicular to the magnetic field
        for kz0 in self.k0:
            ky0 = fsolve(lambda ky_numeric : self.dispersion.en_numeric(0,ky_numeric,(np.dot(kz0,B) - ky_numeric*B[1])/B[2]),-np.pi/dispersion.a+0.1,factor=0.1,maxfev=500000)[0] #figure out why the [0] is here

            if abs(self.dispersion.en_numeric(0,ky0,(np.dot(kz0,B) - ky0*B[1])/B[2]))<1e-3 : self.klist1.append(np.array([0,ky0,(np.dot(kz0,B) - ky0*B[1])/B[2]])) #append solved point only if it lies on the FS

            ky1 = fsolve(lambda ky_numeric : self.dispersion.en_numeric(0,ky_numeric,(np.dot(kz0,B) - ky_numeric*B[1])/B[2]),np.pi/dispersion.a-0.1,factor=0.1,maxfev=500000)[0] #figure out why the [0] is here

            if abs(self.dispersion.en_numeric(0,ky1,(np.dot(kz0,B) - ky1*B[1])/B[2]))<1e-3 : self.klist2.append(np.array([0,ky1,(np.dot(kz0,B) - ky1*B[1])/B[2]])) #append solved point only if it lies on the FS

        # endtime = time()
        #time measuring diagnostic #print(f"Time to create initialpoints: {endtime-starttime}")


class Orbits:
    def __init__(self,dispersion,initialpoints):
        self.dispersion = dispersion
        self.klist1 = initialpoints.klist1
        self.klist2 = initialpoints.klist2


    def createOrbits(self,B,termination_resolution = 0.05,sampletimes = np.linspace(0,400,100000),mult_factor=1):
        """
        Inputs:
        B (magnetic field 3-vector)
        termination_resolution (radius in which integration terminates, default value 0.05)
        sampletimes (times at which to sample solution)
        mult_factor (multiplication factor for B during integration: larger means faster integration, but greater than 10 and integration breaks ihavenoideawhy)

        Creates:
        List of orbits, orbits
        """
        #start timer
        # starttime = time()

        self.B = np.array(B)
        B_normalized = np.array(B)/np.linalg.norm(B)
        self.termination_resolution = termination_resolution

        #the force v \cross B term but now with fixed B
        RHS_withB = lambda t,k : self.dispersion.RHS_numeric(k,mult_factor*B_normalized)

        #timestamps on which to sample the solution
        t_span = (sampletimes[0],sampletimes[-1])

        #iterate over different initial conditions, each corresponding to a given orbit
        def createsingleorbit(initial):
            # solve differential equation to find a single orbit
            event_fun = lambda t,k : np.linalg.norm(np.array(k)-np.array(initial)) - termination_resolution # define event function
            event_fun.terminal = True # make event function terminal
            event_fun.direction = -1 #event function only triggered when it is decreasing
            return (np.transpose(solve_ivp(RHS_withB, t_span, initial, t_eval = sampletimes, dense_output=True, events=event_fun,method='LSODA',rtol=1e-9,atol=1e-15).y))  # use dense_output=True and events argument

        self.orbits1 = [createsingleorbit(initial) for initial in self.klist1]
        self.orbits2 = [createsingleorbit(initial) for initial in self.klist2]
        #self.orbits = Parallel(n_jobs=int(cpus/2))(delayed(createsingleorbit)(initial) for initial in self.k0)

        #end timer
        #endtime = time()

        #print(f"Time to create orbits: {endtime-starttime}")

    def createOrbitsEQS(self,resolution = 0.051,checkingtolerance = 1):
        """
        Inputs:
        resolution (integration resolution)

        Creates:
        List of equally spaced orbits curvesEQS
        """
        #now we create curves with equally spaced points out of the orbits
        self.orbitsEQS = []
        self.resolution = resolution

        #if self.resolution <= self.termination_resolution: print("Integration resolution less than termination resolution")

        #create equally spaced orbit out of a single orbit
        def appendSingleOrbitEQS(curve):
            #add initial point to equally spaced orbit
            startingpoint = curve[0]
            currentpoint  = startingpoint
            orbit1EQS = np.array([currentpoint],ndmin=2)

            #keep iterating over points in the orbit
            for point in curve:
                #if this point is sufficiently far away from previous point, add point to list
                if np.linalg.norm(currentpoint - point) > resolution:
                    currentpoint = point
                    orbit1EQS = np.append(orbit1EQS,np.array([currentpoint],ndmin=2),axis=0)

                    #this condition breaks the loop when you return to the starting point
                    if np.linalg.norm(currentpoint -startingpoint) < resolution:
                        break

            #if orbits1EQS has less than three points, we discard the orbit as numerical path derivatives won't be well defined
            if not orbit1EQS.shape[0] < 3:
                self.orbitsEQS.append(orbit1EQS)

        #check if both orbits are the same, and append only distinct orbits to orbitsEQS:
        tolerance =checkingtolerance

        def checkandappend(orbit1,orbit2,tolerance):
            #MAKE THIS GENERAL
            #this check only works if the two inital points for each plane are diametrically opposite points on the fermi surface
            #diagnostics #print("starting point on orbit1:",orbit1[0],"ending point of orbit1:",orbit1[-1])
            #diagnostics #print("starting point on orbit2:",orbit2[0],"ending point of orbit2:",orbit2[-1],"opposite point located at",int(orbit2.shape[0]/2),"opposite point:",orbit2[int(orbit2.shape[0]/2)])
            if(np.linalg.norm(orbit2[int(orbit2.shape[0]/2)] - orbit1[0]) < tolerance): #this means that the orbits are the same
                appendSingleOrbitEQS(orbit1) #only do this for a single orbit
            else: #this means orbits are different
                appendSingleOrbitEQS(orbit1)
                appendSingleOrbitEQS(orbit2)

        for id,orbit1 in enumerate(self.orbits1):
            orbit2 = self.orbits2[id]
            checkandappend(orbit1,orbit2,tolerance)


    def plotOrbits(self):
        #CURRENTLY DOES NOT WORK PLEASE FIX FOR ORBIT1 and ORBIT2
        ax = plt.figure().add_subplot(projection='3d')

        for orbit1 in self.orbits1:
            ax.scatter(orbit1[:,0],orbit1[:,1], orbit1[:,2], label='parametric curve')

        plt.show()

    def plotOrbitsEQS(self):
        ax = plt.figure().add_subplot(projection='3d')

        for orbit1 in self.orbitsEQS:
            ax.scatter(orbit1[:,0],orbit1[:,1], orbit1[:,2], label='parametric curve')

        plt.show()