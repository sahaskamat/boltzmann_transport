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
        B: magnetic field. Note that code is currently only compatible with Bx = 0

        """
        self.dispersion = dispersion

        if not isinstance(doublefermisurface,bool):
            raise Exception("Argument doublefermisurface is not a boolean")

        c = self.dispersion.c/(1+int(doublefermisurface)) #this makes c = dispersion.c/2 if doublefermisurface is True
        starttime = time()
        
        self.k0 = [np.array([0,0,kz0]) for kz0 in np.linspace(-(np.pi)/c,(np.pi)/c,n+1)] 
        #this is the list of initial conditions evenly spaced in the z direction, lying along x=y=0.
        #these will be used to create initial conditions lying on the fermi surface

        self.klist1 = []
        self.klist2 = []

        #solve numeric function along the kx = 0 lying in the plane of kz0 and perpendicular to the magnetic field
        for iter_num, kz0 in self.k0:
            ky0 = fsolve(lambda ky_numeric : self.dispersion.en_numeric(0,ky_numeric,(np.dot(kz0,B) - ky_numeric*B[1])/B[2]),-np.pi/dispersion.a + 0.1)[0] #figure out why the [0] is here
            self.klist1.append(np.array([0,ky0,(np.dot(kz0,B) - ky0*B[1])/B[2]]))

            ky1 = fsolve(lambda ky_numeric : self.dispersion.en_numeric(0,ky_numeric,(np.dot(kz0,B) - ky_numeric*B[1])/B[2]),np.pi/dispersion.a - 0.1)[0] #figure out why the [0] is here
            self.klist2.append(np.array([0,ky1,(np.dot(kz0,B) - ky1*B[1])/B[2]]))

        endtime = time()
        print(f"Time to create initialpoints: {endtime-starttime}")


class Orbits:
    def __init__(self,dispersion,initialpoints):
        self.dispersion = dispersion
        self.k0 = initialpoints.k0


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
        starttime = time()

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
            return (np.transpose(solve_ivp(RHS_withB, t_span, initial, t_eval = sampletimes, dense_output=True, events=event_fun,method='LSODA',rtol=1e-9,atol=1e-10).y))  # use dense_output=True and events argument

        self.orbits = [createsingleorbit(initial) for initial in self.k0]
        #self.orbits = Parallel(n_jobs=int(cpus/2))(delayed(createsingleorbit)(initial) for initial in self.k0)

        #end timer
        endtime = time()

        print(f"Time to create orbits: {endtime-starttime}")

    def createOrbitsEQS(self,resolution = 0.051):
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

        #check to make sure createOrbits() has been executed:

        #iterate over different orbits
        for curve in self.orbits:

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

    def plotOrbits(self):
        ax = plt.figure().add_subplot(projection='3d')

        for orbit1 in self.orbits:
            ax.scatter(orbit1[:,0],orbit1[:,1], orbit1[:,2], label='parametric curve')

        plt.show()

    def plotOrbitsEQS(self):
        ax = plt.figure().add_subplot(projection='3d')

        for orbit1 in self.orbitsEQS:
            ax.scatter(orbit1[:,0],orbit1[:,1], orbit1[:,2], label='parametric curve')

        plt.show()