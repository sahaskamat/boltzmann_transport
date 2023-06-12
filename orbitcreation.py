import numpy as np
from scipy.optimize import fsolve
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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
        self.k0 = [] #this is the list of initial conditions evenly spaced in the z direction
        self.dkz = [] #list of differences of starting points of orbits (vector)
        self.dispersion = dispersion

        if not isinstance(doublefermisurface,bool):
            raise Exception("Argument doublefermisurface is not a boolean")

        c = self.dispersion.c/(1+int(doublefermisurface)) #this makes c = dispersion.c/2 if doublefermisurface is True

        #solve numeric function along the line kz = ky = 0
        for iter_num, kz0 in enumerate(np.linspace(-(np.pi)/c,(np.pi)/c,n+1)):

            if iter_num==0: #dont append first solution, this is only used to keep dkz consistent
                kx0 = fsolve(lambda kx_numeric : self.dispersion.en_numeric(kx_numeric,0,kz0),0.6)[0]
                previousk0 = np.array([kx0,0,kz0])
            else:
                kx0 = fsolve(lambda kx_numeric : self.dispersion.en_numeric(kx_numeric,0,kz0),0.6)[0]
                thisk0 = np.array([kx0,0,kz0])

                self.k0.append(thisk0)
                self.dkz.append(thisk0 - previousk0)
                previousk0 = thisk0


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
        #list of orbits
        sol = []
        self.B = np.array(B)
        #B_normalized = np.array(B)/np.linalg.norm(B)
        B_normalized = np.array(B)/np.linalg.norm(B)
        self.termination_resolution = termination_resolution

        #the force v \cross B term but now with fixed B
        RHS_withB = lambda t,k : self.dispersion.RHS_numeric(k,mult_factor*B_normalized)

        #timestamps on which to sample the solution
        t_span = (sampletimes[0],sampletimes[-1])

        #iterate over different initial conditions, each corresponding to a given orbit
        for initial in self.k0:
            # solve differential equation to find a single orbit
            event_fun = lambda t,k : np.linalg.norm(np.array(k)-np.array(initial)) - termination_resolution # define event function
            event_fun.terminal = True # make event function terminal
            event_fun.direction = -1 #event function only triggered when it is decreasing
            sol.append(np.transpose(solve_ivp(RHS_withB, t_span, initial, t_eval = sampletimes, dense_output=True, events=event_fun,method='LSODA',rtol=1e-9,atol=1e-10).y))  # use dense_output=True and events argument

        self.orbits = sol

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