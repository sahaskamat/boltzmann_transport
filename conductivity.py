import numpy as np
import scipy as sp
import dispersion
from time import time

class Conductivity:
    """
    Inputs:
    dispersionInstance (object of type dispersion)
    orbitsInstance (object of type orbits)
    initialPointsInstance (object of type initialpoints)

    Contains methods to calculate the Amatrix, alpha, and sigma
    """
    def __init__(self,dispersionInstance,FSorbitsInstance,invtau_iso = 12.595,invtau_aniso = 63.823):
        self.dispersionInstance = dispersionInstance
        self.FSorbitsInstance = FSorbitsInstance
        self.invtau_iso = invtau_iso
        self.invtau_aniso = invtau_aniso

    def createAmatrix_Bindependent(self):
        """
        Creates lists and functions used for the Amatrix calculations that are independent of the magnetic field
        Then populates an Amatrix with terms independent of the magnetic field
        """
        #n is the number of total points in our list, which is also the number of states in our Hilbert space,
        #and hence n x n is the size of our A matrix
        self.n = self.FSorbitsInstance.n_points*self.FSorbitsInstance.n_cuts

        #create A matrix to populate with numbers:
        self.A_Bindependent = np.zeros((self.n,self.n))

        #Amatrixpositionlist[i,j] gives the row (or column) in self.A that corresponds to state self.orbitsInstance.orbitsEQS[i][j]
        self.Amatrixpositionlist = np.arange(0, self.FSorbitsInstance.n_points*self.FSorbitsInstance.n_cuts).reshape(self.FSorbitsInstance.n_points, self.FSorbitsInstance.n_cuts)

        #find state[i+1] - state[i-1] for states as you traverse an "in-plane" orbit, then compute their norms and unit vectors used to compute gradients
        deltaplist_inplane = []
        for orbit in self.FSorbitsInstance.FSorbits:
            orbit_plus1 = np.roll(orbit,-1,axis=0)
            orbit_minus1 = np.roll(orbit,1,axis=0)
            deltaplist_inplane.append(orbit_plus1 - orbit_minus1)
        deltaparray_inplane =   np.array([value for sublist in deltaplist_inplane for value in sublist]) #this can be simplified, fix later
        self.deltaparray_inplane_norms = np.linalg.norm(deltaparray_inplane,axis=1)
        self.deltaparray_inplane_unitvectors = deltaparray_inplane/self.deltaparray_inplane_norms[:,None]

        #find state[i+1] - state[i-1] for states as you traverse an "out of plane" orbit, then compute their norms and unit vectors used to compute gradients
        deltaplist_outofplane = []
        for id,orbit in enumerate(self.FSorbitsInstance.FSorbits):
            orbit_plus1 = self.FSorbitsInstance.FSorbits[(id+1)%len(self.FSorbitsInstance.FSorbits)]
            orbit_minus1 = self.FSorbitsInstance.FSorbits[(id-1)%len(self.FSorbitsInstance.FSorbits)]
            BZlength = (2*np.pi)/self.FSorbitsInstance.c
            deltaplist_outofplane.append((orbit_plus1 - orbit_minus1 + BZlength/2)%(BZlength) - (BZlength/2))
        deltaparray_outofplane =   np.array([value for sublist in deltaplist_outofplane for value in sublist]) #this can be simplified, fix later
        self.deltaparray_outofplane_norms = np.linalg.norm(deltaparray_outofplane,axis=1)
        self.deltaparray_outofplane_unitvectors = deltaparray_outofplane/self.deltaparray_outofplane_norms[:,None]

        #creates a list of states in the same order as they would appear in the double loop
        stateslist = np.array([state for orbit in self.FSorbitsInstance.FSorbits for state in orbit]) #stateslist[i] = ith state
        self.invtaulist = self.dispersionInstance.invtau(np.transpose(stateslist),invtau_iso=self.invtau_iso,invtau_aniso=self.invtau_aniso) #invtaulist[i]  = invtau(stateslist[i])
        self.dedk_list = np.transpose(self.dispersionInstance.dedk(np.transpose(stateslist))) #self.dedk_list[i] = dedk(stateslist[i])

        #now populate the Amatrix with B independent terms
        i=0 #i and j correspond to the ith orbit and jth state on that orbit that is being iterated
        for orbit in self.FSorbitsInstance.FSorbits:
            j=0
            for state in orbit:
                #diagonal term coming from scattering out
                Amatrixposition = self.Amatrixpositionlist[i,j]
                self.A_Bindependent[Amatrixposition,Amatrixposition] += self.invtaulist[Amatrixposition]
                j+= 1
            i+= 1

    def createAmatrix_Bdependent(self,B):
        self.A = np.copy(self.A_Bindependent) #create a copy of the B independent Amatrix to populate with B dependent terms
        self.B = B
        crosslist  = np.cross(self.dedk_list,self.B) #crosslist[i]  = dedk(state[i]) x B
        dotlist_inplane = np.sum(crosslist*self.deltaparray_inplane_unitvectors,axis=1) #dotlist_inplane[i] = (dedk(state[i]) x B) . unitvec(state[i+1] - state[i-1])
        dotlist_outofplane = np.sum(crosslist*self.deltaparray_outofplane_unitvectors,axis=1) #dotlist_outofplane[i] = (dedk(state[i]) x B) . unitvec(state[i+1] - state[i-1]) for out of plane states
        graddatalist_inplane = dotlist_inplane/(self.deltaparray_inplane_norms*(6.582119569**2)) #graddatalist[i] = (dedk(state[i]) x B) . unitvec(state[i+1] - state[i-1])/norm(state[i+1] - state[i-1])
        graddatalist_outofplane = dotlist_outofplane/(self.deltaparray_outofplane_norms*(6.582119569**2)) #graddatalist[i] = (dedk(state[i]) x B) . unitvec(state[i+1] - state[i-1])/norm(state[i+1] - state[i-1]) for out of plane states

        i=0 #i and j correspond to the ith orbit and jth state on that orbit that is being iterated
        n = len(self.FSorbitsInstance.FSorbits) #n is the number of orbits
        for orbit in self.FSorbitsInstance.FSorbits:
            m = len(orbit) #m is the number of states on the current orbit
            j=0

            for state in orbit:
                Amatrixposition = self.Amatrixpositionlist[i,j]

                #off diagonal terms that simulate the derivative term from the boltzmann equation, in plane
                next_Amatrixposition_inplane = self.Amatrixpositionlist[i,(j+1)%m]
                prev_Amatrixposition_inplane = self.Amatrixpositionlist[i,(j-1)%m]

                graddata_inplane = graddatalist_inplane[Amatrixposition]

                self.A[Amatrixposition,next_Amatrixposition_inplane] += graddata_inplane
                self.A[Amatrixposition,prev_Amatrixposition_inplane] += -graddata_inplane

                #off diagonal terms that simulate the derivative term from the boltzmann equation, out of plane
                next_Amatrixposition_outofplane = self.Amatrixpositionlist[(i+1)%n,j]
                prev_Amatrixposition_outofplane = self.Amatrixpositionlist[(i-1)%n,j]

                graddata_outofplane = graddatalist_outofplane[Amatrixposition]

                self.A[Amatrixposition,next_Amatrixposition_outofplane] += graddata_outofplane
                self.A[Amatrixposition,prev_Amatrixposition_outofplane] += -graddata_outofplane

                j += 1
            i += 1


        #now add in scattering in terms coming from forward scattering

        #first create matrix whose {i,j} element is {k_i,k_j}, which will be an input to the scattering-in formula
        plist = np.concatenate(self.FSorbitsInstance.FSorbits) #list of all momentum vectors in correct order

        indices = np.indices([self.n,self.n]) #list of indices {i,j} to be extracted from plist

        pi_minus_pj = (plist[indices])[0] - (plist[indices])[1] #the i,jth element of this matrix is p_i - p_j (directly features into the scattering in matrix)


    def createAlpha(self):
        #creates an array of the cartesian components of the velocity at each point on the discretized fermi surface
        self.moddedk_array = self.dedk_list/np.linalg.norm(self.dedk_list,axis=1)[:,None]

        #multiply Ainv with the ath component of dedk to obtain alpha
        #multiplying by Ainv directly replaced by solving the equation
        self.alpha = sp.linalg.solve(self.A,self.dedk_list)

    def createSigma(self):
        #this creates the matrix sigma_mu_nu
        #mu and nu range from 0 to 2, with 0 being x, 1 being y and 2 being z
        self.sigma = np.zeros([3,3])

        #perptermlist[i] is a vector that lies along the fermi surface, pointing from the ith point to the orbit above it
        perptermlist = self.dispersionInstance.dkperp(self.FSorbitsInstance.dkz,self.FSorbitsInstance.dkz,self.dedk_list)

        #nextstatepointerarray[i] is a vector that lies along the fermi surface and points from the ith point to the succeeding point on a given orbit
        nextstatepointerlist = []
        for orbit in self.FSorbitsInstance.FSorbits:
            orbit_plus1 = np.roll(orbit,-1,axis=0)
            nextstatepointerlist.append(orbit - orbit_plus1)
        nextstatepointerarray =   np.array([value for sublist in nextstatepointerlist for value in sublist])

        #patcharealist[i] is the integration patch area corresponding to the ith point
        patcharealist = np.linalg.norm(np.cross(nextstatepointerarray,perptermlist),axis=1)

        for mu in range(3):
            for nu in range(3):
                #this keeps track of the total area over which we integrate
                self.areasum = 0
                self.sigma[mu,nu] = (3.699/(4*(np.pi**3)))*np.sum(self.moddedk_array[:,mu]*self.alpha[:,nu]*patcharealist)

                self.areasum = np.sum(patcharealist)


