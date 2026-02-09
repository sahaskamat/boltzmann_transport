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
    def __init__(self,dispersionInstance,orbitsInstace,initialPointsInstance):
        self.dispersionInstance = dispersionInstance
        self.orbitsInstance = orbitsInstace
        self.initialPointsInstance = initialPointsInstance

    def createAMatrix(self):
        #n is the number of total points in our list, which is also the number of states in our Hilbert space,
        #and hence n x n is the size of our A matrix
        self.n = self.initialPointsInstance.n_points*self.initialPointsInstance.n_cuts

        #create A matrix to populate with numbers:
        self.A = np.zeros((self.n,self.n))

        #Amatrixpositionlist[i,j] gives the row (or column) in self.A that corresponds to state self.orbitsInstance.orbitsEQS[i][j]
        Amatrixpositionlist = np.arange(0, self.initialPointsInstance.n_points*self.initialPointsInstance.n_cuts).reshape(self.initialPointsInstance.n_points, self.initialPointsInstance.n_cuts)

        #find state[i+1] - state[i-1] for states as you traverse an "in-plane" orbit, then compute their norms and unit vectors used to compute gradients
        deltaplist_inplane = []
        for orbit in self.orbitsInstance.orbitsEQS:
            orbit_plus1 = np.roll(orbit,-1,axis=0)
            orbit_minus1 = np.roll(orbit,1,axis=0)
            deltaplist_inplane.append(orbit_plus1 - orbit_minus1)
        deltaparray_inplane =   np.array([value for sublist in deltaplist_inplane for value in sublist]) #this can be simplified, fix later
        deltaparray_inplane_norms = np.linalg.norm(deltaparray_inplane,axis=1)
        deltaparray_inplane_unitvectors = deltaparray_inplane/deltaparray_inplane_norms[:,None]

        #find state[i+1] - state[i-1] for states as you traverse an "out of plane" orbit, then compute their norms and unit vectors used to compute gradients
        deltaplist_outofplane = []
        for id,orbit in enumerate(self.orbitsInstance.orbitsEQS):
            orbit_plus1 = self.orbitsInstance.orbitsEQS[(id+1)%len(self.orbitsInstance.orbitsEQS)]
            orbit_minus1 = self.orbitsInstance.orbitsEQS[(id-1)%len(self.orbitsInstance.orbitsEQS)]
            BZlength = (2*np.pi)/self.initialPointsInstance.c
            deltaplist_outofplane.append((orbit_plus1 - orbit_minus1 + BZlength/2)%(BZlength) - (BZlength/2))
        deltaparray_outofplane =   np.array([value for sublist in deltaplist_outofplane for value in sublist]) #this can be simplified, fix later
        deltaparray_outofplane_norms = np.linalg.norm(deltaparray_outofplane,axis=1)
        deltaparray_outofplane_unitvectors = deltaparray_outofplane/deltaparray_outofplane_norms[:,None]

        #number that denotes the index of the beginning of the current submatrix
        submatrixindex = 0
        submatrixindexlist = [0] #keeps track of all submatrix starting points

        #creates a list of states in the same order as they would appear in the double loop
        stateslist = np.array([state for orbit in self.orbitsInstance.orbitsEQS for state in orbit]) #stateslist[i] = ith state
        invtaulist = self.dispersionInstance.invtau(np.transpose(stateslist)) #invtaulist[i]  = invtau(stateslist[i])
        self.dedk_list = np.transpose(self.dispersionInstance.dedk(np.transpose(stateslist))) #self.dedk_list[i] = dedk(stateslist[i])
        crosslist  = np.cross(self.dedk_list,self.orbitsInstance.B) #crosslist[i]  = dedk(state[i]) x B
        dotlist_inplane = np.sum(crosslist*deltaparray_inplane_unitvectors,axis=1) #dotlist_inplane[i] = (dedk(state[i]) x B) . unitvec(state[i+1] - state[i-1])
        dotlist_outofplane = np.sum(crosslist*deltaparray_outofplane_unitvectors,axis=1) #dotlist_outofplane[i] = (dedk(state[i]) x B) . unitvec(state[i+1] - state[i-1]) for out of plane states
        graddatalist_inplane = dotlist_inplane/(deltaparray_inplane_norms*(6.582119569**2)) #graddatalist[i] = (dedk(state[i]) x B) . unitvec(state[i+1] - state[i-1])/norm(state[i+1] - state[i-1])
        graddatalist_outofplane = dotlist_outofplane/(deltaparray_outofplane_norms*(6.582119569**2)) #graddatalist[i] = (dedk(state[i]) x B) . unitvec(state[i+1] - state[i-1])/norm(state[i+1] - state[i-1]) for out of plane states

        i=0 #i and j correspond to the ith orbit and jth state on that orbit that is being iterated
        for orbit in self.orbitsInstance.orbitsEQS:
            m = len(orbit) #m x m is the size of the submatrix for this orbit
            j=0

            for state_id,state in enumerate(orbit):
                #diagonal term coming from scattering out
                Amatrixposition = Amatrixpositionlist[i,j]
                self.A[Amatrixposition,Amatrixposition] = invtaulist[Amatrixposition]

                #off diagonal terms that simulate the derivative term from the boltzmann equation
                next_Amatrixposition_inplane = Amatrixpositionlist[i,(j+1)%m]
                prev_Amatrixposition_inplane = Amatrixpositionlist[i,(j-1)%m]
 
                graddata = graddatalist_inplane[Amatrixposition]

                self.A[Amatrixposition,next_Amatrixposition_inplane] = graddata

                #TEST BY CHANGING DIFFERENTIATION METHOD:
                #self.A[i,i_next] += np.linalg.norm(np.cross(self.dispersionInstance.dedk(state),self.orbitsInstance.B))/(dispersion.deltap(orbit[i_next-submatrixindex],orbit[i - submatrixindex])*43.32)

                #testing print
                #print(deltap(orbit1[i_next-submatrixindex],orbit1[i_prev - submatrixindex]))
                #print(state)

                self.A[Amatrixposition,prev_Amatrixposition_inplane] = -graddata

                #TEST BY CHANGING DIFFERENTIATION METHOD:
                #self.A[i,i] += -self.A[i,i_next]
                j += 1
            i += 1


        #now add in scattering in terms coming from forward scattering

        #first create matrix whose {i,j} element is {k_i,k_j}, which will be an input to the scattering-in formula
        plist = np.concatenate(self.orbitsInstance.orbitsEQS) #list of all momentum vectors in correct order

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
        perptermlist = self.dispersionInstance.dkperp(self.orbitsInstance.B,self.initialPointsInstance.dkz,self.dedk_list)

        #nextstatepointerarray[i] is a vector that lies along the fermi surface and points from the ith point to the succeeding point on a given orbit
        nextstatepointerlist = []
        for orbit in self.orbitsInstance.orbitsEQS:
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


