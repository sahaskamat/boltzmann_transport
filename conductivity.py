import numpy as np
import dispersion

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
        n = 0
        for orbit in self.orbitsInstance.orbitsEQS:
            n  += len(orbit)

        #units of A are ps-1
        self.A = np.zeros((n,n))

        #i is an iterator that iterates over the hilbert space
        i=0

        #number that denotes the index of the beginning of the current submatrix
        submatrixindex = 0 
        submatrixindexlist = [0] #keeps track of all submatrix starting points

        for orbit in self.orbitsInstance.orbitsEQS:
            m = len(orbit) #m x m is the size of the submatrix for this orbit
                
            for state in orbit:
                #diagonal term coming from scattering out
                self.A[i,i]  = self.dispersionInstance.invtau(state)

                #off diagonal terms that simulate the derivative term from the boltzmann equation
                i_next = ((i + 1) - submatrixindex)%m + submatrixindex
                i_prev = ((i - 1) - submatrixindex)%m + submatrixindex
                self.A[i,i_next] = np.linalg.norm(np.cross(self.dispersionInstance.dedk(state),self.orbitsInstance.B))/(dispersion.deltap(orbit[i_next-submatrixindex],orbit[i_prev - submatrixindex])*43.32)

                #testing print
                #print(deltap(orbit1[i_next-submatrixindex],orbit1[i_prev - submatrixindex]))
                #print(state)

                self.A[i,i_prev] = -self.A[i,i_next]

                i += 1

            submatrixindex += len(orbit)
            submatrixindexlist.append(submatrixindex) 

        self.Ainv = np.matrix(np.linalg.inv(self.A))

    def createAlpha(self):
        #creates an array of the cartesian components of the velocity at each point on the discretized fermi surface
        dedk_list = []
        moddedk_list = []

        for curve in self.orbitsInstance.orbitsEQS:
            for state in curve:
                dedk_list.append(self.dispersionInstance.dedk(state))
                moddedk_list.append(self.dispersionInstance.dedk(state)/np.linalg.norm(self.dispersionInstance.dedk(state)))

        #convert list to nparray
        self.dedk_array = np.matrix(dedk_list)
        self.moddedk_array = np.matrix(moddedk_list)

        #multiply Ainv with the ath component of dedk to obtain alpha
        self.alpha = np.matmul(self.Ainv,self.dedk_array)

    def createSigma(self):
        #this creates the matrix sigma_mu_nu
        #mu and nu range from 0 to 2, with 0 being x, 1 being y and 2 being z
        self.sigma = np.zeros([3,3])

        for mu in range(3):
            for nu in range(3):
                #this keeps track of the total area over which we integrate
                self.areasum = 0
                #i is an iterator that iterates over the hilbert space
                i=0
                for curvenum,curve in enumerate(self.orbitsInstance.orbitsEQS):
                    #we are iterating over the curvenum'th orbit
                    #j is an iterator that iterates over the current orbit
                    for j,state in enumerate(curve):
                        nextpoint = curve[(j+1)%len(curve)]
                        patcharea = np.linalg.norm(np.cross(state-nextpoint,self.dispersionInstance.dkperp(state,self.orbitsInstance.B,self.initialPointsInstance.dkz[curvenum])))

                        self.sigma[mu,nu] += (3.699/(4*(np.pi**3)))*self.moddedk_array[i,mu]*self.alpha[i,nu]*patcharea 
                        self.areasum += patcharea

                        i+=1 
