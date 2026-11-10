import numpy as np
import scipy as sp
import transport.dispersion as dispersion
from time import time

def kronecker_delta(x,y,atol=1e-5):
    """Takes in two numbers and returns 1 if they are the same, 0 otherwise"""
    return np.isclose(x,y,atol=atol).astype(int)

class scatteringMatrix:
    """
    Inputs:
    dispersionInstance (object of type dispersion)
    fermiSurfaceInstance (object of type fermiSurface)

    Contains methods to calculate the Amatrix
    """
    def __init__(self,dispersionInstance,fermiSurfaceInstance):
        self.dispersionInstance = dispersionInstance
        self.fermiSurfaceInstance = fermiSurfaceInstance

    def createAmatrix_Bindependent_isotropic(self,invtau_iso):
        """
        Creates lists and functions used for the Amatrix calculations that are independent of the magnetic field and depend only on the isotropic (electron-electron) scattering rate
        Then populates an Amatrix with terms independent of the magnetic field
        """
        #n is the number of total points in our list, which is also the number of states in our Hilbert space,
        #and hence n x n is the size of our A matrix
        self.n = self.fermiSurfaceInstance.n_points*self.fermiSurfaceInstance.n_cuts
        self.invtau_iso = invtau_iso #isotropic scattering rate

        #create A matrix to populate with numbers:
        self.A_Bindependent = np.zeros((self.n,self.n))

        #Amatrixpositionlist[i,j] or Amatrixposition(i,j) gives the row (or column) in self.A that corresponds to state self.orbitsInstance.orbitsEQS[i][j]
        self.Amatrixpositionlist = np.arange(0, self.fermiSurfaceInstance.n_points*self.fermiSurfaceInstance.n_cuts).reshape(self.fermiSurfaceInstance.n_points, self.fermiSurfaceInstance.n_cuts)

        #find state[i+1] - state[i-1] for states as you traverse an "in-plane" orbit, then compute their norms and unit vectors used to compute gradients
        deltaplist_inplane = []
        for orbit in self.fermiSurfaceInstance.FSorbits:
            orbit_plus1 = np.roll(orbit,-1,axis=0)
            orbit_minus1 = np.roll(orbit,1,axis=0)
            deltaplist_inplane.append(orbit_plus1 - orbit_minus1)
        deltaparray_inplane =   np.array([value for sublist in deltaplist_inplane for value in sublist]) #this can be simplified, fix later
        self.deltaparray_inplane_norms = np.linalg.norm(deltaparray_inplane,axis=1)
        self.deltaparray_inplane_unitvectors = deltaparray_inplane/self.deltaparray_inplane_norms[:,None]

        #find state[i+1] - state[i-1] for states as you traverse an "out of plane" orbit, then compute their norms and unit vectors used to compute gradients
        deltaplist_outofplane = []
        for id,orbit in enumerate(self.fermiSurfaceInstance.FSorbits):
            orbit_plus1 = self.fermiSurfaceInstance.FSorbits[(id+1)%len(self.fermiSurfaceInstance.FSorbits)]
            orbit_minus1 = self.fermiSurfaceInstance.FSorbits[(id-1)%len(self.fermiSurfaceInstance.FSorbits)]
            BZlength = (2*np.pi)/self.fermiSurfaceInstance.c
            deltaplist_outofplane.append((orbit_plus1 - orbit_minus1 + BZlength/2)%(BZlength) - (BZlength/2))
        deltaparray_outofplane =   np.array([value for sublist in deltaplist_outofplane for value in sublist]) #this can be simplified, fix later
        self.deltaparray_outofplane_norms = np.linalg.norm(deltaparray_outofplane,axis=1)
        self.deltaparray_outofplane_unitvectors = deltaparray_outofplane/self.deltaparray_outofplane_norms[:,None]

        #list of fermi surface areas associated with each point
        self.patcharealist = (self.deltaparray_inplane_norms*self.deltaparray_outofplane_norms)/4

        #list of in-plane fermi surface lengths associated with each point
        self.inplanelengthlist = self.deltaparray_inplane_norms/2

        #scattering out list used for diagnostic plot
        self.dgdt_out_list = np.zeros_like(self.patcharealist)

        #creates a list of states in the same order as they would appear in the double loop
        self.stateslist = np.array([state for orbit in self.fermiSurfaceInstance.FSorbits for state in orbit]) #stateslist[i] = ith state
        self.invtau_iso_list = np.ones(len(self.stateslist))*self.invtau_iso #invtau_iso_list[i] = isotropic scattering rate at state i
        self.dedk_list = np.transpose(self.dispersionInstance.dedk(np.transpose(self.stateslist))) #self.dedk_list[i] = dedk(self.stateslist[i])

        #now populate the Amatrix with B independent terms
        i=0 #i and j correspond to the ith orbit and jth state on that orbit that is being iterated
        for orbit in self.fermiSurfaceInstance.FSorbits:
            j=0
            for state in orbit:
                #diagonal term coming from scattering out
                Amatrixposition = self.Amatrixpositionlist[i,j]
                self.A_Bindependent[Amatrixposition,Amatrixposition] += self.invtau_iso_list[Amatrixposition]
                j+= 1
            i+= 1

    def create_Hfunc(self,scatteringmodel,delta_in_k,**kwargs):
        """
        Inputs:
        scatteringmodel (string corresponding to a function name from scattering_kernels.py)

        Creates a function self.H_func(k,k') that takes in k (3-vector) and k' (list of 3-vectors) and outputs a list of scattering matrix elements
        Each matrix element corresponds to a 3-vector in k'
        """
        self.delta_in_k = delta_in_k

        #import a scattering kernel h(deltak) that takes in deltak (list of 3-vectors) and outputs a list of scalars corresponding to the scattering matrix values
        import transport.scattering_kernels as scattering_kernels

        #reciprocal lattice vectors
        g = np.array([self.fermiSurfaceInstance.g1,self.fermiSurfaceInstance.g2,self.fermiSurfaceInstance.g3])

        def H_func(k,kprime): #should be vectorized in kprime
            deltak = k-kprime #array of 3-vectors if kprime is a 3-vec array

            #reducing deltak to its minimum value by accounting for periodic boundaries for the BZ
            deltak = deltak - np.round(deltak/g)*g #reduces deltak to its minimum value by accounting for periodic boundaries for the BZ
            
            kernel = getattr(scattering_kernels, scatteringmodel) #imports a function that takes deltak,g (list of reciprocal lattice vectors g1,g2,g3),**kwargs as input and outputs scattering matrix elements corresponding to each deltak
            h_func_val =  kernel(deltak,g,**kwargs)
            
            return np.nan_to_num(h_func_val, nan=0.0, posinf=0.0)

        self.H_func = H_func

    def createAmatrix_Bindependent_fwdscatter_out(self):
        """
            Creates lists and functions used for the Amatrix calculations that are independent of the magnetic field and depend only on the forward scattering
            Then populates an Amatrix with terms independent of the magnetic field
        """

        #first create the scattering out terms

        #i and j correspond to the ith orbit and jth state on that orbit that is being iterated
        for i,orbit in enumerate(self.fermiSurfaceInstance.FSorbits):
            for j,state in enumerate(orbit):
                #diagonal term coming from scattering out
                Amatrixposition = self.Amatrixpositionlist[i,j]
                
                if self.delta_in_k:
                    Hlist = self.H_func(state,self.stateslist)*kronecker_delta(state[2],self.stateslist[:,2]) #creates a list of H(k,k') where k is the state being iterated and k' are all the other states, but only if their kz are equal
                    dgdt_out = np.sum((Hlist*self.inplanelengthlist)/np.linalg.norm(self.dedk_list[:,:2],axis=1)) #dgdt_out = sum(inplanelength(k')*H(k,k')/dEdk(inplane k'))
                else:
                    Hlist = self.H_func(state,self.stateslist) #creates a list of H(k,k') where k is the state being iterated and k' are all the other states
                    dgdt_out = np.sum((Hlist*self.patcharealist)/np.linalg.norm(self.dedk_list,axis=1)) #dgdt_out = sum(patcharea(k')*H(k,k')/dEdk(k'))

                self.A_Bindependent[Amatrixposition,Amatrixposition] += dgdt_out
                self.dgdt_out_list[Amatrixposition] += (dgdt_out)


    def createAmatrix_Bindependent_fwdscatter_in(self):
        #now create scattering in terms
        #first iterate over row number of the Amatrix
        #i_row and j_row correspond to the ith orbit and jth state on that orbit that is being iterated, which corresponds to the Amatrix[i_row,j_row] row of the Amatrix
        for i_row,orbit_row in enumerate(self.fermiSurfaceInstance.FSorbits):
            for j_row,state_row in enumerate(orbit_row):
                Amatrixposition_row = self.Amatrixpositionlist[i_row,j_row] 

                if self.delta_in_k:
                    Hlist = self.H_func(state_row,self.stateslist)*kronecker_delta(state_row[2],self.stateslist[:,2]) #scattering term is only non-zero if both states have the same kz
                    row_to_add = (Hlist*self.inplanelengthlist)/np.linalg.norm(self.dedk_list[:,:2],axis=1) #computes the scattering in terms on this row of amatrix
                else:
                    Hlist = self.H_func(state_row,self.stateslist)
                    row_to_add = (Hlist*self.patcharealist)/np.linalg.norm(self.dedk_list,axis=1) #computes the scattering in terms on this row of amatrix
                    
                self.A_Bindependent[Amatrixposition_row,:] = self.A_Bindependent[Amatrixposition_row,:] - row_to_add #add this row to the A matrix

    def createAmatrix_Bdependent(self,B):
        #creates a sparse matrix self.A_Bdependent that is added to self.A_Bindependent to get the total self.A (total scattering-matrix)
        self.A_Bdependent = np.zeros((self.n,self.n))

        if np.linalg.norm(B) == 0:
            #no terms to be added if B=0
            return

        self.B = B
        crosslist  = np.cross(self.dedk_list,self.B) #crosslist[i]  = dedk(state[i]) x B
        dotlist_inplane = np.sum(crosslist*self.deltaparray_inplane_unitvectors,axis=1) #dotlist_inplane[i] = (dedk(state[i]) x B) . unitvec(state[i+1] - state[i-1])
        dotlist_outofplane = np.sum(crosslist*self.deltaparray_outofplane_unitvectors,axis=1) #dotlist_outofplane[i] = (dedk(state[i]) x B) . unitvec(state[i+1] - state[i-1]) for out of plane states
        graddatalist_inplane = dotlist_inplane/(self.deltaparray_inplane_norms*(6.582119569**2)) #graddatalist[i] = (dedk(state[i]) x B) . unitvec(state[i+1] - state[i-1])/norm(state[i+1] - state[i-1])
        graddatalist_outofplane = dotlist_outofplane/(self.deltaparray_outofplane_norms*(6.582119569**2)) #graddatalist[i] = (dedk(state[i]) x B) . unitvec(state[i+1] - state[i-1])/norm(state[i+1] - state[i-1]) for out of plane states

        i=0 #i corresponds to the ith orbit that is being iterated
        n = len(self.fermiSurfaceInstance.FSorbits) #n is the number of orbits
        for orbit in self.fermiSurfaceInstance.FSorbits:
            m = len(orbit) #m is the number of states on the current orbit

            stateindices = np.arange(m) #stateindex[i] corresponds to the ith state on orbit

            Amatrixpositions = self.Amatrixpositionlist[i,stateindices] #Amatrixpositions[i] corresponds to the index of stateindex[i] in the Amatrix

            #off diagonal terms that simulate the derivative term from the boltzmann equation, in plane
            next_Amatrixpositions_inplane = self.Amatrixpositionlist[i,(stateindices+1)%m] #next_Amatrixpositions_inplane[i] corresponds to index of the next in plane state to stateindex[i] in the Amatrix
            prev_Amatrixpositions_inplane = self.Amatrixpositionlist[i,(stateindices-1)%m] #prev_Amatrixpositions_inplane[i] corresponds to index of the previous in plane state to stateindex[i] in the Amatrix

            graddata_inplane = graddatalist_inplane[Amatrixpositions] #graddata_inplane[i] corresponds to the derivative term in plane for stateindex[i]

            self.A_Bdependent[Amatrixpositions,next_Amatrixpositions_inplane] += graddata_inplane
            self.A_Bdependent[Amatrixpositions,prev_Amatrixpositions_inplane] += -graddata_inplane

            #off diagonal terms that simulate the derivative term from the boltzmann equation, out of plane
            next_Amatrixpositions_outofplane = self.Amatrixpositionlist[(i+1)%n,stateindices]
            prev_Amatrixpositions_outofplane = self.Amatrixpositionlist[(i-1)%n,stateindices]

            graddata_outofplane = graddatalist_outofplane[Amatrixpositions]

            self.A_Bdependent[Amatrixpositions,next_Amatrixpositions_outofplane] += graddata_outofplane
            self.A_Bdependent[Amatrixpositions,prev_Amatrixpositions_outofplane] += -graddata_outofplane

            i += 1