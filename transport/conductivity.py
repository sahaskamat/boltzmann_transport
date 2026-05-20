import numpy as np
import scipy as sp
import transport.dispersion as dispersion
from time import time

class Conductivity:
    """
    Inputs:
    dispersionInstance (object of type dispersion)
    orbitsInstance (object of type orbits)
    initialPointsInstance (object of type initialpoints)

    Contains methods to calculate the Amatrix, alpha, and sigma
    """
    def __init__(self,dispersionInstance,FSorbitsInstance,invtau_iso = 12.595,plotScattering=False):
        self.dispersionInstance = dispersionInstance
        self.FSorbitsInstance = FSorbitsInstance
        self.invtau_iso = invtau_iso #isotropic scattering rate
        self.plotScattering = plotScattering

    def createAmatrix_Bindependent_isotropic(self):
        """
        Creates lists and functions used for the Amatrix calculations that are independent of the magnetic field and depend only on the isotropic (electron-electron) scattering rate
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

        #list of fermi surface areas associated with each point
        self.patcharealist = (self.deltaparray_inplane_norms*self.deltaparray_outofplane_norms)/4

        #creates a list of states in the same order as they would appear in the double loop
        self.stateslist = np.array([state for orbit in self.FSorbitsInstance.FSorbits for state in orbit]) #stateslist[i] = ith state
        self.invtau_iso_list = np.ones(len(self.stateslist))*self.invtau_iso #invtau_iso_list[i] = isotropic scattering rate at state i
        self.dedk_list = np.transpose(self.dispersionInstance.dedk(np.transpose(self.stateslist))) #self.dedk_list[i] = dedk(self.stateslist[i])

        #now populate the Amatrix with B independent terms
        i=0 #i and j correspond to the ith orbit and jth state on that orbit that is being iterated
        for orbit in self.FSorbitsInstance.FSorbits:
            j=0
            for state in orbit:
                #diagonal term coming from scattering out
                Amatrixposition = self.Amatrixpositionlist[i,j]
                self.A_Bindependent[Amatrixposition,Amatrixposition] += self.invtau_iso_list[Amatrixposition]
                j+= 1
            i+= 1

    def create_Hfunc(self,scatteringmodel,**kwargs):
        """
        Creates a function self.H_func(k,k') that takes in k (3-vector) and k' (list of 3-vectors) and outputs a list of scattering matrix elements
        Each matrix element corresponds to a 3-vector in k'
        """

        #import a scattering kernel h(deltak) that takes in deltak (list of 3-vectors) and outputs a list of scalars corresponding to the scattering matrix values
        import transport.scattering_kernels as scattering_kernels

        #reciprocal lattice vectors
        g1,g2,g3 = self.FSorbitsInstance.g1,self.FSorbitsInstance.g2,self.FSorbitsInstance.g3

        def H_func(k,kprime): #should be vectorized in kprime
            deltak = k-kprime #array of 3-vectors if kprime is a 3-vec array

            #reducing deltak to its minimum value by accounting for periodic boundaries for the BZ
            deltak[:,0] = deltak[:,0] - g1*np.round(deltak[:,0]/g1)
            deltak[:,1] = deltak[:,1] - g2*np.round(deltak[:,1]/g2)
            deltak[:,2] = deltak[:,2] - g3*np.round(deltak[:,2]/g3)
            
            kernel = getattr(scattering_kernels, scatteringmodel) #imports a function that takes deltak,g (list of reciprocal lattice vectors g1,g2,g3),**kwargs as input and outputs scattering matrix elements corresponding to each deltak
            h_func_val =  kernel(deltak,[g1,g2,g3],**kwargs)
            
            return np.nan_to_num(h_func_val, nan=0.0, posinf=0.0)

        self.H_func = H_func

    def createAmatrix_Bindependent_fwdscatter_out(self):
        """
            Creates lists and functions used for the Amatrix calculations that are independent of the magnetic field and depend only on the forward scattering
            Then populates an Amatrix with terms independent of the magnetic field
        """

        #first create the scattering out terms

        #scattering out list used for diagnostic plot
        dgdt_out_list = []

        #i and j correspond to the ith orbit and jth state on that orbit that is being iterated
        for i,orbit in enumerate(self.FSorbitsInstance.FSorbits):
            for j,state in enumerate(orbit):
                #diagonal term coming from scattering out
                Amatrixposition = self.Amatrixpositionlist[i,j]
                
                Hlist = self.H_func(state,self.stateslist) #creates a list of H(k,k') where k is the state being iterated and k' are all the other states
                dgdt_out = np.sum((Hlist*self.patcharealist)/np.linalg.norm(self.dedk_list,axis=1)) #dgdt_out = sum(patcharea(k')*H(k,k')/dEdk(k'))

                self.A_Bindependent[Amatrixposition,Amatrixposition] += dgdt_out
                dgdt_out_list.append(dgdt_out)

        if self.plotScattering:
            self.plotScatteringOut(dgdt_out_list)

    def createAmatrix_Bindependent_fwdscatter_in(self):
        #now create scattering in terms
        #first iterate over row number of the Amatrix
        #i_row and j_row correspond to the ith orbit and jth state on that orbit that is being iterated, which corresponds to the Amatrix[i_row,j_row] row of the Amatrix
        for i_row,orbit_row in enumerate(self.FSorbitsInstance.FSorbits):
            for j_row,state_row in enumerate(orbit_row):
                Amatrixposition_row = self.Amatrixpositionlist[i_row,j_row] 

                Hlist = self.H_func(state_row,self.stateslist)
                row_to_add = (Hlist*self.patcharealist)/np.linalg.norm(self.dedk_list,axis=1) #computes the scattering in terms on this row of amatrix
                self.A_Bindependent[Amatrixposition_row,:] = self.A_Bindependent[Amatrixposition_row,:] - row_to_add #add this row to the A matrix
        

    def LUdecomp(self,B):
        #performs an LU decomposition on self.A_Bindependent + self.A_Bdependent(B) for a certain magnetic field that is used to speed up future solves. B can be zero.
        self.createAmatrix_Bdependent(B)
        lu,piv = sp.linalg.lu_factor(self.A_Bindependent+self.A_Bdependent)
        print("LU decomposition completed")

        #create a scipy "Linear Operator" that solves self.A_Bindependent*x = b, given b. Approximately returns self.A^-1 @ b. Used to precondition GMRES solver
        def preconditioner(b):
            return sp.linalg.lu_solve((lu,piv),b) 
        
        self.preconditioner_LO = sp.sparse.linalg.LinearOperator(shape=(self.n,self.n),matvec=preconditioner,dtype=np.float64)

    def createAmatrix_Bdependent(self,B):
        #creates a sparse matrix self.A_Bdependent that is added to self.A_Bindependent to get the total self.A (total scattering-matrix)
        self.A_Bdependent = sp.sparse.lil_matrix((self.n,self.n))

        if np.linalg.norm(B) == 0:
            #no terms to be added if B=0
            return

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

                self.A_Bdependent[Amatrixposition,next_Amatrixposition_inplane] += graddata_inplane
                self.A_Bdependent[Amatrixposition,prev_Amatrixposition_inplane] += -graddata_inplane

                #off diagonal terms that simulate the derivative term from the boltzmann equation, out of plane
                next_Amatrixposition_outofplane = self.Amatrixpositionlist[(i+1)%n,j]
                prev_Amatrixposition_outofplane = self.Amatrixpositionlist[(i-1)%n,j]

                graddata_outofplane = graddatalist_outofplane[Amatrixposition]

                self.A_Bdependent[Amatrixposition,next_Amatrixposition_outofplane] += graddata_outofplane
                self.A_Bdependent[Amatrixposition,prev_Amatrixposition_outofplane] += -graddata_outofplane

                j += 1
            i += 1

        self.A_Bdependent = self.A_Bdependent.tocsr()


    def createAlpha(self,gmres=False):
        if gmres:
            #create a scipy "LinearOperator (LO)" object that takes returns self.A @ x (where x is an input)
            def A_times_x(x): 
                return self.A_Bdependent@x + self.A_Bindependent@x #returns self.A@x
            
            A_times_x_LO = sp.sparse.linalg.LinearOperator(shape=(self.n,self.n),matvec=A_times_x,dtype=np.float64) #converts A_times_x to LO object

            #solve self.A@self.alpha = self.dedk_list using GMRES
            self.alpha = np.empty_like(self.dedk_list)
            
            for i in range(3):
                self.alpha[:,i], info = sp.sparse.linalg.gmres(A_times_x_LO,b=self.dedk_list[:,i],M=self.preconditioner_LO)
            if info>0:print(info) #print number of iterations if convergence tolerance not reached
        else:
            self.A = self.A_Bdependent + self.A_Bindependent
            self.alpha = sp.linalg.solve(self.A,self.dedk_list)

    def createSigma(self):
        #creates an array of the cartesian components of the velocity at each point on the discretized fermi surface
        self.moddedk_array = self.dedk_list/np.linalg.norm(self.dedk_list,axis=1)[:,None]

        #this creates the matrix sigma_mu_nu
        #mu and nu range from 0 to 2, with 0 being x, 1 being y and 2 being z
        self.sigma = np.zeros([3,3])

        for mu in range(3):
            for nu in range(3):
                #this keeps track of the total area over which we integrate
                self.areasum = 0
                self.sigma[mu,nu] = (3.699/(4*(np.pi**3)))*np.sum(self.moddedk_array[:,mu]*self.alpha[:,nu]*self.patcharealist)

        self.areasum = np.sum(self.patcharealist)

    def plotScatteringOut(self,dgdt_out_list):
        #creates a plot of scattering out rate vs angle
        import matplotlib.pyplot as plt
        fig= plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1_twin = ax1.twinx()
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        #pick an orbit in the middle,start and halfway point of the FS
        orbit_numbers = [0,self.FSorbitsInstance.n_points//8,self.FSorbitsInstance.n_points//4,self.FSorbitsInstance.n_points//2]

        for id,orbit_num in enumerate(orbit_numbers):
            #iterate over this orbit and plot scattering rate
            for (j,state) in enumerate(self.FSorbitsInstance.FSorbits[orbit_num]):
                theta = np.arctan2(state[1],state[0])
                r = np.sqrt(state[1]**2 + state[0]**2) 
                dgdt_out = dgdt_out_list[self.Amatrixpositionlist[orbit_num,j]]
                dos = 1/np.linalg.norm(self.dedk_list,axis=1)[self.Amatrixpositionlist[orbit_num,j]]
                vz = -self.dedk_list[self.Amatrixpositionlist[orbit_num,j],2]

                ax1.scatter(theta,dgdt_out+self.invtau_iso,color=f"C{id}",s=5)
                ax1.scatter(theta,10+189.85506941378708*np.cos(2*theta)**12,color=f"black",s=5)
                #ax1_twin.scatter(theta,vz,color=f"C{id+4}",s=5,marker="v")

                ax2.scatter(state[0],state[1],color=f"C{id}",s=5)
                ax3.scatter(theta,dos,color=f"C{id}",s=5)

        #reciprocal lattice vectors
        g1,g2,g3 = self.FSorbitsInstance.g1,self.FSorbitsInstance.g2,self.FSorbitsInstance.g3
        ax2.plot([-g1/2,g1/2,g1/2,-g1/2,-g1/2],[g2/2,g2/2,-g2/2,-g2/2,g2/2],color='black')

        ax1.set_title("Scattering out rate")
        ax1.set_xlim(-np.pi/2,np.pi/2)
        ax1.plot(0,0)
        ax1_twin.set_ylim(-0.1,0.5)
        ax2.set_title("Fermi Surface")
        ax3.set_title("Density of states")
        ax3.set_xlim(-np.pi/2,np.pi/2)
        ax3.plot(0,0)

        plt.tight_layout()
        plt.show()

