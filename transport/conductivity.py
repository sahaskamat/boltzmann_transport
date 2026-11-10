import numpy as np
import scipy as sp
import transport.dispersion as dispersion
from time import time

def kronecker_delta(x,y,atol=1e-5):
    """Takes in two numbers and returns 1 if they are the same, 0 otherwise"""
    return np.isclose(x,y,atol=atol).astype(int)

class Conductivity:
    """
    Inputs:
    dispersionInstance (object of type dispersion)
    orbitsInstance (object of type orbits)
    initialPointsInstance (object of type initialpoints)

    Contains methods to calculate the Amatrix, alpha, and sigma
    """
    def __init__(self,scatteringMatrixInstance):
        self.SMInstance = scatteringMatrixInstance

    def createAlpha(self,initial_guess=None):
        """Calculate"""
        A = self.SMInstance.A_Bdependent + self.SMInstance.A_Bindependent
        self.alpha = sp.linalg.solve(A,self.SMInstance.dedk_list)

    def createSigma(self):
        #creates an array of unit vectors of the velocity at each point on the discretized fermi surface
        dedk_list = self.SMInstance.dedk_list
        moddedk_list = dedk_list/np.linalg.norm(dedk_list,axis=1)[:,None]

        #this creates the matrix sigma_mu_nu
        #mu and nu range from 0 to 2, with 0 being x, 1 being y and 2 being z
        self.sigma = np.zeros([3,3])
        #IMPORTANT:Sigma is in units of S/m (i.e. SI units of conductivity)

        for mu in range(3):
            for nu in range(3):
                self.areasum = 0 #this keeps track of the total area over which we integrate
                self.sigma[mu,nu] = (3.699e-10/(4*(np.pi**3)))*np.sum(moddedk_list[:,mu]*self.alpha[:,nu]*self.SMInstance.patcharealist)

        self.areasum = np.sum(self.SMInstance.patcharealist)

    def plotScatteringOut(self):
        dgdt_out_list = self.dgdt_out_list
        #creates a plot of scattering out rate vs angle
        import matplotlib.pyplot as plt
        fig= plt.figure(figsize=(11, 10))
        ax1 = fig.add_subplot(2, 3, 1)
        ax1_twin = ax1.twinx()
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        #pick an orbit in the middle,start and halfway point of the FS
        orbit_numbers = [self.FSorbitsInstance.n_points//2,self.FSorbitsInstance.n_points//4,self.FSorbitsInstance.n_points//8,0]

        #reciprocal lattice vectors
        g1,g2,g3 = self.FSorbitsInstance.g1,self.FSorbitsInstance.g2,self.FSorbitsInstance.g3
        ax2.plot([-g1/2,g1/2,g1/2,-g1/2,-g1/2],[g2/2,g2/2,-g2/2,-g2/2,g2/2],color='black')
        ax4.plot([-g1/2,g1/2,g1/2,-g1/2,-g1/2],[g2/2,g2/2,-g2/2,-g2/2,g2/2],color='black')

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
                ax1_twin.scatter(theta,vz,color=f"C{id+4}",s=5,marker="v")

                ax2.scatter(state[0],state[1],color=f"C{id}",s=5)
                ax3.scatter(theta,dos,color=f"C{id}",s=5)

            #plot space of all momentum transfers
            array_of_states = self.FSorbitsInstance.FSorbits[orbit_num]
            diff = array_of_states[:, None, :] - array_of_states[None, :, :] 
            diff = diff.reshape(-1, 3) #diff is state minus every other state
            diff[:,0] = diff[:,0] - g1*np.round(diff[:,0]/g1) #modulo reciprocal lattice
            diff[:,1] = diff[:,1] - g2*np.round(diff[:,1]/g2) #modulo reciprocal lattice
            ax4.scatter(diff[:, 0], diff[:, 1], color=f"C{id}", s=5)

            #now put contours of h_func on this
            x,y = np.meshgrid(np.linspace(-g1/2,g1/2,100),np.linspace(-g2/2,g2/2,100))
            z = np.vectorize(self.H_func)(x,y)
            print(z.shape)
            ax4.contour(x,y,z,levels=10)

        ax1.set_title("Scattering out rate")
        ax1.set_xlim(-np.pi/2,np.pi/2)
        ax1_twin.set_xlim(-np.pi/2,np.pi/2)
        ax1.plot(0,0)
        ax1_twin.set_ylim(-0.1,0.5)
        ax2.set_title("Fermi Surface")
        ax3.set_title("Density of states")
        ax3.set_xlim(-np.pi/2,np.pi/2)
        ax3.plot(0,0)

        fig.tight_layout()

