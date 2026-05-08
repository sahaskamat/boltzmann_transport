import sympy as symp
import numpy as np
from math import sqrt
from numba import njit

class LSCOdispersion:
    """
    Inputs:
    T (energy scale of tight binding model)
    T1,T11,Tz (multiply by T to get unitfull tight binding parameters)
    mumultvalue (multiplicative factor that sets doping and hence chemical potential)

    mumultvalue = 0.8243(critical point) or 1.15(far from lifshits singularity)
    Class represents LSCO dispersion (remember to set doublefermisurface = True!)
    Contains symbolic calculations that are lambdified to generate numeric values of important dispersion parameters
    Default params are values from fitting in Gael's paper
    """
    def __init__(self,T= 160e-3,T1multvalue=-0.1364,T11multvalue=0.0682,Tzmultvalue=0.0651,mumultvalue=0.8243):

        #############################################################################################
        #LSCO specific functions are below
        #############################################################################################

        #define lattice constants in angstroms
        self.a = 3.75
        self.b= self.a
        self.c = 2*6.6

        #multiply parameters by T to get unitfull parameters in eV:
        T1 = T1multvalue*T
        T11 = T11multvalue*T
        Tz = Tzmultvalue*T
        self.mu = -mumultvalue*T #this is the critical point value
        #mu = -1.15*T #this is a value far from the lifshits singularity (and hence the fermi surface does not cross the van hove points)
        #now we symbolically define the dispersion
        kx, ky, kz = symp.symbols('kx ky kz')

        #energy in eV, k in (angstrom-1)
        en = -self.mu - 2*T*(symp.cos(kx*self.a) + symp.cos(ky*self.a)) - 4*T1*symp.cos(kx*self.a)*symp.cos(ky*self.a) - 2*T11*(symp.cos(2*kx*self.a) + symp.cos(2*ky*self.a)) - 2*Tz*symp.cos((kx*self.a)/2)*symp.cos((ky*self.a)/2)*symp.cos((kz*self.c)/2)*((symp.cos(kx*self.a) - symp.cos(ky*self.a))**2)
        graden = [symp.diff(en,kx),symp.diff(en,ky),symp.diff(en,kz)]

        #############################################################################################
        #end LSCO specific functions
        #############################################################################################

        from sympy.vector import CoordSys3D
        #now we write the RHS of the equation of motion, v \cross B

        R = CoordSys3D('R')
        gradvec = graden[0]*R.i + graden[1]*R.j + graden[2]*R.k

        Bx,By,Bz  =symp.symbols('Bx By Bz')
        Bvec = Bx*R.i + By*R.j + Bz*R.k

        #this is v \cross B converted to a scipy matrix
        force = gradvec.cross(Bvec).to_matrix(R)
        force = force.transpose()

        #this converts v \cross B into a numerical function that can be passed to scipy.odeint
        force_numeric = njit(symp.lambdify([kx,ky,kz,Bx,By,Bz],force))
        self.RHS_numeric = njit(lambda k,B : force_numeric(k[0],k[1],k[2],B[0],B[1],B[2])[0])

        #first convert symbolic dispersion to numeric function
        self.en_numeric = (symp.lambdify([kx,ky,kz],en,"numpy"))

        #define functions used in the A matrix calculations
        graden_numeric  = (symp.lambdify([kx,ky,kz],graden,"numpy"))
        self.graden_numeric = graden_numeric

        self.dedk = (lambda p: graden_numeric(p[0],p[1],p[2]))


class FREEdispersion:
    """
    Inputs:
    mumultvalue (multiplicative factor that sets doping and hence chemical potential)

    Class represents a cylindrical free electron dispersion in the plane with some z axis warping
    Contains symbolic calculations that are lambdified to generate numeric values of important dispersion parameters
    """
    def __init__(self,mu=7):

        #############################################################################################
        #FREE electron specific functions are below
        #############################################################################################

        #define lattice constants in angstroms
        self.a = 1
        self.b= self.a
        self.c = 2
        self.mu=mu
        
        #now we symbolically define the dispersion
        kx, ky, kz = symp.symbols('kx ky kz')

        en = en = 3.8099820794*(kx**2 + ky**2 + 0.3*symp.cos(kz*(self.c))) - self.mu #2d free electron dispersion, en is in eV and k is in (angstrom-1)
        graden = [symp.diff(en,kx),symp.diff(en,ky),symp.diff(en,kz)]

        #############################################################################################
        #end LSCO specific functions
        #############################################################################################

        from sympy.vector import CoordSys3D
        #now we write the RHS of the equation of motion, v \cross B

        R = CoordSys3D('R')
        gradvec = graden[0]*R.i + graden[1]*R.j + graden[2]*R.k

        Bx,By,Bz  =symp.symbols('Bx By Bz')
        Bvec = Bx*R.i + By*R.j + Bz*R.k

        #this is v \cross B converted to a scipy matrix
        force = gradvec.cross(Bvec).to_matrix(R)
        force = force.transpose()

        #this converts v \cross B into a numerical function that can be passed to scipy.odeint
        force_numeric = njit(symp.lambdify([kx,ky,kz,Bx,By,Bz],force))
        self.RHS_numeric = njit(lambda k,B : force_numeric(k[0],k[1],k[2],B[0],B[1],B[2])[0])

        #first convert symbolic dispersion to numeric function
        self.en_numeric = (symp.lambdify([kx,ky,kz],en,"numpy"))

        #define functions used in the A matrix calculations
        graden_numeric  = (symp.lambdify([kx,ky,kz],graden,"numpy"))
        self.graden_numeric = graden_numeric

        self.dedk = (lambda p: graden_numeric(p[0],p[1],p[2]))

    #function that defines the angle dependence of invtau, to be multiplied with invtau_aniso

    @staticmethod
    @njit
    def invtau(p):
        #scattering rate(inverse scattering time)
        #units of tau are ps, invtau are ps-1

        return np.ones(np.shape(p)[1])*(1/1000)

    @staticmethod
    def dkperp(B,dkz,dedk):
        #this calculates the length element lying along the fermi surface for integration
        #dkz is any point on the plane containing the next orbit
        nvec = np.cross(dedk,np.cross(dedk,B)) #nvec = dedk x (dedk x B)
        scalar_term = (np.dot(dkz,B))/(np.dot(nvec,B))
        dkperp = scalar_term[:,None]*nvec
        return dkperp