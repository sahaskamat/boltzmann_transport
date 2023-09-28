import sympy as symp
import numpy as np
from math import sqrt
from numba import njit

def deltap(p1,p2):
    #takes input p1 and p2 as lists and returns magnitude of their difference
    #works only for p1 and p2 of length 3
    return norm([p1[i]-p2[i] for i in range(3)])

def cross(a, b):
    #manually defined cross product since np.cross is very slow
    result = [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]

    return result

def dot(a,b):
    #manually defined dot product since numpy is very slow
    result = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    return result

def norm(p):
    #manually define norm without going through numpy
    result = sqrt(p[0]**2 + p[1]**2 + p[2]**2)
    return result

class LSCOdispersion:
    """
    Inputs:
    mumultvalue (multiplicative factor that sets doping and hence chemical potential)
    mumultvalue = 0.8243(critical point) or 1.15(far from lifshits singularity)

    Class represents LSCO dispersion (remember to set doublefermisurface = True!)
    Contains symbolic calculations that are lambdified to generate numeric values of important dispersion parameters
    """
    def __init__(self,mumultvalue=0.8243):

        #############################################################################################
        #LSCO specific functions are below
        #############################################################################################

        #define lattice constants in angstroms
        self.a = 3.75
        self.b= self.a
        self.c = 2*6.6

        #hopping parameters in eV from Yawen and Gael's paper:
        T = (160)*(10**(-3))
        T1 = -0.1364*T
        T11 = 0.0682*T
        Tz = 0.0651*T
        self.mu = -mumultvalue*T #this is the critical point value
        #mu = -1.15*T #this is a value far from the lifshits singularity (and hence the fermi surface does not cross the van hove points)
        #now we symbolically define the dispersion
        kx, ky, kz = symp.symbols('kx ky kz')

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

    #function that defines the angle dependence of invtau, to be multiplied with invtau_aniso

    @staticmethod
    @njit
    def invtau(p):
        #scattering rate(inverse scattering time)
        #units of tau are ps, invtau are ps-1
        invtau_iso = 9.628
        invtau_aniso = 63.929
        nu=12

        angledependence = np.float_power(np.abs((p[1]**2-p[0]**2)/(p[1]**2+p[0]**2)),nu)

        return (invtau_iso + invtau_aniso*angledependence)

    @staticmethod
    def dkperp(B,dkz,dedk):
        #this calculates the length element lying along the fermi surface for integration
        #dkz is any point on the plane containing the next orbit
        nvec = np.cross(dedk,np.cross(dedk,B)) #nvec = dedk x (dedk x B)
        scalar_term = (np.dot(dkz,B))/(np.dot(nvec,B))
        dkperp = scalar_term[:,None]*nvec
        return dkperp
    
class FREEdispersion:
    """
    Inputs:
    mumultvalue (multiplicative factor that sets doping and hence chemical potential)
    mumultvalue = 0.8243(critical point) or 1.15(far from lifshits singularity)

    Class represents LSCO dispersion (remember to set doublefermisurface = True!)
    Contains symbolic calculations that are lambdified to generate numeric values of important dispersion parameters
    """
    def __init__(self,mumultvalue=0.8243):

        #############################################################################################
        #LSCO specific functions are below
        #############################################################################################

        #define lattice constants in angstroms
        self.a = 1
        self.b= self.a
        self.c = 2

        #hopping parameters in eV from Yawen and Gael's paper:
        self.mu=7
        #now we symbolically define the dispersion
        kx, ky, kz = symp.symbols('kx ky kz')

        en = en = 3.8099820794*(kx**2 + ky**2 + 0.0000001*symp.cos(kz*self.c)) - self.mu #2d free electron dispersion, en is in eV and k is in (angstrom-1)
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

        return 1/100

    @staticmethod
    def dkperp(B,dkz,dedk):
        #this calculates the length element lying along the fermi surface for integration
        #dkz is any point on the plane containing the next orbit
        nvec = np.cross(dedk,np.cross(dedk,B)) #nvec = dedk x (dedk x B)
        scalar_term = (np.dot(dkz,B))/(np.dot(nvec,B))
        dkperp = scalar_term[:,None]*nvec
        return dkperp