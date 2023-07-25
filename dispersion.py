import sympy as symp
import numpy as np

def deltap(p1,p2):
    #takes input p1 and p2 as lists and returns magnitude of their difference
    return np.linalg.norm(np.array(p1) - np.array(p2))

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

class FreeElectronDispersion:
    """
    Inputs:
    a (denotes both in plane lattice constants)
    c (out of plane lattice constant)
    mu (chemical potential)

    Class represents a two dimensional free electron dispersion
    Contains symbolic calculations that are lambdified to generate numeric values of important dispersion parameters
    """
    def __init__(self,a,c,mu):
        #define lattice constants in angstroms
        self.a = a
        self.b = a
        self.c = c
        self.mu = mu

        #now we symbolically define the dispersion
        kx, ky, kz = symp.symbols('kx ky kz')

        en = 3.8099820794*(kx**2 + ky**2 + 0.0000001*symp.cos(kz*c)) - mu #2d free electron dispersion, en is in eV and k is in (angstrom-1)
        graden = [symp.diff(en,kx),symp.diff(en,ky),symp.diff(en,kz)]

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
        force_numeric = symp.lambdify([kx,ky,kz,Bx,By,Bz],force)
        self.RHS_numeric = lambda k,B : force_numeric(k[0],k[1],k[2],B[0],B[1],B[2])[0]

        #first convert symbolic dispersion to numeric function
        self.en_numeric = symp.lambdify([kx,ky,kz],en)

        #define functions used in the A matrix calculations
        self.graden_numeric  = symp.lambdify([kx,ky,kz],graden)

    def invtau(self,p):
        #units of tau are ps, invtau are ps-1
        return 1/100 #placeholder value of time constant for isotropic scattering out matrix

    def dedk(self,p):
        #takes input p as a list of [px,py,pz] and returns of de/dk at that p
        return np.array(self.graden_numeric(p[0],p[1],p[2]))

    def dkperp(self,p,B,dkz):
        #this calculates the length element lying along the fermi surface for integration
        #dkz is any point on the plane containing the next orbit
        nvec = np.cross(self.dedk(p),np.cross(self.dedk(p),B)) #nvec = dedk x (dedk x B)
        scalar_term = (np.dot(dkz,B))/(np.dot(nvec,B))
        dkperp = scalar_term*nvec
        return dkperp

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
        a = 3.75
        b = a
        c = 2*6.6

        self.a,self.b,self.c = a,b,c

        #hopping parameters in eV from Yawen and Gael's paper:
        T = (160)*(10**(-3))
        T1 = -0.1364*T
        T11 = 0.0682*T
        Tz = 0.0651*T
        mu = -mumultvalue*T #this is the critical point value
        #mu = -1.15*T #this is a value far from the lifshits singularity (and hence the fermi surface does not cross the van hove points)

        #############################################################################################
        #now define functions that have been spit out by mathematica code
        #############################################################################################

        def RHS_numeric(k,B):
            kx,ky,kz = k[0],k[1],k[2]
            Bx,By,Bz = B[0],B[1],B[2]
            return ( [( a * Bz * Tz * np.cos( 1/2 * a * kx ) * ( np.cos( a * kx ) )**( 2 ) * np.cos( 1/2 * c * kz ) * np.sin( 1/2 * a * ky ) + ( -2 * a * Bz * Tz * np.cos( 1/2 * a * kx ) * np.cos( a * kx ) * np.cos( a * ky ) * np.cos( 1/2 * c * kz ) * np.sin( 1/2 * a * ky ) + ( a * Bz * Tz * np.cos( 1/2 * a * kx ) * ( np.cos( a * ky ) )**( 2 ) * np.cos( 1/2 * c * kz ) * np.sin( 1/2 * a * ky ) + ( 2 * a * Bz * T * np.sin( a * ky ) + ( 4 * a * Bz * T1 * np.cos( a * kx ) * np.sin( a * ky ) + ( -4 * a * Bz * Tz * np.cos( 1/2 * a * kx ) * np.cos( a * kx ) * np.cos( 1/2 * a * ky ) * np.cos( 1/2 * c * kz ) * np.sin( a * ky ) + ( 4 * a * Bz * Tz * np.cos( 1/2 * a * kx ) * np.cos( 1/2 * a * ky ) * np.cos( a * ky ) * np.cos( 1/2 * c * kz ) * np.sin( a * ky ) + ( 4 * a * Bz * T11 * np.sin( 2 * a * ky ) + ( -1 * By * c * Tz * np.cos( 1/2 * a * kx ) * ( np.cos( a * kx ) )**( 2 ) * np.cos( 1/2 * a * ky ) * np.sin( 1/2 * c * kz ) + ( 2 * By * c * Tz * np.cos( 1/2 * a * kx ) * np.cos( a * kx ) * np.cos( 1/2 * a * ky ) * np.cos( a * ky ) * np.sin( 1/2 * c * kz ) + -1 * By * c * Tz * np.cos( 1/2 * a * kx ) * np.cos( 1/2 * a * ky ) * ( np.cos( a * ky ) )**( 2 ) * np.sin( 1/2 * c * kz ) ) ) ) ) ) ) ) ) ) ),( -1 * a * Bz * Tz * ( np.cos( a * kx ) )**( 2 ) * np.cos( 1/2 * a * ky ) * np.cos( 1/2 * c * kz ) * np.sin( 1/2 * a * kx ) + ( 2 * a * Bz * Tz * np.cos( a * kx ) * np.cos( 1/2 * a * ky ) * np.cos( a * ky ) * np.cos( 1/2 * c * kz ) * np.sin( 1/2 * a * kx ) + ( -1 * a * Bz * Tz * np.cos( 1/2 * a * ky ) * ( np.cos( a * ky ) )**( 2 ) * np.cos( 1/2 * c * kz ) * np.sin( 1/2 * a * kx ) + ( -2 * a * Bz * T * np.sin( a * kx ) + ( -4 * a * Bz * T1 * np.cos( a * ky ) * np.sin( a * kx ) + ( -4 * a * Bz * Tz * np.cos( 1/2 * a * kx ) * np.cos( a * kx ) * np.cos( 1/2 * a * ky ) * np.cos( 1/2 * c * kz ) * np.sin( a * kx ) + ( 4 * a * Bz * Tz * np.cos( 1/2 * a * kx ) * np.cos( 1/2 * a * ky ) * np.cos( a * ky ) * np.cos( 1/2 * c * kz ) * np.sin( a * kx ) + ( -4 * a * Bz * T11 * np.sin( 2 * a * kx ) + ( Bx * c * Tz * np.cos( 1/2 * a * kx ) * ( np.cos( a * kx ) )**( 2 ) * np.cos( 1/2 * a * ky ) * np.sin( 1/2 * c * kz ) + ( -2 * Bx * c * Tz * np.cos( 1/2 * a * kx ) * np.cos( a * kx ) * np.cos( 1/2 * a * ky ) * np.cos( a * ky ) * np.sin( 1/2 * c * kz ) + Bx * c * Tz * np.cos( 1/2 * a * kx ) * np.cos( 1/2 * a * ky ) * ( np.cos( a * ky ) )**( 2 ) * np.sin( 1/2 * c * kz ) ) ) ) ) ) ) ) ) ) ),( a * By * Tz * ( np.cos( a * kx ) )**( 2 ) * np.cos( 1/2 * a * ky ) * np.cos( 1/2 * c * kz ) * np.sin( 1/2 * a * kx ) + ( -2 * a * By * Tz * np.cos( a * kx ) * np.cos( 1/2 * a * ky ) * np.cos( a * ky ) * np.cos( 1/2 * c * kz ) * np.sin( 1/2 * a * kx ) + ( a * By * Tz * np.cos( 1/2 * a * ky ) * ( np.cos( a * ky ) )**( 2 ) * np.cos( 1/2 * c * kz ) * np.sin( 1/2 * a * kx ) + ( 2 * a * By * T * np.sin( a * kx ) + ( 4 * a * By * T1 * np.cos( a * ky ) * np.sin( a * kx ) + ( 4 * a * By * Tz * np.cos( 1/2 * a * kx ) * np.cos( a * kx ) * np.cos( 1/2 * a * ky ) * np.cos( 1/2 * c * kz ) * np.sin( a * kx ) + ( -4 * a * By * Tz * np.cos( 1/2 * a * kx ) * np.cos( 1/2 * a * ky ) * np.cos( a * ky ) * np.cos( 1/2 * c * kz ) * np.sin( a * kx ) + ( 4 * a * By * T11 * np.sin( 2 * a * kx ) + ( -1 * a * Bx * Tz * np.cos( 1/2 * a * kx ) * ( np.cos( a * kx ) )**( 2 ) * np.cos( 1/2 * c * kz ) * np.sin( 1/2 * a * ky ) + ( 2 * a * Bx * Tz * np.cos( 1/2 * a * kx ) * np.cos( a * kx ) * np.cos( a * ky ) * np.cos( 1/2 * c * kz ) * np.sin( 1/2 * a * ky ) + ( -1 * a * Bx * Tz * np.cos( 1/2 * a * kx ) * ( np.cos( a * ky ) )**( 2 ) * np.cos( 1/2 * c * kz ) * np.sin( 1/2 * a * ky ) + ( -2 * a * Bx * T * np.sin( a * ky ) + ( -4 * a * Bx * T1 * np.cos( a * kx ) * np.sin( a * ky ) + ( 4 * a * Bx * Tz * np.cos( 1/2 * a * kx ) * np.cos( a * kx ) * np.cos( 1/2 * a * ky ) * np.cos( 1/2 * c * kz ) * np.sin( a * ky ) + ( -4 * a * Bx * Tz * np.cos( 1/2 * a * kx ) * np.cos( 1/2 * a * ky ) * np.cos( a * ky ) * np.cos( 1/2 * c * kz ) * np.sin( a * ky ) + -4 * a * Bx * T11 * np.sin( 2 * a * ky ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ),] )

        def en_numeric(kx,ky,kz):
            return ( -1 * mu + ( -4 * T1 * np.cos( a * kx ) * np.cos( a * ky ) + ( -2 * T * ( np.cos( a * kx ) + np.cos( a * ky ) ) + ( -2 * T11 * ( np.cos( 2 * a * kx ) + np.cos( 2 * a * ky ) ) + -2 * Tz * np.cos( 1/2 * a * kx ) * np.cos( 1/2 * a * ky ) * ( ( np.cos( a * kx ) + -1 * np.cos( a * ky ) ) )**( 2 ) * np.cos( 1/2 * c * kz ) ) ) ) )

        def graden_numeric(kx,ky,kz):
            return ( [( a * Tz * np.cos( 1/2 * a * ky ) * ( ( np.cos( a * kx ) + -1 * np.cos( a * ky ) ) )**( 2 ) * np.cos( 1/2 * c * kz ) * np.sin( 1/2 * a * kx ) + ( 2 * a * T * np.sin( a * kx ) + ( 4 * a * T1 * np.cos( a * ky ) * np.sin( a * kx ) + ( 4 * a * Tz * np.cos( 1/2 * a * kx ) * np.cos( 1/2 * a * ky ) * ( np.cos( a * kx ) + -1 * np.cos( a * ky ) ) * np.cos( 1/2 * c * kz ) * np.sin( a * kx ) + 4 * a * T11 * np.sin( 2 * a * kx ) ) ) ) ),( a * Tz * np.cos( 1/2 * a * kx ) * ( ( np.cos( a * kx ) + -1 * np.cos( a * ky ) ) )**( 2 ) * np.cos( 1/2 * c * kz ) * np.sin( 1/2 * a * ky ) + ( 2 * a * T * np.sin( a * ky ) + ( 4 * a * T1 * np.cos( a * kx ) * np.sin( a * ky ) + ( -4 * a * Tz * np.cos( 1/2 * a * kx ) * np.cos( 1/2 * a * ky ) * ( np.cos( a * kx ) + -1 * np.cos( a * ky ) ) * np.cos( 1/2 * c * kz ) * np.sin( a * ky ) + 4 * a * T11 * np.sin( 2 * a * ky ) ) ) ) ),c * Tz * np.cos( 1/2 * a * kx ) * np.cos( 1/2 * a * ky ) * ( ( np.cos( a * kx ) + -1 * np.cos( a * ky ) ) )**( 2 ) * np.sin( 1/2 * c * kz ),] )

        self.RHS_numeric,self.en_numeric,self.graden_numeric = RHS_numeric,en_numeric,graden_numeric

    #function that defines the angle dependence of invtau, to be multiplied with invtau_aniso
    
    def angledependence(self,p):
        return np.float_power(np.abs((p[1]**2-p[0]**2)/(p[1]**2+p[0]**2)),12)

    def invtau(self,p):
        #scattering rate(inverse scattering time)
        #units of tau are ps, invtau are ps-1
        invtau_iso = 9.6
        invtau_aniso = 64

        return (invtau_iso + invtau_aniso*self.angledependence(p))

    def dedk(self,p):
        #takes input p as a list of [px,py,pz] and returns of de/dk at that p
        return np.array(self.graden_numeric(p[0],p[1],p[2]))

    def dkperp(self,B,dkz,dedk):
        #this calculates the length element lying along the fermi surface for integration
        #dkz is any point on the plane containing the next orbit
        nvec = cross(dedk,cross(dedk,B)) #nvec = dedk x (dedk x B)
        scalar_term = (dot(dkz,B))/(dot(nvec,B))
        dkperp = [scalar_term*nvec[i] for i in range(3)]
        return dkperp