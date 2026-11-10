from transport.dispersion import LSCOdispersion
from transport.fermiSurface import fermiSurface
from transport.scatteringMatrix import scatteringMatrix
from transport.conductivity import Conductivity
from joblib import delayed, Parallel
import numpy as np

def create_dispersion(T,T1multvalue,T11multvalue,Tzmultvalue,mumultvalue):
    dispersionInstance = LSCOdispersion(T,T1multvalue,T11multvalue,Tzmultvalue,mumultvalue)
    return dispersionInstance

def create_FS(dispersionInstance,res_z,res_xy):
    fermiSurfaceInstance = fermiSurface(res_z,res_xy,dispersion=dispersionInstance,doublefermisurface=True)
    fermiSurfaceInstance.createFS(tilingformat="variable",alpha=0.1,parallelised=False) 
    return fermiSurfaceInstance

def create_SM(dispersionInstance,fermiSurfaceInstance,invtau_iso,scattering_params_list):
    """
    scattering_params_list[i] should be a list of the form
    [str:scattering kernel name,bool:delta in k,dict:kwargs dict for each scattering kernel]
    see examples/example_ADMR_with_wrapper.py for a usecase
    """
    scatteringMatrixInstance = scatteringMatrix(dispersionInstance=dispersionInstance,fermiSurfaceInstance=fermiSurfaceInstance)
    scatteringMatrixInstance.createAmatrix_Bindependent_isotropic(invtau_iso)

    for scattering_params in scattering_params_list:
        scatteringmodel = scattering_params[0]
        delta_in_k = scattering_params[1]
        kwargs = scattering_params[2]

        scatteringMatrixInstance.create_Hfunc(scatteringmodel,delta_in_k,**kwargs)
        scatteringMatrixInstance.createAmatrix_Bindependent_fwdscatter_out()
        scatteringMatrixInstance.createAmatrix_Bindependent_fwdscatter_in()

    return scatteringMatrixInstance

def calculate_sigma(scatteringMatrixInstance):
    conductivityInstance = Conductivity(scatteringMatrixInstance)
    conductivityInstance.createAlpha()
    conductivityInstance.createSigma()
    return conductivityInstance.sigma

def polar_to_cartesian(r,theta,phi):
    vec = np.array([
        r*np.sin(np.radians(theta))*np.cos(np.radians(phi)),
        r*np.sin(np.radians(theta))*np.sin(np.radians(phi)),
        r*np.cos(np.radians(theta))
        ])
    return vec

def rhozz_vs_theta(B,phi,theta_array,scatteringMatrixInstance):

    def calculate_rhozz(theta):
        scatteringMatrixInstance.createAmatrix_Bdependent(polar_to_cartesian(B,theta,phi))
        conductivityInstance = Conductivity(scatteringMatrixInstance)
        conductivityInstance.createAlpha()
        conductivityInstance.createSigma()
        rho = np.linalg.inv(conductivityInstance.sigma)
        return rho[2,2]

    rholist = makelist_parallel(calculate_rhozz,theta_array)
    return rholist

def makelist_parallel(function,inputslist,workers=4):
    """#operate function over inputslist paralelly to create functionlist"""
    functionlist = Parallel(n_jobs=int(workers))(delayed(function)(inputinstance) for inputinstance in inputslist)

    return functionlist



