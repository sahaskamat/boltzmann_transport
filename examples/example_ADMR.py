"""first create a dispersion that encodes the dispersion and associated gradients v(k) and v(k)xB"""
from transport.dispersion import LSCOdispersion
dispersionInstance = LSCOdispersion(T=190e-3,T1multvalue=-0.134,T11multvalue=0.066,Tzmultvalue=0.033,mumultvalue=0.81)

"""then create a discretized fermi surface using this dispersion"""
from transport.fermiSurface import fermiSurface
fermiSurfaceInstance = fermiSurface(res_z=20,res_xy=100,dispersion=dispersionInstance,doublefermisurface=True)
fermiSurfaceInstance.createFS(tilingformat="variable",alpha=0.1,parallelised=False) 

"""then create a scattering matrix"""
from transport.scatteringMatrix import scatteringMatrix
scatteringMatrixInstance = scatteringMatrix(dispersionInstance=dispersionInstance,fermiSurfaceInstance=fermiSurfaceInstance)

#first create the isotropic scattering matrix
scatteringMatrixInstance.createAmatrix_Bindependent_isotropic(invtau_iso=8)

#then create a scattering kernel H(k,k') that is used to create the scattering matrix
kwargs={"strength":13.4,"spread_xy":0.136,"n":5.167}
scatteringMatrixInstance.create_Hfunc(scatteringmodel="pipidelta_exp",delta_in_k=True,**kwargs)

#add scattering in and out terms based on the scattering kernel
scatteringMatrixInstance.createAmatrix_Bindependent_fwdscatter_out()
scatteringMatrixInstance.createAmatrix_Bindependent_fwdscatter_in()

"""for each phi,theta, implement the magnetic field as an addition to the scattering matrix and calculate the conductivity"""
#ADMR for B=41.5
import numpy as np
from matplotlib import pyplot as plt
from transport.conductivity import Conductivity

for phi in [0,45]:
    rhozz_array,theta_array = np.zeros(20),np.linspace(0,90,20)
    for idx,theta in enumerate(theta_array):
        #magnetic field vector in cartesian coordinates
        B = 41.5*np.array([np.sin(np.radians(theta))*np.cos(np.radians(phi)),np.sin(np.radians(theta))*np.sin(np.radians(phi)),np.cos(np.radians(theta))])
        #implement the magnetic field as a scattering term
        scatteringMatrixInstance.createAmatrix_Bdependent(B)
        #calculate the conductivity
        conductivityInstance = Conductivity(scatteringMatrixInstance)
        conductivityInstance.createAlpha()
        conductivityInstance.createSigma()
        rhozz_array[idx] = np.linalg.inv(conductivityInstance.sigma)[2,2]/10000 #outputs resistivity in mOhm-cm
    plt.plot(theta_array,rhozz_array)

plt.show()

    

