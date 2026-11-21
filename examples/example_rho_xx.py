"""first create a dispersion that encodes the dispersion and associated gradients v(k) and v(k)xB"""
from transport.dispersion import LSCOdispersion
dispersionInstance = LSCOdispersion(T=190e-3,T1multvalue=-0.128,T11multvalue=0.064,Tzmultvalue=0.033,mumultvalue=0.87)

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
kwargs={"strength":14,"spread_xy":0.13,"n":5}
scatteringMatrixInstance.create_Hfunc(scatteringmodel="pipidelta_exp",delta_in_k=True,**kwargs)

#add scattering in and out terms based on the scattering kernel
scatteringMatrixInstance.createAmatrix_Bindependent_fwdscatter_out()
scatteringMatrixInstance.createAmatrix_Bindependent_fwdscatter_in()

#add in the magnetic field as a scattering term (in this case, 9T of out-of-plane field)
scatteringMatrixInstance.createAmatrix_Bdependent([0,0,9])

"""use the scattering matrix to calculate the conductivity"""

from transport.conductivity import Conductivity
conductivityInstance = Conductivity(scatteringMatrixInstance)
conductivityInstance.createAlpha()
conductivityInstance.createSigma()

from numpy.linalg import inv
print("rho_xx =",inv(conductivityInstance.sigma)[0,0]/10,"uOhm-cm")
