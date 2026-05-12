#computation libraries
import numpy as np
import matplotlib.pyplot as plt

#benchmarking libraries
from time import time
import cProfile, io
import pstats

#homemade libraries
import transport.dispersion as dispersion
import transport.orbitcreation as orbitcreation
import transport.conductivity as conductivity

def main():

    dispersionInstance = dispersion.LSCOdispersion(T= 190e-3,T1multvalue=-0.134,T11multvalue=0.067,Tzmultvalue=0.077,mumultvalue=0.805)
    FSorbitsInstance = orbitcreation.fermiSurfaceOrbits(200,1000,dispersionInstance,True)

    starttime = time()
    FSorbitsInstance.createFS(parallelised=False)
    doping = FSorbitsInstance.calculateDoping()
    endtime = time()

    print(f"Time taken to create FSorbits = {endtime - starttime}, Number of orbits found = {len(FSorbitsInstance.FSorbits)}, Doping = {doping}")
    FSorbitsInstance.plotpoints()

    theta = np.deg2rad(45)
    phi = np.deg2rad(0)
    B = [45*np.sin(theta)*np.cos(phi),45*np.sin(theta)*np.sin(phi),45*np.cos(theta)]

    #plottingintersections
    #ax.scatter(intersections[:,0],intersections[:,1],intersections[:,2],c='#FF0000',s=10)

    """
    starttime = time()
    conductivityInstance = conductivity.Conductivity(dispersionInstance,FSorbitsInstance)
    endtime = time()
    print(f"Time taken to create conductivityInstance =  {endtime - starttime}")

    starttime = time()
    conductivityInstance.createAmatrix_Bindependent()
    endtime = time()
    print(f"Time taken to create B independent Amatrix with shape {conductivityInstance.A_Bindependent.shape}=  {endtime - starttime}")

    starttime = time()
    conductivityInstance.createAmatrix_Bdependent(B)
    endtime = time()
    print(f"Time taken to create B dependent Amatrix =  {endtime - starttime}")

    starttime = time()
    conductivityInstance.createAlpha()
    endtime = time()
    print(f"Time taken to create Alpha =  {endtime - starttime}")

    starttime = time()
    conductivityInstance.createSigma()
    endtime = time()
    print(f"Time taken to calculate conductivity = {endtime - starttime}")
    print(f"Calculated rho_zz: {np.linalg.inv(conductivityInstance.sigma)[2,2]*10e-5} mOhm cm")
    """
cProfile.run('main()',filename='stats.prof')
#main()
#plt.show()
