import numpy as np
import matplotlib.pyplot as plt
import dispersion
import orbitcreation
import conductivity
from makesigmalist import makelist_parallel
from time import time

def rhozzvstheta(theta_num,planeanchor_num,int_resolution):
    """
    Calculates the zz component of the resistivity tensor for a rotating magnetic field

    Parameters:
    -----------
    theta_num: number of angles equally spaced between zero and eighty degrees to solve for
    planeanchor_num: number of plane anchors in the z direction to start from (this roughly corresponds to number of orbits)
    int_resolution: resolution of the diff eq solver over the orbit. roughly corresponds to point spacing within a orbit

    Returns:
    ------------
    rhozzlist: a list containing the zzth component of the resistivity tensor for a given angle
    arealist: a list of the total fermi surface area calculated for a given angle
    thetalist: a list of the angles solved for
    """
    starttime_global = time()
    thetalist = np.linspace(0,80,theta_num)
    phi = 0

    phi_rad = np.deg2rad(phi)
    dispersionInstance = dispersion.LSCOdispersion()
    initialpointsInstance = orbitcreation.InterpolatedCurves(200,dispersionInstance,True)
    starttime = time()
    initialpointsInstance.solveforpoints(parallelised=False)
    initialpointsInstance.extendedZoneMultiply(5)
    initialpointsInstance.createPlaneAnchors(planeanchor_num)
    endtime = time()
    print(f"Time taken to create initialcurves = {endtime - starttime}")

    starttime = time()
    def getsigma(theta):
        B = [45*np.sin(np.deg2rad(theta))*np.cos(phi_rad),45*np.sin(np.deg2rad(theta))*np.sin(phi_rad),45*np.cos(np.deg2rad(theta))]
        orbitsinstance = orbitcreation.NewOrbits(dispersionInstance,initialpointsInstance)
        orbitsinstance.createOrbits(B,termination_resolution=int_resolution,mult_factor=15)
        orbitsinstance.createOrbitsEQS(integration_resolution=int_resolution)
        #print(f'orbitcreation completed for {theta} degrees')
        #if theta>=60: orbitsinstance.plotOrbitsEQS() #enable plotting for diagnostic purposes
        conductivityInstance = conductivity.Conductivity(dispersionInstance,orbitsinstance,initialpointsInstance)
        conductivityInstance.createAMatrix()
        conductivityInstance.createAlpha()
        conductivityInstance.createSigma()

        #orbitsinstance.orbitdiagnosticplot()
        if theta==80: print(f'matrixinversion performed for {theta}, A matrix dimension {conductivityInstance.A.shape}')
        #print(f"Theta={theta},Calculated total area: {conductivityInstance.areasum}, number of orbits used {len(conductivityInstance.orbitsInstance.orbitsEQS)}")
        return conductivityInstance.sigma,conductivityInstance.areasum

    sigmalist,rholist,arealist = makelist_parallel(getsigma,thetalist)
    endtime = time()
    print("Time taken to create orbits:",endtime-starttime)

    rhozzlist= [rho[2,2]*10e-5 for rho in rholist]

    endtime_global = time()
    print(f"execution time: {endtime_global-starttime_global}")

    #diagnostic saving/plotting below

    #np.savetxt("rhoxyvst.dat",np.transpose([thetalist,rhozzlist]))

    #plt.plot(thetalist,rhozzlist,ls="",marker="o",ms=2)
    #plt.ylabel(r"$\rho_{zz}$ ($m\Omega$ cm )")
    #plt.xlabel(r'$\theta$')
    #plt.show()

    #plt.scatter(thetalist,arealist)
    #plt.ylabel(r"$Area (\AA)$ cm )")
    #plt.xlabel(r'$\theta$')
    #plt.show()

    return rhozzlist,arealist,thetalist

if __name__ == "__main__":
    fig1, axes1 = plt.subplots()
    fig2, axes2 = plt.subplots()
    fig3,axes3 = plt.subplots()

    #first do a low res, many point scan to check for devations
    rhozzlist,arealist,thetalist = rhozzvstheta(500,50,0.07)
    axes1.plot(thetalist,rhozzlist,ls="",marker="o",ms=2,label="Res 1")
    axes2.plot(thetalist,arealist,ls="",marker="o",ms=2,label="Res 1")
    axes3.plot(thetalist,np.array(rhozzlist)*np.array(arealist),ls="",marker="o",ms=2,label="Res 1")

    #decrease res and keep points same
    rhozzlist,arealist,thetalist = rhozzvstheta(500,30,0.1)
    axes1.plot(thetalist,rhozzlist,ls="",marker="o",ms=2,label="Res 2")
    axes2.plot(thetalist,arealist,ls="",marker="o",ms=2,label="Res 2")
    axes3.plot(thetalist,np.array(rhozzlist)*np.array(arealist),ls="",marker="o",ms=2,label="Res 2")

    #increase res to quite high, decrease number of orbits
    rhozzlist,arealist,thetalist = rhozzvstheta(50,100,0.03)
    axes1.plot(thetalist,rhozzlist,ls="",marker="o",ms=4,label="Res 3")
    axes2.plot(thetalist,arealist,ls="",marker="o",ms=4,label="Res 3")
    axes3.plot(thetalist,np.array(rhozzlist)*np.array(arealist),ls="",marker="o",ms=4,label="Res 3")

    #max res, min orbits
    rhozzlist,arealist,thetalist = rhozzvstheta(20,200,0.01)
    axes1.plot(thetalist,rhozzlist,ls="",marker="o",ms=5,label="Res 4")
    axes2.plot(thetalist,arealist,ls="",marker="o",ms=5,label="Res 4")
    axes3.plot(thetalist,np.array(rhozzlist)*np.array(arealist),ls="",marker="o",ms=5,label="Res 4")

    axes1.set_ylabel(r"$\rho_{zz}$ ($m\Omega$ cm )")
    axes1.set_xlabel(r'$\theta$')
    axes1.legend()

    axes2.set_ylabel(f"Area (\u212B)")
    axes2.set_xlabel(r'$\theta$')
    axes2.legend()

    axes3.set_ylabel(r"$\rho_{zz}/$" + f"Area (\u212B)")
    axes3.set_xlabel(r'$\theta$')
    axes3.legend()
    
    plt.show()