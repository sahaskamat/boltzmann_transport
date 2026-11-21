from transport.wrappers import create_dispersion,create_FS
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations

fig = plt.figure(figsize=(15, 10))

# First row: 3D plots
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax3 = fig.add_subplot(2, 3, 3, projection='3d')

# Second row: conventional 2D plots
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)

for doping,mu,axes in zip([20,24,27],[0.77,0.82,0.87],[ax1,ax2,ax3]):
    kbT = 8.6e-5*100 #kbT at 100K in eV
    for offset,color in zip([-kbT,0,kbT],["C0","C1","C2"]):
        T=190e-3
        dispersionInstance = create_dispersion(T=T,T1multvalue=-0.134,T11multvalue=0.067,Tzmultvalue=0.03,mumultvalue=mu+offset/T)
        FSinstance = create_FS(dispersionInstance,res_z=20,res_xy=100)
        FSinstance.calculateDoping()

        for curve in FSinstance.initialcurvesList:
            axes.scatter(curve[:,0],curve[:,1], curve[:,2],s=5,color=color)

    #making unit cell
    hx, hy, hz = np.pi/dispersionInstance.a, np.pi/dispersionInstance.b, np.pi/FSinstance.c  

    pts = list(product([-hx, hx], [-hy, hy], [-hz, hz]))

    for s, e in combinations(pts, 2):
        if sum(abs(a-b) for a,b in zip(s,e)) in (2*hx, 2*hy, 2*hz):
            axes.plot3D(*zip(s,e), color="k")

    axes.set_title(str(doping))

for doping,mu,axes in zip([20,24,27],[0.77,0.82,0.87],[ax4,ax5,ax6]):
    kbT = 8.6e-5*100 #kbT at 100K in eV
    for offset,color in zip([-kbT,0,kbT],["C0","C1","C2"]):
        T=190e-3
        dispersionInstance = create_dispersion(T=T,T1multvalue=-0.134,T11multvalue=0.067,Tzmultvalue=0.03,mumultvalue=mu+offset/T)
        FSinstance = create_FS(dispersionInstance,res_z=20,res_xy=100)
        FSinstance.calculateDoping()

        curve = FSinstance.initialcurvesList[len(FSinstance.initialcurvesList)//4]
        axes.scatter(curve[:,0],curve[:,1],s=5,color=color)

    #making unit cell
    hx, hy, hz = np.pi/dispersionInstance.a, np.pi/dispersionInstance.b, np.pi/FSinstance.c  

    pts = list(product([-hx, hx], [-hy, hy]))

    for s, e in combinations(pts, 2):
        if sum(abs(a-b) for a,b in zip(s,e)) in (2*hx, 2*hy):
            axes.plot(*zip(s,e), color="k")

    axes.set_title(str(doping))
    
plt.show()