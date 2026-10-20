import numpy as np
import matplotlib.pyplot as plt

Temp = [35, 30, 25, 20]

Real_resistivities = [4.7, 4.5, 4.3, 4.1]

Predicted_resistivities = [5.05, 4.5, 4.0, 3.5]

Invau_iso = [10.2, 9.05, 8.0, 6.9]

fig,axes = plt.subplots(1,2,figsize=(12,6))

slope_real, intercept_real = np.polyfit(Temp, Real_resistivities, 1)
slope_predicted, intercept_predicted = np.polyfit(Temp, Predicted_resistivities, 1)
axes[0].plot(Temp,Real_resistivities,"o-",color='b',label=r"$\rho_{zz}$ (Measured)")
axes[0].plot(Temp,Predicted_resistivities,"o-",color='g',label=r"$\rho_{zz}$ (Predicted)")
axes[0].text(0.1,0.8, f"m(Predicted)/m(Measured) = {slope_predicted/slope_real:.3f}",transform=axes[0].transAxes)

slope, intercept = np.polyfit(Temp, Invau_iso, 1)
naturalunitslope = slope/0.13
axes[1].plot(
    Temp, Invau_iso, "o-", color='r',
    label=r"$\tau^{-1}$"+ f"\nm = {naturalunitslope:.3f}"+r" $k_B/\hbar$"
)

axes[0].set_xlabel("T (K)")
axes[0].set_ylabel(r"$\rho$ (m$\Omega$-cm)")
axes[1].set_xlabel("T (K)")
axes[1].set_ylabel(r"$\tau^{-1}$ (ps$^{-1}$)")

axes[0].legend()
axes[1].legend()

plt.show()