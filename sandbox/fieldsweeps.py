"""Generate single-field ADMR curves from an easy-to-edit configuration."""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sandbox.config.dispersionconfig import DispersionConfig
from sandbox.config.scatteringconfig import ScatteringConfig
from sandbox.config.ADMRconfig import ADMRConfig
from sandbox.config.experimentaldataconfig import ExperimentalDataConfig

from sandbox.calculateADMR import rhoZZ_vs_theta
from sandbox.createFSandScattering import build_scattering,build_FS
from sandbox.loaddata import load_experimental_data
from sandbox.plotting import normalize_data

fig,axes = plt.subplots(figsize=(8,4))

for invtau_iso in [10,15,20]:

    DISPERSION_CONFIG = DispersionConfig(Tzmultvalue = 0.09)
    SCATTERING_CONFIG = ScatteringConfig(
            invtau_iso=invtau_iso,
            scatteringmodel=["pipidelta_exp","pipi"],
            delta_in_k=[True,False],
            scattering_kwargs=[{"strength":14,"spread_xy":0.13,"n":5},{"strength":0,"spread_xy":0.4,"n":2}],
            plot_scattering=False
        )   
    ADMR_CONFIGS = (ADMRConfig(phi=5, field=41.5),ADMRConfig(phi=10, field=41.5),ADMRConfig(phi=30, field=41.5),ADMRConfig(phi=45, field=41.5))

    dispersion, fermi_surface = build_FS(DISPERSION_CONFIG)

    scattering = build_scattering(SCATTERING_CONFIG,dispersion,fermi_surface)

    Blist = np.linspace(0,60,20)
    rholist = np.zeros(20)
    for idx,B in enumerate(np.linspace(0,40,20)):
        conductivityinstance = deepcopy(scattering)

        conductivityinstance.createAmatrix_Bdependent([0,0,B])
        conductivityinstance.createAlpha()
        conductivityinstance.createSigma()
        rho = np.linalg.inv(conductivityinstance.sigma)[0,0]

        rholist[idx] = rho

    axes.plot(Blist,rholist-rholist[0])

#theta_exp,rho_exp = load_experimental_data(experimental_data)
#axes.plot(theta_exp,rho_exp)
#axes2.plot(normalize_data(theta_exp,rho_exp)[0],normalize_data(theta_exp,rho_exp)[1])

axes.set_xlabel(r"B (T)")
axes.set_ylabel(r"$\rho$")

fig.tight_layout()
plt.show()
#plot_curves(theoretical_results=results, experimental_data=experimental_data)