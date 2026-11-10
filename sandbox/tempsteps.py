"""Generate single-field ADMR curves from an easy-to-edit configuration."""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

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

DISPERSION_CONFIG = DispersionConfig(Tzmultvalue = 0.039)
ADMR_CONFIG = ADMRConfig(phi=45, field=41.5)

def main() -> None:
    dispersion, fermi_surface = build_FS(DISPERSION_CONFIG)

    fig,(axes,axes2) = plt.subplots(1,2,figsize=(8,4))
    #[10,15,20,25,30,35,40,50,60,75,100,125]
    for temp in  [10,15,20,25,30,35,40,50,60,75,100,125]:
        SCATTERING_CONFIG = ScatteringConfig(
        invtau_iso=4+0.5*(temp*0.3 + 0.002*temp**2),
        scatteringmodel=["pipidelta_exp","only_z_dependent"],
        delta_in_k=[True,False],
        scattering_kwargs=[{"strength":6,"spread_xy":0.136,"n":5.167},{"strength":0,"spread_z":0.5}],
        plot_scattering=False
        )   
        scattering = build_scattering(SCATTERING_CONFIG,dispersion,fermi_surface)

        theta,rho = rhoZZ_vs_theta(ADMR_CONFIG,scattering)
        axes.plot(theta,rho,label=f"$T$={temp}")
        axes2.plot(normalize_data(theta,rho)[0],normalize_data(theta,rho)[1])

        #theta_exp,rho_exp = load_experimental_data(experimental_data)
        #axes.plot(theta_exp,rho_exp)
        #axes2.plot(normalize_data(theta_exp,rho_exp)[0],normalize_data(theta_exp,rho_exp)[1])

    axes.set_xlabel(r"$\theta$ ($^\circ$)")
    axes.set_ylabel(r"$\rho$ (m$\Omega$-cm)")
    axes2.set_xlabel(r"$\theta$ ($^\circ$)")
    axes2.set_ylabel(r"$\rho/\rho_0$")
    axes.legend()

    fig.tight_layout()
    plt.show()
    #plot_curves(theoretical_results=results, experimental_data=experimental_data)


if __name__ == "__main__":
    main()