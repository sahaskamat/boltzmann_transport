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

# Edit these blocks for new curves. The dispersion is built once and shared.
DISPERSION_CONFIG = DispersionConfig(Tzmultvalue = 0.05)
SCATTERING_CONFIG = ScatteringConfig(
        invtau_iso=9,
        scatteringmodel=["pipidelta_exp"],
        delta_in_k=[True],
        scattering_kwargs=[{"strength":13.4,"spread_xy":0.136,"n":5.167}],
        plot_scattering=False
    )   
ADMR_CONFIGS = (ADMRConfig(phi=0, field=45.0),ADMRConfig(phi=45, field=45.0))

EXPERIMENTAL_DATA_CONFIGS = (
    ExperimentalDataConfig(sample="2511A",temperature_kelvin=30,phi=0,field=45.0),
    ExperimentalDataConfig(sample="2511A",temperature_kelvin=30,phi=45,field=45.0),
)

def main() -> None:
    dispersion, fermi_surface = build_FS(DISPERSION_CONFIG)
    scattering = build_scattering(SCATTERING_CONFIG,dispersion,fermi_surface)

    fig,(axes,axes2) = plt.subplots(1,2,figsize=(8,4))
    for admr_config,experimental_data in zip(ADMR_CONFIGS, EXPERIMENTAL_DATA_CONFIGS):

        theta,rho = rhoZZ_vs_theta(admr_config,scattering)
        axes.plot(theta,rho)
        axes2.plot(normalize_data(theta,rho)[0],normalize_data(theta,rho)[1])

        theta_exp,rho_exp = load_experimental_data(experimental_data)
        axes.plot(theta_exp,rho_exp)
        axes2.plot(normalize_data(theta_exp,rho_exp)[0],normalize_data(theta_exp,rho_exp)[1])

    axes.set_xlabel(r"$\theta$ ($^\circ$)")
    axes.set_ylabel(r"$\rho$ (m$\Omega$-cm)")
    axes2.set_xlabel(r"$\theta$ ($^\circ$)")
    axes2.set_ylabel(r"$\rho/\rho_0$")

    plt.show()
    #plot_curves(theoretical_results=results, experimental_data=experimental_data)


if __name__ == "__main__":
    main()