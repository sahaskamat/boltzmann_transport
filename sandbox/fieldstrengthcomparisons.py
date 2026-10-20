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
DISPERSION_CONFIG = DispersionConfig(Tzmultvalue = 0.033369871920650995)
SCATTERING_CONFIG = ScatteringConfig(invtau_iso=9.0583774685104,scattering_kwargs={"strength":13.24,"spread_xy":0.13598510117993956,"n":5.167})
ADMR_CONFIGS = (ADMRConfig(phi=0, field=45),ADMRConfig(phi=45, field=45),
                ADMRConfig(phi=0, field=35),ADMRConfig(phi=45, field=35),
                ADMRConfig(phi=0, field=20),ADMRConfig(phi=45, field=20),
                ADMRConfig(phi=0, field=10),ADMRConfig(phi=45, field=10)
                )

def main() -> None:
    dispersion, fermi_surface = build_FS(DISPERSION_CONFIG)
    scattering = build_scattering(SCATTERING_CONFIG,dispersion,fermi_surface)

    fig,(axes,axes2) = plt.subplots(1,2,figsize=(12,6))

    colors = ["C0","C0","C1","C1","C2","C2","C3","C3"]
    labels = ["B = 45 T",None,"B = 35 T",None,"B = 20 T",None,"B = 10 T",None]
    for admr_config,color,label in zip(ADMR_CONFIGS, colors, labels):

        theta,rho = rhoZZ_vs_theta(admr_config,scattering)
        axes.plot(theta,rho,color=color,label=label)
        axes2.plot(normalize_data(theta,rho)[0],normalize_data(theta,rho)[1],color=color,label=label)

    axes.set_xlabel(r"$\theta$ ($^\circ$)")
    axes.set_ylabel(r"$\rho$ (m$\Omega$-cm)")
    axes2.set_xlabel(r"$\theta$ ($^\circ$)")
    axes2.set_ylabel(r"$\rho/\rho_0$")

    axes.legend()
    axes.text(0.65,0.8, "LSCO 2511A (model)\n$T=30$ K\n$\phi=0^\circ$ and $\phi=45^\circ$",transform=axes.transAxes)

    fig.tight_layout()
    plt.show()
    #plot_curves(theoretical_results=results, experimental_data=experimental_data)


if __name__ == "__main__":
    main()