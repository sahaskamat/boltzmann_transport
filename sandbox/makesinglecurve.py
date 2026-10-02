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

# Edit these blocks for new curves. The dispersion is built once and shared.
DISPERSION_CONFIG = DispersionConfig()
SCATTERING_CONFIG = ScatteringConfig()
ADMR_CONFIGS = (ADMRConfig(phi=0, field=45.0),ADMRConfig(phi=45, field=45.0))

EXPERIMENTAL_DATA_CONFIGS = (
    ExperimentalDataConfig(
        repository_directory=REPOSITORY_ROOT,
        sample="2511A",
        temperature_kelvin=30,
        phi=0,
        field=45.0,
        label="Data, phi=0",
    ),
    ExperimentalDataConfig(
        repository_directory=REPOSITORY_ROOT,
        sample="2511A",
        temperature_kelvin=30,
        phi=30,
        field=45.0,
        label="Data, phi=30",
    ),
)


def main() -> None:
    fig,axes = plt.subplots()
    dispersion, fermi_surface = build_FS(DISPERSION_CONFIG)
    scattering = build_scattering(SCATTERING_CONFIG,dispersion,fermi_surface)
    for admr_config in ADMR_CONFIGS:
        theta,rho = rhoZZ_vs_theta(admr_config,dispersion,fermi_surface,scattering)
        axes.plot(theta,rho)

    plt.show()
    #plot_curves(theoretical_results=results, experimental_data=experimental_data)


if __name__ == "__main__":
    main()