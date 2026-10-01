"""Generate single-field ADMR curves from an easy-to-edit configuration."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sandbox.config.dispersionconfig import DispersionConfig
from sandbox.config.scatteringconfig import ScatteringConfig
from sandbox.config.ADMRconfig import ADMRConfig
from sandbox.config.experimentaldataconfig import ExperimentalDataConfig

from sandbox.calculations import calculate_all_curves
from sandbox.data import load_and_interpolate
from sandbox.model import build_model
from sandbox.plotting import plot_curves

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
    model = build_model(DISPERSION_CONFIG)
    results = calculate_all_curves(ADMR_CONFIGS, SCATTERING_CONFIG, model)
    experimental_data = [
        load_and_interpolate(config, result.theta)
        for config, result in zip(EXPERIMENTAL_DATA_CONFIGS, results)
    ]
    plot_curves(theoretical_results=results, experimental_data=experimental_data)


if __name__ == "__main__":
    main()