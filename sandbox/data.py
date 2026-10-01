from pathlib import Path
from dataclasses import dataclass

import numpy as np

from .config import ExperimentalDataConfig


@dataclass(frozen=True)
class ExperimentalData:
    theta: np.ndarray
    rho: np.ndarray
    label: str


def _data_path(config: ExperimentalDataConfig) -> Path:
    return (
        Path(config.repository_directory)
        / "data"
        / "admr_data"
        / config.sample
        / f"{config.sample}_phi{config.phi:g}_T{config.temperature_kelvin}K_B{config.field:.1f}T.txt"
    )


def load_experimental_data(
    config: ExperimentalDataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one experimental file and return its sorted theta/rho arrays."""
    path = _data_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Experimental data file not found: {path}")
    data_theta, data_rho = np.loadtxt(
        path, unpack=True, skiprows=1, delimiter=",", usecols=(0, 1)
    )
    order = np.argsort(data_theta)
    return data_theta[order], data_rho[order]


def load_and_interpolate(
    config: ExperimentalDataConfig, theta: np.ndarray
) -> ExperimentalData:
    """Load data and interpolate it onto a theoretical theta grid."""
    data_theta, data_rho = load_experimental_data(config)
    theta = np.asarray(theta)
    return ExperimentalData(theta, np.interp(theta, data_theta, data_rho), config.label)
