from pathlib import Path
from dataclasses import dataclass

import numpy as np

from .config import ExperimentalDataConfig

def data_path(config: ExperimentalDataConfig):
    return (
        Path(config.repository_directory)
        / "data"
        / "admr_data"
        / config.sample
        / f"{config.sample}_phi{config.phi:g}_T{config.temperature_kelvin}K_B{config.field:.1f}T.txt"
    )

def load_experimental_data(config: ExperimentalDataConfig):
    """Load one experimental file and return its sorted theta/rho arrays."""
    path = data_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Experimental data file not found: {path}")
    data_theta, data_rho = np.loadtxt(
        path, unpack=True, skiprows=1, delimiter=",", usecols=(0, 1)
    )
    order = np.argsort(data_theta)
    return data_theta[order], data_rho[order]

def interpolate_experimental_data(data_theta,data_rho,theta):
    """Load data and interpolate it onto a theoretical theta grid."""
    data_theta = np.asarray(data_theta)
    data_rho = np.asarray(data_rho)
    theta = np.asarray(theta)
    return np.interp(theta, data_theta, data_rho)
