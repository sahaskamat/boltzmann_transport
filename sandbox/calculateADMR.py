from dataclasses import dataclass
from time import time
from copy import deepcopy

import numpy as np

from transport.makesigmalist import makelist_parallel

from .config import ADMRConfig, ScatteringConfig


def make_theta_list(config: ADMRConfig) -> np.ndarray:
    theta = np.linspace(config.theta_min, config.theta_max, config.n_thetas)
    return np.sort(np.concatenate([theta, [0.0]]))


def magnetic_field(theta: float, phi: float, magnitude: float) -> list:
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    return [
        magnitude * np.sin(theta_rad) * np.cos(phi_rad),
        magnitude * np.sin(theta_rad) * np.sin(phi_rad),
        magnitude * np.cos(theta_rad),
    ]


def rhoZZ_vs_theta(admr_config: ADMRConfig,dispersion_instance,fsOrbits_instance,conductivity_instance_noB):
    conductivity_instance = deepcopy(conductivity_instance_noB)
    theta = np.linspace(admr_config.theta_min, admr_config.theta_max, admr_config.n_thetas)

    def get_sigma(theta_value):
        conductivity_instance.createAmatrix_Bdependent(
            magnetic_field(theta_value, admr_config.phi, admr_config.field)
        )
        conductivity_instance.createAlpha()
        conductivity_instance.createSigma()
        return conductivity_instance.sigma,conductivity_instance.areasum

    sigmalist, rholist, areas = makelist_parallel(
        get_sigma, theta, workers=admr_config.workers
    )

    rhozz = np.asarray([rho[2, 2] * 10e-5 for rho in rholist])

    return theta,rhozz
