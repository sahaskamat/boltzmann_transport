from dataclasses import dataclass
from time import time

import numpy as np

from transport.makesigmalist import makelist_parallel

from .config import ADMRConfig, ScatteringConfig
from .createFSandScattering import FermiSurface, build_scattering


@dataclass
class CurveResult:
    config: ADMRConfig
    theta: np.ndarray
    rhozz: np.ndarray
    rho_zero: np.ndarray
    rho_9T: np.ndarray
    areas: list

    @property
    def rhozz_normalized(self) -> np.ndarray:
        return self.rhozz / self.rhozz[np.argmin(self.theta**2)]


def make_theta_list(config: ADMRConfig) -> np.ndarray:
    theta = np.linspace(config.theta_min, config.theta_max, config.n_thetas)
    return np.sort(np.concatenate([theta, [0.0]]))


def _magnetic_field(theta: float, phi: float, magnitude: float) -> list:
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    return [
        magnitude * np.sin(theta_rad) * np.cos(phi_rad),
        magnitude * np.sin(theta_rad) * np.sin(phi_rad),
        magnitude * np.cos(theta_rad),
    ]


def calculate_curve(
    admr_config: ADMRConfig,
    scattering_config: ScatteringConfig,
    model: FermiSurface,
    theta: np.ndarray,
    ) -> CurveResult:
    start = time()
    conductivity_instance = build_scattering(scattering_config, model)

    def get_sigma(theta_value: float, field: float = admr_config.field):
        conductivity_instance.createAmatrix_Bdependent(
            _magnetic_field(theta_value, admr_config.phi, field)
        )
        conductivity_instance.createAlpha()
        conductivity_instance.createSigma()
        return conductivity_instance.sigma, conductivity_instance.areasum

    sigmalist, rholist, areas = makelist_parallel(
        get_sigma, theta, workers=admr_config.workers
    )
    sigma_zero, _ = get_sigma(theta_value=0.0, field=0.0)
    sigma_9T, _ = get_sigma(theta_value=0.0, field=9.0)
    rho_zero = np.linalg.inv(sigma_zero)
    rho_9T = np.linalg.inv(sigma_9T)
    rhozz = np.asarray([rho[2, 2] * 10e-5 for rho in rholist])

    print(f"rho_xx = {rho_zero[0, 0] * 10e-2}")
    print(f"rho_xy (9 T) = {rho_9T[0, 1] * 10e-2}")
    print(f"Curve calculation time = {time() - start}")
    return CurveResult(admr_config, theta, rhozz, rho_zero, rho_9T, areas)


def calculate_all_curves(
    admr_configs, scattering_config, model: FermiSurface):

    return [
        calculate_curve(
            admr_config,
            scattering_config,
            model,
            make_theta_list(admr_config),
        )
        for admr_config in admr_configs
    ]