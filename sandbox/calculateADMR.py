from dataclasses import dataclass
from time import time
from copy import deepcopy

import numpy as np

from joblib import delayed, Parallel

from .config import ADMRConfig, ScatteringConfig

def polar_to_cartesian(theta: float, phi: float, magnitude: float) -> list:
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    return [
        magnitude * np.sin(theta_rad) * np.cos(phi_rad),
        magnitude * np.sin(theta_rad) * np.sin(phi_rad),
        magnitude * np.cos(theta_rad),
    ]

def makelist_parallel(function,inputslist,workers):
    """#operate function over inputslist paralelly to create functionlist"""
    functionlist = Parallel(n_jobs=int(workers))(delayed(function)(inputinstance) for inputinstance in inputslist)

    return functionlist

def rhoZZ_vs_theta(admr_config: ADMRConfig,conductivity_instance_noB):
    conductivity_instance = deepcopy(conductivity_instance_noB)
    theta = np.linspace(admr_config.theta_min, admr_config.theta_max, admr_config.n_thetas)

    def get_sigma(theta_value):
        conductivity_instance.createAmatrix_Bdependent(polar_to_cartesian(theta_value, admr_config.phi, admr_config.field))
        conductivity_instance.createAlpha()
        conductivity_instance.createSigma()
        return conductivity_instance.sigma

    sigmalist= makelist_parallel(get_sigma, theta, workers=admr_config.workers)
    rholist = [np.linalg.inv(sigma) for sigma in sigmalist]

    rhozz = np.asarray([rho[2, 2] * 10e-5 for rho in rholist]) #rhozz in mOhm-cm

    return theta,rhozz
