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

from sandbox.calculateADMR import polar_to_cartesian, makelist_parallel
from sandbox.createFSandScattering import build_scattering,build_FS
from sandbox.loaddata import load_experimental_data
from sandbox.plotting import normalize_data
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation as R

# Edit these blocks for new curves. The dispersion is built once and shared.
DISPERSION_CONFIG = DispersionConfig(Tzmultvalue = 0.033369871920650995)
SCATTERING_CONFIG = ScatteringConfig(invtau_iso=9.0583774685104,scattering_kwargs={"strength":13.24,"spread_xy":0.13598510117993956,"n":5.167})
ADMR_CONFIG = ADMRConfig(phi=0, field=45.0)

EXPERIMENTAL_DATA_CONFIGS = (
    ExperimentalDataConfig(sample="2511A",temperature_kelvin=35,phi=0,field=45.0),
    ExperimentalDataConfig(sample="2511A",temperature_kelvin=35,phi=45,field=45.0),
)

def rhoZZ_vs_theta(admr_config: ADMRConfig,conductivity_instance_noB,rotation):

    conductivity_instance = deepcopy(conductivity_instance_noB)
    theta = np.linspace(admr_config.theta_min, admr_config.theta_max, admr_config.n_thetas)

    def get_sigma(theta_value):
        field_vector = polar_to_cartesian(theta_value, admr_config.phi, admr_config.field)
        rotated_field_vector = rotation.apply(field_vector)
        conductivity_instance.createAmatrix_Bdependent(rotated_field_vector)
        conductivity_instance.createAlpha()
        conductivity_instance.createSigma()
        return conductivity_instance.sigma

    sigmalist= makelist_parallel(get_sigma, theta, workers=admr_config.workers)
    rholist = [np.linalg.inv(sigma) for sigma in sigmalist]

    rhozz = np.asarray([rho[2, 2] * 10e-5 for rho in rholist]) #rhozz in mOhm-cm

    return theta,rhozz

def main() -> None:
    dispersion, fermi_surface = build_FS(DISPERSION_CONFIG)
    scattering = build_scattering(SCATTERING_CONFIG,dispersion,fermi_surface)

    fig,axes = plt.subplots(figsize=(6,4))
    theta,rho = rhoZZ_vs_theta(ADMR_CONFIG,scattering,rotation = R.from_euler('z', 0, degrees=True))
    axes.plot(theta,rho,label="Precise Alignment")

    theta,rho = rhoZZ_vs_theta(ADMR_CONFIG,scattering,rotation = R.from_euler('y', 5, degrees=True))
    axes.plot(theta,rho,label="5$^\circ$ misalignment in plane of rotation")

    theta,rho = rhoZZ_vs_theta(ADMR_CONFIG,scattering,rotation = R.from_euler('x', 5, degrees=True))
    axes.plot(theta,rho,label="5$^\circ$ misalignment out of plane of rotation")

    axis = np.array([1, 1, 0]) / np.sqrt(2)
    rotation = R.from_rotvec(np.deg2rad(5) * axis)
    theta,rho = rhoZZ_vs_theta(ADMR_CONFIG,scattering,rotation)
    axes.plot(theta,rho,label="5$^\circ$ misalignment 45 deg from plane of rotation")

    axes.set_xlabel(r"$\theta$ ($^\circ$)")
    axes.set_ylabel(r"$\rho$ (m$\Omega$-cm)")
    axes.text(0.6,0.8, "LSCO 2511A (model)\n$T=30$ K, $B=45$ T\n$\phi=0^\circ$",transform=axes.transAxes)
    axes.legend(frameon=False,loc='lower left')

    plt.show()
    #plot_curves(theoretical_results=results, experimental_data=experimental_data)


if __name__ == "__main__":
    main()