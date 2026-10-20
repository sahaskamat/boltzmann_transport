from dataclasses import dataclass
import numpy as np

import transport.conductivity as conductivity
import transport.dispersion as dispersion
import transport.orbitcreation as orbitcreation

from .config import DispersionConfig, ScatteringConfig


def build_FS(config: DispersionConfig):
    dispersion_instance = dispersion.LSCOdispersion(
        T=config.temperature_energy,
        T1multvalue=config.T1multvalue,
        T11multvalue=config.T11multvalue,
        Tzmultvalue=config.Tzmultvalue,
        mumultvalue=config.mumultvalue,
    )
    fsOrbits_instance = orbitcreation.fermiSurfaceOrbits(
        config.res_z, config.res_xy, dispersion_instance, True
    )
    fsOrbits_instance.createFS(
        tilingformat="variable", alpha=config.tiling_alpha, parallelised=False
    )
    doping = fsOrbits_instance.calculateDoping()
    return dispersion_instance, fsOrbits_instance

def build_scattering(config: ScatteringConfig, dispersion_instance, fsOrbits_instance):
    conductivity_instance = conductivity.Conductivity(
        dispersion_instance,
        fsOrbits_instance,
        invtau_iso=config.invtau_iso
    )
    conductivity_instance.createAmatrix_Bindependent_isotropic()

    for scatteringmodel,delta_in_k,scattering_kwargs in zip(config.scatteringmodel,config.delta_in_k,config.scattering_kwargs):
        conductivity_instance.delta_in_k = delta_in_k
        conductivity_instance.create_Hfunc(
            scatteringmodel=scatteringmodel,
            **scattering_kwargs
        )
        conductivity_instance.createAmatrix_Bindependent_fwdscatter_out()
        conductivity_instance.createAmatrix_Bindependent_fwdscatter_in()

    if config.plot_scattering:
        conductivity_instance.plotScatteringOut()

    if config.print_in_plane_rho:
        conductivity_instance.createAmatrix_Bdependent([0,0,0])
        conductivity_instance.createAlpha()
        conductivity_instance.createSigma()
        sigma_0 = conductivity_instance.sigma
        print("rho_xx =",np.linalg.inv(sigma_0)[0,0]/10,"uOhm-cm")

    return conductivity_instance