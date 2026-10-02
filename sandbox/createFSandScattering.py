from dataclasses import dataclass

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
    print(f"Doping={doping}")
    return dispersion_instance, fsOrbits_instance



def build_scattering(config: ScatteringConfig, dispersion_instance, fsOrbits_instance):
    conductivity_instance = conductivity.Conductivity(
        dispersion_instance,
        fsOrbits_instance,
        invtau_iso=config.invtau_iso,
        delta_in_k=config.delta_in_k,
    )
    conductivity_instance.createAmatrix_Bindependent_isotropic()
    conductivity_instance.create_Hfunc(
        scatteringmodel=config.scatteringmodel,
        **config.scattering_kwargs,
    )
    conductivity_instance.createAmatrix_Bindependent_fwdscatter_out()
    conductivity_instance.createAmatrix_Bindependent_fwdscatter_in()
    if config.plot_scattering:
        conductivity_instance.plotScatteringOut()
    return conductivity_instance