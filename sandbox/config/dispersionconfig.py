from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DispersionConfig:
    """Settings to create the dispersion and Fermi surface"""

    # Dispersion parameters
    temperature_energy: float = 190e-3
    Tzmultvalue: float = 0.033369871920650995
    T1multvalue: float = -0.132
    T11multvalue: float = 0.066
    mumultvalue: float = 0.81

    # Discretization parameters for FS
    res_z: int = 20
    res_xy: int = 100
    tiling_alpha: float = 0.1