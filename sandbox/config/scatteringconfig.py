from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ScatteringConfig:
    """Settings that define the scattering model used by a curve."""

    scatteringmodel: list[str] = field(
        default_factory = lambda: ["pipidelta_exp"]
    ) #takes in multiple scattering models as a list of strings
    delta_in_k: list[bool] = field(
        default_factory=lambda: [True]
    ) #takes in multiple values of whether scattering events can change z-momentum
    scattering_kwargs: list[dict[str, Any]] = field(
        default_factory=lambda: [{"strength": 13.4,"spread_xy": 0.136,"n": 5.167}]
    ) #takes in multiple scattering kwargs as a list of dicts 

    plot_scattering: bool = False
    print_in_plane_rho: bool = True
    invtau_iso: float = 9.0583774685104