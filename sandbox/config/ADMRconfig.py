from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ADMRConfig:
    """Settings to create a single ADMR curve."""

    temperature_kelvin: int = 30
    phi: float = 0
    field: float = 45.0
    theta_min: float = -14
    theta_max: float = 99
    n_thetas: int = 40
    workers: int = 5
