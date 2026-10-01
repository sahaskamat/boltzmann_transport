from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ScatteringConfig:
    """Settings that define the scattering model used by a curve."""

    scatteringmodel: str = "pipidelta_exp"
    delta_in_k: bool = True
    plot_scattering: bool = False
    invtau_iso: float = 9.0583774685104
    scattering_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "strength": 13.4,
            "spread_xy": 0.13598510117993956,
            "n": 5.1668009899239635,
        }
    )