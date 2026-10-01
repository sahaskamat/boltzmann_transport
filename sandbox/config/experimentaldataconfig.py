from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ExperimentalDataConfig:
    """Location and metadata for one experimental ADMR curve."""

    repository_directory: Path
    sample: str
    temperature_kelvin: int
    phi: float
    field: float
    label: str