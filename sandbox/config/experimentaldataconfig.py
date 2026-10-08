from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

@dataclass(frozen=True)
class ExperimentalDataConfig:
    """Location and metadata for one experimental ADMR curve."""

    repository_directory: Path = REPOSITORY_ROOT
    sample: str = "2511A"
    temperature_kelvin: int = 30
    phi: float = 0.0
    field: float = 45.0