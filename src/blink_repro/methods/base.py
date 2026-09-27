from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MethodOutput:
    cleaned: np.ndarray
    estimated_artifact: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)

