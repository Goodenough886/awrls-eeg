from dataclasses import dataclass, field, replace, asdict
import math
from awrls2.config import Config as LegacyConfig


@dataclass
class Config:
    base: LegacyConfig = field(default_factory=LegacyConfig)
    estimator: str = "aligned_ls"
    ridge: float = 0.003
    self_penalty: float = 3.0
    dictionary: bool = True
    background: bool = True
    confidence_gain: bool = True
    rescue: bool = False
    template_match_min: float = 0.65
    confidence_floor: float = 0.35
    background_ridge: float = 0.05
    background_frequencies: int = 2
    correction_lowpass_hz: float = 12.0

    def with_updates(self, **kw):
        return replace(self, **kw)

    def validate(self):
        self.base.validate()
        if self.estimator not in ("aligned_ls", "aligned_gls"):
            raise ValueError("estimator must be aligned_ls or aligned_gls")
        for name in ("ridge", "self_penalty", "background_ridge"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name)
        for name in ("template_match_min", "confidence_floor"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(name)
        if not 0 < self.correction_lowpass_hz < self.base.fs / 2:
            raise ValueError("correction_lowpass_hz")
        if self.background_frequencies not in (0, 1, 2, 3):
            raise ValueError("background_frequencies")

    def to_dict(self):
        return asdict(self)
