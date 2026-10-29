"""AWRLS-2: adaptive wavelet pseudo-reference blink removal, revised pipeline."""
from .config import Config, ablation_variants, sensitivity_grid, V1_REPRODUCTION
from .core import clean, process_epoch
from . import evaluate, wavelet_bank

__all__ = ["Config", "ablation_variants", "sensitivity_grid", "V1_REPRODUCTION", "clean", "process_epoch",
           "evaluate", "wavelet_bank"]
__version__ = "2.0.0-dev"
