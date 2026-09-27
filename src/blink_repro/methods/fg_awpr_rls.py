"""Load the unchanged single-input reference core; legacy adapters excluded."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=3)
def _load(filename: str, module_name: str) -> ModuleType:
    if filename != "fg_awpr_rls_simulated_reference.py":
        raise ValueError("Only the strict single-input core is distributed.")
    path = PROJECT_ROOT / "proposed_reference" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load FG-AWPR-RLS reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
