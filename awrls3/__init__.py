"""AWRLS-3 research implementation; historical AWRLS-2 is vendored unchanged."""
from .config import Config
from .core import clean
from .dictionary import ShapeDictionary

__version__ = "3.0.0.dev1"
__all__ = ["Config", "clean", "ShapeDictionary"]
