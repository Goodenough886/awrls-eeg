"""Runtime data locations can change without editing the frozen split manifest."""
from pathlib import Path
import os

def paths(protocol):
    return (Path(os.environ.get('AWRLS_PUBLIC_DATA',protocol['paths']['public'])),
            Path(os.environ.get('AWRLS_INTERNAL_DATA',protocol['paths']['internal'])))
