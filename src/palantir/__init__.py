import importlib.metadata

from . import config
from . import core
from . import presults
from . import io
from . import preprocess
from . import utils
from . import plot

__version__ = importlib.metadata.version("palantir")

__all__ = [
    "config",
    "core",
    "presults",
    "io",
    "preprocess",
    "utils",
    "plot",
    "__version__",
]
