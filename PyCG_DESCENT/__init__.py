from __future__ import absolute_import
from ._pycgd import CGDescent

def get_include():
    """Directory with the CG_DESCENT C sources and the wrapper headers, for
    building extensions that compile CG_DESCENT (e.g. basinvolume). Works for
    installed packages and for in-place builds of a source checkout."""
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    installed = os.path.join(here, "source")
    return installed if os.path.isdir(installed) else os.path.join(os.path.dirname(here), "source")
