"""echo1 -- a faceted radar-cross-section predictor.

A working reconstruction of what Lockheed's ECHO 1 did in 1975: take a shape
built from flat panels and predict its radar cross section, using physical
optics for the panels and Ufimtsev's physical theory of diffraction for their
edges.

    >>> from echo1 import shapes, monostatic_rcs
    >>> import numpy as np
    >>> body = shapes.hopeless_diamond()
    >>> r = monostatic_rcs(body, 10e9, az_deg=np.arange(0, 360, 1.0))
    >>> print(r.summary("VV"))            # doctest: +SKIP
"""

from .analytic import from_dbsm, to_dbsm, wavelength
from .geometry import EdgeSet, Mesh
from .meshio import load_mesh, save_mesh
from .po import gordon_integral, po_amplitude
from .ptd import fringe_coefficients, ptd_amplitude
from .shadow import edge_illumination, facet_illumination
from .solver import RcsResult, bistatic_rcs, look_vectors, monostatic_rcs
from . import analytic, shapes

__version__ = "0.1.0"

__all__ = [
    "Mesh", "EdgeSet", "shapes", "analytic",
    "monostatic_rcs", "bistatic_rcs", "RcsResult", "look_vectors",
    "load_mesh", "save_mesh",
    "to_dbsm", "from_dbsm", "wavelength",
    "gordon_integral", "po_amplitude", "fringe_coefficients", "ptd_amplitude",
    "facet_illumination", "edge_illumination",
    "__version__",
]
