"""
cxi-pipeline-ray: Ray-based CPU post-processing pipeline for PeakNet inference results.

This package provides a scalable, distributed post-processing pipeline that:
- Consumes PeakNet inference results from Ray queues (Q2)
- Performs CPU-based peak finding using scipy
- Writes results to CXI files for crystallography analysis
"""

from .version import __version__

__all__ = ["__version__"]
