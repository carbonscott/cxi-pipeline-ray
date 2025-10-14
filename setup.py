"""
Legacy setup.py for backwards compatibility.

Modern installation should use pyproject.toml:
    pip install -e .
"""

from setuptools import setup, find_packages

# Read version from version.py
with open('cxi_pipeline_ray/version.py') as f:
    exec(f.read())

setup(
    name="cxi-pipeline-ray",
    version=__version__,  # noqa: F821
    packages=find_packages(),
    python_requires=">=3.9",
)
