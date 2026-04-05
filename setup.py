"""Shim for editable installs: ``pip install -e .``

All package metadata lives in pyproject.toml.  This file exists solely so
that ``pip install -e .`` works with older pip versions that do not yet
support PEP 660 editable installs via pyproject.toml alone.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
