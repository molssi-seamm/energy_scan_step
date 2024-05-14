# -*- coding: utf-8 -*-

"""
energy_scan_step
A SEAMM plug-in for calculating energy profiles along coordinates
"""

# Bring up the classes so that they appear to be directly in
# the energy_scan_step package.

from .energy_scan import EnergyScan
from .energy_scan_parameters import EnergyScanParameters
from .energy_scan_step import EnergyScanStep
from .tk_energy_scan import TkEnergyScan

from .metadata import metadata

# Handle versioneer
from ._version import get_versions

__author__ = "Paul Saxe"
__email__ = "psaxe@molssi.org"
versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
