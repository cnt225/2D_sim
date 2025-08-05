"""
Robot Simulation Package

This package provides real-time robot simulation capabilities including:
- Core simulation engine
- Control policies  
- Legacy systems
"""

# Import key modules for easy access
from . import core
from . import control
from . import legacy
from .config_loader import *

__version__ = "0.1.0"
__all__ = ["core", "control", "legacy"]
