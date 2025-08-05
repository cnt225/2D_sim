"""
Data Generator Package

This package provides tools for generating robot simulation data including:
- Environment generation (pointcloud)
- Rigid body pose generation 
- Reference trajectory planning
"""

# Import key modules for easy access
from . import pointcloud
from . import pose  
from . import reference_planner

__version__ = "0.1.0"
__all__ = ["pointcloud", "pose", "reference_planner"]
