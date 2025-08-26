"""
Data Generator Package

This package provides tools for generating robot simulation data including:
- Environment generation (pointcloud)
- Rigid body pose generation 
- Reference trajectory planning
"""

# Import key modules for easy access
from . import pointcloud
# from . import pose  # 임시 비활성화 - pose 모듈 정리 중
# from . import reference_planner

__version__ = "0.1.0"
__all__ = ["pointcloud", "pose", "reference_planner"]
