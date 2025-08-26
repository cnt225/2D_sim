"""
Pointcloud module for Box2D robot simulation.

This module provides functionality to:
1. Load point clouds and convert them to Box2D environments
2. Support various file formats (ply, json)

Usage:
    from pointcloud import PointcloudLoader
    
    # Load point cloud and create environment
    loader = PointcloudLoader()
    new_world = loader.load_and_create_world("my_env")
"""

from .utils.pointcloud_loader import PointcloudLoader
from .utils.pointcloud_extractor import PointcloudExtractor

__all__ = ['PointcloudLoader', 'PointcloudExtractor']
