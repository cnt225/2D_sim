"""
Pointcloud utilities module (legacy).

Core pointcloud processing utilities:
- pointcloud_extractor: Extract pointclouds from Box2D environments  
- pointcloud_loader: Load pointclouds and create Box2D environments
"""

from .pointcloud_extractor import PointcloudExtractor
from .pointcloud_loader import PointcloudLoader

__all__ = ['PointcloudExtractor', 'PointcloudLoader']
