# env.py - Environment Creation Module (SE(3) Rigid Body Simulation)
from Box2D.b2 import world, dynamicBody, staticBody, polygonShape, revoluteJointDef
import math
import os
import numpy as np
from typing import List, Tuple, Optional, Union
try:
    from data_generator.pointcloud import PointcloudLoader
except ImportError:
    try:
        from pointcloud import PointcloudLoader
    except ImportError:
        print("Warning: PointcloudLoader not available. Pointcloud environments will not work.")
        PointcloudLoader = None
from robot_simulation.config_loader import get_rigid_body_config, get_se3_simulation_config

def create_ellipse_vertices(width, height, num_points=16):
    """타원형 버텍스 생성"""
    vertices = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = width * math.cos(angle)
        y = height * math.sin(angle)
        vertices.append((x, y))
    return vertices

def make_world(geometry_id=0, env_file=None, initial_pose=None):
    """
    Create world with SE(3) rigid body robot and environment
    
    Args:
        geometry_id: Rigid body configuration ID (0-3)
        env_file: Pointcloud environment file (.ply). If None, uses static environment
        initial_pose: SE(3) pose [x, y, z, roll, pitch, yaw]. If None, uses default position.
        
    Returns:
        tuple: (world, robot_body, obstacles, robot_config)
    """
    # Get SE(3) rigid body configuration
    try:
        robot_config = get_rigid_body_config(geometry_id)
        if robot_config is None:
            raise ValueError(f"Rigid body configuration not found for ID {geometry_id}")
        print(f"Using SE(3) rigid body {geometry_id}: {robot_config['name']}")
    except (ValueError, TypeError) as e:
        print(f"Error: {e}")
        print("Using default rigid body (ID: 0)")
        robot_config = get_rigid_body_config(0)
        geometry_id = 0
    
    if env_file and env_file != 'static':
        return _make_pointcloud_world(robot_config, env_file, initial_pose, geometry_id)
    else:
        return _make_static_world(robot_config, initial_pose, geometry_id)

def _make_static_world(robot_config, initial_pose=None, geometry_id=0):
    """Create static environment with SE(3) rigid body"""
    W = world(gravity=(0, 0), doSleep=True)

    # Create SE(3) rigid body
    robot_body = _create_rigid_body(W, robot_config, initial_pose)
    obstacles = _create_static_obstacles(W)
    return W, robot_body, obstacles, robot_config

def _create_rigid_body(W, robot_config, initial_pose=None):
    """
    Create SE(3) rigid body robot
    
    Args:
        W: Box2D world
        robot_config: Rigid body configuration
        initial_pose: SE(3) pose [x, y, z, roll, pitch, yaw]. If None, uses default position.
        
    Returns:
        Box2D body representing the rigid body robot
    """
    # Extract configuration
    semi_major_axis = robot_config['semi_major_axis']
    semi_minor_axis = robot_config['semi_minor_axis']
    mass = robot_config['mass']
    
    # Default position if not specified
    if initial_pose is None:
        x, y, yaw = 2.0, 5.0, 0.0  # Default position
    else:
        x, y, z, roll, pitch, yaw = initial_pose
        # Use only x, y, yaw for 2D simulation (ignore z, roll, pitch)
    
    # Create dynamic body
    robot_body = W.CreateDynamicBody(position=(x, y), angle=yaw)
    
    # Create ellipse fixture (Box2D max vertices = 16)
    ellipse_vertices = create_ellipse_vertices(semi_major_axis, semi_minor_axis, num_points=12)
    robot_body.CreatePolygonFixture(vertices=ellipse_vertices, density=mass, friction=0.3, restitution=0.1)
    
    print(f"Created SE(3) rigid body: {robot_config['name']} at ({x:.2f}, {y:.2f}, {yaw:.2f})")
    
    return robot_body

def _create_static_obstacles(W):
    """Create static obstacles in the world"""
    obstacles = []
    
    # Static obstacles for testing
    for pos in [(4,3), (6,1), (8,3)]:
        obs = W.CreateStaticBody(position=pos)
        obs.CreatePolygonFixture(box=(0.3, 0.3))
        obstacles.append(obs)
    
    return obstacles

def _make_pointcloud_world(robot_config, pointcloud_file, initial_pose=None, geometry_id=0):
    """Create world from pointcloud data with SE(3) rigid body"""
    if pointcloud_file is None:
        raise ValueError("pointcloud_file must be specified for pointcloud environment")
    
    # Load pointcloud and create world with obstacles
    if PointcloudLoader is None:
        print("Warning: PointcloudLoader not available, falling back to static environment")
        return _make_static_world(robot_config, initial_pose, geometry_id)
    
    loader = PointcloudLoader()
    
    try:
        # pointcloud_file은 이미 full path (예: "data/pointcloud/circles_only/circles_only.ply")
        # 또는 상대 경로
        if pointcloud_file.startswith('data/pointcloud/'):
            # Remove the data/pointcloud/ prefix since PointcloudLoader expects relative path
            relative_path = pointcloud_file.replace('data/pointcloud/', '')
            # Remove .ply extension if present since PointcloudLoader adds it
            if relative_path.endswith('.ply'):
                relative_path = relative_path[:-4]
        else:
            relative_path = pointcloud_file
            if relative_path.endswith('.ply'):
                relative_path = relative_path[:-4]
        
        W = loader.load_and_create_world(relative_path)
        print(f"Loaded pointcloud environment from {relative_path}")
        
    except Exception as e:
        print(f"Failed to load pointcloud environment: {e}")
        print("Falling back to empty environment")
        W = world(gravity=(0, 0), doSleep=True)
    
    # Create SE(3) rigid body in the pointcloud world
    robot_body = _create_rigid_body(W, robot_config, initial_pose)
    
    # Extract obstacles from world (all static bodies are obstacles)
    obstacles = [body for body in W.bodies if body.type == staticBody]
    
    return W, robot_body, obstacles, robot_config


def list_available_pointclouds(data_dir="data/pointcloud"):
    """List available pointcloud files"""
    if PointcloudLoader is None:
        print("Warning: PointcloudLoader not available")
        return []
    
    loader = PointcloudLoader(data_dir)
    return loader.list_available_pointclouds()