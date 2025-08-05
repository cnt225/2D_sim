# env.py - Environment Creation Module (모듈 2: 시뮬레이션 실행)
from Box2D.b2 import world, dynamicBody, staticBody, polygonShape, revoluteJointDef
import math
import os
import numpy as np
from typing import List, Tuple, Optional
from pointcloud import PointcloudLoader
from robot_simulation.config_loader import get_geometry_config

def create_ellipse_vertices(width, height, num_points=16):
    """타원형 버텍스 생성"""
    vertices = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = width * math.cos(angle)
        y = height * math.sin(angle)
        vertices.append((x, y))
    return vertices

def make_world(geometry_id=0, env_file=None):
    """
    Create world with robot and environment
    
    Args:
        geometry_id: Robot geometry configuration ID (0-5)
        env_file: Pointcloud environment file (.ply). If None, uses static environment
        
    Returns:
        tuple: (world, links, obstacles)
    """
    # Get robot geometry configuration
    try:
        geometry_config = get_geometry_config(geometry_id)
        print(f"Using geometry {geometry_id}: {geometry_config['name']}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Using default geometry (ID: 0)")
        geometry_config = get_geometry_config(0)
    
    if env_file and env_file != 'static':
        return _make_pointcloud_world(geometry_config, env_file)
    else:
        return _make_static_world(geometry_config)

def _make_static_world(geometry_config):
    """Create static environment with predefined obstacles"""
    W = world(gravity=(0, 0), doSleep=True)

    # 고정 베이스
    base = W.CreateStaticBody(position=(0,0))

    # Get robot configuration
    link_lengths = geometry_config['link_lengths']
    link_widths = geometry_config['link_widths']
    link_shape = geometry_config['link_shape']
    
    links = []
    for i, length in enumerate(link_lengths):
        x = sum(link_lengths[:i]) + length/2
        body = W.CreateDynamicBody(position=(x, 0), angle=0)
        
        if link_shape == "ellipse":
            # 타원형 링크 - geometry config의 width를 기반으로 타원 크기 계산
            ellipse_width = length * 0.5  # 링크 길이에 비례한 타원 폭
            ellipse_height = link_widths[i]  # geometry config의 width를 높이로 사용
            vertices = create_ellipse_vertices(ellipse_width, ellipse_height)
            body.CreatePolygonFixture(vertices=vertices, density=1, friction=0.3)
        else:
            # 기본 사각형 링크
            body.CreatePolygonFixture(box=(length/2, link_widths[i]/2), density=1, friction=0.3)
        
        links.append(body)

    # 관절 연결 (베이스-1, 1-2, 2-3)
    W.CreateJoint(revoluteJointDef(bodyA=base,   bodyB=links[0],
                                   localAnchorA=(0,0),
                                   localAnchorB=(-link_lengths[0]/2,0)))
    for i in range(len(links)-1):
        W.CreateJoint(revoluteJointDef(bodyA=links[i], bodyB=links[i+1],
                                       localAnchorA=(link_lengths[i]/2,0),
                                       localAnchorB=(-link_lengths[i+1]/2,0)))

    # 장애물 (static box)
    obstacles = []
    for pos in [(4,3), (6,1), (8,3)]:
        obs = W.CreateStaticBody(position=pos)
        obs.CreatePolygonFixture(box=(0.3, 0.3))
        obstacles.append(obs)

    return W, links, obstacles


def _make_pointcloud_world(geometry_config, pointcloud_file):
    """Create world from pointcloud data"""
    if pointcloud_file is None:
        raise ValueError("pointcloud_file must be specified for pointcloud environment")
    
    # Load pointcloud and create world with obstacles
    loader = PointcloudLoader()
    
    try:
        # Convert environment name to proper path format: env_name -> env_name/env_name
        env_name = pointcloud_file
        if env_name.endswith('.ply'):
            env_name = env_name[:-4]  # Remove .ply extension if provided
        
        # Use the folder/file format: env_name/env_name
        pointcloud_path = f"{env_name}/{env_name}"
        
        W = loader.load_and_create_world(pointcloud_path)
        print(f"Loaded pointcloud environment from {pointcloud_path}")
        
    except Exception as e:
        print(f"Failed to load pointcloud environment: {e}")
        print("Falling back to empty environment")
        W = world(gravity=(0, 0), doSleep=True)
    
    # Create robot in the pointcloud world
    links = _create_robot_links(W, geometry_config)
    
    # Extract obstacles from world (all static bodies except robot base)
    # Robot base is the first static body created
    obstacles = []
    robot_base = None
    for body in W.bodies:
        if body.type == staticBody:
            if robot_base is None:
                robot_base = body  # First static body is the robot base
            else:
                obstacles.append(body)  # Others are obstacles
    
    return W, links, obstacles


def _create_robot_links(W, geometry_config):
    """Create robot links in given world"""
    # 고정 베이스
    base = W.CreateStaticBody(position=(0,0))

    # Get robot configuration
    link_lengths = geometry_config['link_lengths']
    link_widths = geometry_config['link_widths']
    link_shape = geometry_config['link_shape']
    
    links = []
    for i, length in enumerate(link_lengths):
        x = sum(link_lengths[:i]) + length/2
        body = W.CreateDynamicBody(position=(x, 0), angle=0)
        
        if link_shape == "ellipse":
            # 타원형 링크 - geometry config의 width를 기반으로 타원 크기 계산
            ellipse_width = length * 0.5  # 링크 길이에 비례한 타원 폭
            ellipse_height = link_widths[i]  # geometry config의 width를 높이로 사용
            vertices = create_ellipse_vertices(ellipse_width, ellipse_height)
            body.CreatePolygonFixture(vertices=vertices, density=1, friction=0.3)
        else:
            # 기본 사각형 링크
            body.CreatePolygonFixture(box=(length/2, link_widths[i]/2), density=1, friction=0.3)
        
        links.append(body)

    # 관절 연결 (베이스-1, 1-2, 2-3)
    W.CreateJoint(revoluteJointDef(bodyA=base,   bodyB=links[0],
                                   localAnchorA=(0,0),
                                   localAnchorB=(-link_lengths[0]/2,0)))
    for i in range(len(links)-1):
        W.CreateJoint(revoluteJointDef(bodyA=links[i], bodyB=links[i+1],
                                       localAnchorA=(link_lengths[i]/2,0),
                                       localAnchorB=(-link_lengths[i+1]/2,0)))
    
    return links


def list_available_pointclouds(data_dir="data/pointcloud"):
    """List available pointcloud files"""
    loader = PointcloudLoader(data_dir)
    return loader.list_available_pointclouds()