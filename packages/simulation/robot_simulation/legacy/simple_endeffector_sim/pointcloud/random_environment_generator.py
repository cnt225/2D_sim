#!/usr/bin/env python3
"""
Random Environment Generator
로봇팔 검증을 위한 다양한 랜덤 환경 생성

다양한 장애물 타입:
- 사각형 (회전 가능)
- 원형
- 다각형 (3-8각형)
- 곡선형 (스플라인 기반)
"""

import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional
from Box2D.b2 import world, staticBody, polygonShape, circleShape
from scipy.interpolate import splprep, splev


class RandomEnvironmentGenerator:
    """랜덤 환경 생성 클래스"""
    
    def __init__(self, workspace_bounds: Tuple[float, float, float, float] = (0, 10, 0, 8),
                 robot_base_pos: Tuple[float, float] = (0, 0),
                 robot_max_reach: float = 8.0,
                 seed: Optional[int] = None):
        """
        Args:
            workspace_bounds: (min_x, max_x, min_y, max_y) 작업공간 경계
            robot_base_pos: 로봇 베이스 위치
            robot_max_reach: 로봇 최대 도달 거리
            seed: 랜덤 시드 (재현 가능한 환경 생성용)
        """
        self.workspace_bounds = workspace_bounds
        self.robot_base_pos = robot_base_pos
        self.robot_max_reach = robot_max_reach
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 장애물 생성 범위 설정 (로봇 주변 여유공간 확보)
        min_x, max_x, min_y, max_y = workspace_bounds
        self.obstacle_bounds = (
            min_x + 1.0,  # 로봇 주변 1m 여유공간
            max_x - 0.5,
            min_y + 0.5,
            max_y - 0.5
        )
    
    def generate_random_environment(self, 
                                  num_obstacles: int = None,
                                  obstacle_types: List[str] = None,
                                  difficulty: str = 'medium') -> Tuple[world, List]:
        """
        랜덤 환경 생성
        
        Args:
            num_obstacles: 장애물 개수 (None이면 난이도에 따라 자동 설정)
            obstacle_types: 장애물 타입 리스트 ['rectangle', 'circle', 'polygon', 'curve']
            difficulty: 'easy', 'medium', 'hard' 중 하나
            
        Returns:
            (Box2D world, obstacle_list)
        """
        # 기본값 설정
        if obstacle_types is None:
            obstacle_types = ['rectangle', 'circle', 'polygon', 'curve']
        
        if num_obstacles is None:
            difficulty_settings = {
                'easy': (3, 5),      # 3-5개 장애물
                'medium': (5, 8),    # 5-8개 장애물 
                'hard': (8, 12)      # 8-12개 장애물
            }
            min_obs, max_obs = difficulty_settings.get(difficulty, (5, 8))
            num_obstacles = random.randint(min_obs, max_obs)
        
        # 월드 생성
        W = world(gravity=(0, 0), doSleep=True)
        obstacles = []
        
        print(f"Generating {num_obstacles} obstacles with difficulty: {difficulty}")
        print(f"Available obstacle types: {obstacle_types}")
        
        # 장애물 위치 겹침 방지를 위한 리스트
        occupied_positions = []
        
        for i in range(num_obstacles):
            # 랜덤 장애물 타입 선택
            obstacle_type = random.choice(obstacle_types)
            
            # 겹치지 않는 위치 찾기
            position = self._find_free_position(occupied_positions)
            if position is None:
                print(f"Could not find free position for obstacle {i+1}, skipping...")
                continue
            
            # 장애물 생성
            try:
                if obstacle_type == 'rectangle':
                    obstacle = self._create_random_rectangle(W, position)
                elif obstacle_type == 'circle':
                    obstacle = self._create_random_circle(W, position)
                elif obstacle_type == 'polygon':
                    obstacle = self._create_random_polygon(W, position)
                elif obstacle_type == 'curve':
                    obstacle = self._create_random_curve(W, position)
                else:
                    print(f"Unknown obstacle type: {obstacle_type}, using rectangle")
                    obstacle = self._create_random_rectangle(W, position)
                
                if obstacle:
                    obstacles.append(obstacle)
                    occupied_positions.append(position)
                    print(f"Created {obstacle_type} obstacle at {position}")
                
            except Exception as e:
                print(f"Failed to create {obstacle_type} obstacle: {e}")
                continue
        
        print(f"Successfully created {len(obstacles)} obstacles")
        return W, obstacles
    
    def _find_free_position(self, occupied_positions: List[Tuple[float, float]], 
                           max_attempts: int = 50) -> Optional[Tuple[float, float]]:
        """겹치지 않는 위치 찾기"""
        min_x, max_x, min_y, max_y = self.obstacle_bounds
        min_distance = 1.5  # 최소 거리
        
        for _ in range(max_attempts):
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            position = (x, y)
            
            # 로봇 베이스로부터 최소 거리 확인
            robot_distance = math.sqrt((x - self.robot_base_pos[0])**2 + 
                                     (y - self.robot_base_pos[1])**2)
            if robot_distance < 2.0:  # 로봇 주변 2m는 비워둠
                continue
            
            # 다른 장애물과 최소 거리 확인
            if all(math.sqrt((x - ox)**2 + (y - oy)**2) > min_distance 
                   for ox, oy in occupied_positions):
                return position
        
        return None
    
    def _create_random_rectangle(self, W: world, position: Tuple[float, float]):
        """랜덤 사각형 장애물 생성"""
        x, y = position
        
        # 랜덤 크기
        width = random.uniform(0.3, 1.2)
        height = random.uniform(0.3, 1.0)
        
        # 랜덤 회전 각도
        angle = random.uniform(0, 2 * math.pi)
        
        # Body 생성
        body = W.CreateStaticBody(position=(x, y), angle=angle)
        
        # 사각형 fixture 생성
        body.CreatePolygonFixture(box=(width/2, height/2), density=1.0, friction=0.5)
        
        return body
    
    def _create_random_circle(self, W: world, position: Tuple[float, float]):
        """랜덤 원형 장애물 생성"""
        x, y = position
        
        # 랜덤 반지름
        radius = random.uniform(0.2, 0.8)
        
        # Body 생성
        body = W.CreateStaticBody(position=(x, y))
        
        # 원형 fixture 생성
        body.CreateCircleFixture(radius=radius, density=1.0, friction=0.5)
        
        return body
    
    def _create_random_polygon(self, W: world, position: Tuple[float, float]):
        """랜덤 다각형 장애물 생성"""
        x, y = position
        
        # 랜덤 변의 수 (3-8각형)
        num_sides = random.randint(3, 8)
        
        # 랜덤 크기
        radius = random.uniform(0.3, 0.8)
        
        # 다각형 정점 생성
        vertices = []
        for i in range(num_sides):
            angle = 2 * math.pi * i / num_sides
            # 약간의 불규칙성 추가
            r = radius * random.uniform(0.8, 1.2)
            vx = r * math.cos(angle)
            vy = r * math.sin(angle)
            vertices.append((vx, vy))
        
        # 랜덤 회전
        rotation_angle = random.uniform(0, 2 * math.pi)
        cos_a, sin_a = math.cos(rotation_angle), math.sin(rotation_angle)
        
        rotated_vertices = []
        for vx, vy in vertices:
            rx = vx * cos_a - vy * sin_a
            ry = vx * sin_a + vy * cos_a
            rotated_vertices.append((rx, ry))
        
        # Body 생성
        body = W.CreateStaticBody(position=(x, y))
        
        # 다각형 fixture 생성
        body.CreatePolygonFixture(vertices=rotated_vertices, density=1.0, friction=0.5)
        
        return body
    
    def _create_random_curve(self, W: world, position: Tuple[float, float]):
        """랜덤 곡선형 장애물 생성 (스플라인 기반)"""
        x, y = position
        
        # 제어점 개수
        num_points = random.randint(4, 7)
        
        # 랜덤 제어점 생성
        base_radius = random.uniform(0.4, 0.9)
        control_points = []
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            # 불규칙한 반지름으로 곡선 형태 생성
            r = base_radius * random.uniform(0.6, 1.4)
            px = r * math.cos(angle)
            py = r * math.sin(angle)
            control_points.append([px, py])
        
        # 닫힌 곡선 만들기
        control_points.append(control_points[0])
        control_points = np.array(control_points)
        
        try:
            # 스플라인 곡선 생성
            tck, u = splprep([control_points[:, 0], control_points[:, 1]], s=0, per=1)
            
            # 곡선을 다각형 근사
            num_curve_points = 16
            u_new = np.linspace(0, 1, num_curve_points, endpoint=False)
            curve_x, curve_y = splev(u_new, tck)
            
            # 정점 리스트 생성
            vertices = [(float(curve_x[i]), float(curve_y[i])) for i in range(len(curve_x))]
            
            # 너무 작은 다각형 방지
            if len(vertices) < 3:
                return self._create_random_circle(W, position)  # fallback to circle
            
            # Box2D 정점 수 제한 (최대 8개)
            if len(vertices) > 8:
                # 균등하게 샘플링
                indices = np.linspace(0, len(vertices)-1, 8, dtype=int)
                vertices = [vertices[i] for i in indices]
            
            # Body 생성
            body = W.CreateStaticBody(position=(x, y))
            
            # 다각형 fixture 생성
            body.CreatePolygonFixture(vertices=vertices, density=1.0, friction=0.5)
            
            return body
            
        except Exception as e:
            print(f"Failed to create curve obstacle, falling back to circle: {e}")
            return self._create_random_circle(W, position)
    
    def generate_environment_metadata(self, obstacles: List, difficulty: str) -> Dict:
        """생성된 환경의 메타데이터 생성"""
        obstacle_info = []
        
        for i, obstacle in enumerate(obstacles):
            info = {
                'id': i,
                'position': [obstacle.position.x, obstacle.position.y],
                'angle': obstacle.angle,
                'type': 'unknown'  # fixture에서 타입 추론 가능
            }
            
            # Fixture 정보 추가
            if obstacle.fixtures:
                fixture = obstacle.fixtures[0]
                if hasattr(fixture.shape, 'radius'):
                    info['type'] = 'circle'
                    info['radius'] = fixture.shape.radius
                elif hasattr(fixture.shape, 'vertices'):
                    info['type'] = 'polygon'
                    info['vertex_count'] = len(fixture.shape.vertices)
            
            obstacle_info.append(info)
        
        return {
            'environment_type': 'random',
            'difficulty': difficulty,
            'num_obstacles': len(obstacles),
            'obstacles': obstacle_info,
            'workspace_bounds': self.workspace_bounds,
            'robot_base_position': self.robot_base_pos,
            'generation_timestamp': None  # 호출하는 곳에서 설정
        }


# 편의 함수들
def create_random_environment(difficulty: str = 'medium', 
                            obstacle_types: List[str] = None,
                            seed: Optional[int] = None) -> Tuple[world, List, Dict]:
    """
    랜덤 환경 생성 편의 함수
    
    Returns:
        (world, obstacles, metadata)
    """
    generator = RandomEnvironmentGenerator(seed=seed)
    world_obj, obstacles = generator.generate_random_environment(
        obstacle_types=obstacle_types, 
        difficulty=difficulty
    )
    
    metadata = generator.generate_environment_metadata(obstacles, difficulty)
    
    return world_obj, obstacles, metadata


if __name__ == "__main__":
    # 테스트 코드
    print("Testing Random Environment Generator...")
    
    # 다양한 난이도 테스트
    for difficulty in ['easy', 'medium', 'hard']:
        print(f"\n=== Testing {difficulty} difficulty ===")
        world_obj, obstacles, metadata = create_random_environment(
            difficulty=difficulty, 
            seed=42  # 재현 가능한 결과를 위한 고정 시드
        )
        
        print(f"Generated {len(obstacles)} obstacles")
        print(f"Metadata: {metadata}") 