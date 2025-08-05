#!/usr/bin/env python3
"""
Concave Environment Generator
Concave 형태의 장애물들을 다양한 크기와 방향으로 환경에 배치하는 통합 시스템

특징:
- 기존 원형/사각형 장애물과 concave 장애물 혼합 배치
- 다양한 크기와 회전 적용
- Box2D 호환 삼각분할/convex 분해
- JSON 형태 라이브러리 관리
"""

import numpy as np
import math
import random
import os
from typing import List, Tuple, Dict, Optional, Union
from Box2D.b2 import world, staticBody
from dataclasses import dataclass

try:
    from .concave_shape_generator import ConcaveShapeGenerator, ConcaveShapeType, create_concave_shape
    from .concave_box2d_integration import ConcaveBox2DAdapter, generate_concave_shape_library
    from .random_environment_generator import RandomEnvironmentGenerator
except ImportError:
    from concave_shape_generator import ConcaveShapeGenerator, ConcaveShapeType, create_concave_shape
    from concave_box2d_integration import ConcaveBox2DAdapter, generate_concave_shape_library
    from random_environment_generator import RandomEnvironmentGenerator


@dataclass
class ConcaveEnvironmentConfig:
    """Concave 환경 설정"""
    num_total_obstacles: int
    concave_ratio: float  # 전체 장애물 중 concave 비율 (0.0-1.0)
    concave_types: List[str]  # 사용할 concave 타입들
    size_range: Tuple[float, float]
    allow_rotation: bool = True
    allow_scaling: bool = True
    use_mixed_obstacles: bool = True  # 기존 장애물과 혼합 여부
    workspace_bounds: Tuple[float, float, float, float] = (0, 10, 0, 8)
    

class ConcaveEnvironmentGenerator:
    """Concave 장애물 환경 생성기"""
    
    def __init__(self, 
                 concave_library_file: str = "concave_shapes.json",
                 workspace_bounds: Tuple[float, float, float, float] = (0, 10, 0, 8),
                 robot_base_pos: Tuple[float, float] = (0, 0),
                 seed: Optional[int] = None):
        """
        Args:
            concave_library_file: concave 형태 라이브러리 JSON 파일
            workspace_bounds: 작업공간 경계
            robot_base_pos: 로봇 베이스 위치
            seed: 랜덤 시드
        """
        self.workspace_bounds = workspace_bounds
        self.robot_base_pos = robot_base_pos
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Concave 어댑터 초기화
        self.concave_adapter = ConcaveBox2DAdapter()
        
        # 라이브러리 로드 (없으면 생성)
        self.concave_library_file = concave_library_file
        self._ensure_concave_library()
        
        # 기존 환경 생성기
        self.basic_generator = RandomEnvironmentGenerator(
            workspace_bounds=workspace_bounds,
            robot_base_pos=robot_base_pos,
            seed=seed
        )
        
        # 장애물 생성 범위 설정
        min_x, max_x, min_y, max_y = workspace_bounds
        self.obstacle_bounds = (
            min_x + 1.0,  # 로봇 주변 여유공간
            max_x - 1.0,
            min_y + 1.0,
            max_y - 1.0
        )
    
    def _ensure_concave_library(self):
        """Concave 라이브러리 파일이 없으면 생성"""
        
        if not os.path.exists(self.concave_library_file):
            print(f"Concave library not found, generating: {self.concave_library_file}")
            generate_concave_shape_library(self.concave_library_file, shapes_per_type=3)
        
        # 라이브러리 로드
        self.concave_adapter.load_shape_library(self.concave_library_file)
        
        if not self.concave_adapter.list_shapes():
            print("Warning: No concave shapes available, generating basic library")
            generate_concave_shape_library(self.concave_library_file, shapes_per_type=2)
            self.concave_adapter.load_shape_library(self.concave_library_file)
    
    def generate_mixed_environment(self, config: ConcaveEnvironmentConfig) -> Tuple[world, List, Dict]:
        """
        Concave와 기존 장애물을 혼합한 환경 생성
        
        Args:
            config: 환경 설정
            
        Returns:
            (Box2D world, obstacle_list, metadata)
        """
        
        print(f"Generating mixed environment:")
        print(f"  - Total obstacles: {config.num_total_obstacles}")
        print(f"  - Concave ratio: {config.concave_ratio}")
        print(f"  - Concave types: {config.concave_types}")
        
        # World 생성
        W = world(gravity=(0, 0), doSleep=True)
        obstacles = []
        
        # 장애물 개수 분배
        num_concave = int(config.num_total_obstacles * config.concave_ratio)
        num_basic = config.num_total_obstacles - num_concave
        
        print(f"  - Concave obstacles: {num_concave}")
        print(f"  - Basic obstacles: {num_basic}")
        
        # 위치 계획
        positions = self._plan_obstacle_positions(config.num_total_obstacles)
        
        obstacle_metadata = []
        
        # Concave 장애물 생성
        for i in range(num_concave):
            if i < len(positions):
                position = positions[i]
                
                try:
                    obstacle, metadata = self._create_concave_obstacle(W, position, config)
                    if obstacle:
                        obstacles.append(obstacle)
                        obstacle_metadata.append(metadata)
                except Exception as e:
                    print(f"Failed to create concave obstacle {i}: {e}")
                    continue
        
        # 기존 타입 장애물 생성 (혼합 모드인 경우)
        if config.use_mixed_obstacles and num_basic > 0:
            used_positions = positions[:num_concave]
            
            for i in range(num_basic):
                pos_idx = num_concave + i
                if pos_idx < len(positions):
                    position = positions[pos_idx]
                    
                    try:
                        obstacle, metadata = self._create_basic_obstacle(W, position, config)
                        if obstacle:
                            obstacles.append(obstacle)
                            obstacle_metadata.append(metadata)
                    except Exception as e:
                        print(f"Failed to create basic obstacle {i}: {e}")
                        continue
        
        # 환경 메타데이터 생성
        environment_metadata = {
            'environment_type': 'mixed_concave',
            'total_obstacles': len(obstacles),
            'concave_count': num_concave,
            'basic_count': len(obstacles) - num_concave,
            'concave_ratio': config.concave_ratio,
            'concave_types_used': config.concave_types,
            'configuration': {
                'size_range': config.size_range,
                'allow_rotation': config.allow_rotation,
                'allow_scaling': config.allow_scaling,
                'use_mixed_obstacles': config.use_mixed_obstacles
            },
            'obstacles': obstacle_metadata,
            'workspace_bounds': config.workspace_bounds
        }
        
        print(f"Successfully created {len(obstacles)} obstacles")
        return W, obstacles, environment_metadata
    
    def generate_concave_only_environment(self, config: ConcaveEnvironmentConfig) -> Tuple[world, List, Dict]:
        """Concave 장애물만으로 구성된 환경 생성"""
        
        config.use_mixed_obstacles = False
        config.concave_ratio = 1.0
        
        return self.generate_mixed_environment(config)
    
    def _plan_obstacle_positions(self, num_obstacles: int) -> List[Tuple[float, float]]:
        """장애물 위치 계획"""
        
        min_x, max_x, min_y, max_y = self.obstacle_bounds
        positions = []
        max_attempts = 1000
        min_distance = 2.0  # 장애물 간 최소 거리
        
        for _ in range(max_attempts):
            if len(positions) >= num_obstacles:
                break
            
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            
            # 로봇과의 거리 확인
            robot_distance = math.sqrt((x - self.robot_base_pos[0])**2 + 
                                     (y - self.robot_base_pos[1])**2)
            if robot_distance < 2.0:
                continue
            
            # 다른 장애물과의 거리 확인
            valid = True
            for px, py in positions:
                distance = math.sqrt((x - px)**2 + (y - py)**2)
                if distance < min_distance:
                    valid = False
                    break
            
            if valid:
                positions.append((x, y))
        
        return positions
    
    def _create_concave_obstacle(self, world_obj: world, position: Tuple[float, float], 
                                config: ConcaveEnvironmentConfig) -> Tuple[Optional[staticBody], Dict]:
        """Concave 장애물 생성"""
        
        # 사용 가능한 형태에서 랜덤 선택
        available_shapes = [name for name in self.concave_adapter.list_shapes() 
                          if any(shape_type in name for shape_type in config.concave_types)]
        
        if not available_shapes:
            # 사용 가능한 형태가 없으면 모든 형태에서 선택
            available_shapes = self.concave_adapter.list_shapes()
        
        if not available_shapes:
            return None, {}
        
        shape_name = random.choice(available_shapes)
        
        # 크기와 회전 결정
        scale = random.uniform(*config.size_range) if config.allow_scaling else 1.0
        rotation = random.uniform(0, 2 * math.pi) if config.allow_rotation else 0.0
        
        # Body 생성
        body = self.concave_adapter.create_body_from_library(
            world_obj, shape_name, position, scale, rotation, method='auto'
        )
        
        # 메타데이터
        shape_data = self.concave_adapter.get_shape(shape_name)
        metadata = {
            'type': 'concave',
            'shape_name': shape_name,
            'shape_type': shape_data.shape_type if shape_data else 'unknown',
            'position': position,
            'scale': scale,
            'rotation': rotation,
            'num_fixtures': len(body.fixtures) if body else 0
        }
        
        return body, metadata
    
    def _create_basic_obstacle(self, world_obj: world, position: Tuple[float, float],
                              config: ConcaveEnvironmentConfig) -> Tuple[Optional[staticBody], Dict]:
        """기존 타입 장애물 생성"""
        
        obstacle_type = random.choice(['rectangle', 'circle'])
        
        if obstacle_type == 'rectangle':
            return self.basic_generator._create_random_rectangle(world_obj, position), {
                'type': 'rectangle',
                'position': position
            }
        else:
            return self.basic_generator._create_random_circle(world_obj, position), {
                'type': 'circle',
                'position': position
            }
    
    def generate_predefined_config(self, preset: str) -> ConcaveEnvironmentConfig:
        """미리 정의된 설정 생성"""
        
        available_types = list(set(name.split('_')[0] + '_' + name.split('_')[1] 
                                 for name in self.concave_adapter.list_shapes()))
        
        presets = {
            'simple_concave': ConcaveEnvironmentConfig(
                num_total_obstacles=5,
                concave_ratio=0.6,
                concave_types=['L_shape', 'T_shape'],
                size_range=(0.8, 1.5),
                allow_rotation=True,
                allow_scaling=True,
                use_mixed_obstacles=True,
                workspace_bounds=self.workspace_bounds
            ),
            'complex_concave': ConcaveEnvironmentConfig(
                num_total_obstacles=12,
                concave_ratio=0.8,
                concave_types=available_types[:5] if len(available_types) >= 5 else available_types,
                size_range=(0.5, 2.0),
                allow_rotation=True,
                allow_scaling=True,
                use_mixed_obstacles=True,
                workspace_bounds=self.workspace_bounds
            ),
            'concave_only': ConcaveEnvironmentConfig(
                num_total_obstacles=8,
                concave_ratio=1.0,
                concave_types=available_types,
                size_range=(0.7, 1.8),
                allow_rotation=True,
                allow_scaling=True,
                use_mixed_obstacles=False,
                workspace_bounds=self.workspace_bounds
            ),
            'maze_like': ConcaveEnvironmentConfig(
                num_total_obstacles=15,
                concave_ratio=0.9,
                concave_types=['L_shape', 'T_shape', 'U_shape', 'cross_shape'],
                size_range=(1.0, 2.5),
                allow_rotation=True,
                allow_scaling=False,
                use_mixed_obstacles=True,
                workspace_bounds=self.workspace_bounds
            )
        }
        
        return presets.get(preset, presets['simple_concave'])
    
    def list_available_shapes(self) -> List[str]:
        """사용 가능한 concave 형태 목록"""
        return self.concave_adapter.list_shapes()
    
    def get_shape_types(self) -> List[str]:
        """사용 가능한 형태 타입들"""
        return list(set(name.split('_')[0] + '_' + name.split('_')[1] 
                       for name in self.concave_adapter.list_shapes()))


def create_concave_environment(preset: str = 'simple_concave',
                             concave_library_file: str = "concave_shapes.json",
                             seed: Optional[int] = None) -> Tuple[world, List, Dict]:
    """
    Concave 환경 생성 헬퍼 함수
    
    Args:
        preset: 'simple_concave', 'complex_concave', 'concave_only', 'maze_like'
        concave_library_file: concave 형태 라이브러리 파일
        seed: 랜덤 시드
        
    Returns:
        (Box2D world, obstacle_list, metadata)
    """
    
    generator = ConcaveEnvironmentGenerator(
        concave_library_file=concave_library_file,
        seed=seed
    )
    
    config = generator.generate_predefined_config(preset)
    
    return generator.generate_mixed_environment(config)


if __name__ == "__main__":
    # 테스트 실행
    print("Testing Concave Environment Generator...")
    
    # 라이브러리 생성 (테스트용)
    test_library = "test_concave_env_shapes.json"
    if not os.path.exists(test_library):
        generate_concave_shape_library(test_library, shapes_per_type=2)
    
    # 다양한 preset 테스트
    presets = ['simple_concave', 'complex_concave', 'concave_only', 'maze_like']
    
    for preset in presets:
        print(f"\n=== Testing {preset} preset ===")
        
        try:
            test_world, test_obstacles, test_metadata = create_concave_environment(
                preset=preset,
                concave_library_file=test_library,
                seed=42
            )
            
            print(f"Generated environment with {len(test_obstacles)} obstacles")
            print(f"Concave count: {test_metadata.get('concave_count', 0)}")
            print(f"Basic count: {test_metadata.get('basic_count', 0)}")
            
        except Exception as e:
            print(f"Failed to generate {preset}: {e}")
    
    print("\nTesting completed!") 