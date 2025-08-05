#!/usr/bin/env python3
"""
Circle Environment Generator
원형 장애물 전용 환경 생성기 - 10,000개의 다양한 환경 생성을 위한 고급 난이도 시스템

고급 난이도 시스템:
- 장애물 개수: 2-30개 범위에서 세분화
- 장애물 크기: 소/중/대 크기 범위  
- 배치 밀도: 느슨함/보통/조밀함
- 복잡도: 단순/중간/복잡
- 공간 활용도: 중앙집중/분산/가장자리
"""

import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional
from Box2D.b2 import world, staticBody
from enum import Enum
import dataclasses


@dataclasses.dataclass
class EnvironmentConfig:
    """환경 설정 클래스"""
    num_obstacles: int
    obstacle_size_range: Tuple[float, float]  # (min_radius, max_radius)
    density_level: str  # 'sparse', 'medium', 'dense'
    complexity_level: str  # 'simple', 'medium', 'complex'
    spatial_distribution: str  # 'center', 'distributed', 'edge'
    workspace_utilization: float  # 0.0-1.0, 작업공간 활용도
    min_distance_factor: float  # 장애물 간 최소 거리 배수


class CircleEnvironmentGenerator:
    """원형 장애물 전용 환경 생성기"""
    
    def __init__(self, 
                 workspace_bounds: Tuple[float, float, float, float] = (0, 10, 0, 8),
                 robot_base_pos: Tuple[float, float] = (0, 0),
                 robot_max_reach: float = 8.0,
                 seed: Optional[int] = None):
        """
        Args:
            workspace_bounds: (min_x, max_x, min_y, max_y) 작업공간 경계
            robot_base_pos: 로봇 베이스 위치
            robot_max_reach: 로봇 최대 도달 거리
            seed: 랜덤 시드
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
        
        # 난이도 매트릭스 정의
        self._init_difficulty_system()
    
    def _init_difficulty_system(self):
        """고급 난이도 시스템 초기화"""
        
        # 1. 장애물 개수별 난이도 레벨
        self.obstacle_count_levels = {
            'minimal': (2, 4),      # 최소한의 장애물
            'few': (3, 6),          # 적은 장애물
            'light': (5, 8),        # 가벼운 수준
            'moderate': (7, 12),    # 보통 수준
            'busy': (10, 16),       # 바쁜 수준
            'crowded': (14, 20),    # 혼잡한 수준
            'dense': (18, 25),      # 조밀한 수준
            'packed': (22, 30),     # 빽빽한 수준
        }
        
        # 2. 장애물 크기별 카테고리
        self.size_categories = {
            'tiny': (0.1, 0.2),      # 아주 작은
            'small': (0.15, 0.35),   # 작은
            'medium_small': (0.25, 0.45),  # 중간-작은
            'medium': (0.35, 0.65),  # 중간
            'medium_large': (0.5, 0.8),   # 중간-큰
            'large': (0.7, 1.0),     # 큰
            'huge': (0.9, 1.3),      # 아주 큰
            'mixed_small': (0.1, 0.5),     # 작은 것들 혼합
            'mixed_medium': (0.2, 0.8),    # 중간 크기들 혼합
            'mixed_large': (0.4, 1.2),     # 큰 것들 혼합
            'mixed_all': (0.1, 1.0),       # 모든 크기 혼합
        }
        
        # 3. 배치 밀도별 파라미터
        self.density_params = {
            'sparse': {
                'min_distance_factor': 2.0,
                'max_attempts': 100,
                'workspace_utilization': 0.4
            },
            'medium': {
                'min_distance_factor': 1.2,
                'max_attempts': 80,
                'workspace_utilization': 0.6
            },
            'dense': {
                'min_distance_factor': 0.5,
                'max_attempts': 60,
                'workspace_utilization': 0.8
            }
        }
        
        # 4. 복잡도별 파라미터
        self.complexity_params = {
            'simple': {
                'clustering_factor': 0.2,    # 클러스터링 정도
                'size_variation': 0.3,       # 크기 변화량
                'overlap_tolerance': 0.1     # 겹침 허용도
            },
            'medium': {
                'clustering_factor': 0.5,
                'size_variation': 0.5,
                'overlap_tolerance': 0.2
            },
            'complex': {
                'clustering_factor': 0.8,
                'size_variation': 0.8,
                'overlap_tolerance': 0.3
            }
        }
        
        # 5. 공간 활용 패턴
        self.spatial_patterns = {
            'center': 'concentrate_center',
            'distributed': 'uniform_random',
            'edge': 'concentrate_edges',
            'clusters': 'multiple_clusters',
            'line': 'linear_arrangement',
            'arc': 'arc_arrangement'
        }
    
    def generate_predefined_difficulty_config(self, difficulty_name: str) -> EnvironmentConfig:
        """미리 정의된 난이도별 설정 생성"""
        
        predefined_configs = {
            'tutorial': EnvironmentConfig(
                num_obstacles=3,
                obstacle_size_range=(0.2, 0.4),
                density_level='sparse',
                complexity_level='simple',
                spatial_distribution='center',
                workspace_utilization=0.3,
                min_distance_factor=2.0
            ),
            'easy': EnvironmentConfig(
                num_obstacles=random.randint(3, 6),
                obstacle_size_range=random.choice([(0.15, 0.35), (0.25, 0.45)]),
                density_level='sparse',
                complexity_level='simple',
                spatial_distribution=random.choice(['center', 'distributed']),
                workspace_utilization=0.4,
                min_distance_factor=1.8
            ),
            'medium': EnvironmentConfig(
                num_obstacles=random.randint(5, 10),
                obstacle_size_range=random.choice([(0.2, 0.6), (0.3, 0.7)]),
                density_level='medium',
                complexity_level='medium',
                spatial_distribution=random.choice(['distributed', 'clusters']),
                workspace_utilization=0.6,
                min_distance_factor=1.2
            ),
            'hard': EnvironmentConfig(
                num_obstacles=random.randint(8, 16),
                obstacle_size_range=random.choice([(0.1, 0.8), (0.2, 1.0)]),
                density_level='dense',
                complexity_level='complex',
                spatial_distribution=random.choice(['distributed', 'clusters', 'edge']),
                workspace_utilization=0.8,
                min_distance_factor=0.8
            ),
            'expert': EnvironmentConfig(
                num_obstacles=random.randint(12, 25),
                obstacle_size_range=(0.1, 1.2),
                density_level='dense',
                complexity_level='complex',
                spatial_distribution=random.choice(['clusters', 'edge', 'line']),
                workspace_utilization=0.9,
                min_distance_factor=0.5
            )
        }
        
        if difficulty_name not in predefined_configs:
            difficulty_name = 'medium'  # 기본값
        
        return predefined_configs[difficulty_name]
    
    def generate_random_config(self) -> EnvironmentConfig:
        """완전 랜덤한 환경 설정 생성"""
        
        # 랜덤 장애물 개수 (가중치 적용)
        count_weights = [0.1, 0.2, 0.3, 0.25, 0.1, 0.03, 0.015, 0.005]  # minimal ~ packed
        count_level = random.choices(list(self.obstacle_count_levels.keys()), weights=count_weights)[0]
        num_obstacles = random.randint(*self.obstacle_count_levels[count_level])
        
        # 랜덤 크기 범위
        size_category = random.choice(list(self.size_categories.keys()))
        obstacle_size_range = self.size_categories[size_category]
        
        # 랜덤 밀도
        density_level = random.choice(['sparse', 'medium', 'dense'])
        
        # 랜덤 복잡도
        complexity_level = random.choice(['simple', 'medium', 'complex'])
        
        # 랜덤 공간 분포
        spatial_distribution = random.choice(list(self.spatial_patterns.keys()))
        
        # 작업공간 활용도 (장애물 개수와 상관관계)
        base_utilization = min(0.3 + (num_obstacles - 2) * 0.03, 0.9)
        workspace_utilization = base_utilization + random.uniform(-0.1, 0.1)
        workspace_utilization = max(0.2, min(0.95, workspace_utilization))
        
        # 최소 거리 계수 (밀도와 반비례)
        density_to_distance = {'sparse': 2.0, 'medium': 1.2, 'dense': 0.6}
        base_distance_factor = density_to_distance[density_level]
        min_distance_factor = base_distance_factor + random.uniform(-0.3, 0.3)
        min_distance_factor = max(0.1, min(3.0, min_distance_factor))
        
        return EnvironmentConfig(
            num_obstacles=num_obstacles,
            obstacle_size_range=obstacle_size_range,
            density_level=density_level,
            complexity_level=complexity_level,
            spatial_distribution=spatial_distribution,
            workspace_utilization=workspace_utilization,
            min_distance_factor=min_distance_factor
        )
    
    def generate_environment_from_config(self, config: EnvironmentConfig) -> Tuple[world, List, Dict]:
        """설정에 따른 환경 생성"""
        
        print(f"Generating environment with config:")
        print(f"  - Obstacles: {config.num_obstacles}")
        print(f"  - Size range: {config.obstacle_size_range}")
        print(f"  - Density: {config.density_level}")
        print(f"  - Complexity: {config.complexity_level}")
        print(f"  - Distribution: {config.spatial_distribution}")
        
        # 월드 생성
        W = world(gravity=(0, 0), doSleep=True)
        obstacles = []
        
        # 위치 생성 방법 선택
        positions = self._generate_positions_by_pattern(config)
        
        # 장애물 생성
        for i, position in enumerate(positions):
            if i >= config.num_obstacles:
                break
            
            # 크기 결정 (복잡도에 따라 변화량 조정)
            complexity_params = self.complexity_params[config.complexity_level]
            size_variation = complexity_params['size_variation']
            
            base_size = random.uniform(*config.obstacle_size_range)
            variation = random.uniform(-size_variation, size_variation) * base_size
            radius = max(0.05, base_size + variation)
            
            try:
                obstacle = self._create_circle_obstacle(W, position, radius)
                if obstacle:
                    obstacles.append(obstacle)
            except Exception as e:
                print(f"Failed to create obstacle {i}: {e}")
                continue
        
        # 메타데이터 생성
        metadata = self._generate_metadata(config, obstacles)
        
        print(f"Successfully created {len(obstacles)} obstacles")
        return W, obstacles, metadata
    
    def _generate_positions_by_pattern(self, config: EnvironmentConfig) -> List[Tuple[float, float]]:
        """공간 분포 패턴에 따른 위치 생성"""
        
        min_x, max_x, min_y, max_y = self.obstacle_bounds
        pattern = config.spatial_distribution
        
        positions = []
        max_attempts = 1000
        
        if pattern == 'center':
            # 중앙 집중
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            spread_x = (max_x - min_x) * config.workspace_utilization * 0.3
            spread_y = (max_y - min_y) * config.workspace_utilization * 0.3
            
            for _ in range(max_attempts):
                if len(positions) >= config.num_obstacles:
                    break
                
                x = center_x + random.uniform(-spread_x, spread_x)
                y = center_y + random.uniform(-spread_y, spread_y)
                
                if self._is_position_valid(x, y, positions, config):
                    positions.append((x, y))
        
        elif pattern == 'distributed':
            # 균등 분산
            for _ in range(max_attempts):
                if len(positions) >= config.num_obstacles:
                    break
                
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                
                if self._is_position_valid(x, y, positions, config):
                    positions.append((x, y))
        
        elif pattern == 'edge':
            # 가장자리 집중
            edge_margin = min((max_x - min_x), (max_y - min_y)) * 0.3
            
            for _ in range(max_attempts):
                if len(positions) >= config.num_obstacles:
                    break
                
                # 가장자리 근처 위치 생성
                edge_choice = random.choice(['left', 'right', 'top', 'bottom'])
                
                if edge_choice == 'left':
                    x = random.uniform(min_x, min_x + edge_margin)
                    y = random.uniform(min_y, max_y)
                elif edge_choice == 'right':
                    x = random.uniform(max_x - edge_margin, max_x)
                    y = random.uniform(min_y, max_y)
                elif edge_choice == 'top':
                    x = random.uniform(min_x, max_x)
                    y = random.uniform(max_y - edge_margin, max_y)
                else:  # bottom
                    x = random.uniform(min_x, max_x)
                    y = random.uniform(min_y, min_y + edge_margin)
                
                if self._is_position_valid(x, y, positions, config):
                    positions.append((x, y))
        
        elif pattern == 'clusters':
            # 클러스터 배치
            num_clusters = random.randint(2, 4)
            obstacles_per_cluster = config.num_obstacles // num_clusters
            
            # 클러스터 중심 생성
            cluster_centers = []
            for _ in range(num_clusters):
                cx = random.uniform(min_x + 1, max_x - 1)
                cy = random.uniform(min_y + 1, max_y - 1)
                cluster_centers.append((cx, cy))
            
            # 각 클러스터에 장애물 배치
            for i, (cx, cy) in enumerate(cluster_centers):
                cluster_radius = random.uniform(0.8, 2.0)
                target_count = obstacles_per_cluster + (1 if i < config.num_obstacles % num_clusters else 0)
                
                for _ in range(max_attempts // num_clusters):
                    if len([p for p in positions if self._distance(p, (cx, cy)) < cluster_radius]) >= target_count:
                        break
                    
                    angle = random.uniform(0, 2 * math.pi)
                    radius = random.uniform(0, cluster_radius)
                    x = cx + radius * math.cos(angle)
                    y = cy + radius * math.sin(angle)
                    
                    if (min_x <= x <= max_x and min_y <= y <= max_y and 
                        self._is_position_valid(x, y, positions, config)):
                        positions.append((x, y))
        
        # 목표 개수에 못 미치면 균등 분산으로 채움
        while len(positions) < config.num_obstacles:
            for _ in range(100):
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                
                if self._is_position_valid(x, y, positions, config):
                    positions.append((x, y))
                    break
            else:
                break  # 더 이상 배치할 수 없음
        
        return positions[:config.num_obstacles]
    
    def _is_position_valid(self, x: float, y: float, existing_positions: List[Tuple[float, float]], 
                          config: EnvironmentConfig) -> bool:
        """위치가 유효한지 확인"""
        
        # 로봇 베이스와의 거리 확인
        robot_distance = math.sqrt((x - self.robot_base_pos[0])**2 + 
                                 (y - self.robot_base_pos[1])**2)
        if robot_distance < 1.5:  # 로봇 주변 1.5m는 비워둠
            return False
        
        # 다른 장애물과의 거리 확인
        min_distance = config.min_distance_factor * (config.obstacle_size_range[0] + config.obstacle_size_range[1]) / 2
        
        for ex, ey in existing_positions:
            distance = math.sqrt((x - ex)**2 + (y - ey)**2)
            if distance < min_distance:
                return False
        
        return True
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """두 점 사이의 거리"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _create_circle_obstacle(self, W: world, position: Tuple[float, float], radius: float):
        """원형 장애물 생성"""
        x, y = position
        
        # Body 생성
        body = W.CreateStaticBody(position=(x, y))
        
        # 원형 fixture 생성
        body.CreateCircleFixture(radius=radius, density=1.0, friction=0.5)
        
        return body
    
    def _generate_metadata(self, config: EnvironmentConfig, obstacles: List) -> Dict:
        """환경 메타데이터 생성"""
        
        obstacle_info = []
        for i, obstacle in enumerate(obstacles):
            fixture = obstacle.fixtures[0] if obstacle.fixtures else None
            info = {
                'id': i,
                'position': [obstacle.position.x, obstacle.position.y],
                'radius': fixture.shape.radius if fixture else 0.0,
                'type': 'circle'
            }
            obstacle_info.append(info)
        
        return {
            'environment_type': 'circle_only',
            'config': dataclasses.asdict(config),
            'num_obstacles': len(obstacles),
            'obstacles': obstacle_info,
            'workspace_bounds': self.workspace_bounds,
            'robot_base_position': self.robot_base_pos,
            'generation_timestamp': None  # 호출하는 곳에서 설정
        }


def create_circle_environment(difficulty: str = 'medium', 
                            config: Optional[EnvironmentConfig] = None,
                            seed: Optional[int] = None) -> Tuple[world, List, Dict]:
    """
    원형 장애물 환경 생성 헬퍼 함수
    
    Args:
        difficulty: 'tutorial', 'easy', 'medium', 'hard', 'expert', 'random'
        config: 직접 설정 제공 (difficulty 무시)
        seed: 랜덤 시드
    
    Returns:
        (Box2D world, obstacle_list, metadata)
    """
    generator = CircleEnvironmentGenerator(seed=seed)
    
    if config is None:
        if difficulty == 'random':
            config = generator.generate_random_config()
        else:
            config = generator.generate_predefined_difficulty_config(difficulty)
    
    return generator.generate_environment_from_config(config)


if __name__ == "__main__":
    # 테스트 실행
    print("Testing Circle Environment Generator...")
    
    # 다양한 난이도 테스트
    difficulties = ['tutorial', 'easy', 'medium', 'hard', 'expert', 'random']
    
    for difficulty in difficulties:
        print(f"\n=== Testing {difficulty} difficulty ===")
        test_world, test_obstacles, test_metadata = create_circle_environment(
            difficulty=difficulty,
            seed=42
        )
        print(f"Generated {len(test_obstacles)} obstacles")
        print(f"Config: {test_metadata['config']}") 