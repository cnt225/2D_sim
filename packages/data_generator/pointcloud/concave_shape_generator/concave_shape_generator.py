#!/usr/bin/env python3
"""
Concave Shape Generator
다양한 방법으로 concave(오목한) 형태의 장애물을 생성하는 모듈

구현된 방법들:
1. 템플릿 기반: L자, T자, U자, C자, 십자 등
2. Boolean 연산: 다각형 결합/빼기/교집합  
3. 다중 Convex 결합: 여러 convex 형태를 겹쳐서 배치
4. 패턴 기반: 미로, 통로, 복잡한 경로
"""

import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json

try:
    from shapely.geometry import Polygon, Point, MultiPolygon
    from shapely.ops import unary_union, cascaded_union
    from shapely import affinity
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: Shapely not available. Some concave generation methods will be limited.")


class ConcaveShapeType(Enum):
    """Concave 형태 타입"""
    L_SHAPE = "L_shape"
    T_SHAPE = "T_shape" 
    U_SHAPE = "U_shape"
    C_SHAPE = "C_shape"
    CROSS_SHAPE = "cross_shape"
    STAR_SHAPE = "star_shape"
    SPIRAL_SHAPE = "spiral_shape"
    MAZE_SEGMENT = "maze_segment"
    MULTI_CONVEX = "multi_convex"
    BOOLEAN_COMBO = "boolean_combo"


@dataclass
class ConcaveShapeConfig:
    """Concave 형태 설정"""
    shape_type: ConcaveShapeType
    size_range: Tuple[float, float]  # (min_size, max_size)
    complexity: str  # 'simple', 'medium', 'complex'
    rotation: float  # 회전 각도 (라디안)
    scale_x: float = 1.0  # X축 스케일
    scale_y: float = 1.0  # Y축 스케일
    thickness: float = 0.5  # 두께 (L, T, U 형태용)
    detail_level: int = 3  # 디테일 레벨 (1-5)


class ConcaveShapeGenerator:
    """Concave 형태 생성기"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: 랜덤 시드
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 템플릿 정의
        self._init_templates()
    
    def _init_templates(self):
        """기본 템플릿 초기화"""
        
        # L자 형태 템플릿 (정규화된 좌표)
        self.l_template = [
            (0, 0), (0.7, 0), (0.7, 0.3), (0.3, 0.3), (0.3, 1.0), (0, 1.0)
        ]
        
        # T자 형태 템플릿
        self.t_template = [
            (0, 0), (1.0, 0), (1.0, 0.3), (0.65, 0.3), (0.65, 1.0), (0.35, 1.0), (0.35, 0.3), (0, 0.3)
        ]
        
        # U자 형태 템플릿
        self.u_template = [
            (0, 0), (0.3, 0), (0.3, 0.7), (0.7, 0.7), (0.7, 0), (1.0, 0), 
            (1.0, 1.0), (0, 1.0)
        ]
        
        # C자 형태 템플릿 (원호 근사)
        self.c_template = self._generate_c_shape_template()
        
        # 십자 형태 템플릿
        self.cross_template = [
            (0.35, 0), (0.65, 0), (0.65, 0.35), (1.0, 0.35), (1.0, 0.65),
            (0.65, 0.65), (0.65, 1.0), (0.35, 1.0), (0.35, 0.65), (0, 0.65),
            (0, 0.35), (0.35, 0.35)
        ]
    
    def _generate_c_shape_template(self) -> List[Tuple[float, float]]:
        """C자 형태 템플릿 생성 (원호 기반)"""
        outer_radius = 0.5
        inner_radius = 0.25
        gap_angle = math.pi * 0.6  # 108도 갭
        
        points = []
        
        # 외부 호 (반시계방향)
        start_angle = gap_angle / 2
        end_angle = 2 * math.pi - gap_angle / 2
        
        num_points = 20
        for i in range(num_points + 1):
            angle = start_angle + (end_angle - start_angle) * i / num_points
            x = 0.5 + outer_radius * math.cos(angle)
            y = 0.5 + outer_radius * math.sin(angle)
            points.append((x, y))
        
        # 내부 호 (시계방향)
        for i in range(num_points + 1):
            angle = end_angle - (end_angle - start_angle) * i / num_points
            x = 0.5 + inner_radius * math.cos(angle)
            y = 0.5 + inner_radius * math.sin(angle)
            points.append((x, y))
        
        return points
    
    def generate_shape(self, config: ConcaveShapeConfig, 
                      position: Tuple[float, float] = (0, 0)) -> List[Tuple[float, float]]:
        """
        설정에 따른 concave 형태 생성
        
        Args:
            config: 형태 설정
            position: 중심 위치
            
        Returns:
            정점 리스트 (Box2D 호환 형식)
        """
        shape_type = config.shape_type
        
        if shape_type == ConcaveShapeType.L_SHAPE:
            return self._generate_l_shape(config, position)
        elif shape_type == ConcaveShapeType.T_SHAPE:
            return self._generate_t_shape(config, position)
        elif shape_type == ConcaveShapeType.U_SHAPE:
            return self._generate_u_shape(config, position)
        elif shape_type == ConcaveShapeType.C_SHAPE:
            return self._generate_c_shape(config, position)
        elif shape_type == ConcaveShapeType.CROSS_SHAPE:
            return self._generate_cross_shape(config, position)
        elif shape_type == ConcaveShapeType.STAR_SHAPE:
            return self._generate_star_shape(config, position)
        elif shape_type == ConcaveShapeType.SPIRAL_SHAPE:
            return self._generate_spiral_shape(config, position)
        elif shape_type == ConcaveShapeType.MAZE_SEGMENT:
            return self._generate_maze_segment(config, position)
        elif shape_type == ConcaveShapeType.MULTI_CONVEX:
            return self._generate_multi_convex(config, position)
        elif shape_type == ConcaveShapeType.BOOLEAN_COMBO:
            return self._generate_boolean_combo(config, position)
        else:
            # 기본값: L자 형태
            return self._generate_l_shape(config, position)
    
    def _generate_l_shape(self, config: ConcaveShapeConfig, position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """L자 형태 생성"""
        template = self.l_template.copy()
        
        # 복잡도에 따른 변형
        if config.complexity == 'medium':
            # 모서리에 작은 들여쓰기 추가
            template = self._add_indentations(template, 2)
        elif config.complexity == 'complex':
            # 더 많은 들여쓰기와 돌출부 추가
            template = self._add_indentations(template, 4)
            template = self._add_protrusions(template, 2)
        
        return self._transform_template(template, config, position)
    
    def _generate_t_shape(self, config: ConcaveShapeConfig, position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """T자 형태 생성"""
        template = self.t_template.copy()
        
        # 복잡도에 따른 변형
        if config.complexity != 'simple':
            template = self._add_indentations(template, 2 if config.complexity == 'medium' else 4)
        
        return self._transform_template(template, config, position)
    
    def _generate_u_shape(self, config: ConcaveShapeConfig, position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """U자 형태 생성"""
        template = self.u_template.copy()
        
        # 복잡도에 따른 변형
        if config.complexity != 'simple':
            template = self._add_indentations(template, 2 if config.complexity == 'medium' else 3)
        
        return self._transform_template(template, config, position)
    
    def _generate_c_shape(self, config: ConcaveShapeConfig, position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """C자 형태 생성"""
        template = self.c_template.copy()
        
        # 복잡도에 따른 디테일 조정
        if config.complexity == 'simple':
            # 점 수 줄이기
            template = template[::2]  # 절반만 사용
        elif config.complexity == 'complex':
            # 더 부드러운 곡선과 들여쓰기
            template = self._add_indentations(template, 2)
        
        return self._transform_template(template, config, position)
    
    def _generate_cross_shape(self, config: ConcaveShapeConfig, position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """십자 형태 생성"""
        template = self.cross_template.copy()
        
        # 복잡도에 따른 변형
        if config.complexity != 'simple':
            template = self._add_indentations(template, 2 if config.complexity == 'medium' else 4)
        
        return self._transform_template(template, config, position)
    
    def _generate_star_shape(self, config: ConcaveShapeConfig, position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """별 형태 생성"""
        num_points = config.detail_level + 2  # 3-7개 꼭지점
        outer_radius = 0.5
        inner_radius = 0.2
        
        points = []
        for i in range(num_points * 2):
            angle = 2 * math.pi * i / (num_points * 2)
            radius = outer_radius if i % 2 == 0 else inner_radius
            
            # 복잡도에 따른 반지름 변화
            if config.complexity == 'medium':
                radius *= random.uniform(0.8, 1.2)
            elif config.complexity == 'complex':
                radius *= random.uniform(0.6, 1.4)
            
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append((x, y))
        
        return self._transform_template(points, config, position)
    
    def _generate_spiral_shape(self, config: ConcaveShapeConfig, position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """나선 형태 생성"""
        num_turns = 2 + config.detail_level * 0.5
        points_per_turn = 20
        total_points = int(num_turns * points_per_turn)
        
        outer_radius = 0.4
        inner_radius = 0.1
        width = 0.05  # 나선 두께
        
        points = []
        
        # 외부 나선
        for i in range(total_points):
            t = i / points_per_turn
            angle = 2 * math.pi * t
            radius = outer_radius - (outer_radius - inner_radius) * t / num_turns
            
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append((x, y))
        
        # 내부 나선 (역방향)
        for i in range(total_points):
            t = (total_points - 1 - i) / points_per_turn
            angle = 2 * math.pi * t
            radius = outer_radius - (outer_radius - inner_radius) * t / num_turns - width
            radius = max(radius, inner_radius)
            
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append((x, y))
        
        return self._transform_template(points, config, position)
    
    def _generate_maze_segment(self, config: ConcaveShapeConfig, position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """미로 세그먼트 생성"""
        # 격자 기반 미로 패턴
        grid_size = config.detail_level + 2  # 3-7 격자
        cell_size = 1.0 / grid_size
        wall_thickness = cell_size * 0.3
        
        # 랜덤한 미로 패턴 생성
        pattern = self._generate_maze_pattern(grid_size, config.complexity)
        
        # 패턴을 다각형으로 변환
        points = self._maze_pattern_to_polygon(pattern, cell_size, wall_thickness)
        
        return self._transform_template(points, config, position)
    
    def _generate_multi_convex(self, config: ConcaveShapeConfig, position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """다중 convex 형태 결합"""
        if not SHAPELY_AVAILABLE:
            # Shapely가 없으면 단순한 L자 형태로 대체
            return self._generate_l_shape(config, position)
        
        # 2-4개의 convex 다각형 생성
        num_shapes = 2 + config.detail_level // 2
        shapes = []
        
        for i in range(num_shapes):
            # 각 convex 형태 생성
            center_x = random.uniform(0.2, 0.8)
            center_y = random.uniform(0.2, 0.8)
            radius = random.uniform(0.15, 0.35)
            num_vertices = random.randint(3, 6)
            
            vertices = []
            for j in range(num_vertices):
                angle = 2 * math.pi * j / num_vertices + random.uniform(-0.5, 0.5)
                r = radius * random.uniform(0.7, 1.3)
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)
                vertices.append((x, y))
            
            if len(vertices) >= 3:
                try:
                    poly = Polygon(vertices)
                    if poly.is_valid:
                        shapes.append(poly)
                except:
                    continue
        
        if not shapes:
            return self._generate_l_shape(config, position)
        
        # 모든 형태를 결합
        try:
            combined = unary_union(shapes)
            if hasattr(combined, 'exterior'):
                points = list(combined.exterior.coords[:-1])  # 마지막 점 제거 (중복)
            else:
                # MultiPolygon인 경우 가장 큰 부분 사용
                largest = max(combined.geoms, key=lambda x: x.area)
                points = list(largest.exterior.coords[:-1])
            
            return self._transform_template(points, config, position)
        except:
            return self._generate_l_shape(config, position)
    
    def _generate_boolean_combo(self, config: ConcaveShapeConfig, position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Boolean 연산으로 복잡한 형태 생성"""
        if not SHAPELY_AVAILABLE:
            return self._generate_cross_shape(config, position)
        
        try:
            # 기본 사각형
            base_rect = Polygon([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)])
            
            # 빼낼 형태들 생성
            cutouts = []
            
            # 복잡도에 따른 cutout 개수
            num_cutouts = 1 + config.detail_level
            
            for i in range(num_cutouts):
                cut_x = random.uniform(0.2, 0.8)
                cut_y = random.uniform(0.2, 0.8)
                cut_w = random.uniform(0.1, 0.3)
                cut_h = random.uniform(0.1, 0.3)
                
                if random.choice([True, False]):
                    # 사각형 cutout
                    cutout = Polygon([
                        (cut_x - cut_w/2, cut_y - cut_h/2),
                        (cut_x + cut_w/2, cut_y - cut_h/2),
                        (cut_x + cut_w/2, cut_y + cut_h/2),
                        (cut_x - cut_w/2, cut_y + cut_h/2)
                    ])
                else:
                    # 원형 cutout (다각형 근사)
                    center = Point(cut_x, cut_y)
                    cutout = center.buffer(min(cut_w, cut_h) / 2)
                
                cutouts.append(cutout)
            
            # Boolean 연산 수행
            result = base_rect
            for cutout in cutouts:
                result = result.difference(cutout)
            
            if hasattr(result, 'exterior') and result.is_valid:
                points = list(result.exterior.coords[:-1])
                return self._transform_template(points, config, position)
            else:
                return self._generate_cross_shape(config, position)
                
        except:
            return self._generate_cross_shape(config, position)
    
    def _add_indentations(self, template: List[Tuple[float, float]], count: int) -> List[Tuple[float, float]]:
        """템플릿에 들여쓰기 추가"""
        if len(template) < 4:
            return template
        
        new_points = []
        added_count = 0
        
        for i in range(len(template)):
            new_points.append(template[i])
            
            if added_count < count and random.random() < 0.3:
                # 다음 점과의 중점에서 들여쓰기 생성
                next_i = (i + 1) % len(template)
                mid_x = (template[i][0] + template[next_i][0]) / 2
                mid_y = (template[i][1] + template[next_i][1]) / 2
                
                # 중심 방향으로 들여쓰기
                center_x, center_y = 0.5, 0.5
                indent_factor = 0.05
                
                dx = center_x - mid_x
                dy = center_y - mid_y
                length = math.sqrt(dx*dx + dy*dy)
                
                if length > 0:
                    dx /= length
                    dy /= length
                    
                    indent_x = mid_x + dx * indent_factor
                    indent_y = mid_y + dy * indent_factor
                    new_points.append((indent_x, indent_y))
                    added_count += 1
        
        return new_points
    
    def _add_protrusions(self, template: List[Tuple[float, float]], count: int) -> List[Tuple[float, float]]:
        """템플릿에 돌출부 추가"""
        if len(template) < 4:
            return template
        
        new_points = []
        added_count = 0
        
        for i in range(len(template)):
            new_points.append(template[i])
            
            if added_count < count and random.random() < 0.2:
                # 돌출부 생성
                next_i = (i + 1) % len(template)
                mid_x = (template[i][0] + template[next_i][0]) / 2
                mid_y = (template[i][1] + template[next_i][1]) / 2
                
                # 중심 반대 방향으로 돌출
                center_x, center_y = 0.5, 0.5
                protrusion_factor = 0.08
                
                dx = mid_x - center_x
                dy = mid_y - center_y
                length = math.sqrt(dx*dx + dy*dy)
                
                if length > 0:
                    dx /= length
                    dy /= length
                    
                    prot_x = mid_x + dx * protrusion_factor
                    prot_y = mid_y + dy * protrusion_factor
                    new_points.append((prot_x, prot_y))
                    added_count += 1
        
        return new_points
    
    def _generate_maze_pattern(self, size: int, complexity: str) -> List[List[bool]]:
        """미로 패턴 생성 (True = 벽, False = 빈공간)"""
        pattern = [[True for _ in range(size)] for _ in range(size)]
        
        # 간단한 패턴 생성
        if complexity == 'simple':
            # 단순한 L자 통로
            for i in range(size // 2):
                pattern[i][0] = False
                pattern[size - 1][i] = False
        elif complexity == 'medium':
            # 좀 더 복잡한 패턴
            for i in range(size):
                if i % 2 == 0:
                    pattern[i][i // 2] = False
                else:
                    pattern[i][size - 1 - i // 2] = False
        else:  # complex
            # 랜덤 패턴
            for i in range(size):
                for j in range(size):
                    if random.random() < 0.4:
                        pattern[i][j] = False
        
        return pattern
    
    def _maze_pattern_to_polygon(self, pattern: List[List[bool]], 
                               cell_size: float, wall_thickness: float) -> List[Tuple[float, float]]:
        """미로 패턴을 다각형으로 변환"""
        # 단순화: 외곽선만 추출
        points = []
        size = len(pattern)
        
        # 외곽 경계 생성
        for i in range(size + 1):
            points.append((i * cell_size, 0))
        for i in range(1, size + 1):
            points.append((size * cell_size, i * cell_size))
        for i in range(size - 1, -1, -1):
            points.append((i * cell_size, size * cell_size))
        for i in range(size - 1, 0, -1):
            points.append((0, i * cell_size))
        
        return points
    
    def _transform_template(self, template: List[Tuple[float, float]], 
                          config: ConcaveShapeConfig, 
                          position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """템플릿을 설정에 따라 변환"""
        transformed = []
        
        size = random.uniform(*config.size_range)
        cx, cy = position
        
        for x, y in template:
            # 중심을 (0.5, 0.5)에서 (0, 0)으로 이동
            x -= 0.5
            y -= 0.5
            
            # 스케일 적용
            x *= config.scale_x * size
            y *= config.scale_y * size
            
            # 회전 적용
            cos_r = math.cos(config.rotation)
            sin_r = math.sin(config.rotation)
            x_rot = x * cos_r - y * sin_r
            y_rot = x * sin_r + y * cos_r
            
            # 위치 이동
            x_final = x_rot + cx
            y_final = y_rot + cy
            
            transformed.append((x_final, y_final))
        
        return transformed
    
    def generate_random_config(self, shape_type: Optional[ConcaveShapeType] = None) -> ConcaveShapeConfig:
        """랜덤한 concave 형태 설정 생성"""
        
        if shape_type is None:
            shape_type = random.choice(list(ConcaveShapeType))
        
        size_range = random.choice([
            (0.5, 1.0),   # 작은 크기
            (0.8, 1.5),   # 중간 크기
            (1.2, 2.0),   # 큰 크기
            (0.3, 2.5)    # 다양한 크기
        ])
        
        complexity = random.choice(['simple', 'medium', 'complex'])
        rotation = random.uniform(0, 2 * math.pi)
        scale_x = random.uniform(0.7, 1.3)
        scale_y = random.uniform(0.7, 1.3)
        thickness = random.uniform(0.3, 0.7)
        detail_level = random.randint(1, 5)
        
        return ConcaveShapeConfig(
            shape_type=shape_type,
            size_range=size_range,
            complexity=complexity,
            rotation=rotation,
            scale_x=scale_x,
            scale_y=scale_y,
            thickness=thickness,
            detail_level=detail_level
        )
    
    def save_shape_as_svg(self, vertices: List[Tuple[float, float]], 
                         filename: str, size: Tuple[int, int] = (400, 400)):
        """형태를 SVG 파일로 저장 (시각화용)"""
        width, height = size
        
        # 좌표 정규화
        if not vertices:
            return
        
        min_x = min(x for x, y in vertices)
        max_x = max(x for x, y in vertices)
        min_y = min(y for x, y in vertices)
        max_y = max(y for x, y in vertices)
        
        scale_x = width * 0.8 / (max_x - min_x) if max_x > min_x else 1
        scale_y = height * 0.8 / (max_y - min_y) if max_y > min_y else 1
        scale = min(scale_x, scale_y)
        
        # SVG 생성
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
<polygon points="'''
        
        for x, y in vertices:
            svg_x = (x - min_x) * scale + width * 0.1
            svg_y = height - ((y - min_y) * scale + height * 0.1)  # Y축 뒤집기
            svg_content += f"{svg_x:.2f},{svg_y:.2f} "
        
        svg_content += '''" fill="lightblue" stroke="darkblue" stroke-width="2"/>
</svg>'''
        
        with open(filename, 'w') as f:
            f.write(svg_content)


def create_concave_shape(shape_type: Union[str, ConcaveShapeType] = None,
                        complexity: str = 'medium',
                        size_range: Tuple[float, float] = (1.0, 2.0),
                        position: Tuple[float, float] = (0, 0),
                        seed: Optional[int] = None) -> Tuple[List[Tuple[float, float]], Dict]:
    """
    Concave 형태 생성 헬퍼 함수
    
    Args:
        shape_type: 형태 타입 ('L_shape', 'T_shape', 등) 또는 None (랜덤)
        complexity: 'simple', 'medium', 'complex'
        size_range: 크기 범위
        position: 중심 위치
        seed: 랜덤 시드
        
    Returns:
        (vertices, metadata): 정점 리스트와 메타데이터
    """
    generator = ConcaveShapeGenerator(seed=seed)
    
    # shape_type 변환
    if isinstance(shape_type, str):
        try:
            shape_type = ConcaveShapeType(shape_type)
        except ValueError:
            shape_type = None
    
    if shape_type is None:
        config = generator.generate_random_config()
    else:
        config = ConcaveShapeConfig(
            shape_type=shape_type,
            size_range=size_range,
            complexity=complexity,
            rotation=random.uniform(0, 2 * math.pi),
            detail_level=random.randint(2, 4)
        )
    
    vertices = generator.generate_shape(config, position)
    
    metadata = {
        'shape_type': config.shape_type.value,
        'complexity': config.complexity,
        'size_range': config.size_range,
        'position': position,
        'rotation': config.rotation,
        'detail_level': config.detail_level,
        'num_vertices': len(vertices)
    }
    
    return vertices, metadata


if __name__ == "__main__":
    # 테스트 실행
    print("Testing Concave Shape Generator...")
    
    generator = ConcaveShapeGenerator(seed=42)
    
    # 모든 형태 타입 테스트
    for shape_type in ConcaveShapeType:
        print(f"\nTesting {shape_type.value}...")
        
        config = ConcaveShapeConfig(
            shape_type=shape_type,
            size_range=(1.0, 2.0),
            complexity='medium',
            rotation=0,
            detail_level=3
        )
        
        vertices = generator.generate_shape(config, (5, 5))
        print(f"Generated {len(vertices)} vertices")
        
        # SVG 저장 (테스트용)
        svg_filename = f"test_{shape_type.value}.svg"
        generator.save_shape_as_svg(vertices, svg_filename)
        print(f"Saved to {svg_filename}")
    
    print("\nTesting completed!") 