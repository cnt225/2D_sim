#!/usr/bin/env python3
"""
Concave Box2D Integration
Concave 형태를 Box2D에서 사용할 수 있도록 변환하고 관리하는 모듈

Box2D는 concave 다각형을 직접 지원하지 않으므로 다음 방법들을 사용:
1. 삼각분할 (Triangulation)
2. Convex 분해 (Convex Decomposition) 
3. 복합 Fixture (Multiple Fixtures per Body)
"""

import numpy as np
import math
import json
import os
from typing import List, Tuple, Dict, Optional, Union
from Box2D.b2 import world, staticBody, polygonShape
from dataclasses import dataclass, asdict

try:
    from .concave_shape_generator import ConcaveShapeGenerator, ConcaveShapeType, create_concave_shape
except ImportError:
    from concave_shape_generator import ConcaveShapeGenerator, ConcaveShapeType, create_concave_shape

try:
    from shapely.geometry import Polygon
    from shapely.ops import triangulate
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


@dataclass
class ConcaveObstacleData:
    """Concave 장애물 데이터"""
    name: str
    shape_type: str
    vertices: List[Tuple[float, float]]  # 원본 concave 정점들
    convex_parts: List[List[Tuple[float, float]]]  # convex 분해 결과
    triangles: List[List[Tuple[float, float]]]  # 삼각분할 결과  
    metadata: Dict
    

class ConcaveBox2DAdapter:
    """Concave 형태를 Box2D에 적용하기 위한 어댑터"""
    
    def __init__(self):
        self.shape_library = {}  # 미리 계산된 형태들 저장
        
    def triangulate_polygon(self, vertices: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        """다각형을 삼각형들로 분할"""
        
        if len(vertices) < 3:
            return []
        
        if SHAPELY_AVAILABLE:
            try:
                # Shapely를 사용한 삼각분할
                poly = Polygon(vertices)
                if not poly.is_valid:
                    # 자기 교차 등의 문제 해결 시도
                    poly = poly.buffer(0)
                
                triangles = []
                # Delaunay 삼각분할 시도
                from shapely.ops import triangulate as shapely_triangulate
                triangulation = shapely_triangulate([poly])
                
                for tri in triangulation:
                    if tri.intersects(poly):
                        tri_coords = list(tri.exterior.coords[:-1])  # 마지막 점 제거
                        if len(tri_coords) == 3:
                            triangles.append(tri_coords)
                
                if triangles:
                    return triangles
                    
            except Exception as e:
                print(f"Shapely triangulation failed: {e}, using fallback method")
        
        # 간단한 fan triangulation 사용 (fallback)
        return self._fan_triangulation(vertices)
    
    def _fan_triangulation(self, vertices: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        """Fan triangulation (단순하지만 모든 다각형에 작동하지 않음)"""
        if len(vertices) < 3:
            return []
        
        triangles = []
        
        # 첫 번째 정점을 중심으로 fan 형태로 삼각분할
        for i in range(1, len(vertices) - 1):
            triangle = [vertices[0], vertices[i], vertices[i + 1]]
            triangles.append(triangle)
        
        return triangles
    
    def convex_decomposition(self, vertices: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        """다각형을 convex 부분들로 분해"""
        
        if len(vertices) < 3:
            return []
        
        if self._is_convex(vertices):
            return [vertices]  # 이미 convex면 그대로 반환
        
        if SHAPELY_AVAILABLE:
            try:
                # Shapely를 사용한 방법
                return self._shapely_convex_decomposition(vertices)
            except Exception as e:
                print(f"Shapely convex decomposition failed: {e}, using simple method")
        
        # 간단한 방법: 삼각분할 후 인접한 삼각형들을 병합
        triangles = self.triangulate_polygon(vertices)
        return self._merge_triangles_to_convex(triangles)
    
    def _shapely_convex_decomposition(self, vertices: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        """Shapely를 사용한 convex 분해"""
        poly = Polygon(vertices)
        if not poly.is_valid:
            poly = poly.buffer(0)
        
        # 단순화: 삼각분할 후 작은 convex 영역들로 병합
        triangles = self.triangulate_polygon(vertices)
        return self._merge_triangles_to_convex(triangles)
    
    def _merge_triangles_to_convex(self, triangles: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
        """삼각형들을 convex 영역으로 병합"""
        if not triangles:
            return []
        
        # 간단한 구현: 각 삼각형을 개별 convex로 처리
        # 더 정교한 구현에서는 인접한 삼각형들을 실제로 병합할 수 있음
        convex_parts = []
        
        for triangle in triangles:
            if len(triangle) >= 3:
                convex_parts.append(triangle)
        
        return convex_parts
    
    def _is_convex(self, vertices: List[Tuple[float, float]]) -> bool:
        """다각형이 convex인지 확인"""
        if len(vertices) < 3:
            return False
        
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        n = len(vertices)
        sign = None
        
        for i in range(n):
            o = vertices[i]
            a = vertices[(i + 1) % n]
            b = vertices[(i + 2) % n]
            
            cp = cross_product(o, a, b)
            
            if abs(cp) > 1e-10:  # 거의 0이 아닌 경우만
                if sign is None:
                    sign = cp > 0
                elif (cp > 0) != sign:
                    return False
        
        return True
    
    def create_box2d_body(self, world_obj: world, vertices: List[Tuple[float, float]], 
                         position: Tuple[float, float] = (0, 0), 
                         method: str = 'convex') -> Optional[staticBody]:
        """
        Concave 형태를 Box2D body로 생성
        
        Args:
            world_obj: Box2D world
            vertices: concave 다각형 정점들
            position: body 위치
            method: 'convex', 'triangles', 'auto'
            
        Returns:
            생성된 Box2D body 또는 None
        """
        
        if len(vertices) < 3:
            return None
        
        try:
            # Body 생성
            body = world_obj.CreateStaticBody(position=position)
            
            if method == 'triangles':
                # 삼각분할 사용
                triangles = self.triangulate_polygon(vertices)
                for triangle in triangles:
                    if len(triangle) == 3:
                        # 삼각형을 body에 fixture로 추가
                        body.CreatePolygonFixture(vertices=triangle, density=1.0, friction=0.5)
                        
            elif method == 'convex':
                # Convex 분해 사용
                convex_parts = self.convex_decomposition(vertices)
                for part in convex_parts:
                    if len(part) >= 3:
                        # Box2D 정점 수 제한 확인 (최대 8개)
                        if len(part) > 8:
                            # 너무 많으면 삼각분할
                            sub_triangles = self.triangulate_polygon(part)
                            for tri in sub_triangles:
                                if len(tri) == 3:
                                    body.CreatePolygonFixture(vertices=tri, density=1.0, friction=0.5)
                        else:
                            body.CreatePolygonFixture(vertices=part, density=1.0, friction=0.5)
            
            else:  # auto
                # 자동 선택: convex면 그대로, 아니면 적절한 방법 선택
                if self._is_convex(vertices) and len(vertices) <= 8:
                    body.CreatePolygonFixture(vertices=vertices, density=1.0, friction=0.5)
                else:
                    # 복잡하면 convex 분해 시도
                    convex_parts = self.convex_decomposition(vertices)
                    for part in convex_parts:
                        if len(part) >= 3 and len(part) <= 8:
                            body.CreatePolygonFixture(vertices=part, density=1.0, friction=0.5)
                        elif len(part) > 8:
                            # 너무 복잡하면 삼각분할
                            triangles = self.triangulate_polygon(part)
                            for tri in triangles:
                                if len(tri) == 3:
                                    body.CreatePolygonFixture(vertices=tri, density=1.0, friction=0.5)
            
            # Fixture가 하나도 없으면 body 삭제
            if not body.fixtures:
                world_obj.DestroyBody(body)
                return None
            
            return body
            
        except Exception as e:
            print(f"Failed to create Box2D body: {e}")
            return None
    
    def process_and_store_shape(self, shape_name: str, vertices: List[Tuple[float, float]], 
                               metadata: Dict) -> ConcaveObstacleData:
        """형태를 처리하고 저장 가능한 데이터로 변환"""
        
        # Convex 분해
        convex_parts = self.convex_decomposition(vertices)
        
        # 삼각분할
        triangles = self.triangulate_polygon(vertices)
        
        # 데이터 객체 생성
        obstacle_data = ConcaveObstacleData(
            name=shape_name,
            shape_type=metadata.get('shape_type', 'unknown'),
            vertices=vertices,
            convex_parts=convex_parts,
            triangles=triangles,
            metadata=metadata
        )
        
        # 라이브러리에 저장
        self.shape_library[shape_name] = obstacle_data
        
        return obstacle_data
    
    def save_shape_library(self, filename: str):
        """형태 라이브러리를 JSON 파일로 저장"""
        
        library_data = {}
        
        for name, obstacle_data in self.shape_library.items():
            library_data[name] = {
                'name': obstacle_data.name,
                'shape_type': obstacle_data.shape_type,
                'vertices': obstacle_data.vertices,
                'convex_parts': obstacle_data.convex_parts,
                'triangles': obstacle_data.triangles,
                'metadata': obstacle_data.metadata
            }
        
        with open(filename, 'w') as f:
            json.dump(library_data, f, indent=2)
        
        print(f"Shape library saved to {filename}")
    
    def load_shape_library(self, filename: str):
        """JSON 파일에서 형태 라이브러리 로드"""
        
        if not os.path.exists(filename):
            print(f"Shape library file not found: {filename}")
            return
        
        with open(filename, 'r') as f:
            library_data = json.load(f)
        
        self.shape_library = {}
        
        for name, data in library_data.items():
            obstacle_data = ConcaveObstacleData(
                name=data['name'],
                shape_type=data['shape_type'],
                vertices=data['vertices'],
                convex_parts=data['convex_parts'],
                triangles=data['triangles'],
                metadata=data['metadata']
            )
            self.shape_library[name] = obstacle_data
        
        print(f"Loaded {len(self.shape_library)} shapes from {filename}")
    
    def get_shape(self, shape_name: str) -> Optional[ConcaveObstacleData]:
        """이름으로 형태 데이터 가져오기"""
        return self.shape_library.get(shape_name)
    
    def list_shapes(self) -> List[str]:
        """사용 가능한 형태 이름들 반환"""
        return list(self.shape_library.keys())
    
    def create_body_from_library(self, world_obj: world, shape_name: str,
                                position: Tuple[float, float] = (0, 0),
                                scale: float = 1.0,
                                rotation: float = 0.0,
                                method: str = 'auto') -> Optional[staticBody]:
        """라이브러리에서 형태를 가져와서 Box2D body 생성"""
        
        obstacle_data = self.get_shape(shape_name)
        if obstacle_data is None:
            print(f"Shape '{shape_name}' not found in library")
            return None
        
        # 변형 적용 (스케일, 회전)
        transformed_vertices = self._transform_vertices(
            obstacle_data.vertices, position, scale, rotation
        )
        
        return self.create_box2d_body(world_obj, transformed_vertices, (0, 0), method)
    
    def _transform_vertices(self, vertices: List[Tuple[float, float]], 
                           position: Tuple[float, float], 
                           scale: float, rotation: float) -> List[Tuple[float, float]]:
        """정점들에 변형 적용"""
        
        transformed = []
        px, py = position
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        
        for x, y in vertices:
            # 스케일 적용
            x *= scale
            y *= scale
            
            # 회전 적용
            x_rot = x * cos_r - y * sin_r
            y_rot = x * sin_r + y * cos_r
            
            # 위치 이동
            x_final = x_rot + px
            y_final = y_rot + py
            
            transformed.append((x_final, y_final))
        
        return transformed


def generate_concave_shape_library(output_file: str = "concave_shapes.json", 
                                 shapes_per_type: int = 5):
    """다양한 concave 형태들의 라이브러리 생성"""
    
    print(f"Generating concave shape library with {shapes_per_type} shapes per type...")
    
    adapter = ConcaveBox2DAdapter()
    generator = ConcaveShapeGenerator()
    
    shape_count = 0
    
    # 각 형태 타입별로 여러 변형 생성
    for shape_type in ConcaveShapeType:
        for i in range(shapes_per_type):
            # 랜덤 설정 생성
            config = generator.generate_random_config(shape_type)
            
            # 형태 생성
            vertices = generator.generate_shape(config, (0, 0))
            
            if len(vertices) >= 3:
                # 이름 생성
                shape_name = f"{shape_type.value}_{i:02d}"
                
                # 메타데이터 준비
                metadata = {
                    'shape_type': shape_type.value,
                    'complexity': config.complexity,
                    'size_range': config.size_range,
                    'detail_level': config.detail_level,
                    'rotation': config.rotation,
                    'scale_x': config.scale_x,
                    'scale_y': config.scale_y
                }
                
                # 처리 및 저장
                adapter.process_and_store_shape(shape_name, vertices, metadata)
                shape_count += 1
                
                print(f"Generated {shape_name}: {len(vertices)} vertices")
    
    # 라이브러리 저장
    adapter.save_shape_library(output_file)
    
    print(f"Shape library generation complete: {shape_count} shapes saved to {output_file}")
    
    return adapter


def test_box2d_integration():
    """Box2D 통합 테스트"""
    
    print("Testing Box2D integration...")
    
    # 간단한 L자 형태 생성
    vertices, metadata = create_concave_shape('L_shape', complexity='medium')
    
    print(f"Generated L-shape with {len(vertices)} vertices")
    
    # Box2D world 생성
    from Box2D.b2 import world
    test_world = world(gravity=(0, 0), doSleep=True)
    
    # Adapter로 Box2D body 생성
    adapter = ConcaveBox2DAdapter()
    
    # 다양한 방법으로 테스트
    methods = ['triangles', 'convex', 'auto']
    
    for method in methods:
        body = adapter.create_box2d_body(test_world, vertices, (0, 0), method)
        if body:
            print(f"Successfully created body using {method} method: {len(body.fixtures)} fixtures")
        else:
            print(f"Failed to create body using {method} method")
    
    print("Box2D integration test completed!")


if __name__ == "__main__":
    # 테스트 실행
    print("Testing Concave Box2D Integration...")
    
    # Box2D 통합 테스트
    test_box2d_integration()
    
    print("\n" + "="*50)
    
    # 형태 라이브러리 생성 테스트
    library_file = "test_concave_shapes.json"
    adapter = generate_concave_shape_library(library_file, shapes_per_type=2)
    
    print(f"\nAvailable shapes: {adapter.list_shapes()}")
    
    print("\nTesting completed!") 