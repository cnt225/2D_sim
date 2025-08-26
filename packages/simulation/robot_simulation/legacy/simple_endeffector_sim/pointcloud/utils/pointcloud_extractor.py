#!/usr/bin/env python3
"""
Pointcloud Extractor
Box2D 환경에서 포인트클라우드를 추출하여 PLY 파일로 저장하는 모듈
"""

import numpy as np
import os
import json
import datetime
from typing import List, Tuple, Optional, Dict
from Box2D.b2 import world as Box2D_world, staticBody, circleShape
import matplotlib.pyplot as plt


class PointcloudExtractor:
    """Box2D 환경에서 포인트클라우드를 추출하는 클래스"""
    
    def __init__(self, resolution: float = 0.05, noise_level: float = 0.01, 
                 data_dir: str = "data/pointcloud"):
        """
        Args:
            resolution: 포인트클라우드 해상도 (미터 단위)
            noise_level: 센서 노이즈 레벨
            data_dir: 데이터 저장 디렉토리
        """
        self.resolution = resolution
        self.noise_level = noise_level
        self.data_dir = data_dir
        
        # 데이터 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
    
    def extract_from_world(self, world: Box2D_world, 
                          workspace_bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Box2D 월드에서 포인트클라우드 추출
        
        Args:
            world: Box2D 월드 객체
            workspace_bounds: (min_x, max_x, min_y, max_y) 작업공간 경계
            
        Returns:
            points: (N, 2) 포인트클라우드 배열
        """
        min_x, max_x, min_y, max_y = workspace_bounds
        
        # 그리드 생성
        x_coords = np.arange(min_x, max_x + self.resolution, self.resolution)
        y_coords = np.arange(min_y, max_y + self.resolution, self.resolution)
        
        points = []
        
        print(f"Scanning {len(x_coords)}x{len(y_coords)} grid points...")
        
        # 진행률 표시
        total_points = len(x_coords) * len(y_coords)
        processed = 0
        
        for i, x in enumerate(x_coords):
            if i % max(1, len(x_coords) // 10) == 0:
                progress = processed / total_points * 100
                print(f"Progress: {progress:.1f}%")
            
            for y in y_coords:
                # 각 그리드 포인트에서 장애물과의 충돌 검사
                if self._is_point_in_obstacle(world, x, y):
                    # 노이즈 추가
                    noisy_x = x + np.random.normal(0, self.noise_level)
                    noisy_y = y + np.random.normal(0, self.noise_level)
                    points.append([noisy_x, noisy_y])
                
                processed += 1
        
        print(f"Extracted {len(points)} points from {total_points} grid points")
        return np.array(points)
    
    def _is_point_in_obstacle(self, world: Box2D_world, x: float, y: float) -> bool:
        """점이 장애물 내부에 있는지 확인"""
        for body in world.bodies:
            if body.type == staticBody:
                # 각 fixture 확인
                for fixture in body.fixtures:
                    if self._point_in_fixture(fixture, body, x, y):
                        return True
        return False
    
    def _point_in_fixture(self, fixture, body, x: float, y: float) -> bool:
        """점이 특정 fixture 내부에 있는지 확인"""
        # 월드 좌표를 바디 로컬 좌표로 변환
        local_point = body.GetLocalPoint((x, y))
        
        if hasattr(fixture.shape, 'radius'):
            # 원형 장애물
            center = fixture.shape.pos
            radius = fixture.shape.radius
            distance = np.sqrt((local_point.x - center.x)**2 + (local_point.y - center.y)**2)
            return distance <= radius
        
        elif hasattr(fixture.shape, 'vertices'):
            # 다각형 장애물 - 점이 다각형 내부에 있는지 ray casting 알고리즘 사용
            vertices = fixture.shape.vertices
            return self._point_in_polygon(local_point.x, local_point.y, vertices)
        
        return False
    
    def _point_in_polygon(self, x: float, y: float, vertices) -> bool:
        """Ray casting 알고리즘으로 점이 다각형 내부에 있는지 확인"""
        n = len(vertices)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = vertices[i].x, vertices[i].y
            xj, yj = vertices[j].x, vertices[j].y
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def save_pointcloud(self, points: np.ndarray, filename: str, 
                       metadata: Optional[Dict] = None) -> str:
        """
        포인트클라우드를 PLY 파일로 저장
        
        Args:
            points: (N, 2) 포인트클라우드 배열
            filename: 파일명 (확장자 제외)
            metadata: 메타데이터 (JSON으로 별도 저장)
            
        Returns:
            저장된 PLY 파일 경로
        """
        # 파일 경로 설정
        ply_path = os.path.join(self.data_dir, f"{filename}.ply")
        
        # PLY 파일 저장
        with open(ply_path, 'w') as f:
            # PLY 헤더
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            # 포인트 데이터
            for point in points:
                f.write(f"{point[0]:.6f} {point[1]:.6f} 0.000000\n")
        
        # 메타데이터 저장
        if metadata is not None:
            metadata['generation_timestamp'] = datetime.datetime.now().isoformat()
            meta_path = os.path.join(self.data_dir, f"{filename}_meta.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return ply_path
    
    def visualize_pointcloud(self, points: np.ndarray, title: str = "Pointcloud") -> None:
        """포인트클라우드 시각화"""
        plt.figure(figsize=(12, 8))
        plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.6, c='blue')
        plt.title(title)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def extract_and_save(self, world: Box2D_world, filename: str,
                        workspace_bounds: Tuple[float, float, float, float],
                        metadata: Optional[Dict] = None) -> Tuple[np.ndarray, str]:
        """
        포인트클라우드 추출 및 저장을 한 번에 수행
        
        Returns:
            (points, filepath): 추출된 포인트와 저장된 파일 경로
        """
        points = self.extract_from_world(world, workspace_bounds)
        filepath = self.save_pointcloud(points, filename, metadata)
        return points, filepath


if __name__ == "__main__":
    # 테스트 코드
    print("Testing PointcloudExtractor...")
    
    # 간단한 테스트 월드 생성
    from Box2D.b2 import world, staticBody
    
    test_world = world(gravity=(0, 0), doSleep=True)
    
    # 테스트 장애물 추가
    obs1 = test_world.CreateStaticBody(position=(5, 5))
    obs1.CreateCircleFixture(radius=1.0, density=1.0)
    
    obs2 = test_world.CreateStaticBody(position=(8, 3))
    obs2.CreatePolygonFixture(box=(0.5, 0.5), density=1.0)
    
    # 포인트클라우드 추출
    extractor = PointcloudExtractor(resolution=0.1, noise_level=0.02)
    points, filepath = extractor.extract_and_save(
        test_world, 
        "test_extraction",
        workspace_bounds=(0, 10, 0, 8)
    )
    
    print(f"Test completed: {len(points)} points saved to {filepath}") 