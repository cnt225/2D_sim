"""
SVG to PLY Converter
SVG 파일의 polygon 정보를 추출하여 PLY 포인트클라우드로 변환하는 모듈
"""
import xml.etree.ElementTree as ET
import numpy as np
import os
from typing import List, Tuple, Optional
import re

class SVGToPLYConverter:
    """SVG 파일을 PLY 포인트클라우드로 변환하는 클래스"""
    
    def __init__(self):
        pass
    
    def parse_svg_polygon(self, svg_filepath: str) -> List[Tuple[float, float]]:
        """
        SVG 파일에서 polygon의 좌표를 추출
        
        Args:
            svg_filepath: SVG 파일 경로
            
        Returns:
            polygon 좌표 리스트 [(x, y), ...]
        """
        tree = ET.parse(svg_filepath)
        root = tree.getroot()
        
        # SVG namespace 처리
        namespace = {'svg': 'http://www.w3.org/2000/svg'}
        
        vertices = []
        
        # polygon 태그 찾기
        for polygon in root.findall('.//polygon') + root.findall('.//svg:polygon', namespace):
            points_str = polygon.get('points', '')
            if points_str:
                # "x1,y1 x2,y2 x3,y3" 형태의 문자열 파싱
                coords = re.findall(r'([0-9.-]+),([0-9.-]+)', points_str)
                for x_str, y_str in coords:
                    x, y = float(x_str), float(y_str)
                    vertices.append((x, y))
        
        return vertices
    
    def vertices_to_pointcloud(self, vertices: List[Tuple[float, float]], 
                             resolution: float = 0.1) -> np.ndarray:
        """
        Polygon vertices를 dense pointcloud로 변환
        
        Args:
            vertices: polygon 좌표 리스트
            resolution: 포인트 간격
            
        Returns:
            pointcloud array (N, 2)
        """
        if len(vertices) < 3:
            return np.array([])
        
        points = []
        
        # 1. Polygon 경계를 따라 포인트 생성
        for i in range(len(vertices)):
            start = vertices[i]
            end = vertices[(i + 1) % len(vertices)]
            
            # 두 점 사이의 거리
            dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            num_points = max(2, int(dist / resolution))
            
            # 선형 보간으로 포인트 생성
            for j in range(num_points):
                t = j / (num_points - 1) if num_points > 1 else 0
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                points.append([x, y])
        
        # 2. 내부 영역을 채우는 포인트 생성 (간단한 grid 방식)
        if len(vertices) >= 3:
            min_x = min(v[0] for v in vertices)
            max_x = max(v[0] for v in vertices)
            min_y = min(v[1] for v in vertices)
            max_y = max(v[1] for v in vertices)
            
            # Grid 포인트 생성
            x_range = np.arange(min_x, max_x + resolution, resolution)
            y_range = np.arange(min_y, max_y + resolution, resolution)
            
            for x in x_range:
                for y in y_range:
                    if self._point_in_polygon(x, y, vertices):
                        points.append([x, y])
        
        return np.array(points) if points else np.array([])
    
    def _point_in_polygon(self, x: float, y: float, 
                         vertices: List[Tuple[float, float]]) -> bool:
        """Ray casting algorithm으로 점이 polygon 내부에 있는지 확인"""
        n = len(vertices)
        inside = False
        
        p1x, p1y = vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def save_as_ply(self, points: np.ndarray, output_filepath: str):
        """
        포인트클라우드를 PLY 파일로 저장
        
        Args:
            points: 포인트클라우드 (N, 2)
            output_filepath: 출력 PLY 파일 경로
        """
        with open(output_filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            for point in points:
                f.write(f"{point[0]} {point[1]} 0.0\n")
    
    def convert_svg_to_ply(self, svg_filepath: str, ply_filepath: str, 
                          resolution: float = 0.1) -> bool:
        """
        SVG 파일을 PLY 포인트클라우드로 변환
        
        Args:
            svg_filepath: 입력 SVG 파일 경로
            ply_filepath: 출력 PLY 파일 경로
            resolution: 포인트클라우드 해상도
            
        Returns:
            변환 성공 여부
        """
        try:
            # 1. SVG에서 polygon 좌표 추출
            vertices = self.parse_svg_polygon(svg_filepath)
            if not vertices:
                print(f"No polygon found in {svg_filepath}")
                return False
            
            # 2. 좌표를 시뮬레이션 좌표계로 변환 (SVG는 보통 픽셀 좌표)
            # SVG 400x400 → 시뮬레이션 10x10 정도로 스케일링
            scale_factor = 10.0 / 400.0
            vertices = [(x * scale_factor, (400 - y) * scale_factor) for x, y in vertices]
            
            # 3. 포인트클라우드 생성
            points = self.vertices_to_pointcloud(vertices, resolution)
            if len(points) == 0:
                print(f"Failed to generate pointcloud from {svg_filepath}")
                return False
            
            # 4. PLY 파일로 저장
            self.save_as_ply(points, ply_filepath)
            print(f"Converted {svg_filepath} → {ply_filepath} ({len(points)} points)")
            return True
            
        except Exception as e:
            print(f"Error converting {svg_filepath}: {e}")
            return False

def main():
    """SVG 파일들을 PLY로 일괄 변환"""
    converter = SVGToPLYConverter()
    
    # 현재 디렉토리의 모든 SVG 파일 찾기
    svg_files = [f for f in os.listdir('.') if f.endswith('.svg')]
    
    # 출력 디렉토리 생성
    output_dir = "data/pointcloud"
    os.makedirs(output_dir, exist_ok=True)
    
    for svg_file in svg_files:
        # PLY 파일명 생성
        base_name = os.path.splitext(svg_file)[0]
        ply_file = os.path.join(output_dir, f"{base_name}.ply")
        
        # 변환 실행
        success = converter.convert_svg_to_ply(svg_file, ply_file)
        if success:
            print(f"✅ {svg_file} → {ply_file}")
        else:
            print(f"❌ Failed to convert {svg_file}")

if __name__ == "__main__":
    main()
