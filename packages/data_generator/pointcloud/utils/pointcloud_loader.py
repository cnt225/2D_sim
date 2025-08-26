"""
Pointcloud Loader Module
포인트클라우드 파일을 로드하고 Box2D 환경으로 변환하는 모듈

저장된 포인트클라우드 데이터를 기반으로 Box2D 시뮬레이션 환경 재생성
"""
import numpy as np
import os
import json
import Box2D
from typing import List, Tuple, Optional, Union
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import warnings


class PointcloudLoader:
    """포인트클라우드 파일을 로드하고 Box2D 환경으로 변환하는 클래스"""
    
    def __init__(self, data_dir: str = "data/pointcloud"):
        """
        Args:
            data_dir: 포인트클라우드 데이터 디렉토리
        """
        self.data_dir = data_dir
        self.world = None
    
    def load_pointcloud(self, filename: str) -> Tuple[np.ndarray, Optional[dict]]:
        """
        PLY 포인트클라우드 파일 로드
        
        Args:
            filename: 파일명 (확장자 포함 또는 제외)
            
        Returns:
            points: (N, 2) numpy array
            metadata: 메타데이터 (있는 경우)
        """
        # 확장자가 이미 있는지 확인
        if filename.endswith('.ply'):
            base_filename = filename[:-4]
            filepath = os.path.join(self.data_dir, filename)
        else:
            base_filename = filename
            filepath = os.path.join(self.data_dir, f"{filename}.ply")
        
        points = self._load_ply(filepath)
        
        # 메타데이터 로드 (있는 경우)
        metadata = None
        meta_filepath = os.path.join(self.data_dir, f"{base_filename}_meta.json")
        if os.path.exists(meta_filepath):
            with open(meta_filepath, 'r') as f:
                metadata = json.load(f)
        
        return points, metadata
    
    def _load_ply(self, filepath: str) -> np.ndarray:
        """PLY 파일 로드"""
        points = []
        
        with open(filepath, 'r') as f:
            # 헤더 스킵
            line = f.readline()
            while line and not line.startswith('end_header'):
                line = f.readline()
            
            # 포인트 데이터 읽기
            for line in f:
                if line.strip():
                    coords = line.split()
                    x, y = float(coords[0]), float(coords[1])
                    points.append([x, y])
        
        return np.array(points)
    
    def create_world_from_pointcloud(self, points: np.ndarray, 
                                   clustering_eps: float = 0.3,
                                   min_samples: int = 5,
                                   obstacle_type: str = 'polygon') -> Box2D.b2World:
        """
        포인트클라우드에서 Box2D 월드 생성
        
        Args:
            points: (N, 2) 포인트클라우드 데이터
            clustering_eps: DBSCAN 클러스터링 거리 임계값
            min_samples: DBSCAN 최소 샘플 수
            obstacle_type: 'polygon', 'circle', 'auto' 중 하나
            
        Returns:
            Box2D world 객체
        """
        # 새 월드 생성
        world = Box2D.b2World(gravity=(0, 0), doSleep=True)
        
        # 포인트 클러스터링으로 개별 장애물 식별
        clusters = self._cluster_points(points, clustering_eps, min_samples)
        
        # 각 클러스터를 장애물로 변환
        for i, cluster_points in enumerate(clusters):
            if len(cluster_points) < 3:  # 최소 3개 포인트 필요
                print(f"Cluster {i}: Skipping - only {len(cluster_points)} points")
                continue
                
            try:
                print(f"\nCluster {i}: Processing {len(cluster_points)} points")
                
                if obstacle_type == 'polygon':
                    print(f"Cluster {i}: Creating polygon (forced)")
                    self._create_polygon_obstacle(world, cluster_points)
                elif obstacle_type == 'circle':
                    print(f"Cluster {i}: Creating circle (forced)")
                    self._create_circle_obstacle(world, cluster_points)
                elif obstacle_type == 'auto':
                    # 단순하고 효과적인 형태 분석
                    center = np.mean(cluster_points, axis=0)
                    distances = np.linalg.norm(cluster_points - center, axis=1)
                    
                    # 1. 거리 변동성 (원형일수록 변동성이 낮음)
                    std_dist = np.std(distances)
                    mean_dist = np.mean(distances)
                    circularity = std_dist / (mean_dist + 1e-6)
                    
                    # 2. Convex Hull 정점 수 (원형일수록 많은 정점)
                    try:
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(cluster_points)
                        hull_vertices = len(hull.vertices)
                    except:
                        hull_vertices = 4  # 기본값
                    
                    # 3. 클러스터 크기 (작은 것은 원형으로)
                    cluster_size = len(cluster_points)
                    
                    # 단순한 규칙 기반 판정
                    is_circle = False
                    reason = ""
                    
                    # 매우 원형적인 특성 (변동성이 매우 낮고 정점이 많음)
                    if circularity < 0.05 and hull_vertices > 15:
                        is_circle = True
                        reason = f"very_circular(circ={circularity:.3f}, hull_v={hull_vertices})"
                    
                    # 작고 둥근 클러스터
                    elif circularity < 0.1 and cluster_size < 40:
                        is_circle = True
                        reason = f"small_round(circ={circularity:.3f}, size={cluster_size})"
                    
                    # 기본적으로는 다각형으로 (더 정확한 재구성)
                    else:
                        is_circle = False
                        reason = f"polygon_default(circ={circularity:.3f}, hull_v={hull_vertices}, size={cluster_size})"
                    
                    if is_circle:
                        print(f"Cluster {i}: Creating circle (auto, {reason})")
                        self._create_circle_obstacle(world, cluster_points)
                    else:
                        print(f"Cluster {i}: Creating polygon (auto, {reason})")
                        self._create_polygon_obstacle(world, cluster_points)
                        
            except Exception as e:
                warnings.warn(f"Failed to create obstacle from cluster {i}: {e}")
                continue
        
        self.world = world
        return world
    
    def _cluster_points(self, points: np.ndarray, eps: float, min_samples: int) -> List[np.ndarray]:
        """DBSCAN을 사용한 포인트 클러스터링"""
        if len(points) == 0:
            return []
        
        # DBSCAN 클러스터링
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_
        
        # 각 클러스터별로 포인트 그룹화
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # 노이즈 포인트 제외
                continue
            
            cluster_mask = (labels == label)
            cluster_points = points[cluster_mask]
            clusters.append(cluster_points)
        
        return clusters
    
    def _create_polygon_obstacle(self, world: Box2D.b2World, points: np.ndarray):
        """포인트 클러스터에서 다각형 장애물 생성"""
        try:
            # Convex Hull로 외곽선 계산
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            # Box2D 정점 형식으로 변환
            vertices = []
            for point in hull_points:
                vertices.append((float(point[0]), float(point[1])))
            
            print(f"Original polygon: {len(vertices)} vertices from {len(points)} points")
            
            # 정점 수 제한 (Box2D는 최대 8개)
            if len(vertices) > 8:
                # 균등하게 샘플링
                indices = np.linspace(0, len(vertices)-1, 8, dtype=int)
                vertices = [vertices[i] for i in indices]
                print(f"Reduced to {len(vertices)} vertices for Box2D")
            
            # 정점이 3개 미만이면 스킵
            if len(vertices) < 3:
                print(f"Skipping polygon with only {len(vertices)} vertices")
                return
            
            # 반시계방향으로 정렬 (Box2D 요구사항)
            vertices = self._ensure_ccw(vertices)
            
            # 중심점과 크기 계산
            center = np.mean(vertices, axis=0)
            # 다각형 크기 확인 (최대 거리)
            max_distance = 0
            for vertex in vertices:
                dist = np.sqrt((vertex[0] - center[0])**2 + (vertex[1] - center[1])**2)
                max_distance = max(max_distance, dist)
            
            print(f"Creating polygon: center=({center[0]:.2f}, {center[1]:.2f}), {len(vertices)} vertices, max_size={max_distance:.2f}")
            
            # 다각형이 너무 작으면 스킵
            if max_distance < 0.1:
                print(f"Skipping polygon with max_size={max_distance:.3f} (too small)")
                return
            
            # Box2D body 생성
            bodyDef = Box2D.b2BodyDef()
            bodyDef.type = Box2D.b2_staticBody
            body = world.CreateBody(bodyDef)
            
            # 다각형 shape 생성
            shape = Box2D.b2PolygonShape()
            shape.vertices = vertices
            
            # Fixture 생성
            fixtureDef = Box2D.b2FixtureDef()
            fixtureDef.shape = shape
            fixtureDef.density = 1.0
            fixtureDef.friction = 0.5
            body.CreateFixture(fixtureDef)
            
        except Exception as e:
            warnings.warn(f"Failed to create polygon obstacle: {e}")
    
    def _create_circle_obstacle(self, world: Box2D.b2World, points: np.ndarray):
        """포인트 클러스터에서 원형 장애물 생성"""
        try:
            # 중심점과 반지름 계산
            center = np.mean(points, axis=0)
            distances = np.linalg.norm(points - center, axis=1)
            
            # 반지름을 최대 거리의 80% 정도로 설정 (더 정확한 근사)
            radius = np.percentile(distances, 85)  # 85 퍼센타일 사용
            
            # 최소/최대 반지름 제한 (더 합리적인 범위)
            radius = max(0.05, min(radius, 5.0))
            
            print(f"Creating circle: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
            
            # Box2D body 생성
            bodyDef = Box2D.b2BodyDef()
            bodyDef.type = Box2D.b2_staticBody
            bodyDef.position = (float(center[0]), float(center[1]))
            body = world.CreateBody(bodyDef)
            
            # 원형 shape 생성
            shape = Box2D.b2CircleShape()
            shape.radius = float(radius)
            
            # Fixture 생성
            fixtureDef = Box2D.b2FixtureDef()
            fixtureDef.shape = shape
            fixtureDef.density = 1.0
            fixtureDef.friction = 0.5
            body.CreateFixture(fixtureDef)
            
        except Exception as e:
            warnings.warn(f"Failed to create circle obstacle: {e}")
    
    def _ensure_ccw(self, vertices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """정점들이 반시계방향으로 정렬되도록 보장"""
        # 면적 계산으로 방향 확인 (Shoelace formula)
        area = 0
        n = len(vertices)
        
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        
        # 면적이 음수면 시계방향이므로 뒤집기
        if area < 0:
            vertices.reverse()
        
        return vertices
    
    def load_and_create_world(self, filename: str) -> Box2D.b2World:
        """
        PLY 포인트클라우드 파일을 로드하고 Box2D 월드 생성
        클러스터링 설정은 메타데이터에서 자동으로 읽음
        
        Args:
            filename: 파일명 (확장자 포함 또는 제외)
            
        Returns:
            Box2D world 객체
        """
        points, metadata = self.load_pointcloud(filename)
        
        print(f"Loaded {len(points)} points from {filename}")
        
        # 메타데이터에서 클러스터링 설정 읽기 (기본값 제공)
        if metadata:
            clustering_eps = metadata.get('clustering_eps', 0.3)
            min_samples = metadata.get('min_samples', 5)
            obstacle_type = metadata.get('obstacle_type', 'auto')
            
            print(f"Using saved settings - eps: {clustering_eps}, min_samples: {min_samples}, type: {obstacle_type}")
            if 'resolution' in metadata:
                print(f"Original resolution: {metadata['resolution']}")
            if 'workspace_bounds' in metadata:
                print(f"Workspace bounds: {metadata['workspace_bounds']}")
        else:
            # 메타데이터가 없는 경우 기본값 사용
            clustering_eps = 0.3
            min_samples = 5
            obstacle_type = 'auto'
            print("No metadata found, using default clustering settings")
        
        world = self.create_world_from_pointcloud(
            points, clustering_eps, min_samples, obstacle_type
        )
        
        print(f"Created world with {len([b for b in world.bodies if b.type == Box2D.b2_staticBody])} obstacles")
        
        return world
    
    def list_available_pointclouds(self) -> List[str]:
        """사용 가능한 PLY 포인트클라우드 파일 목록"""
        files = []
        
        if not os.path.exists(self.data_dir):
            return files
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.ply'):
                name = os.path.splitext(filename)[0]
                files.append(name)
        
        return sorted(files)
    
    def get_pointcloud_info(self, filename: str) -> dict:
        """포인트클라우드 파일 정보 조회"""
        info = {
            'name': filename,
            'available_formats': [],
            'metadata': None
        }
        
        # 사용 가능한 형식 확인
        for format in ['npy', 'ply', 'txt']:
            filepath = os.path.join(self.data_dir, f"{filename}.{format}")
            if os.path.exists(filepath):
                info['available_formats'].append(format)
        
        # 메타데이터 로드
        meta_filepath = os.path.join(self.data_dir, f"{filename}_meta.json")
        if os.path.exists(meta_filepath):
            with open(meta_filepath, 'r') as f:
                info['metadata'] = json.load(f)
        
        return info


# 편의 함수들
def load_pointcloud_world(filename: str, data_dir: str = "data/pointcloud", **kwargs) -> Box2D.b2World:
    """포인트클라우드에서 Box2D 월드를 로드하는 편의 함수"""
    loader = PointcloudLoader(data_dir)
    return loader.load_and_create_world(filename, **kwargs)


def list_pointclouds(data_dir: str = "data/pointcloud") -> List[str]:
    """사용 가능한 포인트클라우드 목록을 반환하는 편의 함수"""
    loader = PointcloudLoader(data_dir)
    return loader.list_available_pointclouds()


if __name__ == "__main__":
    # 테스트 코드
    loader = PointcloudLoader()
    
    # 사용 가능한 포인트클라우드 목록
    available = loader.list_available_pointclouds()
    print("Available pointclouds:", available)
    
    if available:
        # 첫 번째 포인트클라우드로 월드 생성 테스트
        filename = available[0]
        print(f"\nLoading {filename}...")
        
        info = loader.get_pointcloud_info(filename)
        print(f"Info: {info}")
        
        try:
            world = loader.load_and_create_world(filename, format='npy')
            print(f"Successfully created world with {len(list(world.bodies))} bodies")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No pointclouds found. Run pointcloud_extractor.py first.")
