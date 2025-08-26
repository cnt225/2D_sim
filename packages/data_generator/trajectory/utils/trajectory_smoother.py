#!/usr/bin/env python3
"""
궤적 스무딩 처리 모듈
기존 B-spline 스무딩 모듈을 클래스 기반으로 재구성하여 통합
"""

import numpy as np
from scipy import interpolate
from scipy.spatial.distance import cdist
from typing import Optional, Tuple, Dict, Any
import time


def normalize_angle(angle):
    """각도를 [-π, π] 범위로 정규화"""
    return np.arctan2(np.sin(angle), np.cos(angle))


def unwrap_angles(angles):
    """각도 연속성 보장 (unwrapping)"""
    unwrapped = [angles[0]]
    for i in range(1, len(angles)):
        diff = angles[i] - unwrapped[-1]
        # 큰 점프 감지 및 보정
        if diff > np.pi:
            unwrapped.append(angles[i] - 2*np.pi)
        elif diff < -np.pi:
            unwrapped.append(angles[i] + 2*np.pi)
        else:
            unwrapped.append(angles[i])
    return np.array(unwrapped)


def compute_arc_length_parameterization(waypoints):
    """아크 길이 기반 파라미터화"""
    if len(waypoints) < 2:
        return np.array([0.0])
    
    # 각 웨이포인트 간 거리 계산 (x, y만 사용)
    positions = waypoints[:, :2]
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    
    # 누적 거리
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    total_length = cumulative_distances[-1]
    
    if total_length == 0:
        return np.linspace(0, 1, len(waypoints))
    
    return cumulative_distances / total_length


class BSplineTrajectoryProcessor:
    """B-spline 궤적 스무딩 프로세서"""
    
    def __init__(self, degree: int = 3, smoothing_factor: float = 0.0):
        """
        Args:
            degree: B-spline 차수 (3=cubic)
            smoothing_factor: 스무딩 강도 (0=보간, >0=근사)
        """
        self.degree = degree
        self.smoothing_factor = smoothing_factor
        
    def smooth_trajectory(self, 
                         rrt_waypoints: np.ndarray, 
                         num_points: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        RRT 궤적을 B-spline으로 스무딩 (SE(2) 매니폴드 고려)
        
        Args:
            rrt_waypoints: [N, 3] RRT 웨이포인트 [x, y, theta]
            num_points: 출력 궤적 포인트 수 (None이면 원본의 2배)
        
        Returns:
            smooth_trajectory: [num_points, 3] 스무딩된 궤적
            smoothing_info: 스무딩 정보 딕셔너리
        """
        start_time = time.time()
        
        # 입력 검증
        if len(rrt_waypoints) < self.degree + 1:
            print(f"⚠️ 웨이포인트 수({len(rrt_waypoints)})가 차수+1({self.degree+1})보다 작습니다.")
            return rrt_waypoints, {
                'success': False, 
                'error': 'Insufficient waypoints',
                'processing_time': time.time() - start_time
            }
        
        if num_points is None:
            num_points = len(rrt_waypoints) * 2
        
        try:
            # 1. 각도 unwrapping
            angles = unwrap_angles(rrt_waypoints[:, 2])
            
            # 2. 아크 길이 기반 파라미터화
            t_original = compute_arc_length_parameterization(rrt_waypoints)
            t_smooth = np.linspace(0, 1, num_points)
            
            # 3. 각 차원별 B-spline fitting
            # 위치 (x, y)
            x_spline = interpolate.splrep(t_original, rrt_waypoints[:, 0], 
                                        k=self.degree, s=self.smoothing_factor)
            y_spline = interpolate.splrep(t_original, rrt_waypoints[:, 1], 
                                        k=self.degree, s=self.smoothing_factor)
            
            # 각도 (theta) - unwrapped 상태에서 처리
            theta_spline = interpolate.splrep(t_original, angles, 
                                            k=self.degree, s=self.smoothing_factor)
            
            # 4. 스무딩된 궤적 생성
            x_smooth = interpolate.splev(t_smooth, x_spline)
            y_smooth = interpolate.splev(t_smooth, y_spline)
            theta_smooth = interpolate.splev(t_smooth, theta_spline)
            
            # 5. 각도 정규화
            theta_smooth = [normalize_angle(a) for a in theta_smooth]
            
            smooth_trajectory = np.column_stack([x_smooth, y_smooth, theta_smooth])
            
            # 6. 부드러움 메트릭 계산
            original_metrics = self.calculate_smoothness_metrics(rrt_waypoints)
            smoothed_metrics = self.calculate_smoothness_metrics(smooth_trajectory)
            
            processing_time = time.time() - start_time
            
            smoothing_info = {
                'success': True,
                'processing_time': processing_time,
                'original_waypoints': len(rrt_waypoints),
                'smoothed_waypoints': len(smooth_trajectory),
                'degree': self.degree,
                'smoothing_factor': self.smoothing_factor,
                'original_metrics': original_metrics,
                'smoothed_metrics': smoothed_metrics,
                'improvement': {
                    'curvature': (original_metrics['max_curvature'] - smoothed_metrics['max_curvature']) / original_metrics['max_curvature'] * 100 if original_metrics['max_curvature'] > 0 else 0,
                    'acceleration': (original_metrics['max_acceleration'] - smoothed_metrics['max_acceleration']) / original_metrics['max_acceleration'] * 100 if original_metrics['max_acceleration'] > 0 else 0
                }
            }
            
            return smooth_trajectory, smoothing_info
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ B-spline fitting 오류: {e}")
            return rrt_waypoints, {
                'success': False,
                'error': str(e),
                'processing_time': processing_time
            }
    
    def calculate_smoothness_metrics(self, trajectory: np.ndarray) -> Dict[str, float]:
        """궤적의 부드러움 메트릭 계산"""
        if len(trajectory) < 3:
            return {
                'path_length': 0.0,
                'max_curvature': 0.0,
                'avg_curvature': 0.0,
                'max_acceleration': 0.0,
                'avg_acceleration': 0.0,
                'jerk': 0.0
            }
        
        # 위치 정보
        positions = trajectory[:, :2]
        
        # 1. 경로 길이
        path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        
        # 2. 속도 (1차 차분)
        velocities = np.diff(positions, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # 3. 가속도 (2차 차분)
        accelerations = np.diff(velocities, axis=0)
        accel_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        # 4. 곡률 계산
        curvatures = []
        for i in range(1, len(trajectory) - 1):
            # 3점을 이용한 곡률 계산
            p1, p2, p3 = positions[i-1], positions[i], positions[i+1]
            
            # 벡터
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 외적의 크기 (2D에서는 스칼라)
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            
            # 곡률 = |v1 × v2| / (|v1| * |v2| * |v1 + v2|)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            norm_sum = np.linalg.norm(v1 + v2)
            
            if norm_v1 > 0 and norm_v2 > 0 and norm_sum > 0:
                curvature = abs(cross_product) / (norm_v1 * norm_v2 * norm_sum)
                curvatures.append(curvature)
        
        curvatures = np.array(curvatures) if curvatures else np.array([0.0])
        
        # 5. Jerk (3차 차분)
        jerks = np.diff(accelerations, axis=0)
        jerk_magnitudes = np.linalg.norm(jerks, axis=1)
        
        return {
            'path_length': path_length,
            'max_curvature': np.max(curvatures),
            'avg_curvature': np.mean(curvatures),
            'max_acceleration': np.max(accel_magnitudes) if len(accel_magnitudes) > 0 else 0.0,
            'avg_acceleration': np.mean(accel_magnitudes) if len(accel_magnitudes) > 0 else 0.0,
            'jerk': np.mean(jerk_magnitudes) if len(jerk_magnitudes) > 0 else 0.0
        }
    
    def convert_se3_to_se2(self, se3_trajectory: np.ndarray) -> np.ndarray:
        """SE(3) 궤적을 SE(2)로 변환"""
        if se3_trajectory.shape[1] >= 6:
            # [x, y, z, rx, ry, rz] → [x, y, rz]
            return se3_trajectory[:, [0, 1, 5]]
        elif se3_trajectory.shape[1] >= 3:
            # [x, y, z] → [x, y, 0]
            se2_traj = se3_trajectory[:, :2]
            return np.column_stack([se2_traj, np.zeros(len(se2_traj))])
        else:
            raise ValueError(f"Unsupported SE(3) trajectory shape: {se3_trajectory.shape}")
    
    def convert_se2_to_se3(self, se2_trajectory: np.ndarray) -> np.ndarray:
        """SE(2) 궤적을 SE(3)로 변환"""
        se3_trajectory = []
        for point in se2_trajectory:
            x, y, theta = point
            # SE(3) 형식: [x, y, z, rx, ry, rz]
            se3_pose = [x, y, 0.0, 0.0, 0.0, theta]
            se3_trajectory.append(se3_pose)
        
        return np.array(se3_trajectory)


class SPERLTrajectoryProcessor:
    """SPERL 스무딩 프로세서 (미래 구현용)"""
    
    def __init__(self):
        """SPERL 스무딩 초기화"""
        pass
    
    def smooth_trajectory(self, rrt_waypoints: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """SPERL 스무딩 (TODO: 구현 필요)"""
        print("⚠️ SPERL 스무딩은 아직 구현되지 않았습니다. B-spline 사용을 권장합니다.")
        return rrt_waypoints, {'success': False, 'error': 'SPERL not implemented'}


# 팩토리 함수
def create_trajectory_smoother(method: str = 'bspline', **kwargs) -> BSplineTrajectoryProcessor:
    """궤적 스무더 팩토리 함수"""
    if method.lower() == 'bspline':
        return BSplineTrajectoryProcessor(**kwargs)
    elif method.lower() == 'sperl':
        return SPERLTrajectoryProcessor()
    else:
        raise ValueError(f"Unsupported smoothing method: {method}")


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 BSplineTrajectoryProcessor 테스트")
    
    # 테스트 궤적 생성 (지그재그 패턴)
    test_waypoints = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.5],
        [2.0, 1.0, 1.0],
        [3.0, 0.0, -0.5],
        [4.0, 0.0, 0.0]
    ])
    
    # B-spline 프로세서 생성
    processor = BSplineTrajectoryProcessor(degree=3, smoothing_factor=0.0)
    
    # 스무딩 적용
    smoothed_traj, info = processor.smooth_trajectory(test_waypoints, num_points=20)
    
    if info['success']:
        print("✅ 스무딩 성공")
        print(f"   원본 웨이포인트: {info['original_waypoints']}")
        print(f"   스무딩 웨이포인트: {info['smoothed_waypoints']}")
        print(f"   처리 시간: {info['processing_time']:.3f}초")
        print(f"   곡률 개선: {info['improvement']['curvature']:.1f}%")
        print(f"   가속도 개선: {info['improvement']['acceleration']:.1f}%")
    else:
        print(f"❌ 스무딩 실패: {info['error']}")


