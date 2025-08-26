import json
import numpy as np
from scipy.spatial.transform import Rotation


class TrajectoryParser:
    """
    JSON 궤적 데이터 파싱 클래스
    fm-main의 AcronymGrasps와 유사한 역할
    """
    def __init__(self, trajectory_file):
        self.trajectory_file = trajectory_file
        
        # JSON 데이터 로드
        with open(trajectory_file, 'r') as f:
            self.data = json.load(f)
        
        # 메타데이터 추출
        self.env_name = self.data.get('environment', {}).get('name', 'unknown')
        self.pair_id = self.data.get('pair_id', 0)
        
        # 궤적 데이터 파싱
        self.waypoints = self._parse_waypoints()
        self.twist_vectors = self._compute_twist_vectors()
        
        # CFM용 타겟 변환
        self.target_twists = self._extract_target_twists()
        self.target_poses = self._extract_target_poses()
    
    def _parse_waypoints(self):
        """JSON 데이터에서 SE(3) waypoint 추출"""
        path_data = self.data.get('path', {})
        timestamps = path_data.get('timestamps', [])
        poses_flat = path_data.get('data', [])
        
        waypoints = []
        num_waypoints = min(len(timestamps), len(poses_flat))
        
        for i in range(num_waypoints):
            timestamp = timestamps[i] if i < len(timestamps) else 0.0
            pose_6d = poses_flat[i]
            
            if len(pose_6d) >= 6:
                # 6D pose → SE(3) 행렬
                x, y, z, rx, ry, rz = pose_6d[:6]
                
                R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [x, y, z]
                
                waypoints.append({
                    'pose': T,
                    'timestamp': timestamp,
                    'pose_6d': pose_6d
                })
        
        return waypoints
    
    def _compute_twist_vectors(self):
        """연속된 waypoint 간 twist vector 계산"""
        if len(self.waypoints) < 2:
            return []
        
        twists = []
        for i in range(len(self.waypoints) - 1):
            curr_wp = self.waypoints[i]
            next_wp = self.waypoints[i + 1]
            
            dt = next_wp['timestamp'] - curr_wp['timestamp']
            if dt <= 0:
                dt = 0.1  # 기본값
            
            twist = self._compute_se3_twist(
                curr_wp['pose'], next_wp['pose'], dt
            )
            twists.append(twist)
        
        # 마지막 지점은 정지
        twists.append(np.zeros(6))
        
        return np.array(twists)
    
    def _compute_se3_twist(self, T1, T2, dt):
        """
        두 SE(3) pose 간 twist vector 계산 (body frame)
        fm-main의 u_t 함수와 유사한 계산
        """
        # Translation (body frame)
        R1 = T1[:3, :3]
        p1, p2 = T1[:3, 3], T2[:3, 3]
        v_body = R1.T @ (p2 - p1) / dt
        
        # Angular velocity (body frame)
        R2 = T2[:3, :3]
        R_rel = R1.T @ R2  # Body frame 상대 회전
        
        # Rodrigues formula로 axis-angle 추출
        r_rel = Rotation.from_matrix(R_rel)
        w_body = r_rel.as_rotvec() / dt
        
        return np.concatenate([w_body, v_body])  # [ω, v] 순서
    
    def _extract_target_twists(self):
        """CFM 학습용 타겟 twist vectors"""
        # 모든 중간 twist를 타겟으로 사용
        # (fm-main에서 good_grasps 역할)
        return self.twist_vectors
    
    def _extract_target_poses(self):
        """각 waypoint에서의 최종 목표 pose"""
        if len(self.waypoints) == 0:
            return np.array([])
        
        final_pose = self.waypoints[-1]['pose']
        target_poses = np.tile(final_pose, (len(self.waypoints), 1, 1))
        
        return target_poses
    
    def get_environment_id(self):
        """환경 ID 반환"""
        return self.env_name
    
    def get_pointcloud_file(self, pointcloud_root):
        """대응하는 포인트클라우드 파일 경로"""
        import os
        return os.path.join(pointcloud_root, f"{self.env_name}.ply")
    
    def get_statistics(self):
        """궤적 통계 정보"""
        if len(self.waypoints) == 0:
            return {}
        
        poses = np.array([wp['pose'] for wp in self.waypoints])
        twists = self.twist_vectors
        
        # 위치 통계
        positions = poses[:, :3, 3]
        pos_mean = np.mean(positions, axis=0)
        pos_std = np.std(positions, axis=0)
        
        # 속도 통계
        if len(twists) > 0:
            angular_mean = np.mean(twists[:, :3], axis=0)
            angular_std = np.std(twists[:, :3], axis=0)
            linear_mean = np.mean(twists[:, 3:], axis=0)
            linear_std = np.std(twists[:, 3:], axis=0)
        else:
            angular_mean = angular_std = np.zeros(3)
            linear_mean = linear_std = np.zeros(3)
        
        return {
            'num_waypoints': len(self.waypoints),
            'num_twists': len(twists),
            'duration': self.waypoints[-1]['timestamp'] - self.waypoints[0]['timestamp'],
            'position_mean': pos_mean,
            'position_std': pos_std,
            'angular_velocity_mean': angular_mean,
            'angular_velocity_std': angular_std,
            'linear_velocity_mean': linear_mean,
            'linear_velocity_std': linear_std
        }
    
    def validate_data(self):
        """데이터 유효성 검증"""
        issues = []
        
        # 기본 구조 체크
        if len(self.waypoints) < 2:
            issues.append("Less than 2 waypoints")
        
        # Timestamp 체크
        timestamps = [wp['timestamp'] for wp in self.waypoints]
        if not all(t2 >= t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])):
            issues.append("Non-monotonic timestamps")
        
        # SE(3) 행렬 유효성
        for i, wp in enumerate(self.waypoints):
            T = wp['pose']
            if not np.allclose(T[3, :], [0, 0, 0, 1]):
                issues.append(f"Invalid SE(3) matrix at waypoint {i}")
        
        # Twist magnitude 체크
        if len(self.twist_vectors) > 0:
            max_angular = np.max(np.linalg.norm(self.twist_vectors[:, :3], axis=1))
            max_linear = np.max(np.linalg.norm(self.twist_vectors[:, 3:], axis=1))
            
            if max_angular > 10:  # rad/s
                issues.append(f"High angular velocity: {max_angular:.2f}")
            if max_linear > 5:    # m/s
                issues.append(f"High linear velocity: {max_linear:.2f}")
        
        return issues
    
    def export_summary(self):
        """요약 정보 딕셔너리"""
        stats = self.get_statistics()
        issues = self.validate_data()
        
        return {
            'file': self.trajectory_file,
            'env_id': self.env_name,
            'pair_id': self.pair_id,
            'statistics': stats,
            'validation_issues': issues,
            'is_valid': len(issues) == 0
        }