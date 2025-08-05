"""
Pose Generation and Collision Detection Package
SE(3) 기반 rigid body 포즈 생성 및 충돌 검사 관련 모듈들

주요 모듈:
- random_pose_generator: SE(3) 랜덤 포즈 생성  
- collision_detector: SE(3) rigid body 충돌 검사
- batch_pose_generator: SE(3) 배치 포즈 생성
- pose_visualizer: SE(3) 포즈 시각화
- pose_pair_generator: SE(3) 포즈 쌍 생성
- pose_pair_loader: SE(3) 포즈 쌍 로더
"""

from .random_pose_generator import SE3RandomPoseGenerator, RandomPoseGenerator
from .collision_detector import RigidBodyCollisionDetector
from .batch_pose_generator import SE3BatchPoseGenerator, BatchPoseGenerator
from .pose_visualizer import SE3PoseVisualizer, PoseVisualizer
from .pose_pair_generator import SE3PosePairGenerator, PosePairGenerator
from .pose_pair_loader import SE3PosePairLoader, PosePairLoader

__all__ = [
    # SE(3) 클래스들 (주 사용)
    'SE3RandomPoseGenerator',
    'RigidBodyCollisionDetector', 
    'SE3BatchPoseGenerator',
    'SE3PoseVisualizer',
    'SE3PosePairGenerator',
    'SE3PosePairLoader',
    
    # 호환성 별칭들
    'RandomPoseGenerator',
    'BatchPoseGenerator',
    'PoseVisualizer',
    'PosePairGenerator',
    'PosePairLoader'
] 