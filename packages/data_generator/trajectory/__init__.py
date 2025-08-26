"""
궤적 생성 파이프라인 HDF5 통합 모듈
Raw → Smooth 궤적 생성, 충돌 검증, 메타데이터 관리를 통합한 시스템

주요 기능:
- HDF5 기반 궤적 데이터 저장 시스템
- RRT → B-spline 스무딩 파이프라인
- 충돌 검증 시스템 연동
- 환경별 궤적 데이터 체계적 관리
"""

from .trajectory_data_manager import TrajectoryDataManager
from .batch_generate_trajectories import generate_trajectories_for_environment
from .trajectory_validator import TrajectoryValidator

__all__ = [
    'TrajectoryDataManager',
    'generate_trajectories_for_environment', 
    'TrajectoryValidator'
]

__version__ = "1.0.0"


