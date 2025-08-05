"""
Concave Shape Generator Package
복잡한 오목한(concave) 형태의 장애물을 생성하고 Box2D에 통합하는 패키지

주요 모듈:
- concave_shape_generator: 다양한 concave 형태 생성
- concave_box2d_integration: Box2D 통합 및 convex 분해
- concave_environment_generator: 환경에 concave 장애물 배치
"""

from .concave_shape_generator import (
    ConcaveShapeGenerator,
    ConcaveShapeType,
    ConcaveShapeConfig,
    create_concave_shape
)

from .concave_box2d_integration import (
    ConcaveBox2DAdapter,
    ConcaveObstacleData,
    generate_concave_shape_library
)

from .concave_environment_generator import (
    ConcaveEnvironmentGenerator,
    ConcaveEnvironmentConfig,
    create_concave_environment
)

__all__ = [
    'ConcaveShapeGenerator',
    'ConcaveShapeType', 
    'ConcaveShapeConfig',
    'create_concave_shape',
    'ConcaveBox2DAdapter',
    'ConcaveObstacleData',
    'generate_concave_shape_library',
    'ConcaveEnvironmentGenerator',
    'ConcaveEnvironmentConfig',
    'create_concave_environment'
] 