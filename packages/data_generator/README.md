# 📊 Robot Data Generator Package

로봇 시뮬레이션용 데이터 생성 및 관리 패키지

## 📋 주요 기능

- **환경 생성**: 다양한 타입의 시뮬레이션 환경
- **포즈 생성**: 충돌 없는 로봇 포즈 데이터
- **궤적 계획**: RRT-Connect 기반 경로 계획
- **데이터 표준화**: 일관된 JSON/PLY 포맷

## 🗂️ 모듈 구조

```
robot_data_generator/
├── pointcloud/            # 환경 생성
│   ├── create_pointcloud.py
│   ├── circle_environment_generator.py
│   ├── random_environment_generator.py
│   └── concave_shape_generator/
├── pose/                  # 포즈 생성
│   ├── random_pose_generator.py
│   ├── collision_detector.py
│   ├── pose_pipeline.py
│   └── batch_pose_generator.py
└── reference_planner/     # 궤적 계획
    ├── rrt_connect/
    ├── trajectory_player_servo.py
    └── simulation_runner.py
```

## 🚀 사용 방법

### 환경 데이터 생성
```python
from robot_data_generator.pointcloud import create_circle_environment

# 원형 장애물 환경 생성
env_data = create_circle_environment(
    num_circles=10,
    workspace_bounds=(-1, 11, -1, 11)
)
```

### 포즈 데이터 생성
```python
from robot_data_generator.pose import random_pose_generator

# 충돌 없는 랜덤 포즈 생성
poses = random_pose_generator.generate_collision_free_poses(
    environment_file="circles_only.ply",
    robot_geometry=0,
    num_poses=1000
)
```

### 궤적 계획
```python
from robot_data_generator.reference_planner.rrt_connect import RRTPlanner

# RRT-Connect 궤적 계획
planner = RRTPlanner()
trajectory = planner.plan(
    start_pose=[0, 0, 0],
    goal_pose=[0.5, -0.3, 0.8],
    environment="circles_only.ply"
)
```

## 📁 데이터 출력 구조

### 환경 데이터 (`data/pointcloud/`)
```
environment_name/
├── environment_name.ply      # 포인트클라우드
└── environment_name_meta.json # 메타데이터
```

### 포즈 데이터 (`data/pose/`)
```json
{
  "environment": { "name": "circles_only", ... },
  "robot": { "id": 0, "metadata": {...} },
  "poses": {
    "data": [[angle1, angle2, angle3], ...],
    "count": 1000,
    "format": "joint_angles_radians"
  }
}
```

### 궤적 데이터 (`data/trajectories/`)
```json
{
  "trajectory_id": "traj_001",
  "start_pose": [0, 0, 0],
  "goal_pose": [0.5, -0.3, 0.8],
  "path": [[0, 0, 0], [0.1, -0.05, 0.1], ...],
  "planning_method": "RRT-Connect"
}
```

## 🎯 환경 타입

### 1. Circle Environments
- 원형 장애물 배치
- 난이도 조절 가능
- 대량 생성 지원

### 2. Random Environments  
- 다양한 형태의 장애물
- 복잡도 설정 가능

### 3. Concave Shapes
- 복잡한 오목 형태
- SVG 기반 설계
- Boolean 연산 지원

## ⚡ 배치 생성

### 대량 환경 생성
```bash
uv run python -m robot_data_generator.scripts.generate_environments \
  --type circle --count 10000 --output data/pointcloud/
```

### 대량 포즈 생성
```bash
uv run python -m robot_data_generator.scripts.generate_poses \
  --env-dir data/pointcloud/ --robot-geometries 0,1,2
```

### 궤적 생성
```bash
uv run python -m robot_data_generator.scripts.generate_trajectories \
  --pose-dir data/pose/ --method rrt-connect
```

## 🔧 설정

### 환경 생성 설정 (`config/environment_configs.yaml`)
```yaml
circle_environments:
  workspace_bounds: [-1, 11, -1, 11]
  num_circles_range: [5, 20]
  radius_range: [0.2, 0.8]
  
random_environments:
  complexity_levels: [easy, medium, hard]
  obstacle_types: [circle, polygon, mixed]
```

### 로봇 기하학 (`config/robot_geometries.yaml`)
```yaml
robots:
  0:
    link_lengths: [3.0, 2.5, 2.0]
    link_widths: [0.3, 0.25, 0.2]
    max_reach: 7.5
```

## 📊 데이터 품질 관리

### 충돌 검사
- 정확한 기하학적 충돌 검사
- Safety margin 적용
- 검증 통계 제공

### 데이터 검증
```python
from robot_data_generator.pose import collision_detector

detector = collision_detector.CollisionDetector()
is_valid = detector.check_pose_validity(
    joint_angles=[0.5, -0.3, 0.8],
    environment="circles_only.ply",
    robot_geometry=0
)
```

## 📦 의존성

- `numpy>=1.24.4` - 수치 연산
- `matplotlib>=3.5.0` - 시각화
- `scipy>=1.10.1` - 과학 계산
- `ompl` - 경로 계획 라이브러리
- `plyfile>=1.0.0` - PLY 파일 처리
- `box2d>=2.3.10` - 충돌 검사
- `pyyaml>=6.0` - 설정 파일
