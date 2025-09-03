# 🚀 Trajectory Generation Module

2D 시뮬레이션 환경에서 RRT-Connect 알고리즘을 사용하여 궤적을 생성하고 HDF5 형식으로 저장하는 모듈입니다.

## 📁 모듈 구조

```
trajectory/
├── README.md                           # 이 파일
├── requirements.txt                     # Python 의존성
├── batch_generate_raw_trajectories.py  # 🔥 주요 스크립트: Raw 궤적 대량 생성
├── batch_smooth_trajectories.py        # 궤적 스무딩 처리
├── generate_tdot_trajectories.py       # 🔥 주요 스크립트: Tdot 속도 궤적 생성
├── trajectory_data_manager.py          # HDF5 데이터 관리
├── trajectory_validator.py             # 궤적 충돌 검증
├── rrt_connect/                        # RRT-Connect 플래너
│   ├── __init__.py
│   ├── rrt_planner.py                  # SE(3) RRT-Connect 구현
│   └── ompl_setup.py                   # OMPL 환경 설정
├── utils/                              # 유틸리티
│   ├── trajectory_smoother.py          # B-spline 스무딩
│   ├── trajectory_visualizer.py        # HDF5 기반 시각화 (구버전)
│   └── simple_trajectory_visualizer.py # 새 HDF5 구조용 시각화
└── batch_generate_trajectories.py      # 🔥 주요 스크립트: 통합 파이프라인 (RRT + 스무딩)
```

## 🎯 주요 기능

### 1. Raw 궤적 생성 (RRT-Connect)
- **입력**: 환경 포인트클라우드 (.ply), Pose 쌍 데이터 (unified_poses.h5)
- **알고리즘**: OMPL RRT-Connect (SE(3) 공간)
- **출력**: HDF5 형식 Raw 궤적 데이터

### 2. 궤적 스무딩 (B-spline)
- **입력**: Raw 궤적 데이터
- **알고리즘**: SE(2) B-spline 스무딩
- **출력**: 스무딩된 궤적 데이터

### 3. 궤적 검증 (Collision Detection)
- **기능**: 생성된 궤적의 충돌 여부 검증
- **방법**: 커스텀 Python/NumPy 충돌 검출기

### 4. Tdot 속도 궤적 생성 (NEW!)
- **입력**: 스무딩된 궤적 데이터
- **알고리즘**: SE(3) body twist 계산
- **시간 정책**: 균등 할당 또는 곡률 기반 할당
- **출력**: Tdot 속도 궤적 (모델 학습용)

### 5. 시각화
- **정적 이미지**: 환경과 궤적을 함께 표시
- **애니메이션**: 궤적 재생 동영상 생성

## 📊 HDF5 데이터 구조

### 📍 저장 위치
```
root/data/trajectory/
└── {env_set_name}_trajs.h5           # 예: circles_only_trajs.h5
```

### 🏗️ HDF5 파일 구조
```
{env_set_name}_trajs.h5
├── metadata/                          # 전역 메타데이터
│   ├── @creation_time                 # 생성 시간
│   ├── @env_set_name                  # 환경 세트 이름
│   └── @total_environments            # 총 환경 수
├── circle_env_000000/                 # 환경별 그룹
│   ├── 0/                            # 페어 ID별 서브그룹
│   │   ├── raw_trajectory             # Raw RRT 궤적 데이터 [N x 3] (x, y, yaw)
│   │   ├── smooth_trajectory          # 스무딩된 궤적 [M x 3] (선택적)
│   │   ├── @start_pose               # 시작 pose [x, y, yaw]
│   │   ├── @end_pose                 # 목표 pose [x, y, yaw]
│   │   ├── @generation_time          # RRT 계획 시간 (초)
│   │   ├── @path_length              # 경로 길이 (m)
│   │   ├── @waypoint_count           # waypoint 개수
│   │   └── @timestamp                # 생성 시각
│   └── 1/                            # 다음 페어
│       └── ... (동일 구조)
├── circle_env_000001/
│   └── ... (동일 구조)
└── ... (더 많은 환경들)
```

## 🛠️ 사용법

### 1. Tdot 속도 궤적 생성 (NEW!)

```bash
# 균등 시간 할당 (dt=0.01s)
python generate_tdot_trajectories.py --input circles_only_integrated_trajs.h5 --dt 0.01

# 곡률 기반 시간 할당
python generate_tdot_trajectories.py --input circles_only_integrated_trajs.h5 \
    --time-policy curvature \
    --v-ref 0.4 --v-cap 0.5 --a-lat-max 1.0

# 6D 벡터 형식으로 저장 (기본은 4x4 행렬)
python generate_tdot_trajectories.py --input circles_only_integrated_trajs.h5 \
    --save-format 6d --dt 0.01

# 출력: root/data/Tdot/<input_name>_Tdot.h5
```

### 2. Raw 궤적 대량 생성

```bash
# 기본 사용법
python batch_generate_raw_trajectories.py \
    --env-set circles_only \
    --pose-file unified_poses.h5 \
    --env-count 10 \
    --pair-count 2

# 고급 옵션
python batch_generate_raw_trajectories.py \
    --env-set circles_only \
    --pose-file unified_poses.h5 \
    --env-count 1000 \
    --pair-count 2 \
    --start-env-id 0 \
    --rrt-range 0.25 \
    --rrt-max-time 15.0 \
    --rigid-body-id 3
```

### 3. 사용 가능한 환경 목록 확인

```bash
python batch_generate_raw_trajectories.py \
    --pose-file unified_poses.h5 \
    --list-environments
```

### 3. 궤적 스무딩

```bash
python batch_smooth_trajectories.py \
    --trajectory-file circles_only_trajs.h5 \
    --env-name circle_env_000000 \
    --pair-ids 0,1
```

### 4. 궤적 시각화

```bash
# 새 HDF5 구조용 시각화
python utils/simple_trajectory_visualizer.py circle_env_000000 0

# 구 HDF5 구조용 시각화 (레거시)
python utils/trajectory_visualizer.py circle_env_000000 test_pair_000
```

## ⚙️ 설정 및 파라미터

### RRT-Connect 설정
- **range**: `0.25` (확장 거리, 작을수록 정밀하지만 느림)
- **max_planning_time**: `15.0초` (복잡한 케이스 대응)
- **goal_bias**: `0.05` (목표 지향 확률)

### SE(3) 공간 설정
- **Position bounds**: x,y ∈ [-1, 11], z = 0 (고정)
- **Orientation bounds**: roll,pitch = 0 (고정), yaw ∈ [-π, π]

### B-spline 스무딩 설정
- **degree**: 3 (3차 B-spline)
- **smoothing_factor**: 0.1

## 📈 성능 지표

### 현재 테스트 결과 (circles_only, 20 페어)
- **성공률**: 100% (15초 시간 제한)
- **평균 계획 시간**: ~0.8초/페어
- **시간 분포**: 
  - 75%: 0.5초 이내
  - 20%: 0.5~5초
  - 5%: 5~15초 (복잡한 케이스)

## 🔧 의존성

### Python 패키지
```
numpy
matplotlib
h5py
scipy
ompl (OMPL Python bindings)
```

### OMPL 환경 설정
복잡한 OMPL 의존성으로 인해 전용 conda 환경 사용 권장:

```bash
# 전용 환경 활성화 (OMPL 사용 시)
conda activate trajectory_ompl

# 일반 환경 (OMPL 없이 데이터 처리만)
conda activate base
```

## 🚨 주의사항

### 1. OMPL 메모리 이슈
- **증상**: 프로그램 종료 시 `double free or corruption` 오류
- **해결**: 기능적으로는 정상 동작, 종료 시 무시 가능
- **원인**: OMPL Python 바인딩과 Python 3.12 간 호환성 문제

### 2. 시간 제한 설정
- **권장**: 15초 (복잡한 케이스 대응)
- **최소**: 5초 (간단한 케이스용)
- **주의**: 너무 짧으면 실패율 증가

### 3. 환경 파일 경로
- **포인트클라우드**: `root/data/pointcloud/{env_set_name}/{env_name}.ply`
- **Pose 데이터**: `root/data/pose/{pose_file}`
- **출력 궤적**: `root/data/trajectory/{env_set_name}_trajs.h5`

## 🔄 워크플로우

```mermaid
graph TD
    A[Pose Data<br/>unified_poses.h5] --> B[Raw Trajectory Generation<br/>batch_generate_raw_trajectories.py]
    C[Environment Pointcloud<br/>.ply files] --> B
    B --> D[HDF5 Trajectory Data<br/>{env_set}_trajs.h5]
    D --> E[Trajectory Smoothing<br/>batch_smooth_trajectories.py]
    E --> F[Smoothed Trajectories<br/>in HDF5]
    D --> G[Trajectory Visualization<br/>simple_trajectory_visualizer.py]
    F --> G
    G --> H[PNG Images<br/>data/visualized/trajectory/]
```

## 📚 관련 문서

### 상위 모듈
- **pose**: Pose 데이터 생성 및 관리
- **collision_detector**: 충돌 검출 시스템

### 설정 파일
- **Root Rules.md**: 전체 프로젝트 구조 및 환경 설정
- **requirements.txt**: Python 의존성 목록

## 🆕 최근 업데이트

- **2024-08-26**: 15초 시간 제한으로 안정성 향상
- **2024-08-26**: 새로운 HDF5 구조 (`env_set_name_trajs.h5`) 도입
- **2024-08-26**: `simple_trajectory_visualizer.py` 추가
- **2024-08-26**: 동적 환경 로딩 기능 추가
- **2024-08-26**: RRT range 기본값 0.25로 최적화

---

**📞 문의사항이나 이슈가 있으시면 프로젝트 관리자에게 연락해주세요.**