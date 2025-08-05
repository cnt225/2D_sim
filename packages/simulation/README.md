# 🎮 Robot Simulation Package

실시간 로봇 시뮬레이션 및 정책 검증 패키지

## 📋 주요 기능

- **Box2D 물리 시뮬레이션**: 정확한 물리 기반 로봇 동작
- **다양한 제어 정책**: Servo, PD, RFM 정책 지원
- **실시간 시각화**: pygame 기반 시각화
- **데이터 연동**: data_generator 및 rfm_policy 패키지와 연동

## 🗂️ 모듈 구조

```
robot_simulation/
├── core/                   # 핵심 시뮬레이션
│   ├── main.py            # 메인 실행 파일
│   ├── simulation.py      # 시뮬레이션 로직
│   ├── env.py             # 환경 설정
│   ├── render.py          # 시각화
│   └── record_video.py    # 비디오 녹화
├── control/               # 제어 시스템
│   └── policy.py          # 제어 정책들
├── data_loaders/          # 데이터 로딩
│   └── data_interface.py  # 외부 데이터 인터페이스
├── legacy/                # 레거시 시스템
│   └── simple_endeffector_sim/
└── config/                # 설정 파일들
    ├── robot_geometries.yaml
    └── data_paths.yaml
```

## 🚀 사용 방법

### 기본 시뮬레이션 실행
```bash
# Target pose 제어
uv run python -m robot_simulation.core.main --target-pose 0.5 -0.3 0.8

# 특정 로봇 기하학 사용
uv run python -m robot_simulation.core.main --geometry 2 --target-pose 0.5 -0.3 0.8

# 환경 파일 로딩
uv run python -m robot_simulation.core.main --env circles_only --target-pose 0.5 -0.3 0.8
```

### 사용 가능한 옵션 확인
```bash
# 로봇 기하학 목록
uv run python -m robot_simulation.core.main --list-geometries

# 도움말
uv run python -m robot_simulation.core.main --help
```

### 비디오 녹화
```bash
uv run python -m robot_simulation.core.record_video --output simulation.mp4
```

## ⚙️ 설정

### 로봇 기하학 설정 (`config/robot_geometries.yaml`)
- 6가지 로봇 구성 지원
- link_lengths, link_widths 등 커스터마이징 가능

### 데이터 경로 설정 (`config/data_paths.yaml`)
- 외부 패키지 데이터 경로 설정
- 환경, 포즈, 모델 파일 경로 관리

## 🔄 외부 데이터 연동

```python
from robot_simulation.data_loaders.data_interface import get_data_interface

# 데이터 인터페이스 생성
data_interface = get_data_interface()

# 사용 가능한 환경 확인
environments = data_interface.list_available_environments()

# 환경 파일 경로 가져오기
env_path = data_interface.get_environment_path("circles_only")

# 포즈 데이터 경로 가져오기
pose_path = data_interface.get_pose_data_path("circles_only", robot_id=0)
```

## 🎯 제어 정책

### 1. Servo 제어
- Box2D revolute joint motor 사용
- 부드러운 관절 움직임

### 2. PD 제어  
- 비례-미분 제어
- 정밀한 위치 제어

### 3. RFM 정책 (향후)
- Robot Foundation Model 기반
- 학습된 정책 로딩 및 실행

## 📊 시각화 기능

- **실시간 렌더링**: 로봇 및 환경 시각화
- **Target Pose 표시**: 목표 위치 시각적 표시
- **궤적 추적**: End-effector 경로 표시
- **성능 메트릭**: FPS, 제어 성능 표시

## 🔧 개발자 가이드

### 새로운 제어 정책 추가
1. `control/policy.py`에 정책 함수 추가
2. `core/simulation.py`에서 정책 통합
3. 명령행 옵션에 추가

### 새로운 환경 타입 지원
1. `core/env.py`에서 환경 로딩 로직 추가
2. `data_loaders/`에서 데이터 인터페이스 확장

## 📦 의존성

- `box2d>=2.3.10` - 물리 시뮬레이션
- `pygame>=2.6.1` - 실시간 시각화  
- `numpy>=1.24.4` - 수치 연산
- `opencv-python>=4.5.0` - 비디오 처리
- `pyyaml>=6.0` - 설정 파일
- `plyfile>=1.0.0` - PLY 파일 로딩
