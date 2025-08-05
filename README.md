# 🚀 2D Robot Simulation Suite

**SE(3) Rigid Body 시뮬레이션 및 Riemannian Flow Matching 정책 학습 프로젝트**

## 📋 프로젝트 개요

**SE(3) Rigid Body** 환경에서 **Riemannian Flow Matching (RFM)** 기반 AI 정책을 학습하여 **RRT-Connect를 대체하는 학습 기반 경로 계획**을 개발하는 프로젝트입니다.

### 🎯 핵심 목표
- **SE(3) Ellipsoid Rigid Body** 시뮬레이션 환경 구축 
- **Riemannian Flow Matching** 기반 AI 정책 학습
- **RRT-Connect 대비 10x 빠른 추론**, **부드러운 궤적** 생성

---

## 🗂️ 프로젝트 구조

```
2d_sim/
├── packages/                          # 모듈화된 패키지들
│   ├── data_generator/                 # 데이터 생성 도구들
│   │   ├── pointcloud/                 # 환경 포인트클라우드 생성
│   │   ├── pose/                       # SE(3) 포즈 및 포즈 페어 생성
│   │   └── reference_planner/          # RRT-Connect + B-spline 스무딩
│   │       ├── rrt_connect/            # OMPL 기반 RRT 구현
│   │       ├── bspline_smoothing.py    # B-spline 궤적 스무딩
│   │       └── utils/                  # 궤적 시각화 도구
│   ├── simulation/                     # 시뮬레이션 환경
│   │   └── robot_simulation/           # SE(3) rigid body 시뮬레이션
│   └── rfm_policy/                     # Riemannian Flow Matching 모델
│       ├── models/                     # SE(3) RFM 모델 구현 
│       ├── loaders/                    # 데이터 로더
│       ├── losses/                     # 손실 함수들
│       ├── trainers/                   # 훈련 시스템
│       └── utils/                      # ODE solver, SE(3) 유틸리티
├── data/                               # 생성된 데이터
│   ├── pointcloud/circle_envs_10k/     # 10,000개 환경 포인트클라우드
│   ├── pose_pairs/circle_envs_10k/     # Init-target SE(3) 포즈 페어
│   ├── trajectories/                   # 생성된 궤적들
│   │   ├── circle_envs_10k/            # RRT-Connect 궤적 (100개 완료)
│   │   └── circle_envs_10k_bsplined/   # B-spline 스무딩 궤적 (100개 완료)
│   └── visualizations/                 # 시각화 결과
└── fm-main/                            # 참조 모델 (기존 grasp RCFM)
```

---

## 🛠️ 환경 설정

### **패키지 관리: UV**
이 프로젝트는 **UV**로 패키지를 관리합니다.

#### **Mac/Linux (로컬)**
```bash
# 가상환경 활성화
source .venv/bin/activate

# 의존성 설치
uv sync

# 새 패키지 추가 시
uv add <package_name>
```

#### **Ubuntu 서버 (연구실)**
```bash
# 프로젝트 클론
git clone <repository_url>
cd 2d_sim

# UV 설치 (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 가상환경 생성 및 활성화
uv venv
source .venv/bin/activate

# 의존성 설치
uv sync
```

### **핵심 의존성**
- **OMPL**: 경로 계획 (RRT-Connect)
- **PyTorch**: AI 모델 학습
- **NumPy/SciPy**: 수치 계산, B-spline 
- **Open3D**: 포인트클라우드 처리
- **Matplotlib**: 시각화
- **Wandb**: 실험 추적
- **OmegaConf**: 설정 관리

---

## ✅ 완료된 작업

### **1. 데이터 생성 파이프라인** ✅
- ✅ **10,000개 환경**: 원형 장애물 기반 포인트클라우드
- ✅ **10,000개 포즈 페어**: 각 환경별 10개 포즈, 첫 번째 페어 사용
- ✅ **100개 RRT 궤적**: collision margin 0.05m, pair_1 네이밍
- ✅ **100개 B-spline 궤적**: 2x 밀도, 90-100% 스무딩 개선

### **2. SE(3) Riemannian Flow Matching 모델** ✅  
- ✅ **SE3RFM**: fm-main 스타일 RFM 모델
- ✅ **3D DGCNN**: 3D 포인트클라우드 인코더 (1024D→2048D)
- ✅ **SE(3) Encoder**: 4×4 행렬 → 12D 직접 플래튼
- ✅ **Geometry Encoder**: 타원체 형상 (32D)
- ✅ **Velocity Field**: 2105D → 6D twist 벡터
- ✅ **훈련 인프라**: Wandb, 분산 훈련, ODE solver

### **3. 궤적 품질 개선** ✅
- ✅ **RRT-Connect 최적화**: collision margin 0.05m
- ✅ **B-spline 스무딩**: SE(2) 매니폴드 기반
- ✅ **로봇 지오메트리 수정**: 1.2×0.4m 길쭉한 타원체
- ✅ **충돌 검증**: RRT + Post-B-spline 충돌 체크
- ✅ **시각화 개선**: 실제 로봇 크기 반영

---

## 📊 현재 데이터 현황

### **환경 데이터**
- **포인트클라우드**: `data/pointcloud/circle_envs_10k/` (10,000개)
- **포즈 페어**: `data/pose_pairs/circle_envs_10k/` (10,000개)

### **궤적 데이터** 
- **RRT 궤적**: `data/trajectories/circle_envs_10k/` (**100개 완료**)
  - 파일명: `circle_env_XXXXXX_pair_1_traj_rb3.json`
  - 환경: 000000-000099 (첫 100개 환경)
  - 포즈페어: 각 환경의 첫 번째 페어 사용
  - collision margin: 0.05m

- **B-spline 궤적**: `data/trajectories/circle_envs_10k_bsplined/` (**100개 완료**)
  - 파일명: `circle_env_XXXXXX_pair_1_traj_rb3_bsplined.json`
  - 2x 밀도 증가, 90-100% 스무딩 개선
  - 충돌 검증 완료

---

## 🚀 데이터 생성 파이프라인

### **단일 환경 테스트**
```bash
# 환경 활성화
source .venv/bin/activate
cd packages/data_generator/reference_planner

# 1. RRT 궤적 생성 (단일)
python se3_trajectory_generator.py \
  --pointcloud_file ../../../data/pointcloud/circle_envs_10k/circle_env_000000.ply \
  --pose_pairs_file ../../../data/pose_pairs/circle_envs_10k/circle_env_000000_rb_3_pairs.json \
  --rigid_body_id 3 \
  --range 0.05 \
  --output_dir ../../../data/trajectories/circle_envs_10k

# 2. B-spline 스무딩
python bspline_smoothing.py \
  --trajectory_file ../../../data/trajectories/circle_envs_10k/circle_env_000000_pair_1_traj_rb3.json \
  --output_dir ../../../data/trajectories/circle_envs_10k_bsplined

# 3. 시각화
python utils/trajectory_visualizer.py \
  ../../../data/trajectories/circle_envs_10k_bsplined/circle_env_000000_pair_1_traj_rb3_bsplined.json \
  --save_image \
  --output_path ../../../data/visualizations/test_bsplined.png
```

### **대량 생성 (이미 완료)**
```bash
# 100개 환경 생성 (완료됨)
python generate_test_100.py

# 결과 확인
ls ../../../data/trajectories/circle_envs_10k/*pair_1_traj_rb3.json | wc -l      # 100개
ls ../../../data/trajectories/circle_envs_10k_bsplined/*_bsplined.json | wc -l  # 100개
```

### **다음 단계: 대규모 생성**
```bash
# 전체 10,000개 환경 대량 생성 (예정)
python generate_all_10k.py  # 추후 실행
```

---

## 🎯 다음 단계 (연구실 서버)

### **1. RFM 모델 훈련 준비**
```bash
# 모델 테스트
cd packages/rfm_policy
python test_se3_rfm.py

# 설정 확인
cat configs/se3_rfm_config.yaml

# 훈련 시작 (RTX 4090)
python train_se3_rfm.py --config configs/se3_rfm_config.yaml
```

### **2. 데이터 스케일링**
- 100개 → 1,000개 → 10,000개 환경으로 점진적 확장
- 궤적 품질 vs 데이터 규모 분석

### **3. 모델 성능 평가**
- RRT-Connect vs RFM 성능 비교
- 추론 속도, 궤적 품질, 충돌 회피율

---

## 🛠️ 개발 명령어

### **데이터 생성**
```bash
cd packages/data_generator/reference_planner

# RRT 궤적 생성
python se3_trajectory_generator.py --range 0.05

# B-spline 스무딩
python bspline_smoothing.py --trajectory_file <path>

# 충돌 체크
python bspline_collision_checker.py --trajectory_file <path>
```

### **시각화**
```bash
# 궤적 시각화
python utils/trajectory_visualizer.py <trajectory.json> --save_image

# 환경 시각화
python pointcloud/utils/quick_visualize.py <env.ply>
```

### **모델 관련**
```bash
cd packages/rfm_policy

# 모델 테스트
python test_se3_rfm.py

# 훈련 시작
python train_se3_rfm.py --config configs/se3_rfm_config.yaml

# 평가
python evaluation/evaluator.py --model_path <checkpoint>
```

---

## 📈 성공 기준

### **데이터 완료도**
- ✅ **100개 환경**: RRT + B-spline 궤적 완료
- 🎯 **1,000개 환경**: 모델 훈련용 데이터셋
- 🚀 **10,000개 환경**: 최종 대규모 데이터셋

### **모델 성능**  
- 🎯 **90% 충돌 없는 궤적 생성**
- 🎯 **RRT 대비 10x 빠른 추론**
- 🎯 **RRT 대비 2x 부드러운 궤적**

### **시스템 안정성**
- 🎯 **분산 훈련 성공** (RTX 4090)
- 🎯 **Wandb 로깅 안정화**
- 🎯 **재현 가능한 실험**

---

## 📚 참고 자료

- **Riemannian Flow Matching**: Manifold-aware generative modeling
- **SE(3) Lie Groups**: Rigid body transformations
- **B-spline Curves**: Smooth trajectory interpolation
- **OMPL**: Open Motion Planning Library
- **FM-Main**: Reference RCFM implementation

---

**현재 상태**: 데이터 생성 완료, RFM 모델 훈련 준비 중 🚀