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
│   └── policy/                         # Riemannian Flow Matching 모델
│       └── v1/                         # 최신 구현 버전
│           ├── models/                  # Motion RFM 모델 구현 
│           ├── loaders/                 # 데이터 로더 (정규화 포함)
│           ├── trainers/                # 훈련 시스템
│           ├── utils/                   # 정규화, ODE solver, SE(3) 유틸리티
│           ├── configs/                 # 설정 파일들
│           ├── checkpoints/             # 학습된 모델 체크포인트
│           └── debug/                   # 디버그 및 분석 스크립트
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
- ✅ **3,241개 RRT 궤적**: 대규모 데이터셋 생성 완료
- ✅ **3,241개 B-spline 궤적**: T_dot 계산 포함 완료
- ✅ **데이터 정리**: 빈 파일 제거, 일관성 검증, 재인덱싱

### **2. 모션 플래닝 RFM 모델** ✅  
- ✅ **MotionRCFM**: fm-main 기반 적응 (GraspRCFM → MotionRCFM)
- ✅ **3D DGCNN**: 포인트클라우드 인코더 (2048D features)
- ✅ **조건부 입력**: current_T, target_T, time_t, pointcloud
- ✅ **6D Twist 출력**: SE(3) body frame 속도 벡터
- ✅ **개선된 T_dot 계산**: scipy.spatial.transform 기반 정확한 계산

### **3. 훈련 인프라 구축** ✅
- ✅ **TrajectoryDataset**: 실시간 T_dot 계산, PLY 파일 로딩
- ✅ **견고한 오류 처리**: 손상된 파일 대체, fallback 포인트클라우드
- ✅ **Wandb 통합**: motion_planning_rfm 프로젝트
- ✅ **tmux 백그라운드 실행**: SSH 연결 해제 시에도 안전
- ✅ **가상환경 분리**: packages/policy/policy_env

### **4. 모션 RFM 모델 학습 및 최적화** ✅
- ✅ **초기 학습 시작**: tmux 세션으로 백그라운드 실행
- ✅ **학습 속도 문제 진단**: 11 에포크/11시간 → 617일 예상
- ✅ **하이퍼파라미터 최적화**: 
  - `batch_size`: 4 → 32 (8x 빨라짐)
  - `n_epoch`: 1000 → 10 (충분한 테스트)
  - `augment_data`: true → false (안정성 향상)
- ✅ **학습 완료**: 10 에포크, Loss ~5.0

### **5. 추론 파이프라인 구현** ✅
- ✅ **기본 추론 엔진**: `inference.py` 구현
- ✅ **궤적 생성 전략**: `traj_gen_strategy.md` 문서화
- ✅ **ODE 적분 기반 궤적 생성**: RK4 적분, 적응형 타임스텝
- ✅ **초기 추론 테스트**: circle_env_000000, pose pair #2

### **6. 추론 성능 문제 진단 및 정규화 구현** ✅
- ✅ **추론 실패 원인 분석**: 거의 제자리, 극도로 작은 twist 값
- ✅ **학습 데이터 twist 통계 분석**: 
  - 학습 데이터: 평균 5.6 m/s
  - 모델 출력: 0.06 m/s (100배 작음)
- ✅ **정규화 파이프라인 설계 및 구현**:
  - `TwistNormalizer` 클래스 (통계 생성, 정규화, 역정규화)
  - 정규화된 데이터셋 (Dataset에 정규화 적용)
  - 정규화된 추론 엔진 (`inference_normalized.py`)
- ✅ **정규화된 모델 재학습**: 10 에포크 완료

### **7. 코드베이스 정리 및 Git 관리** ✅
- ✅ **디버그/분석 스크립트 정리**: `debug/` 폴더로 이동
- ✅ **.gitignore 업데이트**: fm-main/, policy_env/, debug/ 폴더 등
- ✅ **Git 캐시 정리**: 불필요한 파일 제거
- ✅ **원격 저장소 푸시**: 깔끔한 코드베이스 반영

---

## 📊 현재 데이터 현황

### **환경 데이터**
- **포인트클라우드**: `data/pointcloud/circle_envs_10k/` (10,000개)
- **포즈 페어**: `data/pose_pairs/circle_envs_10k/` (10,000개)

### **궤적 데이터** 
- **RRT 궤적**: `data/trajectories/circle_envs_10k/` (**3,241개 완료**)
  - 파일명: `circle_env_XXXXXX_pair_1_traj_rb3.json`
  - 환경: 000000-003240 (정리된 환경)
  - 포즈페어: 각 환경의 첫 번째 페어 사용
  - collision margin: 0.05m

- **B-spline 궤적**: `data/trajectories/circle_envs_10k_bsplined/` (**3,241개 완료**)
  - 파일명: `circle_env_XXXXXX_pair_1_traj_rb3_bsplined.json`
  - 실시간 T_dot 계산: body frame 기준 6D twist 벡터
  - 정리된 고품질 데이터셋

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

## 🎯 현재 상황 (2025.01.11)

### **🚨 현재 문제: 정규화 모델 추론 실패**
```bash
# 정규화된 모델 추론 테스트 결과
cd packages/policy/v1
source ../policy_env/bin/activate
python inference_test.py

# 결과: 심각한 문제 발견
# 🎯 시작: [4.139, 0.279, 0.000] → 목표: [8.439, 4.652, 0.000] (오른쪽 위)
# 🔴 실제: [3.269, -0.459, 0.000] (**왼쪽 아래로!**)
# 📏 극도로 작은 이동: 평균 5.7mm/스텝
# 🔄 일정한 패턴 반복: 방향성 학습 실패
```

### **✅ 완료된 작업들**
- **정규화 파이프라인**: 완전한 양방향 정규화 구현
- **모델 재학습**: 정규화된 데이터로 10 에포크 완료
- **추론 엔진**: 정규화 적용된 추론 시스템 구현
- **코드베이스 정리**: Git 관리 및 구조 정리 완료

### **🔍 문제 진단 결과**
1. **방향성 문제**: 모델이 목표와 반대 방향으로 이동
2. **스케일 문제**: 정규화 후에도 여전히 작은 twist 값
3. **학습 부족**: 10 에포크로는 복잡한 환경에서 방향성 학습 어려움
4. **환경 인식 문제**: 포인트클라우드 정보를 제대로 활용하지 못함

### **🎯 다음 단계**
1. **학습 강화**: 에포크 수 증가 (50-100)
2. **데이터 검증**: 학습 데이터의 정답 라벨 품질 확인
3. **모델 검증**: velocity field 방향성 분석
4. **단계적 테스트**: 간단한 환경에서 시작하여 복잡도 증가

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

### **모델 관련 (packages/policy/v1)**
```bash
cd packages/policy/v1

# 가상환경 활성화
source ../policy_env/bin/activate

# 정규화된 모델 추론 테스트
python inference_test.py

# 학습 재시작 (필요시)
bash start_normalized_training.sh

# tmux 세션 확인
tmux list-sessions
tmux attach-session -t motion_training_10epochs

# 학습 진행 확인
tail -f training_tmux.log
```

---

## 📈 성공 기준

### **데이터 완료도**
- ✅ **3,241개 환경**: 고품질 궤적 완료
- ✅ **데이터 정리**: 일관성 검증, 오류 처리 완료
- ✅ **T_dot 계산**: body frame 기준 정확한 계산

### **모델 성능**  
- ✅ **정규화 파이프라인**: 완전한 양방향 정규화 구현
- ✅ **학습 완료**: 정규화된 데이터로 10 에포크 완료
- 🚨 **추론 실패**: 방향성 문제, 스케일 문제 해결 필요
- 🎯 **목표**: 90% 충돌 없는 궤적 생성, RRT 대비 10x 빠른 추론

### **시스템 안정성**
- ✅ **안정적 학습 환경**: tmux + 견고한 오류 처리
- ✅ **Wandb 로깅**: 실시간 모니터링
- ✅ **재현 가능한 실험**: 완전한 설정 파일
- ✅ **코드베이스 관리**: Git 기반 버전 관리

---

## 🚨 현재 이슈 및 해결 방안

### **핵심 문제: 정규화 모델 추론 실패**
- **증상**: 완전히 잘못된 방향으로 이동, 극도로 작은 이동 스케일
- **원인**: 10 에포크로는 복잡한 환경에서 방향성 학습 어려움
- **해결 방안**: 
  1. 학습 에포크 증가 (50-100)
  2. 학습 데이터 품질 검증
  3. 단계적 테스트 (간단한 환경부터)

### **기술적 성과**
- ✅ **정규화 파이프라인**: 완전한 구현 및 검증
- ✅ **추론 시스템**: ODE 적분 기반 궤적 생성
- ✅ **데이터 처리**: 견고한 오류 처리 및 fallback 시스템
- ✅ **개발 환경**: 안정적인 학습 및 테스트 환경

---

## 📚 참고 자료

- **Riemannian Flow Matching**: Manifold-aware generative modeling
- **SE(3) Lie Groups**: Rigid body transformations
- **B-spline Curves**: Smooth trajectory interpolation
- **OMPL**: Open Motion Planning Library
- **FM-Main**: Reference RCFM implementation

---

**현재 상태**: 정규화 모델 추론 실패 원인 분석 및 해결 진행 중 🚨 (2025.01.11)

**다음 마일스톤**: 추론 성공 및 궤적 생성 품질 개선 🎯