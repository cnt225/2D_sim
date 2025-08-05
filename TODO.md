# 📋 TODO: SE(3) Riemannian Flow Matching 프로젝트

**마지막 업데이트**: 2024년 (연구실 서버 이관 준비 완료)

---

## ✅ 완료된 작업

### **데이터 생성 파이프라인** ✅
- [x] **환경 생성**: 10,000개 원형 장애물 포인트클라우드 환경
- [x] **포즈 페어 생성**: 각 환경별 10개 포즈, 총 100,000개 페어
- [x] **RRT 궤적 생성**: 첫 100개 환경, collision margin 0.05m
- [x] **B-spline 스무딩**: SE(2) 매니폴드 기반, 2x 밀도, 90-100% 개선
- [x] **충돌 검증**: RRT + Post-B-spline 충돌 체크 완료
- [x] **시각화 시스템**: 실제 로봇 지오메트리 반영 (1.2×0.4m)

### **SE(3) RFM 모델 구현** ✅
- [x] **SE3RFM 아키텍처**: fm-main 스타일 Riemannian Flow Matching
- [x] **3D Point Cloud Encoder**: DGCNN 기반, 1024D→2048D 출력
- [x] **SE(3) Encoder**: 4×4 변환행렬 → 12D 직접 플래튼
- [x] **Geometry Encoder**: 타원체 파라미터 → 32D 임베딩
- [x] **Velocity Field Network**: 2105D → 6D twist 벡터
- [x] **SE(3) Utilities**: exp_map, log_map, compose, inverse 구현
- [x] **ODE Solver**: Euler, RK4, Adaptive 적분기
- [x] **Loss Functions**: Flow matching + collision + regularization
- [x] **Training Infrastructure**: Wandb, 분산 훈련, 설정 관리

### **코드 정리 및 문서화** ✅
- [x] **디렉토리 구조 통일**: fm-main 스타일 (loaders, losses, trainers)
- [x] **B-spline 코드 정리**: 단일 스크립트로 통합
- [x] **모델 아키텍처 문서화**: README, progress.md 업데이트
- [x] **Git 준비**: .gitignore 설정, 데이터 파일 제외

---

## 🔄 진행 중인 작업

### **연구실 서버 이관 준비** (현재)
- [x] **README.md 업데이트**: 환경 설정, 데이터 현황, 사용법
- [x] **TODO.md 작성**: 현재 상황 및 다음 단계 명시
- [x] **.gitignore 설정**: 데이터/결과 파일 제외
- [x] **pyproject.toml 업데이트**: 프로젝트 설명 최신화
- [ ] **Git push**: 코드만 연구실 서버로 전송
- [ ] **Ubuntu 환경 설정**: UV 설치, 의존성 설치

---

## 🎯 다음 단계 (연구실 서버에서 진행)

### **Phase 1: 모델 훈련 준비** (우선순위 1)
- [ ] **환경 검증**: Ubuntu에서 패키지 설치 및 임포트 확인
  ```bash
  cd packages/rfm_policy
  python test_se3_rfm.py  # 모델 로드 및 forward pass 테스트
  ```

- [ ] **데이터 로더 테스트**: 100개 궤적으로 데이터 로딩 확인
  ```bash
  python loaders/se3_trajectory_dataset.py  # 데이터셋 로딩 테스트
  ```

- [ ] **훈련 설정 조정**: RTX 4090 환경에 맞게 하이퍼파라미터 조정
  - Batch size, learning rate, GPU 메모리 최적화
  - Wandb 프로젝트 설정

### **Phase 2: 소규모 훈련 실험** (우선순위 2)
- [ ] **Proof of Concept**: 100개 궤적으로 모델 수렴 확인
  ```bash
  python train_se3_rfm.py --config configs/se3_rfm_config.yaml --debug
  ```

- [ ] **Loss 모니터링**: Flow matching loss 수렴 여부 확인
- [ ] **Overfitting 테스트**: 단일 궤적에 모델 오버피팅 가능한지 확인
- [ ] **성능 벤치마크**: 추론 속도, 메모리 사용량 측정

### **Phase 3: 데이터 스케일링** (우선순위 3)
- [ ] **1,000개 환경**: 다음 900개 환경에 대해 RRT + B-spline 생성
  ```bash
  cd packages/data_generator/reference_planner
  python generate_range_1000.py  # 환경 000100-000999
  ```

- [ ] **10,000개 환경**: 전체 데이터셋 완성
  ```bash
  python generate_all_10k.py  # 나머지 9,000개 환경
  ```

- [ ] **데이터 검증**: 충돌 체크, 궤적 품질 확인
- [ ] **데이터 분할**: Train/Validation/Test split (8:1:1)

### **Phase 4: 대규모 훈련** (우선순위 4)
- [ ] **분산 훈련 설정**: Multi-GPU 훈련 (if available)
- [ ] **하이퍼파라미터 튜닝**: Learning rate, batch size, regularization
- [ ] **모델 비교**: 다양한 아키텍처 실험
- [ ] **수렴 분석**: Loss curves, validation metrics

### **Phase 5: 평가 및 벤치마킹** (우선순위 5)
- [ ] **RRT vs RFM 비교**: 성능, 속도, 궤적 품질
- [ ] **충돌 회피율**: 테스트셋에서 안전성 평가
- [ ] **실시간 성능**: 추론 속도 벤치마크
- [ ] **일반화 성능**: 새로운 환경에서 테스트

---

## 📊 데이터 현황 (Mac 로컬 → 서버 이관 예정)

### **완료된 데이터**
```
data/
├── pointcloud/circle_envs_10k/         # 10,000개 환경 (완료)
├── pose_pairs/circle_envs_10k/         # 100,000개 포즈 페어 (완료)
├── trajectories/
│   ├── circle_envs_10k/                # 100개 RRT 궤적 (완료)
│   └── circle_envs_10k_bsplined/       # 100개 B-spline 궤적 (완료)
└── visualizations/                     # 테스트 시각화 파일들
```

### **필요한 데이터 (서버에서 생성)**
- **900개 추가 RRT + B-spline**: 환경 000100-000999
- **9,000개 최종 데이터**: 환경 001000-009999
- **Train/Val/Test 분할**: 체계적인 데이터 관리

---

## 🛠️ 서버 설정 가이드

### **1. 프로젝트 클론 및 환경 설정**
```bash
# 프로젝트 클론
git clone <repository_url>
cd 2d_sim

# UV 설치 (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 가상환경 생성
uv venv
source .venv/bin/activate

# 의존성 설치
uv sync
```

### **2. 데이터 전송** (Mac → 서버)
```bash
# 로컬 (Mac)에서
rsync -avz --progress data/ server:~/2d_sim/data/
```

### **3. 환경 검증**
```bash
# 모델 로드 테스트
cd packages/rfm_policy
python test_se3_rfm.py

# GPU 확인
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🎯 성공 기준

### **단기 목표 (1-2주)**
- [ ] 서버 환경 설정 완료
- [ ] 100개 데이터로 모델 수렴 확인
- [ ] Wandb 로깅 정상 작동

### **중기 목표 (1개월)**
- [ ] 1,000개 데이터 생성 완료
- [ ] 안정적인 훈련 파이프라인 구축
- [ ] 기본 성능 벤치마크 완료

### **장기 목표 (2-3개월)**
- [ ] 10,000개 완전 데이터셋 훈련
- [ ] RRT 대비 성능 향상 달성
- [ ] 실시간 추론 성능 확보

---

## ⚠️ 주의사항

### **데이터 관리**
- **로컬 백업**: Mac에서 중요 데이터 백업 유지
- **서버 스토리지**: 10,000개 환경 = ~50GB 예상
- **Git 관리**: 데이터는 Git에 포함하지 않음 (.gitignore 설정 완료)

### **훈련 안정성**
- **메모리 모니터링**: RTX 4090 24GB VRAM 효율적 사용
- **Checkpoint 저장**: 정기적인 모델 체크포인트
- **실험 추적**: Wandb로 모든 실험 기록

### **성능 최적화**
- **Batch Size**: GPU 메모리에 맞게 조정
- **Data Loading**: 멀티프로세싱으로 I/O 병목 해결
- **Mixed Precision**: FP16 훈련으로 메모리 절약

---

**현재 상태**: 연구실 서버 이관 준비 완료 → 곧 모델 훈련 시작 🚀