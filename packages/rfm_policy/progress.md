# SE(3) RFM Policy Development Progress

## 📋 프로젝트 개요

**목표**: FM-Main의 Riemannian Conditional Flow Matching (RCFM) 모델을 SE(3) rigid body 제어 시스템에 적용  
**기간**: 2024-07-22 ~ 2024-07-30  
**레퍼런스**: [FM-Main](../../fm-main/) Grasping RCFM 모델

---

## 🔄 개발 버전 히스토리

### **V1: 초기 설계 및 구버전 구현** (2024-07-22)
*구버전 SE(3) RCFM 모델 설계 및 구현 과정*

**주요 특징:**
- 2D 환경 처리 (PointCloud → 2D DGCNN)
- 3DOF 제어 (x, y, yaw)
- Conditional Flow Matching 기반
- 구버전 아키텍처 (se3_rcfm.py, train.py 등)

**핵심 설계:**
- 환경 인코더: DGCNN_2D (512차원 출력)
- 로봇 지오메트리 인코더: 타원체 파라미터 (4차원)
- 상대 좌표 변환: 로봇 중심 좌표계
- Velocity Field: 3차원 출력 (vx, vy, omega_z)

**구현된 파일들:**
- `models/se3_rcfm.py` (삭제됨)
- `train.py` (삭제됨)
- `data_interface/se3_dataset.py` (삭제됨)
- `configs/se3_rcfm.yml` (삭제됨)

---

### **V2: 완전한 SE(3) RFM 모델 구현** (2024-07-30)
*현재 구현된 완전한 SE(3) Riemannian Flow Matching 모델*

#### **주요 개선사항**
- **완전한 SE(3) 지원**: 6차원 twist vector 출력 (3D translation + 3D rotation)
- **3D Point Cloud 처리**: 실제 3D 환경에서의 장애물 회피
- **모듈화된 아키텍처**: PointCloudEncoder, GeometryEncoder, SE3Encoder, VelocityFieldNetwork 분리
- **완전한 훈련 시스템**: Wandb 통합, 평가 시스템, ODE 솔버 포함

#### **핵심 컴포넌트**
- **SE3RFM**: 메인 모델 (`models/se3_rfm.py`)
- **PointCloudEncoder**: 3D DGCNN 기반 환경 인코더
- **GeometryEncoder**: 타원체 로봇 지오메트리 인코더  
- **SE3Encoder**: SE(3) 포즈 인코더
- **VelocityFieldNetwork**: SE(3) twist 예측 네트워크

#### **기술적 특징**
- **SE(3) Lie Group 연산**: 완전한 SE(3) manifold 처리
- **다중 ODE 솔버**: Euler, RK4, Adaptive solver 지원
- **다중 손실 함수**: Flow matching + 충돌 회피 + 정규화
- **완전한 평가 시스템**: 성공률, 경로 효율성, 부드러움 등

#### **현재 상태**
- ✅ **구현 완료**: 모든 핵심 컴포넌트 구현됨
- ✅ **통합 테스트**: `test_se3_rfm.py`로 모든 기능 검증됨
- ✅ **문서화**: README.md에 완전한 사용법 및 아키텍처 설명
- 🔄 **다음 단계**: 대규모 데이터셋으로 훈련 및 성능 평가

#### **파일 구조**
```
packages/rfm_policy/
├── models/
│   ├── se3_rfm.py              # 메인 SE3RFM 모델
│   ├── modules.py              # 신경망 컴포넌트
│   └── __init__.py
├── losses/
│   ├── flow_matching_loss.py   # Flow matching loss
│   ├── collision_loss.py       # Collision avoidance loss
│   ├── regularization_loss.py  # Regularization loss
│   └── multi_task_loss.py      # Combined multi-task loss
├── trainers/
│   ├── optimizers.py           # 옵티마이저
│   └── schedulers.py           # 스케줄러
├── utils/
│   ├── se3_utils.py           # SE(3) 연산
│   └── ode_solver.py          # ODE 솔버
├── loaders/
│   └── se3_trajectory_dataset.py  # 데이터 로딩
├── evaluation/
│   └── evaluator.py           # 평가 시스템
├── configs/
│   └── se3_rfm_config.yaml    # 설정 파일
├── train_se3_rfm.py           # 훈련 스크립트
└── test_se3_rfm.py            # 통합 테스트
```

---

## 🎯 향후 개발 계획

### **V3: 성능 최적화 및 확장** (예정)
- **대규모 훈련**: 실제 데이터셋으로 모델 훈련
- **성능 평가**: 다양한 환경에서의 성능 측정
- **실시간 최적화**: 추론 속도 개선
- **실제 로봇 적용**: Sim-to-real transfer

### **V4: 고급 기능** (예정)
- **다중 로봇 지원**: Multi-agent 시나리오
- **동적 환경**: 움직이는 장애물 처리
- **3D 확장**: 완전한 3D 환경 지원
- **실제 배포**: 실제 로봇 시스템 적용

---

## 📝 개발 노트

### **주요 결정사항**
1. **V1 → V2 전환**: 구버전 RCFM에서 완전한 SE(3) RFM으로 전환
2. **모듈화 설계**: 각 컴포넌트를 독립적으로 개발 및 테스트 가능
3. **완전한 SE(3) 지원**: 6차원 twist vector로 완전한 3D 제어
4. **실용적 접근**: 실제 사용 가능한 훈련 및 평가 시스템 구축

### **학습한 교훈**
1. **설계 문서의 중요성**: V1의 상세한 설계가 V2 구현에 큰 도움
2. **모듈화의 가치**: 각 컴포넌트를 독립적으로 개발할 수 있어 효율적
3. **테스트의 중요성**: 통합 테스트로 구현 오류를 조기에 발견
4. **문서화의 필요성**: README.md로 사용법을 명확히 전달

---

*이 문서는 SE(3) RFM Policy 개발 과정의 히스토리를 기록하며, 향후 개발 방향을 제시합니다.* 