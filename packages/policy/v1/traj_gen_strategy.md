# 🚀 Motion RFM 궤적 생성 전략

## 📋 개요

Riemannian Flow Matching 기반 모션 플래닝에서 학습된 모델을 사용하여 실시간 궤적을 생성하는 전략을 정의합니다.

## 🎯 핵심 아이디어

**"RFM이 예측하는 것은 optimal flow direction"**
- 각 시점에서 **어느 방향으로 움직여야 하는지** 알려줌 (6D twist vector)
- 그 방향을 따라가면서 **자연스럽게 목표에 수렴**
- **적분 과정에서 부드러운 궤적이 자동으로 생성됨**

## 🔄 궤적 생성 파이프라인

### 1️⃣ 입력 & 초기화
```
입력:
- start_pose: SE(3) 4x4 matrix (시작 포즈)
- target_pose: SE(3) 4x4 matrix (목표 포즈)  
- pointcloud: Nx3 points (환경 포인트클라우드)
- config: 생성 파라미터

초기화:
- current_pose = start_pose
- trajectory = [current_pose]
- step = 0
```

### 2️⃣ 메인 생성 루프
```python
while step < max_steps:
    # 1. Progress 계산 (거리 기반)
    progress = calculate_progress(current_pose, start_pose, target_pose)
    
    # 2. 모델 추론: 6D twist vector 예측
    twist_6d = model(current_pose, target_pose, progress, pointcloud)
    
    # 3. 목표 도달 확인 (조기 종료)
    if reached_target(current_pose, target_pose):
        break
    
    # 4. SE(3) 적분: twist → next pose
    current_pose = integrate_se3(current_pose, twist_6d, dt)
    
    # 5. 궤적에 추가 & 안전 체크
    trajectory.append(current_pose)
    step += 1
```

### 3️⃣ 출력
```
결과:
- trajectory: List[SE(3)] - 궤적 포즈들
- success: bool - 목표 도달 성공 여부
- info: dict - 생성 통계 및 디버깅 정보
```

## ⚙️ 핵심 함수들

### A. Progress 계산 (거리 기반)
```python
def calculate_progress(current, start, target):
    total_dist = pose_distance(start, target)
    current_dist = pose_distance(current, target)
    progress = 1.0 - (current_dist / total_dist)
    return torch.clamp(progress, 0.0, 1.0)
```

### B. 목표 도달 판별 (Threshold 기반)
```python
def reached_target(current, target, pos_tol=0.02, rot_tol=0.1):
    pos_error = torch.norm(current[:3, 3] - target[:3, 3])
    rot_error = rotation_angle_between(current[:3, :3], target[:3, :3])
    return (pos_error < pos_tol) and (rot_error < rot_tol)
```

### C. SE(3) 적분 (Exponential Map)
```python
def integrate_se3(pose, twist_6d, dt):
    w = twist_6d[:3]  # angular velocity (body frame)
    v = twist_6d[3:]  # linear velocity (body frame)
    
    # SE(3) exponential map
    w_skew = skew_symmetric(w * dt)
    R_delta = torch.matrix_exp(w_skew)
    
    # 새로운 pose 계산
    new_pose = pose.clone()
    new_pose[:3, :3] = pose[:3, :3] @ R_delta  # rotation 업데이트
    new_pose[:3, 3] = pose[:3, 3] + pose[:3, :3] @ (v * dt)  # position 업데이트
    
    return new_pose
```

## 🎛️ 설정 전략

### 기본 설정 (균형형)
```python
DEFAULT_CONFIG = {
    'dt': 0.02,                    # 적당한 정밀도
    'max_steps': 100,              # 2초 최대 시간
    'pos_tolerance': 0.02,         # 2cm 허용 오차
    'rot_tolerance': 0.1,          # 5.7도 허용 오차
    'early_stop': True,            # 도달 시 조기 종료
    'safety_check': True,          # 발산/충돌 체크
}
```

### 고품질 설정 (정밀형)
```python
HIGH_QUALITY_CONFIG = {
    'dt': 0.005,                   # 고정밀 적분
    'max_steps': 400,              # 충분한 시간
    'pos_tolerance': 0.005,        # 5mm 허용 오차
    'rot_tolerance': 0.05,         # 3도 허용 오차
    'early_stop': True,
    'safety_check': True,
}
```

### 고속 생성 설정 (효율형)
```python
FAST_CONFIG = {
    'dt': 0.05,                    # 큰 스텝
    'max_steps': 50,               # 빠른 종료
    'pos_tolerance': 0.05,         # 5cm 허용 오차
    'rot_tolerance': 0.2,          # 11도 허용 오차
    'early_stop': True,
    'safety_check': False,         # 속도 우선
}
```

## 🛡️ 안전 장치

### 다중 정지 조건
1. **목표 도달**: threshold 기반 성공 판별
2. **최대 스텝**: 무한 루프 방지
3. **발산 감지**: 목표에서 너무 멀어지면 중단
4. **충돌 감지**: 장애물과 충돌 시 중단 (선택적)

### 품질 검증
- **최종 오차**: 실제 도달 정확도 측정
- **궤적 부드러움**: jerk, acceleration 분석
- **스텝 효율성**: 생성 시간 vs 품질 트레이드오프

## 📊 성능 지표

### 성공률 지표
- **도달 성공률**: 허용 오차 내 목표 도달 비율
- **평균 스텝 수**: 효율성 측정
- **평균 생성 시간**: 실시간성 평가

### 품질 지표
- **최종 위치 오차**: ||final_pos - target_pos||
- **최종 회전 오차**: angle_between(final_rot, target_rot)
- **궤적 길이**: 전체 경로 길이
- **부드러움**: 속도/가속도 변화율

## 🔬 실험 계획

### 하이퍼파라미터 튜닝
```python
dt_candidates = [0.005, 0.01, 0.02, 0.05]
tolerance_candidates = [
    (0.005, 0.05),  # 고정밀
    (0.02, 0.1),    # 기본
    (0.05, 0.2),    # 관대
]
```

### 평가 데이터셋
- **테스트 환경**: 100-500개 새로운 환경
- **거리 변화**: 근거리/중거리/원거리 목표
- **복잡도 변화**: 단순/보통/복잡한 환경

## 🎯 기대 효과

### RRT-Connect 대비 장점
1. **속도**: 수초 → 밀리초 (1000x 빠름)
2. **품질**: 꺾이는 궤적 → 부드러운 곡선
3. **일관성**: 랜덤 → 학습된 최적 패턴
4. **확장성**: 다양한 환경에 일반화

### 실용적 가치
- **실시간 로봇 제어**: 즉석 궤적 생성
- **동적 환경 대응**: 빠른 재계획
- **고품질 경로**: 자연스러운 움직임

---

**핵심 철학**: "Flow를 따라가면 자연스럽게 목표에 도달한다" 🌊➡️🎯


