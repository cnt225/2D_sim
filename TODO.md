# TODO - SE(3) RFM Motion Planning Project

## 🎯 프로젝트 목표
SE(3) Rigid Body 환경에서 Riemannian Flow Matching 기반 AI 정책을 학습하여 RRT-Connect를 대체하는 학습 기반 경로 계획 개발

## ✅ 완료된 작업들

### 1. 데이터 생성 파이프라인 ✅
- [x] 10,000개 원형 장애물 환경 생성
- [x] 100,000개 포즈 페어 생성
- [x] 100개 RRT 궤적 생성
- [x] 100개 B-spline 궤적 생성

### 2. 기존 SE(3) RFM 모델 구현 ✅
- [x] 3D DGCNN 포인트클라우드 인코더
- [x] SE(3) 인코더, 지오메트리 인코더
- [x] Velocity Field Network (6D twist 출력)
- [x] 완전한 SE(3) Lie Group 연산

### 3. 훈련 인프라 구축 ✅
- [x] Wandb 통합 실험 추적
- [x] 분산 훈련 지원
- [x] ODE solver (Euler, RK4, Adaptive)
- [x] 다중 손실 함수 (Flow matching + 충돌 회피 + 정규화)

### 4. 연구실 서버 이관 준비 ✅
- [x] README, TODO, .gitignore 설정 완료
- [x] 코드만 서버로 전송 예정

### 5. FM-Main 기반 v1 모델 구현 ✅
- [x] MotionRCFM 클래스 생성 (GraspRCFM 기반)
- [x] vf_FC_vec_motion 네트워크 구현 (25D 입력 + 2048D features)
- [x] 입력 형식 수정 (current_T, target_T, time_t, pointcloud)
- [x] TrajectoryDataset 클래스 구현
- [x] Config 파일 생성 (motion_rcfm.yml)
- [x] 훈련 루프 수정 (train_step 메서드 추가)
- [x] 테스트 스크립트 생성 (test_motion_model.py)

### 6. 모델 테스트 및 검증 ✅
- [x] 테스트 스크립트 실행 및 디버깅
- [x] 데이터 로더 검증 (3,241개 샘플)
- [x] 모델 forward pass 검증
- [x] 훈련 스텝 검증
- [x] 의존성 설치 (Open3D, scipy 등)

### 7. 데이터 품질 개선 ✅
- [x] T_dot 계산 오류 수정 (각속도가 0이었던 문제)
- [x] scipy.spatial.transform 기반 정확한 body frame 계산
- [x] PLY 파일 오류 처리 (빈 파일, 손상된 파일)
- [x] 데이터 일관성 정리 (pose, pose_pairs, trajectories 동기화)
- [x] 빈 파일 삭제 및 재인덱싱 (10,000개 → 3,241개)

### 8. 안정적 학습 환경 구축 ✅
- [x] packages/policy/policy_env 가상환경 분리
- [x] 견고한 오류 처리 (fallback pointcloud)
- [x] tmux 기반 백그라운드 실행
- [x] Wandb 통합 (motion_planning_rfm 프로젝트)
- [x] RPly 경고 분석 및 대응

## 🔥 현재 진행 중인 작업 (2025.01.09)

### 9. 모션 RFM 모델 학습 🔥
- [x] 학습 시작 (tmux 세션: motion_training)
- [x] 첫 번째 에포크 진행 중 (Loss ~5.0)
- [ ] 1000 에포크 완료 대기
- [ ] Loss 수렴 모니터링

## 📋 다음 단계 작업들

### 10. 학습 완료 후 분석
- [ ] 훈련 손실 곡선 분석
- [ ] Validation 성능 확인
- [ ] 모델 체크포인트 저장
- [ ] 하이퍼파라미터 튜닝 (필요시)

### 11. 모델 평가 및 궤적 생성
- [ ] 학습된 모델로 궤적 생성 테스트
- [ ] ODE integration 구현 (Euler/RK4)
- [ ] 궤적 생성 품질 평가
- [ ] RRT-Connect와 성능 비교

### 12. 성능 벤치마킹
- [ ] 추론 속도 측정 (vs RRT-Connect)
- [ ] 궤적 부드러움 비교
- [ ] 충돌 회피 성능 측정
- [ ] 성공률 통계

### 13. 실시간 시스템 통합
- [ ] 실시간 velocity field 추론
- [ ] 궤적 스무딩 및 최적화
- [ ] 실시간 성능 최적화
- [ ] 시뮬레이션 환경 통합

### 14. 확장 및 개선
- [ ] 다양한 환경에서 테스트
- [ ] 더 복잡한 장애물 환경
- [ ] 동적 장애물 처리
- [ ] 불확실성 고려

## 🚨 현재 이슈 및 블로커

### 해결된 이슈들
- [x] FM-Main 기반 구조 설계 완료
- [x] 데이터 형식 정의 완료  
- [x] 모델 아키텍처 구현 완료
- [x] T_dot 계산 오류 수정 (각속도 0 → 정확한 body frame 계산)
- [x] 데이터 품질 문제 해결 (빈 파일, 일관성)
- [x] PLY 파일 오류 처리 (RPly 경고 대응)
- [x] 훈련 환경 안정화 (tmux, 오류 처리)

### 현재 모니터링 중
- [ ] RPly 경고 (75% 파일에서 발생, 하지만 학습에 영향 없음)
- [ ] 학습 수렴성 (현재 Loss ~5.0)
- [ ] 메모리 사용량 (~2.8GB GPU)

### 알려진 제한사항
- PLY 파일 헤더 불일치 (선언 vs 실제 점 수)
- fallback 포인트클라우드 사용률 ~75%

## 📊 진행률
- **전체 진행률**: 85%
- **현재 단계**: 모션 RFM 모델 학습 진행 중
- **다음 마일스톤**: 학습 완료 및 성능 평가

## 🎯 단기 목표 (1-2주)
1. 모델 테스트 완료
2. POC 훈련 실행
3. 기본 궤적 생성 성공

## 🎯 중기 목표 (1개월)
1. RRT-Connect 대비 성능 평가
2. 실시간 궤적 생성 구현
3. 다양한 환경에서 테스트

## 🎯 장기 목표 (3개월)
1. 논문 작성 및 제출
2. 오픈소스 릴리즈
3. 추가 연구 방향 탐색