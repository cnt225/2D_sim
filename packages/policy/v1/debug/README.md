# 🔍 Debug & Analysis Scripts

이 폴더는 Motion RFM 모델의 디버깅, 분석, 시각화를 위한 스크립트들을 포함합니다.

## 📁 파일 구조

```
debug/
├── README.md                           # 이 파일
├── training_data_analysis.py           # 학습 데이터 twist 통계 분석
├── velocity_field_analysis.py          # 모델 출력 velocity field 분석
├── velocity_field_visualization.py     # velocity field 2D 시각화
└── result/                             # 분석 결과 및 이미지
    └── velocity_field_visualization.png
```

## 🛠️ 스크립트 설명

### 1. `training_data_analysis.py`
**목적**: 학습 데이터셋의 twist 벡터 통계 분석
**기능**:
- 모든 궤적에서 twist 벡터 추출
- 선형/각속도 통계 계산 (평균, 표준편차, 최소/최대값)
- 정규화 통계 생성

**사용법**:
```bash
cd debug
python training_data_analysis.py
```

**출력 예시**:
```
📊 학습 데이터 Twist 통계:
   Angular velocity: 평균 0.9833 ± 1.3127 rad/s
   Linear velocity: 평균 5.3123 ± 2.4272 m/s
```

### 2. `velocity_field_analysis.py`
**목적**: 학습된 모델의 velocity field 품질 분석
**기능**:
- 다양한 거리와 진행도에서 twist 벡터 예측
- 방향성 및 크기 분석
- 학습 데이터와의 스케일 비교

**사용법**:
```bash
cd debug
python velocity_field_analysis.py
```

**출력 예시**:
```
🔍 Velocity Field 분석:
   거리 1.0m: twist 크기 0.06 m/s (방향 정확도: 낮음)
   거리 5.0m: twist 크기 0.06 m/s (방향 정확도: 낮음)
```

### 3. `velocity_field_visualization.py`
**목적**: 2D 평면에서 velocity field 시각화
**기능**:
- 그리드 기반 velocity field 생성
- 화살표로 방향과 크기 표시
- 이미지 파일로 저장

**사용법**:
```bash
cd debug
python velocity_field_visualization.py
```

**출력**: `result/velocity_field_visualization.png`

## 📊 분석 결과 요약

### **학습 데이터 특성**
- **선형 속도**: 평균 5.31 m/s (표준편차 2.43 m/s)
- **각속도**: 평균 0.98 rad/s (표준편차 1.31 rad/s)
- **데이터 품질**: 3,241개 고품질 궤적

### **모델 출력 특성**
- **스케일 문제**: 모델 출력이 학습 데이터보다 100배 작음
- **방향성 문제**: 목표와 반대 방향으로 이동
- **일관성**: 모든 상황에서 동일한 패턴 반복

### **문제 진단**
1. **학습 부족**: 10 에포크로는 복잡한 환경에서 방향성 학습 어려움
2. **정규화 한계**: 스케일은 개선되었지만 방향성 문제는 지속
3. **환경 인식**: 포인트클라우드 정보를 제대로 활용하지 못함

## 🔧 해결 방안

### **단기 해결책**
1. **학습 강화**: 에포크 수 증가 (50-100)
2. **데이터 검증**: 학습 데이터의 정답 라벨 품질 확인
3. **단계적 테스트**: 간단한 환경에서 시작

### **장기 개선 방향**
1. **모델 아키텍처**: 포인트클라우드 인코딩 개선
2. **손실 함수**: 방향성 학습을 위한 새로운 손실 함수
3. **데이터 증강**: 다양한 시나리오로 데이터셋 확장

## 📝 사용 시 주의사항

1. **가상환경**: `policy_env` 가상환경 활성화 필요
2. **모델 체크포인트**: 분석 전에 유효한 모델 체크포인트 확인
3. **GPU 메모리**: 대용량 데이터 분석 시 GPU 메모리 모니터링
4. **결과 해석**: 분석 결과는 모델의 현재 상태를 반영하며, 개선 후 재분석 필요

## 🔄 업데이트 이력

- **2025.01.11**: 초기 디버그 스크립트 생성 및 정리
- **2025.01.11**: 정규화 모델 추론 실패 원인 분석 완료
- **2025.01.11**: 문제 진단 및 해결 방안 문서화

---

**현재 상태**: 정규화 모델 추론 실패 원인 분석 완료 🚨

**다음 단계**: 해결 방안 구현 및 모델 성능 개선 🎯

